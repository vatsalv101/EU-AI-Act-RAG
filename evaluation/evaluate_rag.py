# evaluation/evaluate_rag.py
import os
import sys
import json
import time
from typing import List, Dict, Any
from dotenv import load_dotenv

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from retrieval.rag_pipeline import AIActRAG

load_dotenv()

EVAL_DIR = os.path.dirname(__file__)
DATASET_PATH = os.path.join(EVAL_DIR, "eval_dataset.json")
RESULTS_PATH = os.path.join(EVAL_DIR, "eval_results.json")

def section_matches(retrieved_section: str, expected_section: str) -> bool:
    """Check if 'Article 5' matches 'Article 5' or 'Article 5 - Prohibited...'"""
    if not retrieved_section or not expected_section:
        return False
    # Normalize non-breaking spaces (\xa0) and check exact prefix boundary
    ret = retrieved_section.replace('\xa0', ' ').strip().lower()
    exp = expected_section.replace('\xa0', ' ').strip().lower()
    return ret == exp or ret.startswith(exp + " ") or ret.startswith(exp + ":") or ret.startswith(exp + "-")

def compute_retrieval_metrics(retrieved_sections: List[str], expected_section: str) -> Dict[str, float]:
    """Compute Recall@k, Precision@k, and Reciprocal Rank for a single query."""
    if not expected_section:
        return {"recall": 0.0, "precision": 0.0, "rr": 0.0}

    hits = [1 if section_matches(sec, expected_section) else 0 for sec in retrieved_sections]

    recall = 1.0 if sum(hits) > 0 else 0.0
    precision = sum(hits) / len(retrieved_sections) if retrieved_sections else 0.0

    rr = 0.0
    for idx, hit in enumerate(hits, start=1):
        if hit == 1:
            rr = 1.0 / idx
            break

    return {
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "rr": round(rr, 4)
    }

def run_evaluation(top_k: int = 4, run_ragas: bool = True):
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    rag = AIActRAG(top_k=top_k)

    per_query_results = []
    in_scope_records = []
    oos_correct = 0
    oos_total = 0

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    has_openai = bool(openai_key and not openai_key.startswith("sk-proj-your-actual"))

    print(f"Running evaluation on {len(dataset)} benchmark queries (top_k={top_k})...\n")

    for item in dataset:
        q_id = item["id"]
        question = item["question"]
        filter_lang = item.get("filter_language")
        expected_sec = item.get("expected_section")
        category = item["category"]

        start_t = time.time()
        # 1. Run Filtered Retrieval
        nodes = rag.custom_retriever.retrieve(query=question, language=filter_lang)
        retrieval_ms = round((time.time() - start_t) * 1000, 2)

        retrieved_sections = [n.node.metadata.get("section", "Unknown") for n in nodes]
        contexts = [n.node.text for n in nodes]

        # 2. Run End-to-End RAG Generation
        if has_openai:
            try:
                rag_output = rag.ask(question)
                answer = rag_output["answer"]
            except Exception as e:
                answer = f"Error during generation: {e}"
        else:
            answer = "LLM generation skipped (OPENAI_API_KEY not configured in .env)."

        record: Dict[str, Any] = {
            "id": q_id,
            "category": category,
            "question": question,
            "expected_section": expected_sec,
            "retrieved_sections": retrieved_sections,
            "retrieval_latency_ms": retrieval_ms,
            "answer": answer
        }

        if category == "out_of_scope":
            oos_total += 1
            abstained = "cannot answer this based on the retrieved legal context" in answer.lower()
            if abstained:
                oos_correct += 1
            record["abstained_correctly"] = abstained
            print(f"[{q_id}] Out-of-Scope | Abstained Correctly: {abstained}")
        else:
            metrics = compute_retrieval_metrics(retrieved_sections, expected_sec)
            record.update(metrics)
            in_scope_records.append({
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": item["ground_truth"],
                **metrics,
                "latency_ms": retrieval_ms
            })
            print(f"[{q_id}] Recall@{top_k}: {metrics['recall']} | MRR: {metrics['rr']} | Retrieved: {retrieved_sections}")

        per_query_results.append(record)

    # Aggregate Deterministic Metrics
    avg_recall = round(sum(r["recall"] for r in in_scope_records) / len(in_scope_records), 4) if in_scope_records else 0.0
    avg_precision = round(sum(r["precision"] for r in in_scope_records) / len(in_scope_records), 4) if in_scope_records else 0.0
    mrr = round(sum(r["rr"] for r in in_scope_records) / len(in_scope_records), 4) if in_scope_records else 0.0
    avg_latency = round(sum(r["latency_ms"] for r in in_scope_records) / len(in_scope_records), 2) if in_scope_records else 0.0
    abstention_acc = round(oos_correct / oos_total, 4) if oos_total > 0 else 0.0

    summary = {
        "top_k": top_k,
        "num_in_scope_queries": len(in_scope_records),
        "num_out_of_scope_queries": oos_total,
        f"recall_at_{top_k}": avg_recall,
        f"precision_at_{top_k}": avg_precision,
        "mrr": mrr,
        "avg_retrieval_latency_ms": avg_latency,
        "abstention_accuracy": abstention_acc
    }

    # Optional: Run RAGAS Faithfulness & Answer Relevance on In-Scope Queries
    if run_ragas and has_openai:
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import faithfulness, answer_relevancy

            print("\nRunning RAGAS LLM-as-a-Judge evaluation (Faithfulness & Answer Relevancy)...")
            ragas_ds = Dataset.from_dict({
                "question": [r["question"] for r in in_scope_records],
                "answer": [r["answer"] for r in in_scope_records],
                "contexts": [r["contexts"] for r in in_scope_records],
                "ground_truth": [r["ground_truth"] for r in in_scope_records]
            })
            ragas_res = evaluate(ragas_ds, metrics=[faithfulness, answer_relevancy])
            summary["ragas_faithfulness"] = round(float(ragas_res["faithfulness"]), 4)
            summary["ragas_answer_relevancy"] = round(float(ragas_res["answer_relevancy"]), 4)
        except Exception as e:
            print(f"⚠️ RAGAS evaluation skipped or encountered an error: {e}")
    elif not has_openai:
        print("\nℹ️ OPENAI_API_KEY is not set in .env: RAGAS LLM-as-a-Judge evaluation skipped.")

    output_payload = {
        "summary": summary,
        "details": per_query_results
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    print("\n==========================================")
    print("📊 EVALUATION SUMMARY")
    print("==========================================")
    print(json.dumps(summary, indent=2))
    print(f"\n✅ Detailed results saved to {RESULTS_PATH}")

if __name__ == "__main__":
    run_evaluation(top_k=4, run_ragas=True)
