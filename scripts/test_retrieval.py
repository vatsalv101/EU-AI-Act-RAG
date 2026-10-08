# scripts/test_retrieval.py
import os
import sys

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from retrieval.retriever import AIActRetriever

def run_retrieval_tests():
    retriever = AIActRetriever(top_k=2)

    test_cases = [
        {"query": "What are prohibited AI practices?", "lang": "en"},
        {"query": "Welche KI-Praktiken sind verboten?", "lang": "de"},
        {"query": "High-risk AI system classification criteria", "lang": "en"}
    ]

    for case in test_cases:
        print(f"\n==========================================")
        print(f"Query: '{case['query']}' [Filter Lang: {case['lang']}]")
        print(f"==========================================")
        nodes = retriever.retrieve(query=case["query"], language=case["lang"])

        for i, n in enumerate(nodes, 1):
            score = round(n.score, 4) if n.score is not None else "N/A"
            print(f"Hit #{i} (Score: {score})")
            print(f"Section : {n.node.metadata.get('section')}")
            print(f"Text    : {n.node.text[:120]}...\n")

if __name__ == "__main__":
    run_retrieval_tests()
