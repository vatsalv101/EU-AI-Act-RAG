# retrieval/rag_pipeline.py
import os
import sys
import types
from dotenv import load_dotenv
from llama_index.core import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai import OpenAI

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# NEW: Import the Cross-Encoder Reranker
try:
    from llama_index.postprocessor.cross_encoder_rerank import CrossEncoderRerank
except (ImportError, ModuleNotFoundError):
    from llama_index.core.postprocessor import SentenceTransformerRerank as CrossEncoderRerank
    if "llama_index.postprocessor.cross_encoder_rerank" not in sys.modules:
        _m = types.ModuleType("llama_index.postprocessor.cross_encoder_rerank")
        _m.CrossEncoderRerank = CrossEncoderRerank
        sys.modules["llama_index.postprocessor.cross_encoder_rerank"] = _m

from retrieval.retriever import AIActRetriever

load_dotenv()

QA_PROMPT_TMPL = (
    "You are an expert, multilingual legal compliance assistant for the EU AI Act.\n"
    "You must follow these strict rules:\n"
    "1. ONLY use the context information provided below to answer the user's query.\n"
    "2. Do NOT use prior knowledge or external information.\n"
    "3. If the context does not contain enough information to answer the query, "
    "you must state EXACTLY: 'I cannot answer this based on the retrieved legal context.'\n"
    "4. Always cite the specific Section (e.g., 'Article 6') at the end of your answer.\n"
    "5. Answer in the same language as the user's query.\n\n"
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Query: {query_str}\n"
    "Answer: "
)

# Ensure gpt-4o and gpt-4o-mini are registered in llama-index openai model dictionary
import llama_index.llms.openai.utils as openai_utils
if hasattr(openai_utils, "ALL_AVAILABLE_MODELS"):
    openai_utils.ALL_AVAILABLE_MODELS.setdefault("gpt-4o-mini", 128000)
    openai_utils.ALL_AVAILABLE_MODELS.setdefault("gpt-4o", 128000)

class AIActRAG:
    def __init__(self, top_k_retrieve: int = 15, top_k_rerank: int = 4, top_k: int = None):
        if top_k is not None:
            top_k_rerank = top_k

        self.top_k_retrieve = top_k_retrieve
        self.top_k_rerank = top_k_rerank
        self.llm = OpenAI(model="gpt-4o-mini", temperature=0.0)

        # Stage 1: Retrieve a wider net of candidates (e.g., 15)
        self.custom_retriever = AIActRetriever(top_k=top_k_retrieve)

        # Stage 2: Initialize the Multilingual Cross-Encoder Reranker
        # This model is specifically trained to score query-document pairs across languages.
        print("Initializing Cross-Encoder Reranker (BAAI/bge-reranker-v2-m3)...")
        self.reranker = CrossEncoderRerank(
            model="BAAI/bge-reranker-v2-m3",
            top_n=top_k_rerank
        )

        # Construct the Query Engine with the node postprocessor
        self.query_engine = RetrieverQueryEngine.from_args(
            retriever=self.custom_retriever.index.as_retriever(similarity_top_k=top_k_retrieve),
            node_postprocessors=[self.reranker], # Inject the reranker here
            llm=self.llm
        )

        qa_prompt = PromptTemplate(QA_PROMPT_TMPL)
        self.query_engine.update_prompts(
            {"response_synthesizer:text_qa_template": qa_prompt}
        )

    def ask(self, query: str, filter_language: str = None) -> dict:
        """
        If filter_language is provided, we bypass the standard QueryEngine 
        to manually pass filters to our custom retriever, run the reranker, 
        and synthesize the response. For simplicity in the API, we will just 
        use the standard query method without hard filters, relying on the 
        cross-encoder to naturally pull the right language.
        """
        try:
            response = self.query_engine.query(query)
            answer_text = str(response)
            source_nodes = response.source_nodes
        except Exception as e:
            # Fallback if OPENAI_API_KEY is not configured or fails
            from llama_index.core.schema import QueryBundle
            qb = QueryBundle(query_str=query)
            retriever = self.custom_retriever.index.as_retriever(similarity_top_k=self.top_k_retrieve)
            retrieved_nodes = retriever.retrieve(qb)
            source_nodes = self.reranker.postprocess_nodes(retrieved_nodes, query_bundle=qb)
            answer_text = f"[LLM generation unavailable without OPENAI_API_KEY: {e}]"

        sources = []
        for node in source_nodes:
            sources.append({
                "score": round(float(node.score), 4) if node.score is not None else None,
                "section": node.node.metadata.get("section", "Unknown"),
                "language": node.node.metadata.get("language", "Unknown"),
                "text": node.node.text[:150] + "..."
            })

        return {
            "answer": answer_text,
            "sources": sources
        }

if __name__ == "__main__":
    # The first run will download the BAAI reranker model weights.
    rag = AIActRAG()

    print("\n--- TEST: Cross-Lingual Two-Stage Retrieval ---")
    # We ask a German question, hoping the reranker accurately scores English documents as well.
    res = rag.ask("Was sind die Transparenzpflichten für Anbieter von KI-Systemen?")
    print(f"Answer: {res['answer']}\n")
    print("Reranked Sources used:")
    for s in res['sources']:
        print(f" - [{s['language'].upper()}] {s['section']} (Reranker Score: {s['score']})")
