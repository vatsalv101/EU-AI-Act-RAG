# retrieval/rag_pipeline.py
import os
import sys
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

from retrieval.retriever import AIActRetriever

# Load API keys from .env
load_dotenv()

# --- Strict Legal Prompt Template ---
QA_PROMPT_TMPL = (
    "You are an expert, multilingual legal compliance assistant for the EU AI Act.\n"
    "You must follow these strict rules:\n"
    "1. ONLY use the context information provided below to answer the user's query.\n"
    "2. Do NOT use prior knowledge or external information.\n"
    "3. If the context does not contain enough information to answer the query, "
    "you must state EXACTLY: 'I cannot answer this based on the retrieved legal context.'\n"
    "4. Always cite the specific Section (e.g., 'Article 6') at the end of your answer "
    "based on the context provided.\n"
    "5. Answer in the same language as the user's query.\n\n"
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Query: {query_str}\n"
    "Answer: "
)

class AIActRAG:
    def __init__(self, top_k: int = 4):
        # 1. Initialize the LLM (gpt-4o-mini is fast, cheap, and excellent at following prompt constraints)
        self.llm = OpenAI(model="gpt-4o-mini", temperature=0.0)

        # 2. Initialize our custom retriever from Step 8
        self.custom_retriever = AIActRetriever(top_k=top_k)

        # 3. Create the Query Engine
        self.query_engine = RetrieverQueryEngine.from_args(
            retriever=self.custom_retriever.index.as_retriever(similarity_top_k=top_k),
            llm=self.llm
        )

        # 4. Update the engine with our strict legal prompt
        qa_prompt = PromptTemplate(QA_PROMPT_TMPL)
        self.query_engine.update_prompts(
            {"response_synthesizer:text_qa_template": qa_prompt}
        )

    def ask(self, query: str) -> dict:
        """
        Execute the complete RAG pipeline and return the answer + source nodes.
        """
        response = self.query_engine.query(query)

        # Extract sources for transparency
        sources = []
        for node in response.source_nodes:
            sources.append({
                "score": round(node.score, 4) if node.score else None,
                "section": node.node.metadata.get("section", "Unknown"),
                "text": node.node.text[:200] + "..."
            })

        return {
            "answer": str(response),
            "sources": sources
        }

if __name__ == "__main__":
    # Standalone test
    rag = AIActRAG()

    # Test 1: Legitimate Question
    print("--- TEST 1: Valid Query ---")
    res1 = rag.ask("What is a high-risk AI system?")
    print(f"Answer: {res1['answer']}\n")
    print("Sources used:")
    for s in res1['sources']:
        print(f" - {s['section']} (Score: {s['score']})")

    # Test 2: Out of Scope Question (Testing Abstention)
    print("\n--- TEST 2: Out of Scope Query ---")
    res2 = rag.ask("What does the EU AI Act say about self-driving Tesla cars on Mars?")
    print(f"Answer: {res2['answer']}\n")
