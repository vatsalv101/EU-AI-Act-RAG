# scripts/chunking_experiment.py
import os
import json
import tiktoken
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter

PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

def load_clean_data(lang="en"):
    filepath = os.path.join(PROCESSED_DATA_DIR, f"ai_act_{lang}_clean.json")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def create_llama_documents(data):
    """Convert our JSON dictionaries into LlamaIndex Document objects with metadata."""
    documents = []
    for item in data:
        doc = Document(
            text=item["content"],
            metadata={
                "document": item["document"],
                "language": item["language"],
                "section": item["section"],
                "source": item["source"]
            },
            # Exclude metadata from the LLM prompt to save tokens, but keep it for filtering
            excluded_llm_metadata_keys=["document", "source"],
            excluded_embed_metadata_keys=["document", "source"]
        )
        documents.append(doc)
    return documents

def run_chunking_experiments():
    print("Loading EN dataset...")
    en_data = load_clean_data("en")
    documents = create_llama_documents(en_data)
    print(f"Created {len(documents)} source Documents.")

    # --- Strategy 1: Fixed Token Splitting ---
    # Splits strictly by token count. Good for benchmarking.
    fixed_splitter = TokenTextSplitter(
        chunk_size=256,
        chunk_overlap=24,
        tokenizer=tiktoken.get_encoding("cl100k_base").encode
    )

    print("\nRunning Fixed Token Chunking...")
    fixed_nodes = fixed_splitter.get_nodes_from_documents(documents)
    print(f"Result: {len(fixed_nodes)} chunks created.")
    if fixed_nodes:
        print("Sample Fixed Chunk:\n", fixed_nodes[50].text[:150], "...\n")

    # --- Strategy 2: Sentence/Recursive Splitting ---
    # Respects sentence boundaries. Highly recommended for Legal RAG.
    sentence_splitter = SentenceSplitter(
        chunk_size=256,
        chunk_overlap=24
    )

    print("Running Sentence Boundary Chunking...")
    sentence_nodes = sentence_splitter.get_nodes_from_documents(documents)
    print(f"Result: {len(sentence_nodes)} chunks created.")
    if sentence_nodes:
        print("Sample Sentence Chunk:\n", sentence_nodes[50].text[:150], "...\n")

    # Check Metadata propagation
    if sentence_nodes:
        print("Metadata successfully propagated:", sentence_nodes[50].metadata)

if __name__ == "__main__":
    run_chunking_experiments()
