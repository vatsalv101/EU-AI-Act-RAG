# scripts/index_qdrant.py
import os
import json
from qdrant_client import QdrantClient
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore

PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
DB_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "qdrant_db")

def load_clean_data(lang="en"):
    filepath = os.path.join(PROCESSED_DATA_DIR, f"ai_act_{lang}_clean.json")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    for item in data:
        doc = Document(
            text=item["content"],
            metadata={
                "language": item["language"],
                "section": item["section"],
                "source": item["source"]
            },
            # Keep LLM prompt tokens clean
            excluded_llm_metadata_keys=["source"],
            # Prevent metadata strings from polluting the embedding math
            excluded_embed_metadata_keys=["source", "language"]
        )
        documents.append(doc)
    return documents

def main():
    os.makedirs(DB_DIR, exist_ok=True)

    print("Initializing Embedding Model (MiniLM)...")
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    print("Loading Documents...")
    all_docs = load_clean_data("en") + load_clean_data("de")

    print("Setting up Qdrant Client...")
    # Initialize Qdrant locally on disk
    client = QdrantClient(path=DB_DIR)

    # Create a specialized vector store collection
    vector_store = QdrantVectorStore(client=client, collection_name="eu_ai_act")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Define our optimal chunking strategy
    splitter = SentenceSplitter(chunk_size=256, chunk_overlap=24)

    print(f"Indexing {len(all_docs)} documents into Qdrant. This will take a few minutes...")

    # Execute the ingestion pipeline
    VectorStoreIndex.from_documents(
        all_docs,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=[splitter],
        show_progress=True
    )

    print("[SUCCESS] Indexing complete. Vectors stored persistently in data/qdrant_db")

if __name__ == "__main__":
    main()
