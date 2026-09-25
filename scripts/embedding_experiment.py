# scripts/embedding_experiment.py
import os
import json
import time
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

def get_sample_nodes(lang="en", sample_size=10):
    """Quickly load data, parse into nodes, and return a small sample."""
    filepath = os.path.join(PROCESSED_DATA_DIR, f"ai_act_{lang}_clean.json")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)[:sample_size]

    documents = [Document(text=item["content"]) for item in data]
    splitter = SentenceSplitter(chunk_size=256, chunk_overlap=24)
    return splitter.get_nodes_from_documents(documents)

def run_embedding_benchmark():
    nodes = get_sample_nodes("en", sample_size=50) # Use 50 chunks for the benchmark
    texts = [node.text for node in nodes]

    print(f"Benchmarking with {len(texts)} text chunks...\n")

    models_to_test = {
        "MiniLM": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "BGE-M3": "BAAI/bge-m3"
    }

    results = {}

    for name, model_id in models_to_test.items():
        print(f"Loading {name} ({model_id})...")
        # We use CPU by default for the portfolio, but HuggingFaceEmbedding auto-detects GPU/CUDA if available
        embed_model = HuggingFaceEmbedding(model_name=model_id)

        start_time = time.time()
        # LlamaIndex abstracts the batching process
        embeddings = embed_model.get_text_embedding_batch(texts)
        end_time = time.time()

        duration = end_time - start_time
        dim = len(embeddings[0])

        results[name] = {
            "dimensions": dim,
            "time_seconds": round(duration, 2),
            "chunks_per_second": round(len(texts) / duration, 2)
        }

        print(f"[SUCCESS] {name} complete: {dim} dimensions, {results[name]['chunks_per_second']} chunks/sec\n")

    print("--- Benchmark Summary ---")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    # Warning: The first time you run this, it will download several GB of model weights!
    run_embedding_benchmark()
