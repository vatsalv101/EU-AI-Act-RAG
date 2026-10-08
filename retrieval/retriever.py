# retrieval/retriever.py
import os
import sys
from typing import List, Optional
from qdrant_client import QdrantClient
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores.types import (
    MetadataFilters,
    MetadataFilter,
    FilterOperator,
    FilterCondition
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

DB_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "qdrant_db")
COLLECTION_NAME = "eu_ai_act"

class AIActRetriever:
    def __init__(self, top_k: int = 3):
        self.top_k = top_k
        self.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.client = QdrantClient(path=DB_DIR)
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=COLLECTION_NAME
        )
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            embed_model=self.embed_model
        )

    def retrieve(
        self,
        query: str,
        language: Optional[str] = None,
        section: Optional[str] = None
    ) -> List[NodeWithScore]:
        """
        Retrieve relevant legal chunks with optional metadata filtering.
        """
        filters_list = []
        if language:
            filters_list.append(
                MetadataFilter(key="language", value=language, operator=FilterOperator.EQ)
            )
        if section:
            filters_list.append(
                MetadataFilter(key="section", value=section, operator=FilterOperator.EQ)
            )

        metadata_filters = None
        if filters_list:
            metadata_filters = MetadataFilters(
                filters=filters_list,
                condition=FilterCondition.AND
            )

        retriever = self.index.as_retriever(
            similarity_top_k=self.top_k,
            filters=metadata_filters
        )
        return retriever.retrieve(query)

if __name__ == "__main__":
    # Standalone test of the retriever
    test_retriever = AIActRetriever(top_k=3)

    test_query = "What AI practices are prohibited?"
    print(f"Testing Query: '{test_query}' (Language: en)")
    results = test_retriever.retrieve(test_query, language="en")

    for idx, res in enumerate(results, 1):
        score = round(res.score, 4) if res.score is not None else "N/A"
        sec = res.node.metadata.get("section", "Unknown")
        snippet = res.node.text[:140].replace("\n", " ")
        print(f"\n[{idx}] Score: {score} | Section: {sec}")
        print(f"    Excerpt: {snippet}...")
