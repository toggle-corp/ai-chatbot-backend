import logging
import re
import time
from enum import Enum
from functools import wraps
from typing import Any, List, Optional

import requests
from django.conf import settings
from langchain.schema import Document
from langchain_community.document_transformers import LongContextReorder
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables.config import run_in_executor
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi


def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        global execution_time
        execution_time = end_time - start_time
        logging.info(f"{func.__name__} executed in {execution_time:.6f} seconds")
        return result

    return wrapper


class EmbeddingModelType(Enum):
    """Embedding Model Types"""

    SENTENCE_TRANSFORMES = 1
    OLLAMA = 2
    OPENAI = 3


class LLMType(Enum):
    """LLM Types"""

    OLLAMA = 1
    OPENAI = 2


class Reranker:
    def __init__(self, query: str, documents: List[Document]):
        self.query = query
        self.documents = documents

    def _send_request(self, timeout: int = 30):
        """Sends request to get scores"""
        headers = {"Content-Type": "application/json"}
        payload = {"query": self.query, "documents": [d.page_content for d in self.documents]}

        response = requests.post(
            url=f"{settings.EMBEDDING_MODEL_URL}/docs_reranking_scores", headers=headers, json=payload, timeout=timeout
        )
        assert response.status_code == 200, f"Error on request {response.status_code}"
        return response.json()

    def rerank(self):
        """Re-rank documents"""
        scores = self._send_request()
        doc_score_pairs = list(zip(self.documents, scores))
        ranked_documents = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked_documents]


class BM25DocRetriever(BaseRetriever):
    bm25: Any
    all_docs: Any
    k: Any
    threshold: Any
    """BM25 retriever for text based search"""

    def __init__(self, docs: List[Document], k_items: int, threshold: int = 10):
        super().__init__()
        self.all_docs = docs
        self.k = k_items
        self.threshold = threshold
        document_contents = [doc.metadata["page_content"] for doc in docs]
        tokenized_docs = [doc.split() for doc in document_contents]
        self.bm25 = BM25Okapi(tokenized_docs)

    def _get_relevant_documents(self, query: str, run_manager, **kwargs):
        """Get relevant documents"""
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        ranked_docs_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.all_docs[i] for i in ranked_docs_indices if scores[i] > self.threshold][: self.k]  # Returns top k items

    async def _aget_relevant_documents(self, query: str, run_manager=None):
        return await run_in_executor(None, self._get_relevant_documents, query, run_manager=run_manager)


class QdrantDocRetriever:
    """Retrieve documents from Qdrant"""

    def __init__(self, qdrant_client: QdrantClient, collection_name: str, embedding_model: Any):
        self.db_retriever = QdrantVectorStore(
            client=qdrant_client.db_client, collection_name=collection_name, embedding=embedding_model
        )

    def get_qdrant_retriever(self, top_k_items: int, score_threshold: float):
        """Get the Qdrant retriever"""
        retriever = self.db_retriever.as_retriever(
            search_type="similarity_score_threshold", search_kwargs={"k": top_k_items, "score_threshold": score_threshold}
        )
        return retriever


class HybridRetriever(BaseRetriever):
    """Retrieve documents using Hybrid retriever"""

    bm25_retriever: Any
    qdrant_retriever: Any
    chunks_reordering: Any

    def __init__(self, bm25_retriever, qdrant_retriever):
        super().__init__()
        self.bm25_retriever = bm25_retriever
        self.qdrant_retriever = qdrant_retriever
        self.chunks_reordering = LongContextReorder()

    @time_it
    def _get_relevant_documents(self, query: str, run_manager: Optional[Any] = None):
        """
        Get the relevant document based on re-ranking
        """
        bm25_docs = self.bm25_retriever._get_relevant_documents(query, run_manager=run_manager)

        bm25_docs = [
            Document(page_content=d.metadata["page_content"], metadata={"_id": d.metadata["_id"]}) for d in bm25_docs
        ]
        qdrant_docs = self.qdrant_retriever.invoke(input=query)

        qdrant_docs = [Document(page_content=d.page_content, metadata={"_id": d.metadata["_id"]}) for d in qdrant_docs]

        combined_docs = bm25_docs + qdrant_docs

        reordered_documents = self.chunks_reordering.transform_documents(combined_docs)
        unique_reordered_documents = {}
        for doc in reordered_documents:
            doc_id = doc.metadata["_id"]
            if doc_id not in unique_reordered_documents:
                unique_reordered_documents[doc_id] = doc
        return list(unique_reordered_documents.values())

    async def _aget_relevant_documents(self, query: str, run_manager: Optional[Any] = None):
        """
        Asnyc method for getting the relevant documents based on re-ranking
        """
        return await run_in_executor(None, self._get_relevant_documents, query, run_manager=run_manager)


def preprocess_text(texts: list[str]) -> list[str]:
    """
    Preprocessing of the texts
    """
    pattern = r"\s+"

    results = [re.sub(pattern, "", text) for text in texts]
    return results
