import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

import qdrant_client.http.models as q_models
from django.conf import settings
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    VectorParams,
)

logger = logging.getLogger(__name__)


@dataclass
class QdrantDatabase:
    """Qdrant Vector Database"""

    collection_name: str
    host: str
    port: int
    db_client: QdrantClient = field(init=False)

    def __post_init__(self):
        """Initialize database client"""
        self.db_client = QdrantClient(host=self.host, port=self.port)

    def _collection_exists(self, collection_name: str) -> bool:
        """Check if the collection in db already exists"""
        try:
            self.db_client.get_collection(collection_name=collection_name)
            return True
        except UnexpectedResponse:
            return False

    def set_collection(self):
        """Create the database collection"""
        if not self._collection_exists(self.collection_name):
            self.db_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=settings.EMBEDDING_MODEL_VECTOR_SIZE, distance=Distance.COSINE),
            )
        else:
            logger.info(f"Collection {self.collection_name} already exists. Using the existing one.")

    def store_data(self, data: list) -> None:
        """Stores data in vector db"""
        point_vectors = [
            {"id": str(uuid.uuid4()), "vector": v_representation, "payload": metadata} for v_representation, metadata in data
        ]

        response = self.db_client.upsert(collection_name=self.collection_name, points=point_vectors)
        return response

    def data_search(
        self, collection_names: list, query_vector: list, top_n_retrieval: int = 5, score_threshold: float = 0.7
    ):
        """Search data in the database"""
        results = [
            self.db_client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                top=top_n_retrieval,
                score_threshold=score_threshold,
            )
            for collection_name in collection_names
        ]
        # Note the results shall contain score key; sort the results using score key and get top 5 among them.
        return results

    def delete_data_by_src_uuid(self, collection_name: str, key: str, value: Any) -> bool:
        """
        Delete data by source uuid
        Note that the document source key should be doc_uuid
        """
        points_selector = FilterSelector(filter=Filter(must=[FieldCondition(key=key, match=MatchValue(value=value))]))
        result = self.db_client.delete(collection_name=collection_name, points_selector=points_selector)

        return result.status == q_models.UpdateStatus.COMPLETED
