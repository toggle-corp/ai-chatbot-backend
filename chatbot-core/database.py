from typing import List
from dataclasses import dataclass, field
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.exceptions import UnexpectedResponse
from uuid import uuid4

@dataclass
class QdrantDatabase:
    """ Qdrant Vector Database """
    host: str="localhost"
    port: int=6333
    db_client: QdrantClient=field(init=False)
    collection_name: str=field(init=False)

    def initialize_db(self, collection_name: str):
        """ Initialize database client """
        self.db_client = QdrantClient(host=self.host, port=self.port)
        self.collection_name = collection_name

    def _collection_exists(self, collection_name: str) -> bool:
        """ Check if the collection in db already exists """
        try:
            self.db_client.get_collection(collection_name=collection_name)
            return True
        except UnexpectedResponse:
            return False

    def set_collection(self, vector_size: int=384):
        """ Create the database collection """
        if not self._collection_exists(self.collection_name):
            self.db_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
        else:
            print(f"Collection {self.collection_name} already exists. Using the existing one.")

    def store_data(self, data: List) -> None:
        """ Stores data in vector db """
        point_vectors = [
            {
                "id": str(uuid4()),
                "vector": v_representation,
                "payload": metadata
            }
            for v_representation, metadata in data
        ]

        response = self.db_client.upsert(
            collection_name=self.collection_name,
            points=point_vectors
        )
        return response

    def data_search(
        self,
        collection_names: list,
        query_vector: list,
        top_n_retrieval: int=5,
        score_threshold: float=0.7
    ):
        """ Search data in the database """
        results = [
            self.db_client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                top=top_n_retrieval,
                score_threshold=score_threshold
                
            )
            for collection_name in collection_names
        ]
        # Note the results shall contain score key; sort the results using score key and get top 5 among them.
        return results