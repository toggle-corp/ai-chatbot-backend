from dataclasses import dataclass
from typing import List, Optional

import requests
from langchain.embeddings.base import Embeddings
from chatbotcore.utils import EmbeddingModelType


@dataclass
class CustomEmbeddingsWrapper(Embeddings):
    """
    This is an Embeddings model wrapper to send request to the actual
    embedding models
    """

    url= "http://192.168.88.11:8000/get_embeddings/"
    model_name: str
    model_type: EmbeddingModelType
    base_url: Optional[str]

    def __post_init__(self):
        if self.model_type in [EmbeddingModelType.SENTENCE_TRANSFORMES, EmbeddingModelType.OPENAI]:
            if not (self.url and self.model_name):
                raise Exception("Url or model name or both are not provided.")
        elif self.model_type == EmbeddingModelType.OLLAMA:
            if not (self.url and self.model_name and self.base_url):
                raise Exception("Url or base_url or both are not provided.")

    def embed_query(self, text: str, timeout: int = 30) -> List[float]:
        """
        Sends the request to Embedding module to
        embed the query to the vector representation
        """
        payload = {"type_model": self.model_type, "name_model": self.model_name, "texts": text}
        try:
            response = requests.post(url=self.url, json=payload, timeout=timeout)
        except requests.Timeout as e:
            raise Exception(e)
        return response.json()

    def embed_documents(self, texts: List[str], timeout: int = 30) -> List[List[float]]:
        """
        Sends the request to Embedding module to
        embed multiple queries to the vector representation
        """
        payload = {"type_model": self.model_type, "name_model": self.model_name, "texts": texts}
        try:
            response = requests.post(url=self.url, json=payload, timeout=timeout)
        except requests.Timeout as e:
            raise Exception(e)
        return response.json()
