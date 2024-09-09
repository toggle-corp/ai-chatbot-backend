from dataclasses import dataclass, field
from typing import List

import numpy as np
from torch import Tensor

from sentence_transformers import SentenceTransformer

from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv

load_dotenv()

# Note to check if the vector dimensions are same (impt for injecting in vector db)

@dataclass
class SentenceTransformerEmbeddingModel(Embeddings):
    """
    Embedding model using Sentence Transformers
    """
    model: str="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_model: SentenceTransformer = field(init=False)

    def __post_init__(self):
        """
        Post initialization
        """
        self.st_embedding_model = SentenceTransformer(self.model)

    def embed_documents(self, texts: list) -> np.ndarray:
        """
        Generate embeddings for a list of documents
        """
        return self.st_embedding_model.encode(texts)

    def embed_query(self, text: str) -> np.ndarray:
        """
        Generate embedding for a piece of text
        """
        return self.st_embedding_model.encode(text)

    def check_similarity(self, embeddings_1: np.ndarray, embeddings_2: np.ndarray) -> Tensor:
        """
        Computes the cosine similarity between two embeddings
        """
        return self.st_embedding_model.similarity(embeddings_1, embeddings_2)
    
    def get_model(self):
        """ Returns the model """
        return self.st_embedding_model

# TODO: check with Ollama as it always expect llama2 model
@dataclass
class OllamaEmbeddingModel(Embeddings):
    """
    Embedding model using Ollama (locally deployed)
    """
    model: str="all-minilm:latest"
    base_url: str="http://localhost:11434"
    model: str="nomic-embed-text:latest"

    def __post_init__(self):
        """
        Post initialization
        """
        self.ollama_embed_model = OllamaEmbeddings(model=self.model, base_url=self.base_url)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of documents.
        """
        return self.ollama_embed_model.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """
        Generate embedding for a piece of text
        """
        return self.ollama_embed_model.embed_query(text=text, model=self.model)

    def get_model(self):
        """ Returns the model """
        return self.ollama_embed_model

@dataclass
class OpenAIEmbeddingModel(Embeddings):
    """
    Embedding Model using OpenAI
    """
    model: str="text-embedding-3-small"

    def __post_init__(self):
        """
        Post initialization
        """
        self.openai_embed_model = OpenAIEmbeddings(model=self.model)

    def embed_documents(self, texts: List[str]):
        """
        Generate embeddings for a list of documents.
        """
        return self.openai_embed_model.embed_documents(texts=texts)

    def embed_query(self, text: str):
        """
        Generate embedding for a piece of text
        """
        return self.openai_embed_model.embed_query(text=text)

    def get_model(self):
        """ Returns the model """
        return self.openai_embed_model
