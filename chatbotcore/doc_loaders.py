import json
from dataclasses import dataclass, field
from typing import List

import requests
from django.conf import settings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader

from chatbotcore.contextual_chunks import ContextualChunking


@dataclass(kw_only=True)
class DocumentLoader:
    """
    Base Class for Document Loaders
    """

    chunk_size: int = 200
    chunk_overlap: int = 50
    context_retrieval: ContextualChunking = field(init=False)

    def __post_init__(self):
        self.context_retrieval = ContextualChunking()

    def _get_split_documents_with_recursive_char(self, documents: List[Document], multiplier: int = 3):
        """
        Splits documents into multiple chunks using Recursive Character splitter
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size * multiplier, chunk_overlap=self.chunk_overlap * multiplier, length_function=len
        )
        return splitter.split_documents(documents=documents)

    def langchain_document_to_dict(self, doc: Document):
        """
        Converts langchain Document to dictionary
        """
        return {"page_content": doc.page_content, "metadata": doc.metadata}

    def dict_to_langchain_document(self, doc: dict):
        """
        Converts dictionary to Langchain docuemnt
        """
        return Document(page_content=doc["page_content"], metadata=doc["metadata"])

    def _get_split_documents_using_token_based(self, documents: List[Document], timeout: int = 60):
        """
        Splits documents into multiple chunks using Sentence Transformer
        token based.
        """
        url = f"{settings.EMBEDDING_MODEL_URL}/split_docs_based_on_tokens"
        documents_dict = [self.langchain_document_to_dict(d) for d in documents]
        payload = {
            "model": settings.EMBEDDING_MODEL_NAME,
            "documents": json.dumps(documents_dict),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(url=url, headers=headers, json=payload, timeout=timeout)
        data = response.json()
        return [self.dict_to_langchain_document(d) for d in data]


@dataclass
class LoaderFromText(DocumentLoader):
    """
    Document loader for plain texts
    """

    text: str

    def create_document_chunks(self):
        """
        Creates multiple documents from the input texts
        """
        documents = [Document(page_content=self.text)]
        # doc_chunks = self._get_split_documents_using_token_based(documents=documents)
        doc_chunks = self._get_split_documents_with_recursive_char(documents=documents)
        contextualized_chunks = self.context_retrieval.generate_contextualized_chunks(document=self.text, chunks=doc_chunks)
        return contextualized_chunks


@dataclass
class LoaderFromWeb(DocumentLoader):
    """
    Document loader for the web url
    """

    url: str

    def create_document_chunks(self):
        """
        Creates multiple documents from the input url
        """
        loader = WebBaseLoader(web_path=self.url)
        docs = loader.load()
        doc_chunks = self._get_split_documents_using_token_based(documents=docs)
        contextualized_chunks = self.context_retrieval.generate_contextualized_chunks(document=docs, chunks=doc_chunks)
        return contextualized_chunks
