from dataclasses import dataclass, field
from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader

from chatbotcore.contextual_chunks import ContextualChunking


@dataclass(kw_only=True)
class DocumentLoader:
    """
    Base Class for Document Loaders
    """

    chunk_size: int = 100
    chunk_overlap: int = 20
    context_retrieval: ContextualChunking = field(init=False)

    def __post_init__(self):
        self.context_retrieval = ContextualChunking()

    def _get_split_documents(self, documents: List[Document]):
        """
        Splits documents into multiple chunks
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap, length_function=len
        )

        return splitter.split_documents(documents)


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
        doc_chunks = self._get_split_documents(documents=documents)
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
        doc_chunks = self._get_split_documents(documents=docs)
        contextualized_chunks = self.context_retrieval.generate_contextualized_chunks(document=docs, chunks=doc_chunks)
        return contextualized_chunks
