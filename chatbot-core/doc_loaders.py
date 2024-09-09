from typing import List
from dataclasses import dataclass
from langchain.schema import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

@dataclass(kw_only=True)
class DocumentLoader:
    """
    Base Class for Document Loaders
    """
    chunk_size: int=200
    chunk_overlap: int=20

    def _get_split_documents(self, documents: List[Document]):
        """
        Splits documents into multiple chunks
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )

        return splitter.split_documents(documents)

@dataclass
class LoaderFromText(DocumentLoader):
    """
    Document loader for plain texts
    """
    texts: str

    def create_document_chunks(self):
        """
        Creates multiple documents from the input texts
        """
        documents = [Document(page_content=self.texts)]
        doc_chunks = self._get_split_documents(documents=documents)
        return doc_chunks

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
        return doc_chunks
