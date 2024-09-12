import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from custom_embeddings import CustomEmbeddingsWrapper
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import AIMessage, HumanMessage
from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from django.conf import settings

logger = logging.getLogger(__name__)


@dataclass
class LLMBase:
    """LLM Base containing common methods"""

    qdrant_client: QdrantClient = field(init=False)
    llm_model: Any = field(init=False)
    memory: Any = field(init=False)
    embedding_model: CustomEmbeddingsWrapper = field(init=False)
    rag_chain: Optional[Any] = None

    def __post_init__(self, mem_key: str = "chat_history", conversation_max_window: int = 3):
        self.llm_model = None
        self.qdrant_client = None
        self.memory = None

        try:
            self.qdrant_client = QdrantClient(host=settings.QDRANT_DB_HOST, port=settings.QDRANT_DB_PORT)
        except Exception as e:
            raise Exception(f"Qdrant client is not properly setup. {str(e)}")
        self.memory = ConversationBufferWindowMemory(k=conversation_max_window, memory_key=mem_key, return_messages=True)

        self.embedding_model = CustomEmbeddingsWrapper(
            url=settings.EMBEDDING_MODEL_URL,
            model_name=settings.EMBEDDING_MODEL_NAME,
            model_type=settings.EMBEDDING_MODEL_TYPE,
            base_url=settings.OLLAMA_EMBEDDING_MODEL_BASE_URL,
        )

    def _system_prompt_for_retrieval(self):
        """System prompt for information retrieval"""
        return (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

    def _system_prompt_for_response(self):
        """
        System prompt for response generation
        """
        system_prompt = """
                You are an assistant for question-answering tasks.\n,
                Use the following retrieved context to answer the question.\n,
                If you don't get the answer from the provided context, \n,
                say that 'I don't know. How can I help with the office related queries ?'
                \n\n,
                Context: {context}
            """
        return system_prompt

    def get_prompt_template_for_retrieval(self):
        """Get the prompt template"""
        system_prompt = self._system_prompt_for_retrieval()
        context_prompt_template = ChatPromptTemplate.from_messages(
            [("system", system_prompt), MessagesPlaceholder(variable_name="chat_history"), ("human", "{input}")]
        )
        return context_prompt_template

    def get_prompt_template_for_response(self):
        """Get the prompt template for response generation"""
        system_prompt = self._system_prompt_for_response()
        llm_response_prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
        )
        return llm_response_prompt

    def get_db_retriever(self, collection_name: str, top_k_items: int = 5, score_threshold: float = 0.5):
        """Get the database retriever"""
        db_retriever = QdrantVectorStore(
            client=self.qdrant_client, collection_name=collection_name, embedding=self.embedding_model
        )
        retriever = db_retriever.as_retriever(
            search_type="similarity_score_threshold", search_kwargs={"k": top_k_items, "score_threshold": score_threshold}
        )
        return retriever

    def create_chain(self, db_collection_name: str):
        """Creates a llm chain"""
        if not self.llm_model:
            raise Exception("The LLM model is not loaded.")

        context_prompt_template = self.get_prompt_template_for_retrieval()
        response_prompt_template = self.get_prompt_template_for_response()

        retriever = self.get_db_retriever(collection_name=db_collection_name)

        history_aware_retriever = create_history_aware_retriever(self.llm_model, retriever, context_prompt_template)

        chat_response_chain = create_stuff_documents_chain(self.llm_model, response_prompt_template)

        rag_chain = create_retrieval_chain(history_aware_retriever, chat_response_chain)
        return rag_chain

    def execute_chain(self, query: str, db_collection_name: str = settings.QDRANT_DB_COLLECTION_NAME):
        """
        Executes the chain
        """
        if not self.rag_chain:
            self.rag_chain = self.create_chain(db_collection_name=db_collection_name)

        response = self.rag_chain.invoke({"input": query, "chat_history": self.get_message_history()["chat_history"]})
        self.memory.chat_memory.add_message(HumanMessage(content=query))
        self.memory.chat_memory.add_message(AIMessage(content=response["answer"]))
        return response["answer"] if "answer" in response else ""

    def get_message_history(self):
        """
        Returns the historical conversational data
        """
        return self.memory.load_memory_variables({})


@dataclass
class OpenAIHandler(LLMBase):
    """LLM handler using OpenAI for RAG"""

    temperature: float = 0.2

    def __post_init__(self):
        super().__post_init__()
        try:
            self.llm_model = ChatOpenAI(model=settings.LLM_MODEL_NAME, temperature=self.temperature)
        except Exception as e:
            raise Exception(f"OpenAI LLM model is not successfully loaded. {str(e)}")


@dataclass
class OllamaHandler(LLMBase):
    """LLM Handler using Ollama for RAG"""

    temperature: float = 0.2

    def __post_init__(self):
        super().__post_init__()
        try:
            self.llm_model = Ollama(model=settings.LLM_MODEL_NAME, base_url=settings.LLM_OLLAMA_BASE_URL, temperature=self.temperature)
        except Exception as e:
            raise Exception(f"Ollama LLM model is not successfully loaded. {str(e)}")
