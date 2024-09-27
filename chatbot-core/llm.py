import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from custom_embeddings import CustomEmbeddingsWrapper
from django.conf import settings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.memory import ConversationBufferWindowMemory

# from langchain.schema import AIMessage, HumanMessage
from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)


@dataclass
class LLMBase:
    """LLM Base containing common methods"""

    mem_key: str = field(init=False)
    conversation_max_window: int = field(init=False)
    qdrant_client: QdrantClient = field(init=False)
    llm_model: Any = field(init=False)
    user_memory_mapping: dict = field(init=False)
    memory: Any = field(init=False)
    embedding_model: CustomEmbeddingsWrapper = field(init=False)
    rag_chain: Optional[Any] = None

    def __post_init__(self, mem_key: str = "chat_history", conversation_max_window: int = 5):
        self.llm_model = None
        self.qdrant_client = None

        self.mem_key = mem_key
        self.conversation_max_window = conversation_max_window

        try:
            self.qdrant_client = QdrantClient(host=settings.QDRANT_DB_HOST, port=settings.QDRANT_DB_PORT)
        except Exception as e:
            raise Exception(f"Qdrant client is not properly setup. {str(e)}")

        self.user_memory_mapping = {}

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

    def execute_chain(self, user_id: str, query: str, db_collection_name: str = settings.QDRANT_DB_COLLECTION_NAME):
        """
        Executes the chain
        """
        if not self.rag_chain:
            self.rag_chain = self.create_chain(db_collection_name=db_collection_name)

        if "user_id" not in self.user_memory_mapping:
            self.user_memory_mapping[user_id] = ConversationBufferWindowMemory(
                k=self.conversation_max_window, memory_key=self.mem_key, return_messages=True
            )

        memory = self.user_memory_mapping[user_id]

        response = self.rag_chain.invoke(
            {"input": query, "chat_history": self.get_message_history(user_id=user_id)["chat_history"]}
        )
        response_text = response["answer"] if "answer" in response else "I don't know the answer."
        memory.save_context({"input": query}, {"output": response_text})
        self.user_memory_mapping[user_id] = memory

        return response_text

    def get_message_history(self, user_id: str):
        """
        Returns the historical conversational data
        """
        if "user_id" in self.user_memory_mapping:
            return self.user_memory_mapping[user_id].load_memory_variables({})
        return {}

    def delete_message_history_by_user(self, user_id: str) -> bool:
        """Deletes the message history based on user id"""
        if "user_id" in self.user_memory_mapping:
            del self.user_memory_mapping[user_id]
            logger.info(f"Successfully delete the {user_id} conversational history.")
            return True
        return False


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
            self.llm_model = Ollama(
                model=settings.LLM_MODEL_NAME, base_url=settings.LLM_OLLAMA_BASE_URL, temperature=self.temperature
            )
        except Exception as e:
            raise Exception(f"Ollama LLM model is not successfully loaded. {str(e)}")
