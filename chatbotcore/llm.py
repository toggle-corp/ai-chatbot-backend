import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

from django.conf import settings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.llms.ollama import Ollama
from langchain_community.utils.math import cosine_similarity
from langchain_core.messages.ai import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient

from chatbotcore.custom_embeddings import CustomEmbeddingsWrapper
from chatbotcore.database import QdrantDatabase
from chatbotcore.utils import BM25DocRetriever, HybridRetriever, QdrantDocRetriever

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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
    db_retriever: Optional[Any] = None

    def __post_init__(self, mem_key: str = "chat_history", conversation_max_window: int = 3):
        self.llm_model = None
        self.qdrant_client = None

        self.mem_key = mem_key
        self.conversation_max_window = conversation_max_window

        self.default_failure_message = "Sorry, can't answer as relevant context is not available or didn't understand your question. How can I help with other office related queries ?"  # noqa

        try:
            self.qdrant_client = QdrantDatabase(
                collection_name=settings.QDRANT_DB_COLLECTION_NAME,
                host=settings.QDRANT_DB_HOST,
                port=settings.QDRANT_DB_PORT,
            )
        except Exception as e:
            raise Exception(f"Qdrant client is not properly setup. {str(e)}")

        self.user_memory_mapping = {}

        self.embedding_model = CustomEmbeddingsWrapper(
            url=f"{settings.EMBEDDING_MODEL_URL}/get_embeddings",
            model_name=settings.EMBEDDING_MODEL_NAME,
            model_type=settings.EMBEDDING_MODEL_TYPE,
            base_url=settings.OLLAMA_EMBEDDING_MODEL_BASE_URL,
        )

    def get_db_retriever(self, top_k_items: int = 5, score_threshold: float = 0.7):
        """Get the database retriever"""
        all_documents = self.qdrant_client.load_all_documents()

        bm25_retriever = BM25DocRetriever(docs=all_documents, k_items=top_k_items)
        qdrant_retriever = QdrantDocRetriever(
            qdrant_client=self.qdrant_client,
            collection_name=settings.QDRANT_DB_COLLECTION_NAME,
            embedding_model=self.embedding_model,
        )
        hybrid_retriever = HybridRetriever(
            bm25_retriever=bm25_retriever,
            qdrant_retriever=qdrant_retriever.get_qdrant_retriever(top_k_items=top_k_items, score_threshold=score_threshold),
        )
        return hybrid_retriever

    def _system_prompt_for_retrieval(self):
        """System prompt for information retrieval"""
        return """Given a chat history and the latest user question {input} \
            which might reference context in the chat history, formulate a standalone question \
            which can be understood without the chat history. Do NOT answer the question, \
            just reformulate it if needed and otherwise return it as is."""

    def _system_prompt_for_response(self):
        """
        System prompt for response generation
        """
        system_prompt = """
            You are an assistant to answer the office related relevant questions based on provided contexts according to the query {input}.\n,
            Use the retrieved context to answer the question strictly. The response should be concise and to the point.\n,
            If the retrieved context is not available, do not use your own knowledge or the chat history,\n
            You will not invent anything that is not drawn directly from the provided context.\n
            Just say 'Sorry, can't answer as relevant context is not available or didn't understand your question.\n
            How can I help with other office related queries ?'
            \n\n,
            {context}
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

    def create_chain(self):
        """Creates a llm chain"""
        if not self.llm_model:
            raise Exception("The LLM model is not loaded.")

        context_prompt_template = self.get_prompt_template_for_retrieval()
        response_prompt_template = self.get_prompt_template_for_response()

        history_aware_retriever = create_history_aware_retriever(self.llm_model, self.db_retriever, context_prompt_template)

        chat_response_chain = create_stuff_documents_chain(self.llm_model, response_prompt_template)

        rag_chain = create_retrieval_chain(history_aware_retriever, chat_response_chain)
        return rag_chain

    async def filter_relevant_history(self, user_id: str, query: str, similarity_threshold: float = 0.5):

        current_query_vector = self.embedding_model.embed_query(query)

        relevant_history = []

        message_history = self.get_message_history(user_id=user_id)["chat_history"]
        for i in range(1, len(message_history)):

            if isinstance(message_history[i], AIMessage):
                message_content_ai = message_history[i].content
                logger.info(f"message content: {message_content_ai}")
                query_vector_ai = self.embedding_model.embed_query(text=message_content_ai)

                similarity_score_ai = cosine_similarity([query_vector_ai], [current_query_vector])[0][0]
                logger.info(f"the cosine similarity of ai response is {similarity_score_ai}")
                if similarity_score_ai > similarity_threshold:
                    relevant_history.append(message_history[i - 1])
                    relevant_history.append(message_history[i])

        return relevant_history if relevant_history else []

    async def execute_chain(self, user_id: str, query: str):
        """
        Executes the chain
        """
        if not self.db_retriever:
            self.db_retriever = self.get_db_retriever()

        if not self.rag_chain:
            self.rag_chain = self.create_chain()

        if user_id not in self.user_memory_mapping:
            self.user_memory_mapping[user_id] = ConversationBufferWindowMemory(
                k=self.conversation_max_window, memory_key=self.mem_key, return_messages=True
            )

        relevant_history = await self.filter_relevant_history(user_id=user_id, query=query, similarity_threshold=0.7)
        memory = self.user_memory_mapping[user_id]

        response = await self.rag_chain.ainvoke(
            {
                "input": query,
                "chat_history": relevant_history,
            }
        )
        context_documents = response.get("context", [])
        logger.info(f"Documents retrieved by the history aware retriever: {context_documents}")
        response_text = response["answer"] if "answer" in response else self.default_failure_message

        point_ids = [d.metadata["_id"] for d in response["context"]]
        logger.info("the point_ids obtained by qdrant: %s", point_ids)
        relevant_vectors = self.qdrant_client.retrieve_vectors(points=point_ids)

        # page_contexts = [d.metadata.get("page_content") or d.page_content for d in response["context"]]

        postprocess_results = await self.postprocess_response(relevant_vectors=relevant_vectors, llm_response=response_text)
        if postprocess_results:
            memory.save_context({"input": query}, {"output": response_text})
            self.user_memory_mapping[user_id] = memory
            return response_text
        return self.default_failure_message

    async def postprocess_response(
        self, relevant_vectors: List[List[float]], llm_response: str, threshold: float = 0.5
    ) -> bool:
        """
        Check if the generated llm response is in sync with the retrieved contexts
        """
        llm_response_vector = self.embedding_model.embed_query(text=llm_response)
        similarity_results = cosine_similarity(relevant_vectors, [llm_response_vector])
        similarity_results_flattend = [item for sublist in similarity_results for item in sublist]
        logger.info("Similarity scores: %s", similarity_results_flattend)
        results = [item >= threshold for item in similarity_results_flattend]
        if any(results):
            return True
        return False

    def get_message_history(self, user_id: str):
        """
        Returns the historical conversational data
        """
        if user_id in self.user_memory_mapping:
            return self.user_memory_mapping[user_id].load_memory_variables({})
        return {"chat_history": []}

    def delete_message_history_by_user(self, user_id: str) -> bool:
        """Deletes the message history based on user id"""
        if user_id in self.user_memory_mapping:
            del self.user_memory_mapping[user_id]
            logger.info(f"Successfully delete the {user_id} conversational history.")
            return True
        return False


@dataclass
class OpenAIHandler(LLMBase):
    """LLM handler using OpenAI for RAG"""

    temperature: float = 0.1

    def __post_init__(self):
        super().__post_init__()
        try:
            self.llm_model = ChatOpenAI(model=settings.LLM_MODEL_NAME, temperature=self.temperature)
        except Exception as e:
            raise Exception(f"OpenAI LLM model is not successfully loaded. {str(e)}")


@dataclass
class OllamaHandler(LLMBase):
    """LLM Handler using Ollama for RAG"""

    temperature: float = 0.1

    def __post_init__(self):
        super().__post_init__()
        try:
            self.llm_model = Ollama(
                model=settings.LLM_MODEL_NAME, base_url=settings.LLM_OLLAMA_BASE_URL, temperature=self.temperature
            )
        except Exception as e:
            raise Exception(f"Ollama LLM model is not successfully loaded. {str(e)}")
