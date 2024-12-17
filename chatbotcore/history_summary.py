import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List

from django.conf import settings
from langchain_community.llms.ollama import Ollama
from langchain_openai import ChatOpenAI
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.human import HumanMessage
from langchain.memory.summary_buffer import ConversationSummaryBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain


from chatbotcore.utils import LLMType

logger = logging.getLogger(__name__)


@dataclass
class OpenAIHandler:
    """LLM handler using OpenAI for RAG"""

    temperature: float = 0.1
    llm: ChatOpenAI = field(init=False)

    def __post_init__(self):
        try:
            self.llm = ChatOpenAI(model=settings.LLM_MODEL_NAME, temperature=self.temperature)
        except Exception as e:
            raise Exception(f"OpenAI LLM model is not successfully loaded. {str(e)}")


@dataclass
class OllamaHandler:
    """LLM Handler using Ollama"""

    temperature: float = 0.1
    llm: Ollama = field(init=False)

    def __post_init__(self):
        try:
            self.llm = Ollama(
                model=settings.LLM_MODEL_NAME, base_url=settings.LLM_OLLAMA_BASE_URL, temperature=self.temperature
            )
        except Exception as e:
            raise Exception(f"Ollama LLM model is not successfully loaded. {str(e)}")


@dataclass
class HistorySummaryOpenAI:

    model: Any = field(init=False)
    model_type: Enum = LLMType.OLLAMA

    def __post_init__(self):
        if self.model_type == LLMType.OLLAMA:
            self.model = OllamaHandler()
        elif self.model_type == LLMType.OPENAI:
            self.model = OpenAIHandler()
        else:
            logger.error("Wrong LLM Type")
            raise ValueError("Wront LLM Type")
        
    async def create_summary(self, relevant_hist):

        llm = self.model.llm
        summary_memory = ConversationSummaryBufferMemory(
            llm = llm,
            max_token_limit = 1000,
            return_messages = True
            )
        for i in range(1,len(relevant_hist)):

            if isinstance(relevant_hist[i], AIMessage):
                user_input = relevant_hist[i-1].content
                model_output = relevant_hist[i].content
                summary_memory.save_context({"input": user_input}, {"output": model_output})
        #summary = summary_memory.load_memory_variables({})
        #logging.info("the summary is %s", summary)
        previous_summary = ""
        messages = summary_memory.chat_memory.messages[:-2]
        summary = summary_memory.predict_new_summary(messages, previous_summary)


        return summary # This returns the summarized history of the conversation 



@dataclass
class HistorySummary:
    """Context retrieval for the chunk documents"""

    model: Any = field(init=False)
    model_type: Enum = LLMType.OLLAMA

    def __post_init__(self):
        if self.model_type == LLMType.OLLAMA:
            self.model = OllamaHandler()
        elif self.model_type == LLMType.OPENAI:
            self.model = OpenAIHandler()
        else:
            logger.error("Wrong LLM Type")
            raise ValueError("Wront LLM Type")

    
    def get_prompt(self):
        """Creates a prompt for summarizing a conversation between a human and AI."""
        prompt = """
        You are an AI assistant tasked with summarizing a conversation between a human and an AI.
        Below is the conversation:

        {conversation}

        Please provide a concise summary of the conversation, highlighting the key points discussed.
        Make sure the summary is short, accurate, and captures the main intent of both the human and the AI's responses.
        Provide only the summary and return empty if there is no conversation.
        """
        return prompt


    def _generate_summary(self, message_history: List[dict]):
        """Generates a summary of the conversation between human and AI."""
        # Directly pass the raw message history without any processing or formatting
        prompt_template = ChatPromptTemplate.from_messages([("system", self.get_prompt())])
        messages = prompt_template.format_messages(conversation=message_history)

        # Generate summary using the model
        response = self.model.llm.invoke(messages)
        
        return response
 