import logging
from enum import Enum
from typing import List, Any
from dataclasses import dataclass, field
from django.conf import settings

from langchain_community.llms.ollama import Ollama
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

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
                model=settings.LLM_MODEL_NAME,
                base_url=settings.LLM_OLLAMA_BASE_URL,
                temperature=self.temperature
            )
        except Exception as e:
            raise Exception(f"Ollama LLM model is not successfully loaded. {str(e)}")


@dataclass
class ContextualChunking:
    """ Context retrieval for the chunk documents """
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
        """ Creates a prompt """
        prompt =  """
        You are an AI assistant specializing in Human Resources data processing in a company.
        Here is the document:
        <document>
        {document}
        </document>

        Here is the chunk we want to situate within the whole document:
        <chunk>
        {chunk}
        </chunk>

        Please give a short succint context using maximum 20 words to situate this chunk within the overall document\n
        for the purposes of improving search retrieval of the chunk. Answer only with the succint context and nothing else.

        Context:
        """
        return prompt

    def _generate_context(self, document: str, chunk: str):
        """ Generates contextualized document chunk response """
        prompt_template = ChatPromptTemplate.from_messages([("system", self.get_prompt())])
        messages = prompt_template.format_messages(
            document=document,
            chunk=chunk
        )
        response = self.model.llm.invoke(messages)
        return response

    def generate_contextualized_chunks(self, document: str, chunks: List[Document]):
        """ Generates contextualized document chunks """
        contextualized_chunks = []
        for chunk in chunks:
            context = self._generate_context(document, chunk.page_content)
            contextualized_content = f"{context}\n\n\n{chunk.page_content}"
            contextualized_chunks.append(
                Document(page_content=contextualized_content, metadata=chunk.metadata)
            )
        return contextualized_chunks
