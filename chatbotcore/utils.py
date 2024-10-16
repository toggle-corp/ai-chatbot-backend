import re
from enum import Enum


class EmbeddingModelType(Enum):
    """Embedding Model Types"""

    SENTENCE_TRANSFORMES = 1
    OLLAMA = 2
    OPENAI = 3


class LLMType(Enum):
    """LLM Types"""

    OLLAMA = 1
    OPENAI = 2


def preprocess_text(texts: list[str]) -> list[str]:
    """
    Preprocessing of the texts
    """
    pattern = r"\s+"

    results = [re.sub(pattern, "", text) for text in texts]
    return results
