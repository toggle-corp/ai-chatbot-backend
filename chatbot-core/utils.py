import re
from enum import Enum
from typing import List

class EmbeddingModelType(Enum):
    """ Embedding Model Types """
    SENTENCE_TRANSFORMES=1
    OLLAMA=2
    OPENAI=3

def preprocess_text(texts: List[str]) -> List[str]:
    """
    Preprocessing of the texts
    """
    pattern = r'\s+'

    results = [re.sub(pattern, '', text) for text in texts]
    return results
