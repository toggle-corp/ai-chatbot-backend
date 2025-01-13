from typing import List
from langchain_community.utils.math import cosine_similarity
from chatbotcore.custom_embeddings import CustomEmbeddingsWrapper


class InformationGap:
    """
    Stores information about token usage, request count, and cost during the retrieval process.
    """

    def __init__(self, total_tokens: int, prompt_tokens: int, completion_tokens: int, total_cost: float, request_count: int):
        self.total_tokens = total_tokens
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_cost = total_cost
        self.request_count = request_count

    def as_dict(self):
        """
        Returns the information as a dictionary.
        """
        return {
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_cost": self.total_cost,
            "request_count": self.request_count,
        }

    def __repr__(self):
        return f"InformationGap(total_tokens={self.total_tokens}, prompt_tokens={self.prompt_tokens}, " \
               f"completion_tokens={self.completion_tokens}, total_cost={self.total_cost}, " \
               f"request_count={self.request_count})"
    
class Metricsretrieval:
    """
    Calculates retrieval-related metrics such as min and max similarity scores for a set of vectors.
    """

    def min_max_score(self, query: str, relevant_vectors: List[List[float]], embedding_model: CustomEmbeddingsWrapper):
        """
        Computes the minimum and maximum similarity score between the query and the retrieved vectors.
        """

        query_vector = embedding_model.embed_query(text=query)
        similarity_scores = []

        # Compute similarity between query vector and each relevant vector
        for vector in relevant_vectors:
            score = cosine_similarity([query_vector], [vector])[0][0]
            similarity_scores.append(score)

        # Calculate the min and max similarity scores
        min_score = min(similarity_scores) if similarity_scores else None
        max_score = max(similarity_scores) if similarity_scores else None

        return min_score, max_score



