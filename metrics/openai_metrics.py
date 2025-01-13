from typing import List


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

