from collections import Counter

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from chatbotcore.llm import OpenAIHandler


class LLMcoreMetrics:
    def __init__(self, llm_handler: OpenAIHandler):
        # Initialize with an OpenAIHandler (which handles the LLM model)
        self.llm_handler = llm_handler
        self.llm_model = self.llm_handler.llm_model

    def prompt_top_questions(self):

        top_questions_prompt = PromptTemplate(
            input_variables=["questions"],
            template="""
            Here is a list of questions. Please rank them from the most insightful to the least.
            Rank them in order and provide a list of the top questions based on your judgement.

            Questions:
            {questions}
            """,
        )
        return top_questions_prompt

    def create_chain(self, prompt: PromptTemplate):
        """
        Create the LLM chain to run the prompt with the LLM model
        """
        return LLMChain(llm=self.llm_model, prompt=prompt)

    def word_count(self, questions: list):
        """
        Use the LLM model to get the ranking of the questions based on word count (excluding stop words)
        """
        words = [word.lower() for sentence in questions for word in sentence.split()]

        # Count word frequencies
        word_count = Counter(words)

        return word_count.items()

    def top_questions(self, questions: list):
        """
        Use the LLM model to get the ranking of the questions based on word count (excluding stop words)
        """
        prompt_questions = self.prompt_top_questions()
        llm_chain = self.create_chain(prompt_questions)

        # Run the chain with the formatted questions text
        response = llm_chain.run({"questions": questions})

        return response


if __name__ == "__main__":

    # Initialize OpenAIHandler instance
    llm_handler = OpenAIHandler()

    # Create an instance of LLMcoreMetrics
    llm_metrics = LLMcoreMetrics(llm_handler=llm_handler)

    # List of questions to rank
    questions = [
        "What is the capital of France?",
        "How to bake a cake?",
        "What is artificial intelligence?",
        "What is quantum computing?",
        "What are the benefits of exercise?",
    ]

    # Get the ranking of the questions
    response = llm_metrics.word_count(questions)

    # Output the response (which should be the ranking in JSON format)
    print(response)
