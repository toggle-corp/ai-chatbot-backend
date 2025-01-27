from collections import Counter

from django.conf import settings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from nltk.corpus import stopwords

from chatbotcore.llm import OllamaHandler, OpenAIHandler
from chatbotcore.utils import LLMType


class LLMcoreMetrics:

    def __init__(self):
        self.questions = None
        if LLMType(int(settings.LLM_TYPE)) == LLMType.OLLAMA:
            self.llm = OllamaHandler()
        elif LLMType(int(settings.LLM_TYPE)) == LLMType.OPENAI:
            self.llm = OpenAIHandler()

    def top_word_questions(self, questions, num_questions, num_words):
        self.questions = questions
        self.top_questions_from_llm = self.top_questions(questions=questions, num_questions=num_questions)
        self.word_count_counter = self.word_count(questions=questions, num_words=num_words)
        return {"top_questions": self.top_questions_from_llm, "word_cloud": self.word_count_counter}

    def prompt_top_questions(self):

        top_questions_prompt = PromptTemplate(
            input_variables=["questions"],
            template="""
                You are an HR assistant chatbot, designed to help with HR-related queries.
                Given the following list of questions, please rank them from the most insightful to the least insightful.
                Do not add any additional information, commentary, or context.
                Simply return the top {num_ques} questions in ranked order and disregard questions that seem nonsensical.

                Questions:
                {questions}
                """,
        )
        return top_questions_prompt

    def create_chain(self, prompt: PromptTemplate):
        """
        Create the LLM chain to run the prompt with the LLM model
        """
        return LLMChain(llm=self.llm.llm_model, prompt=prompt)

    def word_count(self, questions: list, num_words: int = 10):
        """
        Use the LLM model to get the ranking of the questions based on word count (excluding stop words)
        """
        stop_words = set(stopwords.words("english"))
        words = [word.lower() for sentence in questions for word in sentence.split() if word.lower() not in stop_words]

        # Count word frequencies
        word_count = Counter(words)
        sorted_by_value = sorted(word_count.items(), key=lambda x: x[1], reverse=True)

        # Convert the sorted list of tuples back into an OrderedDict (optional)
        return sorted_by_value[:num_words]

    def top_questions(self, questions: list, num_questions: int = 3):
        """
        Use the LLM model to get the ranking of the questions based on word count (excluding stop words)
        """
        prompt_questions = self.prompt_top_questions()
        llm_chain = self.create_chain(prompt_questions)

        # Run the chain with the formatted questions text
        response = llm_chain.run({"questions": questions, "num_ques": num_questions})
        # print("the response of the llm chain is %s", response)
        # print("----------------------------------------------")
        return response


# llm_metrics = LLM
