# Create your views here.
import asyncio

from django.conf import settings
from rest_framework.generics import GenericAPIView
from rest_framework.response import Response

from chatbotcore.llm import OllamaHandler, OpenAIHandler
from chatbotcore.utils import LLMType
from content.serializers import UserQuerySerializer


class UserQuery(GenericAPIView):
    if LLMType(int(settings.LLM_TYPE)) == LLMType.OLLAMA:
        llm = OllamaHandler()
    elif LLMType(int(settings.LLM_TYPE)) == LLMType.OPENAI:
        llm = OpenAIHandler()
    else:
        raise Exception("Wrong LLM Type")

    def post(self, request, *arg, **kwargs):
        serializer = UserQuerySerializer(data=request.data)
        if serializer.is_valid():
            result = asyncio.run(self.llm.execute_chain(request.data["user_id"], request.data["query"]))
            return Response(result)
        return Response(serializer.errors, 422)
