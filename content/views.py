# Create your views here.
import asyncio

from django.conf import settings
from rest_framework.generics import GenericAPIView
from rest_framework.response import Response

from chat.views import UserSession
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
    Session = UserSession()

    def post(self, request, *arg, **kwargs):
        serializer = UserQuerySerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user_session = self.Session.create_chat_session(request.data)
        user_message = self.Session.create_chat_message(request.data, user_session)
        result = asyncio.run(self.llm.execute_chain(request.data["user_id"], request.data["query"]))
        self.Session.update_chat_message(result, user_message)
        return Response(result)
