# Create your views here.
import asyncio

from rest_framework.generics import GenericAPIView
from rest_framework.response import Response

from chatbotcore.llm import OllamaHandler
from content.serializers import UserQuerySerializer


class UserQuery(GenericAPIView):
    llm = OllamaHandler()

    def post(self, request, *arg, **kwargs):
        serializer = UserQuerySerializer(data=request.data)
        if serializer.is_valid():
            result = asyncio.run(self.llm.execute_chain(request.data["user_id"], request.data["query"]))
            return Response(result)
        return Response(serializer.errors, 422)
