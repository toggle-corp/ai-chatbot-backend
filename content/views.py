# Create your views here.
from rest_framework.decorators import api_view
from rest_framework.response import Response

from chatbotcore.llm import OllamaHandler
from content.serializers import UserQuerySerializer


@api_view(["POST"])
def chat(request):
    data = OllamaHandler()
    serializer = UserQuerySerializer(data=request.data)
    if serializer.is_valid():
        result = data.execute_chain(request.data["query"])
        return Response(result)
    return Response(serializer.errors)
