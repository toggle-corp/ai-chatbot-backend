# Create your views here.
from django.http import JsonResponse, HttpResponse
from chatbotcore.llm import OllamaHandler

def chat(request):
    data = OllamaHandler()
    result=data.execute_chain("Leave policy")
    return HttpResponse(result)
