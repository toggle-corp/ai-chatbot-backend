import requests
from celery import shared_task
from chatbotcore.doc_loaders import LoaderFromText
from chatbotcore.database import QdrantDatabase


@shared_task(blind=True)
def create_embedding_for_content_task(content_id):
    from .models import Content
    content = Content.objects.get(id=content_id)
    url = "http://192.168.88.11:8000/get_embeddings"
    headers = {"Content-Type": "application/json"}

    payload = {
        "type_model": 1,
        "name_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "texts": [content.description],
    }
    loader = LoaderFromText(text=content.description)
    split_docs = loader.create_document_chunks()

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        metadata = [{"source": "rawtext", "page_content": split_docs[i].page_content} for i in range(len(response.json()))]
        db = QdrantDatabase(host="localhost", port=6333, collection_name="test2")
        db.set_collection()
        db.store_data(zip(response.json(), metadata))
        content.document_status = Content.DocumentStatus.ADDED_TO_VECTOR
    else:
        content.document_status = Content.DocumentStatus.FAILURE
