import requests
from celery import shared_task
from django.conf import settings

from chatbotcore.database import QdrantDatabase
from chatbotcore.doc_loaders import LoaderFromText


@shared_task(blind=True)
def create_embedding_for_content_task(content_id):
    from .models import Content

    content = Content.objects.get(id=content_id)
    url = settings.EMBEDDING_MODEL_URL
    headers = {"Content-Type": "application/json"}

    loader = LoaderFromText(text=content.description)
    split_docs = loader.create_document_chunks()

    payload = {
        "type_model": 1,
        "name_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "texts": [split_docs[i].page_content for i in range(len(split_docs))],
    }
    response = requests.post(url, headers=headers, json=payload)
    metadata = [
        {"source": "plain-text", "page_content": split_docs[i].page_content, "uuid": content.uuid}
        for i in range(len(split_docs))
    ]
    if response.status_code == 200:
        db = QdrantDatabase(host="qdrant", port=settings.QDRANT_DB_PORT, collection_name=settings.QDRANT_DB_COLLECTION_NAME)
        db.set_collection()
        db.store_data(zip(response.json(), metadata))
        content.document_status = Content.DocumentStatus.ADDED_TO_VECTOR
    else:
        content.document_status = Content.DocumentStatus.FAILURE
