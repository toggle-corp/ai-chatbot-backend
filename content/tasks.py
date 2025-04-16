import logging

import requests
from celery import shared_task
from django.conf import settings

from chatbotcore.database import QdrantDatabase
from chatbotcore.doc_loaders import LoaderFromText

logger = logging.getLogger(__name__)


@shared_task(bind=True)
def create_embedding_for_content_task(self, content_id):
    from content.models import Content

    content = Content.objects.get(id=content_id)
    url = f"{settings.EMBEDDING_MODEL_URL}/get_embeddings"
    headers = {"Content-Type": "application/json"}
    data = content.extracted_file.read()
    loader = LoaderFromText(text=data)
    split_docs = loader.create_document_chunks()

    payload = {
        "type_model": settings.EMBEDDING_MODEL_TYPE,
        "name_model": settings.EMBEDDING_MODEL_NAME,
        "texts": [split_docs[i].page_content for i in range(len(split_docs))],
    }
    response = requests.post(url=url, headers=headers, json=payload)
    metadata = [
        {"source": "plain-text", "page_content": split_docs[i].page_content, "uuid": content.content_id}
        for i in range(len(split_docs))
    ]
    try:
        db = QdrantDatabase(
            host=settings.QDRANT_DB_HOST, port=settings.QDRANT_DB_PORT, collection_name=settings.QDRANT_DB_COLLECTION_NAME
        )
        db.set_collection()
        db.store_data(zip(response.json(), metadata))
        content.document_status = Content.DocumentStatus.ADDED_TO_VECTOR

    # NOTE: All exceptions have been handled with except
    except Exception:
        logger.error("An error occurred while creating embeddings", exc_info=True)
        content.document_status = Content.DocumentStatus.FAILURE
    content.save()


@shared_task(bind=True)
def delete_content_from_qdrant_task(self, content_id):
    db = QdrantDatabase(
        host=settings.QDRANT_DB_HOST, port=settings.QDRANT_DB_PORT, collection_name=settings.QDRANT_DB_COLLECTION_NAME
    )
    db.delete_data_by_src_uuid(key="uuid", value=str(content_id))
    return logger.info(f"Deleted content {content_id}")
