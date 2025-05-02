import strawberry
import strawberry_django

from content.enums import DocumentStatusTypeEnum, DocumentTypeEnum
from content.models import Content, Tag


@strawberry_django.type(Tag)
class TagType:
    id: strawberry.ID
    name: strawberry.auto
    description: strawberry.auto


@strawberry_django.type(Tag)
class TagNameType:
    id: strawberry.ID
    name: strawberry.auto


@strawberry_django.type(Content)
class ContentType:
    id: strawberry.ID
    title: strawberry.auto
    extracted_file: strawberry.auto
    created_at: strawberry.auto
    tag: list[TagNameType]
    document_status: DocumentStatusTypeEnum
    document_type: DocumentTypeEnum
    is_deleted: strawberry.auto
    content_id: strawberry.auto
    deleted_at: strawberry.auto
    deleted_by: strawberry.auto
    document_file: strawberry.auto
