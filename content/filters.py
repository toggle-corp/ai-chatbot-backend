import strawberry
import strawberry_django

from content.enums import DocumentStatusTypeEnum

from .models import Content, Tag


@strawberry_django.filter_type(Content, lookups=True)
class ContentFilter:
    id: strawberry.auto
    document_status: DocumentStatusTypeEnum


@strawberry_django.filter_type(Tag, lookups=True)
class TagFilter:
    name: strawberry.auto
