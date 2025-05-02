import strawberry
import strawberry_django

from content.enums import DocumentStatusTypeEnum

from .models import Content, Tag


@strawberry_django.filters.filter(Content, lookups=True)
class ContentFilter:
    id: strawberry.auto
    status: DocumentStatusTypeEnum


@strawberry_django.filters.filter(Tag, lookups=True)
class TagFilter:
    name: strawberry.auto
