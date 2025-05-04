from typing import Optional

import strawberry
import strawberry_django

from content.enums import DocumentStatusTypeEnum

from .models import Content, Tag


@strawberry_django.filter_type(Content, lookups=True)
class ContentFilter:
    id: strawberry.auto
    document_status: Optional[DocumentStatusTypeEnum]  # type: ignore[reportInvalidTypeForm]
    created_at: strawberry.auto


@strawberry_django.filter_type(Tag, lookups=True)
class TagFilter:
    name: Optional[str]
