from typing import Optional

import strawberry
import strawberry_django

from content.enums import DocumentStatusTypeEnum

from .models import Content, Tag


@strawberry_django.filter_type(Content, lookups=True)
class ContentFilter:
    id: Optional[strawberry.auto] = strawberry.UNSET
    document_status: Optional[DocumentStatusTypeEnum] = strawberry.UNSET


@strawberry_django.filter_type(Tag, lookups=True)
class TagFilter:
    name: strawberry.auto
