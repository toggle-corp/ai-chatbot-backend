import strawberry
import strawberry_django
from strawberry_django.pagination import OffsetPaginated
from strawberry_django.permissions import IsAuthenticated

from content.filters import ContentFilter, TagFilter
from content.orders import ContentOrder
from content.types import ContentType, TagType


@strawberry.type
class Query:
    contents: OffsetPaginated[ContentType] = strawberry_django.offset_paginated(
        filters=ContentFilter,
        order=ContentOrder,
        extensions=[IsAuthenticated()],
    )

    content: ContentType = strawberry_django.field(
        extensions=[IsAuthenticated()],
    )
    tags: OffsetPaginated[TagType] = strawberry_django.offset_paginated(
        filters=TagFilter,
        extensions=[IsAuthenticated()],
    )

    tag: TagType = strawberry_django.field(
        extensions=[IsAuthenticated()],
    )
