import strawberry_django
from strawberry_django.pagination import OffsetPaginated
from strawberry_django.permissions import IsAuthenticated

from content.filters import TagFilter
from content.types import ContentType, TagType


class Query:
    contents: OffsetPaginated[ContentType] = strawberry_django.offset_paginated(
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
