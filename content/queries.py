import strawberry
import strawberry_django

from content.types import ContentType, TagType
from main.graphql.context import Info
from utils.strawberry.paginations import CountList, pagination_field


@strawberry.type
class PrivateQuery:
    content: CountList[ContentType] = pagination_field(
        pagination=True,
    )

    @strawberry_django.field(description="Return all content")
    async def all_content(self, info: Info) -> list[ContentType]:
        return [content async for content in ContentType.get_queryset(None, None, info)]

    tag: CountList[ContentType] = pagination_field(
        pagination=True,
    )

    @strawberry_django.field()
    async def all_tags(self, info: Info) -> list[TagType]:
        return [tag async for tag in TagType.get_queryset(None, None, info)]
