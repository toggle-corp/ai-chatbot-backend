import strawberry
import strawberry_django

from content.types import ContentType, TagType
from main.graphql.context import Info
from utils.strawberry.paginations import CountList, pagination_field


@strawberry.type
class PrivateQuery:
    contents: CountList[ContentType] = pagination_field(
        pagination=True,
    )

    @strawberry_django.field(description="Return all content")
    async def content(self, info: Info, pk: strawberry.ID) -> ContentType | None:
        return await ContentType.get_queryset(None, None, info).filter(pk=pk).afirst()

    tags: CountList[TagType] = pagination_field(
        pagination=True,
    )

    @strawberry_django.field()
    async def tag(self, info: Info, pk: strawberry.ID) -> TagType | None:
        return await TagType.get_queryset(None, None, info).filter(pk=pk).afirst()
