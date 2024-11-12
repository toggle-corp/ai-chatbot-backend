import strawberry
import strawberry_django

from main.graphql.context import Info
from utils.strawberry.paginations import CountList, pagination_field

from .types import ContentType


@strawberry.type
class PrivateQuery:
    content: CountList[ContentType] = pagination_field(
        pagination=True,
    )

    @strawberry_django.field(description="Return all content")
    async def all_content(self, info: Info) -> list[ContentType]:
        return [content async for content in ContentType.get_queryset(None, None, info).filter(created_by=info.context.user)]
