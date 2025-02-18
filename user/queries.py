import strawberry
import strawberry_django
from asgiref.sync import sync_to_async

from main.graphql.context import Info
from user.types import UserMeType, UserType
from utils.strawberry.paginations import CountList, pagination_field


@strawberry.type
class PublicQuery:
    @strawberry.field
    @sync_to_async
    def me(self, info: Info) -> UserMeType | None:
        user = info.context.request.user
        if user.is_authenticated:
            return user  # type: ignore[reportGeneralTypeIssues]


@strawberry.type
class PrivateQuery:
    noop: strawberry.ID = strawberry.ID("noop")

    users: CountList[UserType] = pagination_field(
        pagination=True,
    )

    @strawberry_django.field()
    async def all_users(self, info: Info) -> list[UserType]:
        return [users async for users in UserType.get_queryset(None, None, info)]
