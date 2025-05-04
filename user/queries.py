import strawberry
import strawberry_django
from asgiref.sync import sync_to_async
from strawberry_django.pagination import OffsetPaginated

from main.graphql.context import Info
from main.graphql.permissions import IsAdminOrSuperuser
from user.filters import UserFilter
from user.orders import UserOrder
from user.types import UserMeType, UserType


class Query:
    @strawberry.field
    @sync_to_async
    def me(self, info: Info) -> UserMeType | None:
        user = info.context.request.user
        if user.is_authenticated:
            return user  # type: ignore[reportGeneralTypeIssues]
        return None

    #  Paginated
    users: OffsetPaginated[UserType] = strawberry_django.offset_paginated(
        filters=UserFilter, order=UserOrder, extensions=[IsAdminOrSuperuser()]
    )

    user: UserType = strawberry_django.field(extensions=[IsAdminOrSuperuser()])
