import strawberry
import strawberry_django

from main.graphql.context import Info
from organization.types import OrganizationType
from utils.strawberry.paginations import CountList, pagination_field


@strawberry.type
class Query:

    Organizations: CountList[OrganizationType] = pagination_field(
        pagination=True,
    )

    @strawberry_django.field()
    async def Organization(self, info: Info, pk: strawberry.ID) -> OrganizationType | None:
        return await OrganizationType.get_queryset(None, None, info).filter(pk=pk).afirst()
