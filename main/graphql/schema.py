import strawberry
from django.core.files.uploadedfile import UploadedFile
from strawberry.django.views import AsyncGraphQLView
from strawberry.file_uploads import Upload
from strawberry_django.optimizer import DjangoOptimizerExtension

# import utils.graphql.monkey_patches  # noqa: F401  type: ignore
from content import mutations as content_mutations
from content import queries as content_queries
from user import mutations as user_mutations
from user import queries as user_queries

from .context import GraphQLContext
from .dataloaders import GlobalDataLoader
from .enums import AppEnumCollection, AppEnumCollectionData

# from strawberry.schema.config import StrawberryConfig


class CustomAsyncGraphQLView(AsyncGraphQLView):
    async def get_context(self, *args, **kwargs) -> GraphQLContext:  # type: ignore[reportIncompatibleMethodOverride]
        return GraphQLContext(
            *args,
            **kwargs,
            dl=GlobalDataLoader(),
        )


@strawberry.type
class Query(
    user_queries.Query,
    content_queries.Query,
):
    enums: AppEnumCollection = strawberry.field(  # type: ignore[reportGeneralTypeIssues]
        resolver=lambda: AppEnumCollectionData(),
    )


@strawberry.type
class Mutation(
    user_mutations.Mutation,
    content_mutations.Mutation,
): ...  # noqa: E701


schema = strawberry.Schema(
    query=Query,
    mutation=Mutation,
    extensions=[
        DjangoOptimizerExtension,
    ],
    scalar_overrides={
        UploadedFile: Upload,
    },
    # config=StrawberryConfig(auto_camel_case=True)
)
