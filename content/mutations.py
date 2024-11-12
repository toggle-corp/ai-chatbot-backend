import strawberry

from main.graphql.context import Info
from utils.common import get_object_or_404_async
from utils.strawberry.mutations import ModelMutation, MutationResponseType

from .serializers import ContentSerializers
from .types import ContentType
from strawberry.file_uploads import Upload

ContentMutation = ModelMutation("Content", ContentSerializers)


@strawberry.type
class PrivateMutation:
    @strawberry.mutation
    async def create_content(
        self,
        data: ContentMutation.InputType,  # type: ignore[reportInvalidTypeForm]
        info: Info,
    ) -> MutationResponseType[ContentType]:
        return await ContentMutation.handle_create_mutation(data, info, None)

    @strawberry.mutation
    def read_file(self, file: Upload) -> str:
        return file.read().decode("utf-8")
