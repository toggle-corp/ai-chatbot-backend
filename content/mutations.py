import typing

import strawberry
from asgiref.sync import sync_to_async
from strawberry.file_uploads import Upload

from content.models import Content
from content.serializers import (
    ArchiveContentSerializer,
    ContentSerializer,
    TagSerializer,
    UpdateContentSerializer,
)
from content.types import ContentType, TagType
from main.graphql.context import Info
from utils.strawberry.mutations import (
    ModelMutation,
    MutationEmptyResponseType,
    MutationResponseType,
    mutation_is_not_valid,
    process_input_data,
)


@strawberry.input
class FolderInput:
    files: typing.List[Upload]


CreateContentMutation = ModelMutation("Content", ContentSerializer)
UpdateMutation = ModelMutation("UpdateContent", UpdateContentSerializer)
DeleteContent = ModelMutation("archive", ArchiveContentSerializer)
CreateTagMutation = ModelMutation("CreateTag", TagSerializer)


@strawberry.type
class PrivateMutation:
    @strawberry.mutation
    async def create_content(
        self,
        data: CreateContentMutation.InputType,  # type: ignore[reportInvalidTypeForm]
        info: Info,
    ) -> MutationResponseType[ContentType]:
        return await CreateContentMutation.handle_create_mutation(data, info, None)

    @strawberry.mutation
    def read_file(self, file: Upload) -> str:
        return file.read().decode("utf-8")

    @strawberry.mutation
    async def update_content_title(
        self,
        id: strawberry.ID,
        data: UpdateMutation.PartialInputType,  # type: ignore[reportInvalidTypeForm]
        info: Info,
    ) -> MutationResponseType[ContentType]:
        try:
            instance = await Content.objects.aget(id=id)
        except Content.DoesNotExist:
            return MutationResponseType(ok=False, errors=["Content not found"])
        serializer = UpdateContentSerializer(
            instance, data=process_input_data(data), context={"request": info.context.request}, partial=True
        )
        if errors := mutation_is_not_valid(serializer):
            return MutationResponseType(ok=False, errors=errors)
        await sync_to_async(serializer.save)()
        return MutationResponseType()

    @strawberry.mutation
    async def archive_content(
        self, id: strawberry.ID, info: Info, data: DeleteContent.PartialInputType  # type: ignore[reportInvalidTypeForm]
    ) -> MutationEmptyResponseType:
        try:
            instance = await Content.objects.aget(id=id)
        except Content.DoesNotExist:
            return MutationEmptyResponseType(ok=False, errors=["Content not found"])

        serializer = ArchiveContentSerializer(
            instance, data=process_input_data(data), context={"request": info.context.request}, partial=True
        )

        if errors := mutation_is_not_valid(serializer):
            return MutationEmptyResponseType(ok=False, errors=errors)

        await sync_to_async(serializer.save)()

        return MutationEmptyResponseType(ok=True)

    @strawberry.mutation
    async def create_tag(
        self,
        data: CreateTagMutation.InputType,  # type: ignore[reportInvalidTypeForm]
        info: Info,
    ) -> MutationResponseType[TagType]:
        return await CreateTagMutation.handle_create_mutation(data, info, None)
