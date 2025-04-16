import typing

import strawberry
from asgiref.sync import sync_to_async
from strawberry.file_uploads import Upload

from content.serializers import (
    ArchiveContentSerializer,
    ContentSerializer,
    RetriggerContentSerializer,
    TagSerializer,
    UpdateContentSerializer,
)
from content.types import ContentType, TagType
from main.graphql.context import Info
from utils.strawberry.mutations import (
    ModelMutation,
    MutationResponseType,
    convert_serializer_to_type,
    mutation_is_not_valid,
    process_input_data,
)


@strawberry.input
class FolderInput:
    files: typing.List[Upload]


CreateContentMutation = ModelMutation("Content", ContentSerializer)
CreateTagMutation = ModelMutation("CreateTag", TagSerializer)
UpdateContentTitleInput = convert_serializer_to_type(UpdateContentSerializer, name="UpdateContentTitleInput")
RetriggerContentInput = convert_serializer_to_type(RetriggerContentSerializer, name="RetriggerContentInput")
ArchiveContentInput = convert_serializer_to_type(ArchiveContentSerializer, name="ArchiveContentInput")


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
    @sync_to_async
    def update_content_title(
        self,
        data: UpdateContentTitleInput,  # type: ignore[reportInvalidTypeForm]
        info: Info,
    ) -> MutationResponseType[ContentType]:
        serializer = UpdateContentSerializer(
            instance=info.context.request.user,
            data=process_input_data(data),
            context={"request": info.context.request},
        )
        if errors := mutation_is_not_valid(serializer):
            return MutationResponseType(
                ok=False,
                errors=errors,
            )
        content = serializer.save()  # type: ignore[reportReturnType]
        return MutationResponseType(
            result=content,  # type: ignore[reportReturnType]
        )

    @strawberry.mutation
    @sync_to_async
    def retrigger_content(
        self,
        data: RetriggerContentInput,  # type: ignore[reportInvalidTypeForm]
        info: Info,
    ) -> MutationResponseType[ContentType]:
        serializer = RetriggerContentSerializer(
            instance=info.context.request.user,
            data=process_input_data(data),
            context={"request": info.context.request},
        )
        if errors := mutation_is_not_valid(serializer):
            return MutationResponseType(
                ok=False,
                errors=errors,
            )
        content = serializer.save()  # type: ignore[reportReturnType]
        return MutationResponseType(
            result=content,  # type: ignore[reportReturnType]
        )

    @strawberry.mutation
    @sync_to_async
    def archive_content(
        self, data: ArchiveContentInput, info: Info  # type: ignore[reportInvalidTypeForm]
    ) -> MutationResponseType[ContentType]:
        serializer = ArchiveContentSerializer(
            instance=info.context.request.user,
            data=process_input_data(data),
            context={"request": info.context.request},
        )
        if errors := mutation_is_not_valid(serializer):
            return MutationResponseType(
                ok=False,
                errors=errors,
            )
        content = serializer.save()  # type: ignore[reportReturnType]
        return MutationResponseType(
            result=content,  # type: ignore[reportReturnType]
        )

    @strawberry.mutation
    async def create_tag(
        self,
        data: CreateTagMutation.InputType,  # type: ignore[reportInvalidTypeForm]
        info: Info,
    ) -> MutationResponseType[TagType]:
        return await CreateTagMutation.handle_create_mutation(data, info, None)
