import strawberry
from asgiref.sync import sync_to_async
from strawberry.types import Info

from organization.serializers import (
    AddOrganizationSerializer,
    UpdateOrganizationSerializer,
)
from organization.types import OrganizationType
from utils.strawberry.mutations import (
    MutationResponseType,
    _CustomErrorType,
    convert_serializer_to_type,
    mutation_is_not_valid,
    process_input_data,
)

AddOrganizationInputType = convert_serializer_to_type(AddOrganizationSerializer, name="AddOrganizationInputType")
UpdateOrganizationInputType = convert_serializer_to_type(UpdateOrganizationSerializer, name="UpdateOrganizationInputType")


@strawberry.type
class PrivateMutation:
    @strawberry.mutation
    @sync_to_async
    def add_organization(
        self, data: AddOrganizationInputType, info: Info  # type: ignore[reportInvalidTypeForm]
    ) -> MutationResponseType[OrganizationType]:
        serializer = AddOrganizationSerializer(data=process_input_data(data), context={"request": info.context.request})
        if errors := mutation_is_not_valid(serializer):
            return MutationResponseType(
                ok=False,
                errors=errors,
            )
        organization = serializer.save()
        return MutationResponseType(result=organization)  # type: ignore[reportInvalidTypeForm]

    @strawberry.mutation
    @sync_to_async
    def update_organization(
        self, data: UpdateOrganizationInputType, info: Info  # type: ignore[reportInvalidTypeForm]
    ) -> MutationResponseType[OrganizationType]:
        serializer = UpdateOrganizationSerializer(data=process_input_data(data), context={"request": info.context.request})
        if errors := mutation_is_not_valid(serializer):
            return MutationResponseType(
                ok=False,
                errors=errors,
            )
        organization = serializer.save()
        return MutationResponseType(result=organization)  # type: ignore[reportInvalidTypeForm]

    @strawberry.mutation
    @sync_to_async
    def delete_organization(
        self,
        id: strawberry.ID,
        info: Info,
    ) -> MutationResponseType[OrganizationType]:
        instance = OrganizationType.get_queryset(None, None, info).filter(id=id).first()
        if instance is None:
            return MutationResponseType(
                ok=False,
                errors=_CustomErrorType.generate_message(message="Organization not found"),
            )
        organization_id = instance.id
        instance.delete()
        instance.id = organization_id
        return MutationResponseType(
            result=instance,  # type: ignore[reportReturnType]
        )
