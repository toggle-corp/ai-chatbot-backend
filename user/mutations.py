import strawberry
from asgiref.sync import sync_to_async
from django.contrib.auth import login, logout
from strawberry.types import Info

from main.graphql.permissions import IsOrganizationAdmin
from user.serializers import (
    AddUserSerializer,
    ChangePasswordSerializer,
    ForgotpasswordSerializer,
    LoginSerializer,
    UpdateMeSerializer,
    UserActivationSerializer,
    UserDeactivationSerializer,
    UserPasswordResetConfirmSerializer,
    UserPasswordResetTriggerSerializer,
    UserRegisterSerializer,
    UserResendInviteSerializer,
)
from user.types import UserMeType, UserType
from utils.strawberry.mutations import (
    MutationEmptyResponseType,
    MutationResponseType,
    mutation_is_not_valid,
    process_input_data,
)
from utils.strawberry.transformers import convert_serializer_to_type

LoginInput = convert_serializer_to_type(LoginSerializer, name="LoginInput")

AddUserInput = convert_serializer_to_type(AddUserSerializer, name="AddUserInput")
RegisterUserInput = convert_serializer_to_type(UserRegisterSerializer, name="RegisterUserInput")
UserResendInviteInput = convert_serializer_to_type(UserResendInviteSerializer, name="UserResendInviteInput")
UserActivationInput = convert_serializer_to_type(UserActivationSerializer, name="UserActivationInput")
UserDeactivationInput = convert_serializer_to_type(UserDeactivationSerializer, name="UserDeactivationInput")
UserPasswordResetTriggerInput = convert_serializer_to_type(
    UserPasswordResetTriggerSerializer, name="UserPasswordResetTriggerInput"
)
UserPasswordReset = convert_serializer_to_type(UserPasswordResetConfirmSerializer, name="UserPasswordReset")
ChangePasswordInput = convert_serializer_to_type(ChangePasswordSerializer, name="ChangePasswordInput")
UpdateMeInput = convert_serializer_to_type(UpdateMeSerializer, name="UserMeInput")
ForgotPasswordInput = convert_serializer_to_type(ForgotpasswordSerializer, name="ResetUserPassword")


@strawberry.type
class PublicMutation:

    @strawberry.mutation
    @sync_to_async
    def login(
        self,
        data: LoginInput,  # type: ignore[reportInvalidTypeForm]
        info: Info,
    ) -> MutationResponseType[UserMeType]:
        serializer = LoginSerializer(data=process_input_data(data), context={"request": info.context.request})
        if errors := mutation_is_not_valid(serializer):
            return MutationResponseType(
                ok=False,
                errors=errors,
            )
        user = serializer.validated_data["user"]
        login(info.context.request, user)
        return MutationResponseType(
            result=user,
        )

    @strawberry.mutation(permission_classes=[IsOrganizationAdmin])
    @sync_to_async
    def add_user(self, info: Info, data: AddUserInput) -> MutationEmptyResponseType:  # type: ignore[reportInvalidTypeForm]
        serializer = AddUserSerializer(data=process_input_data(data), context={"request": info.context.request})
        if errors := mutation_is_not_valid(serializer):
            return MutationEmptyResponseType(
                ok=False,
                errors=errors,
            )
        serializer.save()
        return MutationEmptyResponseType(
            ok=True,  # type: ignore[reportReturnType]
        )

    @strawberry.mutation
    @sync_to_async
    def register_user(
        self, info: Info, data: RegisterUserInput  # type: ignore[reportInvalidTypeForm]
    ) -> MutationResponseType[UserType]:
        serializer = UserRegisterSerializer(data=process_input_data(data), context={"request": info.context.request})
        if errors := mutation_is_not_valid(serializer):
            return MutationResponseType(
                ok=False,
                errors=errors,
            )
        user = serializer.save()
        return MutationResponseType(result=user)  # type: ignore[reportReturnType]

    @strawberry.mutation(permission_classes=[IsOrganizationAdmin])
    @sync_to_async
    def resend_invite(
        self, data: UserResendInviteInput, info: Info  # type: ignore[reportInvalidTypeForm]
    ) -> MutationEmptyResponseType:
        serializer = UserResendInviteSerializer(
            data=process_input_data(data),
            context={"request": info.context.request},
        )
        if errors := mutation_is_not_valid(serializer):
            return MutationEmptyResponseType(
                ok=False,
                errors=errors,
            )

        serializer.save()
        return MutationEmptyResponseType(ok=True)

    @strawberry.mutation
    @sync_to_async
    def account_activation(
        self,
        data: UserActivationInput,  # type: ignore[reportInvalidTypeForm]
        info: Info,
    ) -> MutationEmptyResponseType:
        serializer = UserActivationSerializer(data=process_input_data(data), context={"request": info.context.request})
        if errors := mutation_is_not_valid(serializer):
            return MutationEmptyResponseType(
                ok=False,
                errors=errors,
            )
        serializer.save()
        return MutationEmptyResponseType()

    @strawberry.mutation(permission_classes=[IsOrganizationAdmin])
    @sync_to_async
    def account_deactivation(
        self,
        data: UserDeactivationInput,  # type: ignore[reportInvalidTypeForm]
        info: Info,
    ) -> MutationEmptyResponseType:
        serializer = UserDeactivationSerializer(data=process_input_data(data), context={"request": info.context.request})
        if errors := mutation_is_not_valid(serializer):
            return MutationEmptyResponseType(
                ok=False,
                errors=errors,
            )
        serializer.save()
        return MutationEmptyResponseType()

    @strawberry.mutation(permission_classes=[IsOrganizationAdmin])
    @sync_to_async
    def password_reset_trigger(
        self,
        data: UserPasswordResetTriggerInput,  # type: ignore[reportInvalidTypeForm]
        info: Info,
    ) -> MutationEmptyResponseType:
        serializer = UserPasswordResetTriggerSerializer(
            data=process_input_data(data),
            context={"request": info.context.request},
        )
        if errors := mutation_is_not_valid(serializer):
            return MutationEmptyResponseType(
                ok=False,
                errors=errors,
            )
        serializer.save()
        return MutationEmptyResponseType()

    @strawberry.mutation
    @sync_to_async
    def password_reset(
        self,
        data: UserPasswordReset,  # type: ignore[reportInvalidTypeForm]
        info: Info,
    ) -> MutationEmptyResponseType:
        serializer = UserPasswordResetConfirmSerializer(
            data=process_input_data(data),
            context={"request": info.context.request},
        )
        if errors := mutation_is_not_valid(serializer):
            return MutationEmptyResponseType(
                ok=False,
                errors=errors,
            )
        serializer.save()
        return MutationEmptyResponseType()

    @strawberry.mutation()
    @sync_to_async
    def forgot_password(
        self,
        data: ForgotPasswordInput,  # type: ignore[reportInvalidTypeForm]
        info: Info,
    ) -> MutationEmptyResponseType:
        serializer = ForgotpasswordSerializer(data=process_input_data(data), context={"request": info.context.request})
        if errors := mutation_is_not_valid(serializer):
            return MutationEmptyResponseType(
                ok=False,
                errors=errors,
            )
        serializer.save()
        return MutationEmptyResponseType()


@strawberry.type
class PrivateMutation:
    @strawberry.mutation
    @sync_to_async
    def logout(self, info: Info) -> MutationEmptyResponseType:
        if info.context.request.user.is_authenticated:
            logout(info.context.request)
            return MutationEmptyResponseType(ok=True)
        return MutationEmptyResponseType(ok=False)

    @strawberry.mutation
    @sync_to_async
    def change_password(
        self,
        data: ChangePasswordInput,  # type: ignore[reportInvalidTypeForm]
        info: Info,
    ) -> MutationEmptyResponseType:
        serializer = ChangePasswordSerializer(
            data=process_input_data(data),
            context={"request": info.context.request},
        )
        if errors := mutation_is_not_valid(serializer):
            return MutationEmptyResponseType(
                ok=False,
                errors=errors,
            )
        serializer.save()
        return MutationEmptyResponseType()

    @strawberry.mutation
    @sync_to_async
    def update_me(
        self,
        data: UpdateMeInput,  # type: ignore[reportInvalidTypeForm]
        info: Info,
    ) -> MutationResponseType[UserMeType]:
        serializer = UpdateMeSerializer(
            instance=info.context.request.user,
            data=process_input_data(data),
            context={"request": info.context.request},
            partial=True,
        )
        if errors := mutation_is_not_valid(serializer):
            return MutationResponseType(
                ok=False,
                errors=errors,
            )
        user = serializer.save()
        return MutationResponseType(
            result=user,  # type: ignore[reportReturnType]
        )
