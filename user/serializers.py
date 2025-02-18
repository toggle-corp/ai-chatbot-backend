from django.contrib.auth import authenticate
from django.contrib.auth.password_validation import validate_password
from django.utils.http import urlsafe_base64_decode
from django.utils.translation import gettext
from rest_framework import serializers

from main.token import TokenManager
from user.models import User
from user.utils import resend_account_activation, send_password_reset


class LoginSerializer(serializers.Serializer):
    email = serializers.CharField()
    password = serializers.CharField(write_only=True)

    def validate(self, attrs):
        # NOTE: authenticate only works for active users
        authenticate_user = authenticate(
            email=attrs["email"].lower(),
            password=attrs["password"],
        )
        # User doesn't exists in the system.
        if authenticate_user is None:
            raise serializers.ValidationError("No active account found with the given credentials")
        return {"user": authenticate_user}


def validate_token(attrs, token_generator) -> User:
    try:
        uid = urlsafe_base64_decode(attrs["uuid"]).decode("utf-8")
        user = User.objects.get(pk=uid)
    except (
        TypeError,
        ValueError,
        OverflowError,
        User.DoesNotExist,
    ):
        user = None

    if user is not None and token_generator.check_token(user, attrs["token"]):
        return user
    raise serializers.ValidationError(gettext("Invalid or expired token"))


class AddUserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ("email", "password", "first_name", "last_name", "department")

        def validate_email(self, email) -> str:
            if User.objects.filter(email__iexact=email).exists():
                raise serializers.ValidationError(gettext("This email is already registered."))
            return email.lower()

        def validate_password(self, password):
            validate_password(password=password)
            return password

        def create(self, validated_data):
            password = validated_data.pop("password")
            user = User(**validated_data)
            user.set_password(password)
            user.save()
            return user


class EditUserSerializer(serializers.ModelSerializer):
    id = serializers.IntegerField(
        required=True,
    )

    class Meta:
        model = User
        fields = ["id", "first_name", "last_name", "department", "is_active"]


class UserResendInviteSerializer(serializers.Serializer):
    user_id = serializers.CharField(required=True)

    def validate(self, attrs):
        user_id = attrs["user_id"]
        user = User.objects.filter(id=user_id).first()
        if user is None:
            raise serializers.ValidationError(gettext("User not found."))

        if user.is_active:
            raise serializers.ValidationError(gettext("User is already active. No need to resend an invite."))
        return {
            **attrs,
            "user": user,
        }

    def save(self, **_):
        assert isinstance(self.validated_data, dict)
        user = self.validated_data["user"]
        resend_account_activation(user=user)


class UserActivationSerializer(serializers.Serializer):
    uuid = serializers.CharField(required=True)
    token = serializers.CharField(required=True)

    def validate(self, attrs):
        return {**attrs, "user": validate_token(attrs, TokenManager.account_activation_token_generator)}

    def save(self, **_):
        assert isinstance(self.validated_data, dict)
        user = self.validated_data["user"]
        user.is_active = True
        user.save(update_fields=("is_active",))


class UserDeactivationSerializer(serializers.Serializer):
    user_id = serializers.CharField(required=True)

    def validate(self, attrs):
        user_id = attrs["user_id"]
        user = User.objects.filter(id=user_id).first()
        if user is None:
            raise serializers.ValidationError("User not found.")
        if not user.is_active:
            raise serializers.ValidationError(gettext("User is already inactive."))
        return {
            **attrs,
            "user": user,
        }

    def save(self, **_):
        assert isinstance(self.validated_data, dict)
        user = self.validated_data["user"]
        user.is_active = False
        user.save(update_fields=["is_active"])


class UserPasswordResetTriggerSerializer(serializers.Serializer):
    email = serializers.EmailField(required=True)

    def validate(self, attrs):
        email = attrs["email"].lower()
        user = User.objects.filter(email=email).first()
        if user is None:
            raise serializers.ValidationError(gettext("User with that email doesn't exists!!"))
        return {
            **attrs,
            "user": user,
        }

    def save(self, **_):
        assert isinstance(self.validated_data, dict)
        user = self.validated_data["user"]
        send_password_reset(user=user)


class UserPasswordResetConfirmSerializer(serializers.Serializer):
    uuid = serializers.CharField(required=True)
    token = serializers.CharField(required=True)
    new_password = serializers.CharField(required=True)
    confirm_new_password = serializers.CharField(required=True)

    def validate_new_password(self, password):
        validate_password(password)
        return password

    def validate(self, attrs):
        if attrs["new_password"] != attrs["confirm_new_password"]:
            raise serializers.ValidationError(gettext("Passwords do not match."))
        return {**attrs, "user": validate_token(attrs, TokenManager.password_reset_token_generator)}

    def save(self, **_):
        assert isinstance(self.validated_data, dict)
        user = self.validated_data["user"]
        new_password = self.validated_data["new_password"]
        user.set_password(new_password)
        user.save(update_fields=("password",))


class ChangePasswordSerializer(serializers.Serializer):
    old_password = serializers.CharField(required=True)
    new_password = serializers.CharField(required=True)
    confirm_new_password = serializers.CharField(required=True)

    def validate_old_password(self, password):
        user = self.context["request"].user
        if not user.check_password(password):
            raise serializers.ValidationError(gettext("Invalid old Password"))
        return password

    def validate(self, attrs):
        if attrs["old_password"] == attrs["new_password"]:
            raise serializers.ValidationError(gettext("New and old provided passwords are same"))
        if attrs["new_password"] != attrs["confirm_new_password"]:
            raise serializers.ValidationError(gettext("Passwords do not match."))
        return attrs

    def validate_new_password(self, password):
        validate_password(password)
        return password

    def save(self, **_):
        assert isinstance(self.validated_data, dict)
        user = self.context["request"].user
        new_password = self.validated_data["new_password"]
        user.set_password(new_password)
        user.save(update_fields=("password",))


class UpdateMeSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = (
            "first_name",
            "last_name",
            "email",
        )
