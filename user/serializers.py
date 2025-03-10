from django.contrib.auth import authenticate
from django.contrib.auth.password_validation import validate_password
from django.db import transaction
from django.shortcuts import get_object_or_404
from django.utils.http import urlsafe_base64_decode
from django.utils.translation import gettext
from rest_framework import serializers

from main.token import TokenManager
from user.models import User
from user.tasks import (
    resend_account_activation_task,
    send_account_creation_email_task,
    send_password_reset_email_task,
)


class LoginSerializer(serializers.Serializer):
    email = serializers.CharField()
    password = serializers.CharField(write_only=True)

    def validate(self, attrs):
        authenticate_user = authenticate(
            email=attrs["email"].lower(),
            password=attrs["password"],
        )
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


class AddUserSerializer(serializers.Serializer):
    emails = serializers.CharField(required=True)

    def validate(self, validated_data):
        emails = [email.strip().lower() for email in validated_data["emails"].split(",")]
        registered_emails = User.objects.filter(email__in=emails).values_list("email", flat=True)
        new_emails = list(set(emails) - set(registered_emails))
        if not new_emails:
            raise serializers.ValidationError("All emails are already registered. provide new emails")
        validated_data["new_emails"] = new_emails
        return validated_data

    def create(self, validated_data):
        new_emails = validated_data.get("new_emails")
        new_users = []
        with transaction.atomic():
            for email in new_emails:
                user = User.objects.create_user(
                    email=email,
                    password=None,
                    is_active=False,
                )
                transaction.on_commit(lambda user_id=user.id: send_account_creation_email_task.delay(user_id))
                new_users.append(user)
        return new_users


class UserRegisterSerializer(serializers.ModelSerializer):
    uuid = serializers.CharField(required=True)
    token = serializers.CharField(required=True)
    confirm_password = serializers.CharField(write_only=True, required=True)

    class Meta:
        model = User
        fields = ("uuid", "token", "password", "confirm_password", "first_name", "last_name", "department")
        extra_kwargs = {"password": {"write_only": True}}

    def validate_password(self, password):
        validate_password(password)
        return password

    def validate(self, attrs):
        if attrs["password"] != attrs["confirm_password"]:
            raise serializers.ValidationError(gettext("Passwords do not match."))
        return {**attrs, "user": validate_token(attrs, TokenManager.account_registration_token_generator)}

    def save(self, **_):
        assert isinstance(self.validated_data, dict)
        user = self.validated_data["user"]
        user.first_name = self.validated_data["first_name"]
        user.last_name = self.validated_data["last_name"]
        user.department = self.validated_data["department"]
        user.set_password(self.validated_data["password"])
        user.is_active = True
        user.save(update_fields=("first_name", "last_name", "department", "password", "is_active"))
        return user


class UserResendInviteSerializer(serializers.Serializer):
    user_id = serializers.CharField(required=True)

    def validate(self, attrs):
        user_id = attrs["user_id"]
        user = get_object_or_404(User, id=user_id)
        if user.is_active:
            raise serializers.ValidationError(gettext("User is already active."))
        return {
            **attrs,
            "user": user,
        }

    def save(self, **_):
        assert isinstance(self.validated_data, dict)
        user_id = self.validated_data["user_id"]
        transaction.on_commit(lambda: resend_account_activation_task.delay(user_id))


class UserActivationSerializer(serializers.Serializer):
    uuid = serializers.CharField(required=True)
    token = serializers.CharField(required=True)

    def validate(self, attrs):
        return {**attrs, "user": validate_token(attrs, TokenManager.account_reactivation_token_generator)}

    def save(self, **_):
        assert isinstance(self.validated_data, dict)
        user = self.validated_data["user"]
        user.is_active = True
        user.save(update_fields=("is_active",))


class UserDeactivationSerializer(serializers.Serializer):
    user_id = serializers.CharField(required=True)

    def validate(self, attrs):
        user_id = attrs["user_id"]
        user = get_object_or_404(User, id=user_id)
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
    user_id = serializers.CharField(required=True)

    def validate(self, attrs):
        user_id = attrs["user_id"]
        user = get_object_or_404(User, id=user_id)
        return {
            **attrs,
            "user": user,
        }

    def save(self, **_):
        assert isinstance(self.validated_data, dict)
        user_id = self.validated_data["user_id"]
        transaction.on_commit(lambda: send_password_reset_email_task.delay(user_id))


class ForgotpasswordSerializer(serializers.Serializer):
    email = serializers.EmailField(required=True)

    def validate(self, attrs):
        email = attrs["email"].lower()
        user = User.objects.filter(email=email).first()
        if user is None:
            raise serializers.ValidationError(gettext("User with that email doesn't exists!!"))
        return {
            **attrs,
            "user_id": user.id,
        }

    def save(self, **_):
        assert isinstance(self.validated_data, dict)
        user_id = self.validated_data["user_id"]
        transaction.on_commit(lambda: send_password_reset_email_task.delay(user_id))


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
        )
