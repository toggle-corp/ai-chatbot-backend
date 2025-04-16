from django.db import transaction
from django.utils import timezone
from rest_framework import serializers

from chat.models import UserChatSession
from content.models import Content, Tag
from content.tasks import (
    create_embedding_for_content_task,
    delete_content_from_qdrant_task,
)
from utils.file_check import validate_document_size, validate_file_type


class UserQuerySerializer(serializers.Serializer):
    query = serializers.CharField(required=True, allow_null=False, allow_blank=False)
    user_id = serializers.UUIDField(required=True)
    platform = serializers.IntegerField(default=UserChatSession.Platform.STREAMLIT)


class TagSerializer(serializers.ModelSerializer):
    class Meta:
        model = Tag
        fields = ["name", "description"]


class ContentSerializer(serializers.ModelSerializer):
    tag = serializers.PrimaryKeyRelatedField(queryset=Tag.objects.all(), many=True, required=False)
    document_file = serializers.FileField(required=True)

    class Meta:
        model = Content
        fields = ["title", "document_file", "tag", "document_type", "document_url"]
        read_only_fields = ["created_by", "modified_by"]

    def validate_document_file(self, file):
        validate_document_size(file)
        validate_file_type(file)
        return file

    def create(self, validated_data):
        tags = validated_data.pop("tag", [])
        validated_data["created_by"] = self.context["request"].user
        validated_data["modified_by"] = self.context["request"].user
        content = super().create(validated_data)
        content.tag.set(tags)
        return content


class UpdateContentSerializer(serializers.ModelSerializer):
    # NOTE: Update only the content title for now
    content = serializers.PrimaryKeyRelatedField(queryset=Content.objects.all(), required=True)

    class Meta:
        model = Content
        fields = ["title", "content"]

    def save(self, **_):
        assert isinstance(self.validated_data, dict)
        content = self.validated_data["content"]
        content.title = self.validated_data["title"]
        content.modified_by = self.context["request"].user
        content.save(update_fields=["title", "modified_by"])
        return content


class ArchiveContentSerializer(serializers.ModelSerializer):
    """NOTE: Update the document status to DELETED_FROM_VECTOR in content model and delete the content from qdrant  db"""

    content = serializers.PrimaryKeyRelatedField(queryset=Content.objects.all(), required=True)

    def validate(self, attrs):
        content = attrs["content"]
        if content.document_status == Content.DocumentStatus.DELETED_FROM_VECTOR:
            raise serializers.ValidationError("Content is already deleted from vector.")
        return attrs

    class Meta:
        model = Content
        fields = ["content"]

    def save(self, **_):
        assert isinstance(self.validated_data, dict)
        content = self.validated_data["content"]
        content.document_status = Content.DocumentStatus.DELETED_FROM_VECTOR
        content.deleted_by = self.context["request"].user
        content.deleted_at = timezone.now()
        content.is_deleted = True
        content.save(update_fields=["document_status", "deleted_by", "deleted_at", "is_deleted"])
        transaction.on_commit(lambda: delete_content_from_qdrant_task.delay(content.content_id))
        return content


class RetriggerContentSerializer(serializers.ModelSerializer):
    content = serializers.PrimaryKeyRelatedField(queryset=Content.objects.all(), required=True)

    def validate(self, attrs):
        content = attrs["content"]
        if content.document_status == Content.DocumentStatus.ADDED_TO_VECTOR:
            raise serializers.ValidationError("Content has already been added to vector. No need to trigger it again.")
        return attrs

    class Meta:
        model = Content
        fields = ["content"]

    def save(self, **_):
        assert isinstance(self.validated_data, dict)
        content = self.validated_data["content"]
        transaction.on_commit(lambda: create_embedding_for_content_task.delay(content.id))
        return content
