import os

from rest_framework import serializers

from content.models import Content, Tag


class UserQuerySerializer(serializers.Serializer):
    query = serializers.CharField(required=True, allow_null=False, allow_blank=False)
    user_id = serializers.UUIDField(required=True)


class TagSerializer(serializers.ModelSerializer):
    class Meta:
        model = Tag
        fields = ["name", "description"]


class ContentSerializer(serializers.ModelSerializer):
    tag = serializers.PrimaryKeyRelatedField(queryset=Tag.objects.all(), many=True, required=False)

    class Meta:
        model = Content
        fields = ["title", "document_file", "tag", "document_type"]
        read_only_fields = ["created_by", "modified_by"]

    def validate_document_file(self, file):
        max_size = 5 * 1024 * 1024  # Max file size 5MB
        if file.size > max_size:
            raise serializers.ValidationError("Document size should be less then 5MB.")

        allowed_extensions = ["pdf", "txt"]
        extension = os.path.splitext(file.name)[1].replace(".", "")
        if extension.lower() not in allowed_extensions:
            raise serializers.ValidationError("Only .pdf and .txt files are allowed.")
        return file

    def create(self, validated_data):
        tags = validated_data.pop("tag", [])
        validated_data["created_by"] = self.context["request"].user
        validated_data["modified_by"] = self.context["request"].user
        content = super().create(validated_data)
        content.tag.set(tags)
        return content


class UpdateContentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Content
        fields = ["title"]
        read_only_fields = ["modified_by"]

    def update(self, instance, validated_data):
        validated_data["modified_by"] = self.context["request"].user

        return super().update(instance, validated_data)


class ArchiveContentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Content
        fields = [
            "is_deleted",
        ]
        read_only_fields = ["deleted_at", "deleted_by"]

    def validate(self, data):
        instance = self.instance
        if instance.is_deleted:
            raise serializers.ValidationError("Content is already deleted.")
        return data

    def update(self, instance, validated_data):
        instance.is_deleted = True
        instance.deleted_by = self.context["request"].user
        instance.save(update_fields=("is_deleted", "deleted_by"))
        return instance
