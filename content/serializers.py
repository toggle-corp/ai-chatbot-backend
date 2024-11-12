from rest_framework import serializers

from content.models import Content


class UserQuerySerializer(serializers.Serializer):
    query = serializers.CharField(required=True, allow_null=False, allow_blank=False)
    user_id = serializers.UUIDField(required=True)


class ContentSerializers(serializers.ModelSerializer):
    created_by = serializers.HiddenField(default=serializers.CurrentUserDefault())

    class Meta:
        model = Content
        fields = [
            "title",
            "document_file",
            "tag",
            "created_by",
        ]

    def create(self, validated_data):
        return super().create(validated_data)
