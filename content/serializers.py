from rest_framework import serializers

from chat.models import UserChatSession


class UserQuerySerializer(serializers.Serializer):
    query = serializers.CharField(required=True, allow_null=False, allow_blank=False)
    user_id = serializers.UUIDField(required=True)
    platform = serializers.IntegerField(default=UserChatSession.Platform.STREAMLIT)
