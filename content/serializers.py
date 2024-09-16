from rest_framework import serializers


class UserQuerySerializer(serializers.Serializer):
    message = serializers.CharField(required=True, allow_null=False, allow_blank=False)
