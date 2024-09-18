from rest_framework import serializers


class UserQuerySerializer(serializers.Serializer):
    query = serializers.CharField(required=True, allow_null=False, allow_blank=False)
