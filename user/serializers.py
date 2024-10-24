from django.contrib.auth import authenticate
from rest_framework import serializers



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
