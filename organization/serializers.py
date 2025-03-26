from rest_framework import serializers

from organization.models import Organization


class AddOrganizationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Organization
        fields = (
            "name",
            "slider_bar_color",
            "navbar_color",
            "image",
        )

    def create(self, validated_data):
        validated_data["created_by"] = self.context["request"].user
        validated_data["modified_by"] = self.context["request"].user
        validated_data["image"] = validated_data.get("image", None)
        content = super().create(validated_data)
        return content


class UpdateOrganizationSerializer(serializers.ModelSerializer):
    image = serializers.ImageField(allow_null=True, required=False)
    organization = serializers.PrimaryKeyRelatedField(
        queryset=Organization.objects.all(),
        required=True,
    )

    class Meta:
        model = Organization
        fields = (
            "name",
            "slider_bar_color",
            "navbar_color",
            "image",
            "organization",
        )

    def save(self, **_):
        assert isinstance(self.validated_data, dict)
        # organization is an object/instance representing the related organization
        organization = self.validated_data["organization"]
        organization.modified_by = self.context["request"].user
        organization.name = self.validated_data["name"]
        organization.slider_bar_color = self.validated_data["slider_bar_color"]
        organization.navbar_color = self.validated_data["navbar_color"]
        organization.image = self.validated_data.get("image", organization.image)
        organization.save(update_fields=["name", "slider_bar_color", "navbar_color", "image", "modified_by"])
        return organization
