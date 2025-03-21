import strawberry
import strawberry_django
from django.db import models

from main.graphql.context import Info
from organization.models import Organization
from utils.common import get_queryset_for_model


@strawberry_django.type(Organization)
class OrganizationType:
    id: strawberry.ID
    name: strawberry.auto
    slider_bar_color: strawberry.auto
    navbar_color: strawberry.auto

    @staticmethod
    def get_queryset(_, queryset: models.QuerySet | None, info: Info):
        return get_queryset_for_model(Organization, queryset)
