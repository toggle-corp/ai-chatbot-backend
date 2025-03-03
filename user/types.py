import strawberry
import strawberry_django
from django.db import models

from main.graphql.context import Info
from user.models import User
from utils.common import get_queryset_for_model
from utils.strawberry.enums import enum_field


@strawberry_django.type(User)
class UserType:
    id: strawberry.ID
    email: strawberry.auto
    first_name: strawberry.auto
    last_name: strawberry.auto
    is_active: strawberry.auto
    department = enum_field(User.department)

    @staticmethod
    def get_queryset(_, queryset: models.QuerySet | None, info: Info):
        return get_queryset_for_model(User, queryset)

    @strawberry_django.field
    def display_name(self, root: User) -> str:
        return root.display_name


@strawberry_django.type(User)
class UserMeType:
    id: strawberry.ID
    email: strawberry.auto
    first_name: strawberry.auto
    last_name: strawberry.auto
