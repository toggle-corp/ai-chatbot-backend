import strawberry
import strawberry_django

from user.models import User


@strawberry_django.filters.filter(User, lookups=True)
class UserFilter:
    id: strawberry.auto
    is_active: strawberry.auto
