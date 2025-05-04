from typing import Optional

import strawberry
import strawberry_django

from user.models import User


@strawberry_django.filter_type(User, lookups=True)
class UserFilter:
    id: strawberry.auto
    display_name: strawberry.auto
    is_active: Optional[bool]
