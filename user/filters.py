from typing import Optional

import strawberry
import strawberry_django

from user.models import User


@strawberry_django.filter_type(User, lookups=True)
class UserFilter:
    id: Optional[strawberry.auto] = strawberry.UNSET
    display_name: Optional[strawberry.auto] = strawberry.UNSET
    is_active: Optional[bool] = strawberry.UNSET
