import strawberry
import strawberry_django

from user.models import User


@strawberry_django.order_type(User)
class UserOrder:
    display_name: strawberry.auto
