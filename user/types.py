import strawberry
import strawberry_django

from utils.strawberry.enums import enum_field

from .models import User


@strawberry_django.type(User)
class UserType:
    id: strawberry.ID
    first_name: strawberry.auto
    last_name: strawberry.auto
    department: enum_field(User.department)
    

    @strawberry_django.field
    def display_name(self, root: User) -> str:
        return root.display_name
