import strawberry
import strawberry_django

from user.enums import DepartmentTypeEnum, UserRoleEnum
from user.models import User, UserRole


@strawberry_django.type(User)
class UserType:
    id: strawberry.ID
    email: strawberry.auto
    first_name: strawberry.auto
    last_name: strawberry.auto
    is_active: strawberry.auto
    department: DepartmentTypeEnum
    profile_picture: strawberry.auto
    display_name: strawberry.auto


@strawberry_django.type(User)
class UserMeType:
    id: strawberry.ID
    email: strawberry.auto
    first_name: strawberry.auto
    last_name: strawberry.auto
    profile_picture: strawberry.auto
    display_name: strawberry.auto


@strawberry_django.type(UserRole)
class UserRoleType:
    id: strawberry.ID
    role: UserRoleEnum
    user: strawberry.auto
