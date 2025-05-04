import strawberry

from user.models import User, UserRole
from utils.strawberry.enums import get_enum_name_from_django_field

DepartmentTypeEnum = strawberry.enum(User.Department, name="DepartmentTypeEnum")
UserRoleEnum = strawberry.enum(UserRole.Role, name="UserRoleEnum")

enum_map = {
    get_enum_name_from_django_field(field): enum
    for field, enum in (
        (User.department, DepartmentTypeEnum),
        (UserRole.role, UserRoleEnum),
    )
}
