import strawberry

from user.models import Member, User
from utils.strawberry.enums import get_enum_name_from_django_field

DepartmentTypeEnum = strawberry.enum(User.Department, name="DepartmentTypeEnum")
UserRoleEnum = strawberry.enum(Member.Role, name="UserRoleEnum")

enum_map = {get_enum_name_from_django_field(field): enum for field, enum in ((User.department, DepartmentTypeEnum),)}

enum_map = {
    get_enum_name_from_django_field(field): enum
    for field, enum in (
        (User.department, DepartmentTypeEnum),
        (Member.role, UserRoleEnum),
    )
}
