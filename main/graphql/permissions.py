import typing

from asgiref.sync import sync_to_async
from strawberry.permission import BasePermission
from strawberry.types import Info

from user.models import Member


class IsAuthenticated(BasePermission):
    message = "User is not authenticated"

    @sync_to_async
    def has_permission(self, source: typing.Any, info: Info, **_) -> bool:
        user = info.context.request.user
        return bool(user and user.is_authenticated)


class IsAdmin(BasePermission):
    message = "User is not an admin user  or superuser"

    @sync_to_async
    def has_permission(self, source: typing.Any, info: Info, **_) -> bool:
        user = info.context.request.user
        if not user.is_authenticated:
            return False
        if user.is_authenticated and user.is_superuser:
            return True
        return Member.objects.filter(user=user, role=Member.Role.ADMIN).exists()
