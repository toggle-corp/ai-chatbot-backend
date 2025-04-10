import typing

from asgiref.sync import sync_to_async
from strawberry.permission import BasePermission
from strawberry.types import Info

from organization.models import OrganizationMember


class IsAuthenticated(BasePermission):
    message = "User is not authenticated"

    @sync_to_async
    def has_permission(self, source: typing.Any, info: Info, **_) -> bool:
        user = info.context.request.user
        return bool(user and user.is_authenticated)


class IsSuperAdmin(BasePermission):
    message = "User is not a Super Admin"

    @sync_to_async
    def has_permission(self, source: typing.Any, info: Info, **_) -> bool:
        user = info.context.request.user
        return bool(user and user.is_authenticated and user.is_superuser)


class IsOrganizationAdmin(BasePermission):
    message = "User is not a Super Admin or organization admin."

    @sync_to_async
    def has_permission(self, source: typing.Any, info: Info, **_) -> bool:
        user = info.context.request.user
        if not user or not user.is_authenticated:
            return False
        if user.is_superuser:
            return True

        return OrganizationMember.objects.filter(user=user, role=OrganizationMember.Role.ADMIN).exists()
