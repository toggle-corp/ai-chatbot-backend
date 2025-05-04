from collections.abc import Callable
from typing import Any, ClassVar, Optional

from strawberry.types import Info
from strawberry_django.permissions import DjangoNoPermission, DjangoPermissionExtension
from strawberry_django.resolvers import django_resolver

from user.models import Member


class IsAdminOrSuperuser(DjangoPermissionExtension):
    """Allows access to superusers or users with the admin role."""

    DEFAULT_ERROR_MESSAGE: ClassVar[str] = "User is not a superuser or an admin."

    @django_resolver(qs_hook=None)
    def resolve_for_user(
        self,
        resolver: Callable,
        user: Optional["UserType"],  # noqa: F821
        *,
        info: Info,
        source: Any,
    ):
        if (
            user is None
            or not user.is_authenticated
            or (
                not getattr(user, "is_superuser", False)
                and not Member.objects.filter(user=user, role=Member.Role.ADMIN).exists()
            )
        ):
            raise DjangoNoPermission(self.DEFAULT_ERROR_MESSAGE)

        return resolver()
