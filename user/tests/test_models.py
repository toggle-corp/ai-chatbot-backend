from django.db import IntegrityError
from django.test import TestCase

from user.factories import UserFactory, UserRoleFactory
from user.models import UserRole


class UserRoleTestCase(TestCase):

    def test_create_user_role(self):
        user = UserFactory.create()
        # Create a UserRole for the user with a role of ADMIN
        user_role = UserRoleFactory.create(user=user, role=UserRole.Role.ADMIN)
        self.assertEqual(user_role.user, user)
        self.assertEqual(user_role.role, UserRole.Role.ADMIN)

    def test_user_role_unique_together(self):
        user = UserFactory.create()
        UserRoleFactory.create(user=user, role=UserRole.Role.USER)
        # Create another UserRole with the same user and role
        with self.assertRaises(IntegrityError):
            UserRoleFactory.create(user=user, role=UserRole.Role.USER)

    def test_admin_role_unique_for_user(self):
        # Create a user using the UserFactory
        user = UserFactory.create()
        # Assign the first ADMIN role to the user
        UserRoleFactory.create(user=user, role=UserRole.Role.ADMIN)
        # Assign another ADMIN role to the same user
        with self.assertRaises(IntegrityError):
            UserRoleFactory.create(user=user, role=UserRole.Role.ADMIN)
