from main.test import TestCase
from user.factories import UserFactory, UserRoleFactory
from user.models import UserRole


class TestUserQuery(TestCase):
    class Query:
        ME = """
            query meQuery {
                me {
                    id
                    email
                    firstName
                    lastName
                    displayName
                }
            }
        """
        USERS = """
            query Users($pagination: OffsetPaginationInput) {
                users(pagination: $pagination) {
                    totalCount
                    results {
                        id
                        email
                        firstName
                        lastName
                        isActive
                        department
                        displayName
                    }
                }
            }
        """
        USERS_WITH_FILTER = """
            query($displayName: String!, $pagination: OffsetPaginationInput) {
                users(filters: {displayName: {exact: $displayName}}, pagination: $pagination) {
                    totalCount
                    results {
                        id
                        email
                        firstName
                        lastName
                        isActive
                        department
                        displayName
                    }
                }
            }
        """

    def setUp(self):
        super().setUp()
        self.user = UserFactory.create(email="testuser@gmail.com", first_name="Test", last_name="User")
        self.user2 = UserFactory.create(email="john@gmail.com", first_name="john", last_name="cena")
        self.users = UserFactory.create_batch(3)

        # Admin user (via Member)
        self.admin_user = UserFactory.create(email="admin@gmail.com")
        UserRoleFactory.create(user=self.admin_user, role=UserRole.Role.ADMIN)

        # Superuser
        self.superuser = UserFactory.create(email="super@gmail.com", is_superuser=True)

    def test_me(self):
        # Without authentication
        content = self.query_check(self.Query.ME)
        assert content["data"]["me"] is None
        # With authentication
        self.force_login(self.user)
        content = self.query_check(self.Query.ME)
        assert content["data"]["me"] == dict(
            id=self.gID(self.user.id),
            email=self.user.email,
            firstName=self.user.first_name,
            lastName=self.user.last_name,
            displayName=self.user.display_name,
        )

    def test_users_query_permission_for_non_admin(self):
        # Not authenticated
        response = self.query_check(
            self.Query.USERS,
            variables={"pagination": {"limit": 10, "offset": 0}},
        )
        assert response["data"]["users"]["totalCount"] == 0

        # Authenticated but not admin/superuser
        self.force_login(self.user)
        response = self.query_check(
            self.Query.USERS,
            variables={"pagination": {"limit": 10, "offset": 0}},
        )
        assert response["data"]["users"]["totalCount"] == 0

    def test_users_query_as_admin_user(self):
        self.force_login(self.admin_user)
        response = self.query_check(
            self.Query.USERS,
            variables={"pagination": {"limit": 10, "offset": 0}},
        )
        users_data = response["data"]["users"]
        assert users_data["totalCount"] == 7

    def test_users_query_as_superuser(self):
        self.force_login(self.superuser)
        response = self.query_check(
            self.Query.USERS,
            variables={"pagination": {"limit": 10, "offset": 0}},
        )
        users_data = response["data"]["users"]
        assert users_data["totalCount"] == 7

    def test_users_with_filter_with_admin(self):
        self.force_login(self.admin_user)
        response = self.query_check(
            self.Query.USERS_WITH_FILTER,
            variables={
                "displayName": self.user2.display_name,
                "pagination": {"limit": 10, "offset": 0},
            },
        )
        filtered = response["data"]["users"]
        assert filtered["totalCount"] == 1
        assert filtered["results"][0]["id"] == self.gID(self.user2.id)
