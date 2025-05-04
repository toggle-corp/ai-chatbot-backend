from unittest import mock

from main.test import TestCase
from user.factories import UserFactory, UserRoleFactory
from user.models import UserRole


class TestUserMutations(TestCase):
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

    class Mutation:
        LOGIN_MUTATION = """
            mutation Login($data: LoginInput!) {
                login(data: $data) {
                    ok
                    errors
                    result {
                        id
                        email
                        displayName
                    }
                }
            }
        """
        LOGOUT_MUTATION = """
            mutation {
                logout {
                    ok
                }
            }
        """
        ADD_USER = """
            mutation MyMutation($data: AddUserInput!) {
                addUser(data: $data) {
                    errors
                    ok
                }
            }
        """
        REGISTER_USER = """
            mutation MyMutation($data: RegisterUserInput!) {
                registerUser(data: $data) {
                    errors
                    ok
                    result {
                        id
                        email
                        firstName
                        lastName
                        department
                        isActive
                    }
                }
            }
        """
        RESEND_INVITE = """
            mutation MyMutation($data: UserResendInviteInput!) {
                resendInvite(data: $data) {
                    errors
                    ok
                }
            }
        """
        ACCOUNT_ACTIVATION = """
            mutation Activate($data: UserActivationInput!) {
                accountActivation(data: $data) {
                    ok
                    errors
                }
            }
        """
        ACCOUNT_DEACTIVATION = """
            mutation MyMutation($data: UserDeactivationInput!) {
                accountDeactivation(data: $data) {
                    ok
                    errors
                }
            }
        """
        PASSWORD_RESET_TRIGGER = """
            mutation MyMutation($data: UserPasswordResetTriggerInput!) {
                passwordResetTrigger(data: $data) {
                    ok
                    errors
                }
            }
        """
        FORGOT_PASSWORD = """
            mutation MyMutation($data: ForgotPasswordInput!) {
                forgotPassword(data: $data) {
                    ok
                    errors
                }
            }
        """
        PASSWORD_RESET = """
            mutation PasswordReset($data: UserPasswordReset!) {
                passwordReset(data: $data) {
                    ok
                    errors
                }
            }
        """
        CHANGE_PASSWORD = """
        mutation ChangePassword($data: ChangePasswordInput!) {
            changePassword(data: $data) {
                ok
                errors
            }
        }
        """
        UPDATE_ME = """
        mutation UpdateMe($data: UserMeInput!) {
            updateMe(data: $data) {
                ok
                errors
                result {
                    id
                    firstName
                    lastName
                }
            }
        }
        """
        ASSIGN_ROLE = """
            mutation MyMutation($data: UserRoleInput!) {
                assignRole(data: $data) {
                    ok
                    errors
                    result {
                        id
                        role
                        user {
                            pk
                        }
                    }
                }
            }
        """

    def _query_login(self, data: dict, **kwargs):
        return self.query_check(self.Mutation.LOGIN_MUTATION, variables={"data": data}, **kwargs)

    def _query_logout(self, **kwargs):
        return self.query_check(self.Mutation.LOGOUT_MUTATION, **kwargs)

    def _query_register_user(self, data: dict, **kwargs):
        return self.query_check(
            self.Mutation.REGISTER_USER,
            variables={"data": data},
            **kwargs,
        )

    def _query_add_user(self, emails: str, **kwargs):
        return self.query_check(
            self.Mutation.ADD_USER,
            variables={"data": {"emails": emails}},
            **kwargs,
        )

    def _query_resend_invite(self, user_id: str, **kwargs):
        return self.query_check(
            self.Mutation.RESEND_INVITE,
            variables={"data": {"userId": user_id}},
            **kwargs,
        )

    def _query_account_activation(self, uuid: str, token: str, **kwargs):
        return self.query_check(
            self.Mutation.ACCOUNT_ACTIVATION,
            variables={
                "data": {
                    "uuid": uuid,
                    "token": token,
                }
            },
            **kwargs,
        )

    def _query_account_deactivation(self, user_id: str, **kwargs):
        return self.query_check(
            self.Mutation.ACCOUNT_DEACTIVATION,
            variables={"data": {"userId": user_id}},
            **kwargs,
        )

    def _query_password_reset_trigger(self, user_id: str, **kwargs):
        return self.query_check(
            self.Mutation.PASSWORD_RESET_TRIGGER,
            variables={"data": {"userId": user_id}},
            **kwargs,
        )

    def _query_forgot_password(self, email: str, **kwargs):
        return self.query_check(
            self.Mutation.FORGOT_PASSWORD,
            variables={"data": {"email": email}},
            **kwargs,
        )

    def _query_password_reset(self, uuid, token, new_password, **kwargs):
        return self.query_check(
            self.Mutation.PASSWORD_RESET,
            variables={
                "data": {
                    "uuid": uuid,
                    "token": token,
                    "newPassword": new_password,
                }
            },
            **kwargs,
        )

    def _query_change_password(self, old_password, new_password, **kwargs):
        return self.query_check(
            self.Mutation.CHANGE_PASSWORD,
            variables={
                "data": {
                    "oldPassword": old_password,
                    "newPassword": new_password,
                }
            },
            **kwargs,
        )

    def _query_assign_role(self, user_id, role, **kwargs):
        return self.query_check(
            self.Mutation.ASSIGN_ROLE,
            variables={
                "data": {
                    "user": str(user_id),
                    "role": role,
                }
            },
            **kwargs,
        )

    def _query_update_me(self, data, **kwargs):
        return self.query_check(
            self.Mutation.UPDATE_ME,
            variables={"data": data},
            **kwargs,
        )

    def setUp(self):
        super().setUp()
        self.user = UserFactory.create(is_active=True)
        self.super_admin = UserFactory.create(email="usperadmin@gamil.com", is_superuser=True)
        self.admin_user = UserFactory.create(email="admin123@gmail.com", is_active=True)
        UserRoleFactory.create(user=self.admin_user, role=UserRole.Role.ADMIN)

    def test_login(self):
        password = "StrongPass123!"
        user = UserFactory.create(email="login@example.com", password=password, is_active=True)
        # NOTE: Django's create_user hash passwords, but factory may not. Set manually.
        user.set_password(password)
        user.save()
        # Try logging in with wrong credentials
        response = self._query_login({"email": user.email, "password": "WrongPass!"})
        result = response["data"]["login"]
        assert result["ok"] is False
        assert result["errors"] is not None
        # Try logging in with correct credentials
        response = self._query_login({"email": user.email, "password": password})
        result = response["data"]["login"]
        assert result["ok"] is True
        assert result["errors"] is None
        assert result["result"]["email"] == user.email

    @mock.patch("user.serializers.validate_token")
    def test_register_user(self, mock_validate_token):
        user = UserFactory.create(
            email="register@example.com",
            is_active=False,
        )
        mock_validate_token.return_value = user
        uuid = "dummy-uuid"
        token = "dummy-token"
        data = {
            "uuid": uuid,
            "token": token,
            "password": "StrongPass123!",
            "firstName": "Test",
            "lastName": "User",
            "department": "HR",
        }
        response = self._query_register_user(data)
        result = response["data"]["registerUser"]
        assert result["ok"] is True
        assert result["errors"] is None
        assert result["result"]["email"] == "register@example.com"
        assert result["result"]["firstName"] == "Test"
        assert result["result"]["lastName"] == "User"
        assert result["result"]["department"] == "HR"
        assert result["result"]["isActive"] is True

    @mock.patch("user.serializers.validate_token")
    def test_account_activation(self, mock_validate_token):
        user = UserFactory.create(email="pending@example.com", is_active=False)
        token = "dummy"
        uuid = str(user.id)
        response = self._query_account_activation(uuid=uuid, token=token)
        result = response["data"]["accountActivation"]
        assert result["ok"] is True
        assert result["errors"] is None

    def test_account_deactivation(self):
        active_user = UserFactory.create(is_active=True)
        # Unauthenticated request
        response = self._query_account_deactivation(user_id=str(active_user.id), assert_errors=True)
        assert response["data"] is None
        # Non-admin request
        self.force_login(self.user)
        response = self._query_account_deactivation(user_id=str(active_user.id), assert_errors=True)
        assert response["data"] is None
        # with admin user
        self.force_login(self.admin_user)
        response = self._query_account_deactivation(user_id=str(active_user.id))
        data = response["data"]["accountDeactivation"]
        assert data["ok"] is True
        assert data["errors"] is None
        # refresh user from db
        active_user.refresh_from_db()
        assert active_user.is_active is False

    @mock.patch("user.serializers.send_password_reset_email_task.delay")
    def test_password_reset_trigger(self, mock_send_password_reset_email_task):
        target_user = UserFactory.create(email="target@example.com")
        # Without authentication
        response = self._query_password_reset_trigger(user_id=str(target_user.id), assert_errors=True)
        assert response["data"] is None
        # With non-admin user authentication
        self.force_login(self.user)
        response = self._query_password_reset_trigger(user_id=str(target_user.id), assert_errors=True)
        assert response["data"] is None
        # With admin user authentication
        self.force_login(self.admin_user)
        with self.captureOnCommitCallbacks(execute=True):
            response = self._query_password_reset_trigger(user_id=str(target_user.id))
        data = response["data"]["passwordResetTrigger"]
        assert data["ok"] is True
        assert data["errors"] is None
        mock_send_password_reset_email_task.assert_called_once_with(str(target_user.id))

    @mock.patch("user.serializers.send_password_reset_email_task.delay")
    def test_forgot_password(self, mock_send_reset_email):
        user = UserFactory.create(email="valid@example.com")
        with self.captureOnCommitCallbacks(execute=True):
            response = self._query_forgot_password(email=user.email)
        data = response["data"]["forgotPassword"]
        assert data["ok"] is True
        assert data["errors"] is None
        mock_send_reset_email.assert_called_once_with(user.id)

    def test_forgot_password_invalid_email(self):
        response = self.query_check(
            self.Mutation.FORGOT_PASSWORD,
            variables={"data": {"email": "nono@example.com"}},
        )
        data = response["data"]["forgotPassword"]
        assert data["ok"] is False
        assert data["errors"] is not None

    @mock.patch("user.serializers.validate_token")
    def test_password_reset(self, mock_validate_token):
        user = UserFactory.create()
        uuid = str(user.id)
        token = "dummy-token"
        new_password = "NewStrongPass123!"
        response = self._query_password_reset(uuid, token, new_password)
        result = response["data"]["passwordReset"]
        assert result["ok"] is True
        assert result["errors"] is None

    def test_change_password(self):
        test_user = UserFactory.create(email="change@gmail.com")
        test_user.set_password("old-password123")
        test_user.save()
        # Without authentication
        response = self._query_change_password("old-password123", "new-password456", assert_errors=True)
        assert response["data"] is None
        # With authentication
        self.force_login(test_user)
        response = self._query_change_password("old-password123", "new-password456")
        data = response["data"]["changePassword"]
        assert data["ok"] is True
        assert data["errors"] is None
        # Confirm password was changed
        test_user.refresh_from_db()
        assert test_user.check_password("new-password456")

    def test_update_me(self):
        new_user = UserFactory.create(email="userme@gmail.com", first_name="sandy", last_name="candy")
        response = self._query_update_me({"firstName": "mandy", "lastName": "handy"}, assert_errors=True)
        assert response["data"] is None
        # With authentication
        self.force_login(new_user)
        response = self._query_update_me(
            {"firstName": "mandy", "lastName": "handy"},
        )
        data = response["data"]["updateMe"]
        assert data["ok"] is True
        assert data["errors"] is None
        # Confirm user was updated
        new_user.refresh_from_db()
        assert new_user.first_name == "mandy"
        assert new_user.last_name == "handy"

    def test_assign_role(self):
        target_user = UserFactory.create(email="dummyuser@gmail.com")
        # without authenticated
        response = self._query_assign_role(
            user_id=target_user.pk,
            role="USER",
            assert_errors=True,
        )
        assert response["data"] is None
        # Authenticated but not superuser
        self.force_login(self.admin_user)
        response = self._query_assign_role(
            user_id=target_user.pk,
            role="USER",
            assert_errors=True,
        )
        assert response["data"] is None
        # Authenticated as superuser
        self.force_login(self.super_admin)
        response = self._query_assign_role(
            user_id=target_user.pk,
            role="ADMIN",
            assert_errors=False,
        )
        data = response["data"]["assignRole"]
        assert data["ok"] is True
        assert data["errors"] is None
        assert data["result"]["role"] == "ADMIN"
        assert str(data["result"]["user"]["pk"]) == str(target_user.pk)

    def test_logout(self):
        user = UserFactory.create(email="jack@gmail.com")
        # Check me query without login
        content = self.query_check(self.Query.ME)
        self.assertEqual(content["data"]["me"], None, content)
        # Check me query after login
        self.force_login(user)
        content = self.query_check(self.Query.ME)
        self.assertEqual(content["data"]["me"]["id"], self.gID(user.id), content)
        self.assertEqual(content["data"]["me"]["email"], user.email, content)
        # Perform logout mutation
        content = self._query_logout()
        self.assertTrue(content["data"]["logout"]["ok"], content)
        # Check me query again after logout
        content = self.query_check(self.Query.ME)
        self.assertEqual(content["data"]["me"], None, content)

    @mock.patch("user.serializers.send_account_creation_email_task.delay")
    def test_add_user(self, mock_send_email):
        UserFactory.create(email="existing@example.com")
        # without authentication
        response = self._query_add_user("new1@example.com,new2@example.com", assert_errors=True)
        assert response["data"] is None
        # with admin authentication
        self.force_login(self.super_admin)
        # emails already registered
        response = self._query_add_user("existing@example.com")
        result = response["data"]["addUser"]
        assert result["ok"] is False
        emails = "new@example.com,extra@example.com"
        with self.captureOnCommitCallbacks(execute=True):
            response = self._query_add_user(emails)
        result = response["data"]["addUser"]
        assert result["ok"] is True
        assert result["errors"] is None
        assert mock_send_email.call_count == 2

    @mock.patch("user.serializers.resend_account_activation_task.delay")
    def test_resend_invite(self, mock_resend_account_activation_task):
        inactive_user = UserFactory.create(email="inactive@example.com", is_active=False)
        # Without authentication
        response = self._query_resend_invite(user_id=str(inactive_user.id), assert_errors=True)
        assert response["data"] is None
        # With non-admin user
        self.force_login(self.user)
        response = self._query_resend_invite(user_id=str(inactive_user.id), assert_errors=True)
        assert response["data"] is None
        # with super user
        self.force_login(self.super_admin)
        with self.captureOnCommitCallbacks(execute=True):
            response = self._query_resend_invite(user_id=str(inactive_user.id))
        data = response["data"]["resendInvite"]
        assert data["ok"] is True
        assert data["errors"] is None
        mock_resend_account_activation_task.assert_called_once_with(str(inactive_user.id))
