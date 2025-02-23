from django.conf import settings


class Permalink:
    BASE_URL = f"{settings.APP_FRONTEND_HOST}"

    @classmethod
    def user_registration(cls, uid: str, token: str):
        return f"{cls.BASE_URL}/user-registration/{uid}/{token}"

    @classmethod
    def user_password_reset(cls, uid: str, token: str):
        return f"{cls.BASE_URL}/user-password-reset/{uid}/{token}"

    @classmethod
    def user_activation(cls, uid: str, token: str):
        return f"{cls.BASE_URL}/user-activation/{uid}/{token}"
