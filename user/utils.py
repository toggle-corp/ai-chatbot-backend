from django.conf import settings
from django.core.mail import send_mail
from django.utils.encoding import force_bytes
from django.utils.http import urlsafe_base64_encode

from main.permalink import Permalink
from main.token import TokenManager


def send_account_creation_email(user):

    uid = urlsafe_base64_encode(force_bytes(user.pk))
    token = TokenManager.account_registration_token_generator.make_token(user)
    activation_url = Permalink.user_activation(uid, token)
    subject = "Account Creation"
    message = "Hello from chat-bot cms,\n\n"
    message += f"Click the link below to create your account:\n\n {activation_url}\n\n"
    send_mail(subject, message, settings.EMAIL_HOST_USER, [user.email])


def resend_account_activation(user):

    uid = urlsafe_base64_encode(force_bytes(user.pk))
    token = TokenManager.account_reactivation_token_generator.make_token(user)
    activation_url = Permalink.user_activation(uid, token)
    subject = "Account Activation"
    message = f"Hi {user.first_name} {user.last_name},\n\n"
    message += f"Click the link below to activate your account:\n\n {activation_url}\n\n"
    send_mail(subject, message, settings.EMAIL_HOST_USER, [user.email])


def send_password_reset_email(user):

    uid = urlsafe_base64_encode(force_bytes(user.pk))
    token = TokenManager.password_reset_token_generator.make_token(user)
    password_reset_url = Permalink.user_password_reset(uid, token)
    subject = "Password Reset"
    message = f"Hi {user.first_name} {user.last_name},\n\n"
    message += f"Click the link below to reset your password:\n\n{password_reset_url}\n\n"
    send_mail(subject, message, settings.EMAIL_HOST_USER, [user.email])
