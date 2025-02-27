import logging

from celery import shared_task
from django.contrib.auth import get_user_model
from django.shortcuts import get_object_or_404

from user.utils import resend_account_activation, send_account_creation_email

User = get_user_model()

logger = logging.getLogger(__name__)


@shared_task
def send_account_creation_email_task(user_id):
    user = User.objects.filter(id=user_id).first()
    if user:
        send_account_creation_email(user)
    else:
        logger.error(f"User with ID {user_id} not found")


@shared_task
def resend_account_activation_task(user_id):
    user = get_object_or_404(User, id=user_id)
    if not user.is_active:
        resend_account_activation(user)
        f"Invitation resent to {user.email}"
    else:
        logger.error(f"User with this {user.email} is already active")
