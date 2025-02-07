import sentry_sdk
from celery.exceptions import Retry as CeleryRetry
from django.core.exceptions import PermissionDenied
from sentry_sdk.integrations.celery import CeleryIntegration
from sentry_sdk.integrations.django import DjangoIntegration
from sentry_sdk.integrations.logging import ignore_logger
from sentry_sdk.integrations.redis import RedisIntegration

IGNORED_ERRORS = [
    PermissionDenied,
    CeleryRetry,
]
IGNORED_LOGGERS = [
    "django.core.exceptions.ObjectDoesNotExist",
]

for _logger in IGNORED_LOGGERS:
    ignore_logger(_logger)


def init_sentry(app_type, tags={}, **config):
    integrations = [
        DjangoIntegration(),
        CeleryIntegration(),
        RedisIntegration(),
    ]
    sentry_sdk.init(
        **config,
        ignore_errors=IGNORED_ERRORS,
        integrations=integrations,
    )
    with sentry_sdk.configure_scope() as scope:
        scope.set_tag("app_type", app_type)
        for tag, value in tags.items():
            scope.set_tag(tag, value)
