from django.contrib.auth.models import AbstractUser
from django.db import models
from django.utils.translation import gettext_lazy as _

from .managers import CustomUserManager


# Create your models here.
class User(AbstractUser):
    class Department(models.IntegerChoices):
        HR = 1, _("HR")

    EMAIL_FIELD = USERNAME_FIELD = "email"
    REQUIRED_FIELDS = []

    username = None
    email = models.EmailField(unique=True)
    display_name = models.CharField(
        verbose_name=_("system generated user display name"),
        blank=True,
        max_length=255,
    )
    department = models.PositiveSmallIntegerField(choices=Department.choices, null=True)

    objects: CustomUserManager = CustomUserManager()

    def save(self, *args, **kwargs):
        # Make sure email are store in lowercase
        self.email = self.email.lower()
        if self.pk is None:
            super().save(*args, **kwargs)
        self.display_name = self.get_full_name() or f"User#{self.pk}"
        return super().save(*args, **kwargs)
