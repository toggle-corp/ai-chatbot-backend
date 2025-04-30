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
    profile_picture = models.ImageField(upload_to="profile_pictures/", blank=True, null=True)
    department = models.PositiveSmallIntegerField(choices=Department.choices, null=True)

    objects: CustomUserManager = CustomUserManager()

    def save(self, *args, **kwargs):
        # Make sure email are store in lowercase
        self.email = self.email.lower()
        if self.pk is None:
            super().save(*args, **kwargs)
            # Remove force_insert since we have already inserted
            kwargs.pop("force_insert", None)
        self.display_name = self.get_full_name() or f"User#{self.pk}"
        return super().save(*args, **kwargs)


class Member(models.Model):
    class Role(models.IntegerChoices):
        ADMIN = 1, ("admin")
        USER = 2, ("user")

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    role = models.IntegerField(choices=Role.choices, default=Role.USER)

    def __str__(self):
        return f"{self.user.email} - {self.get_role_display()}"
