from django.db import models
from django.utils.translation import gettext_lazy as _

from common.models import UserResource
from user.models import User

# Create your models here.


class Organization(UserResource):
    name = models.CharField(max_length=40)
    image = models.ImageField(
        upload_to="organization",
        null=True,
        blank=True,
    )
    slider_bar_color = models.CharField(max_length=10, default="#f56f42")
    navbar_color = models.CharField(max_length=10, default="#f81341")

    def __str__(self):
        return self.name


class OrganizationMember(models.Model):
    class Role(models.IntegerChoices):
        USER = 1, _("user")
        ADMIN = 2, _("admin")

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    organization = models.ForeignKey(Organization, on_delete=models.CASCADE)
    role = models.IntegerField(choices=Role.choices, default=Role.USER)

    class Meta:
        unique_together = ("user", "organization")

    def __str__(self):
        return "{} @ {}".format(str(self.user), self.organization.name)

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
