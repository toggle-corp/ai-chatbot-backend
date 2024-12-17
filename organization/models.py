from django.db import models
from common.models import UserResource
# Create your models here.


class Organization(UserResource):
    name = models.CharField(max_length=40)
    image = models.ImageField(upload_to="organization")
    slider_bar_color = models.CharField(max_length=10, default="#f56f42")
    navbar_color = models.CharField(max_length=10, default="#f81341")

    def __str__(self):
        return self.name
