from django.contrib import admin
from .models import Organization

# Register your models here.


@admin.register(Organization)
class OrganizationAdmin(admin.ModelAdmin):
    list_display = ['name', 'navbar_color']
