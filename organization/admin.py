from django.contrib import admin

from .models import Organization, OrganizationMember

# Register your models here.


@admin.register(Organization)
class OrganizationAdmin(admin.ModelAdmin):
    list_display = ["name", "navbar_color"]


@admin.register(OrganizationMember)
class OrganizationMemberAdmin(admin.ModelAdmin):
    list_display = ["user", "organization", "role"]
    list_filter = ["organization", "role"]
