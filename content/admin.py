from django.contrib import admin

from common.admin import UserResourceAdmin
from content.models import Content, Tag

# Register your models here.


@admin.register(Tag)
class TagAdmin(admin.ModelAdmin):
    list_display = ["name"]
    search_fields = ["name"]


@admin.register(Content)
class ContentAdmin(UserResourceAdmin):
    list_display = ["title", "content_id"]
    autocomplete_fields = ["deleted_by", "tag"]
