from django.contrib import admin, messages
from django.utils.safestring import mark_safe

from common.admin import UserResourceAdmin
from content.models import Content, Tag
from content.tasks import retrigger_content_processing

# Register your models here.


@admin.register(Tag)
class TagAdmin(admin.ModelAdmin):
    list_display = ["name"]
    search_fields = ["name"]


@admin.register(Content)
class ContentAdmin(UserResourceAdmin):
    list_display = ["title", "content_id"]
    autocomplete_fields = ["deleted_by", "tag"]
    readonly_fields = ["extracted_file"]
    actions = ["trigger_content_processing"]

    def trigger_content_processing(self, request, queryset):
        retrigger_content_processing(queryset)
        messages.add_message(request, messages.INFO, mark_safe("Successfully Re-trigger content processing! "))

    trigger_content_processing.short_description = "Re-trigger content processing"
