from django.contrib import admin, messages
from django.utils.safestring import mark_safe

from common.admin import UserResourceAdmin
from content.models import Content, Tag
from content.tasks import create_embedding_for_content_task

# Register your models here.


@admin.register(Tag)
class TagAdmin(admin.ModelAdmin):
    list_display = ["name"]
    search_fields = ["name"]


@admin.register(Content)
class ContentAdmin(UserResourceAdmin):
    list_display = [
        "title",
        "content_id",
        "document_status",
    ]
    autocomplete_fields = ["deleted_by", "tag"]
    readonly_fields = ["document_status", "document_type"]

    actions = ["retrigger_content_processing"]

    def retrigger_content_processing(self, request, queryset):
        """
        Trigger content processing for selected content.
        """
        for content in queryset:
            if content.document_status == Content.DocumentStatus.ADDED_TO_VECTOR:
                messages.add_message(request, messages.WARNING, mark_safe(f"Content {content.title} is already processed."))
            else:

                create_embedding_for_content_task.delay(content.id)

                messages.add_message(
                    request, messages.INFO, mark_safe("Successfully triggered content processing for selected contents!")
                )

    retrigger_content_processing.short_description = "Re-trigger content processing"
