import uuid

from django.db import models
from django.utils.translation import gettext_lazy as _

from common.models import UserResource


class Tag(models.Model):
    name = models.CharField(max_length=20)
    description = models.CharField(max_length=50, null=True, blank=True)


class Content(UserResource):
    class DocumentType(models.IntegerChoices):
        WORD = 1, _("Word")
        PDF = 2, _("PDF")

    class DocumenetStatus(models.IntegerChoices):
        PENDING = 1, _("Pending")
        TEXT_EXTRACTED = 2, _("Text extracted")
        ADDED_TO_VECTOR = 3, _("Added to vector")
        DELETED_FROM_VECTOR = 4, _("Deleted from vector")

    title = models.CharField(max_length=100)
    document_type = models.IntegerField(choices=DocumentType.choices)
    document_file = models.FileField(upload_to="documents", blank=True)
    extracted_file = models.FileField(upload_to="documents", blank=True)
    content_id = models.UUIDField(default=uuid.uuid4, editable=False)
    description = models.TextField(help_text="Content text")
    tag = models.ManyToManyField("Tag", blank=True)
    is_deleted = models.BooleanField(default=False)
    deleted_at = models.DateTimeField(blank=True)
    deleted_by = models.ForeignKey("user.User", null=True, blank=True, on_delete=models.PROTECT)
