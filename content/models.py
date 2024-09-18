import uuid

from django.db import models
from django.utils.translation import gettext_lazy as _

from common.models import UserResource


class Tag(models.Model):
    name = models.CharField(max_length=20)
    description = models.CharField(max_length=50, null=True, blank=True)

    def __str__(self):
        return self.name


class Content(UserResource):
    class DocumentType(models.IntegerChoices):
        WORD = 1, _("Word")
        PDF = 2, _("PDF")
        TEXT = 3, _("Text")

    class DocumentStatus(models.IntegerChoices):
        PENDING = 1, _("Pending")
        TEXT_EXTRACTED = 2, _("Text extracted")
        ADDED_TO_VECTOR = 3, _("Added to vector")
        DELETED_FROM_VECTOR = 4, _("Deleted from vector")
        FAILURE = 5, _("Failure")

    title = models.CharField(max_length=100)
    document_type = models.IntegerField(choices=DocumentType.choices, default=DocumentType.TEXT)
    document_file = models.FileField(upload_to="documents")
    extracted_file = models.FileField(upload_to="documents-extracts", null=True, blank=True)
    content_id = models.UUIDField(default=uuid.uuid4, editable=False)
    document_status = models.PositiveSmallIntegerField(choices=DocumentStatus.choices, default=DocumentStatus.PENDING)
    tag = models.ManyToManyField("Tag", blank=True)
    is_deleted = models.BooleanField(default=False)
    deleted_at = models.DateTimeField(null=True, blank=True)
    deleted_by = models.ForeignKey("user.User", null=True, blank=True, on_delete=models.PROTECT)

    def __str__(self):
        return self.title

    def save(self, *args, **kwargs):
        """Save the content to the database."""
        if self.document_type == self.DocumentType.TEXT:
            self.extracted_file = self.document_file
        super().save(*args, **kwargs)
