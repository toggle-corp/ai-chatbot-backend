import uuid

from django.db import models
from django.utils.translation import gettext_lazy as _


class Tag(models.Model):
    name = models.CharField(max_length=20)
    description = models.CharField(max_length=50, null=True, blank=True)


class Content(models.Model):
    # TODO need to create UserResource  models for user model
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
    document_file = models.FileField(upload_to="documents")
    extracted_file = models.FileField(upload_to="documents")
    content_id = models.UUIDField(default=uuid.uuid4, editable=False)
    description = models.TextField()
    tag = models.ManyToManyField("Tag")
    created_at = models.DateTimeField(auto_now_add=True)
    created_by = models.ForeignKey("user.User", on_delete=models.PROTECT)
    is_deleted = models.BooleanField(default=False)
    deleted_at = models.DateTimeField(null=True)
    deleted_by = models.ForeignKey("user.User", null=True, on_delete=models.PROTECT)
