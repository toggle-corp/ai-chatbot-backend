import os

from django.core.exceptions import ValidationError


def validate_document_size(file):
    max_size = 5 * 1024 * 1024  # Max file size 5MB
    if file.size > max_size:
        raise ValidationError("Document size should be less then 5MB.")
    return file


def validate_file_type(file):
    allowed_extensions = ["pdf", "txt"]
    extension = os.path.splitext(file.name)[1].replace(".", "")
    if extension.lower() not in allowed_extensions:
        raise ValidationError("Only .pdf and .txt files are allowed.")
    return file


def validate_image_size(image):
    max_file_size = 2 * 1024 * 1024
    if image.size > max_file_size:
        raise ValidationError("Image size must be less than 2MB.")
    return image
