import strawberry
import strawberry_django
from django.db import models

from main.graphql.context import Info
from utils.common import get_queryset_for_model

from .models import Content


@strawberry_django.type(Content)
class ContentType:
    id: strawberry.ID
    title: strawberry.auto
    document_type: strawberry.auto
    extracted_file: strawberry.auto
    document_status: strawberry.auto
    tag: strawberry.auto

    @staticmethod
    def get_queryset(_, queryset: models.QuerySet | None, info: Info):
        return get_queryset_for_model(Content, queryset)
