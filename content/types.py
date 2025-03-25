import strawberry
import strawberry_django
from django.db import models

from content.models import Content, Tag
from main.graphql.context import Info
from utils.common import get_queryset_for_model
from utils.strawberry.enums import enum_field


@strawberry_django.type(Tag)
class TagType:
    id: strawberry.ID
    name: strawberry.auto
    description: strawberry.auto

    @staticmethod
    def get_queryset(_, queryset: models.QuerySet | None, info: Info) -> models.QuerySet:
        return get_queryset_for_model(Tag, queryset)


@strawberry_django.type(Tag)
class TagNameType:
    id: strawberry.ID
    name: strawberry.auto


@strawberry_django.type(Content)
class ContentType:
    id: strawberry.ID
    title: strawberry.auto
    extracted_file: strawberry.auto
    created_at: strawberry.auto
    tag: list[TagNameType]
    document_status = enum_field(Content.document_status)
    document_type = enum_field(Content.document_type)

    @staticmethod
    def get_queryset(_, queryset: models.QuerySet | None, info: Info):
        return get_queryset_for_model(Content, queryset)
