import strawberry
import strawberry_django

from .models import Tag


@strawberry_django.filters.filter(Tag, lookups=True)
class TagFilter:
    name: strawberry.auto
