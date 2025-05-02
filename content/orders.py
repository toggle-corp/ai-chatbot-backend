import strawberry
import strawberry_django

from content.models import Content


@strawberry_django.ordering.order(Content)
class ContentOrder:
    id: strawberry.auto
