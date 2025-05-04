import strawberry
import strawberry_django

from content.models import Content

@strawberry_django.order_type(Content)
class ContentOrder:
    id : strawberry.auto
    created_at: strawberry.auto