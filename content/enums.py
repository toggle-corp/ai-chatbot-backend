import strawberry

from utils.strawberry.enums import get_enum_name_from_django_field

from .models import Content

DocumentStatusTypeEnum = strawberry.enum(Content.DocumentStatus, name="DocumentStatusTypeEnum")

DocumentTypeEnum = strawberry.enum(Content.DocumentType, name="DocumentTypeEnum")
enum_map = {
    get_enum_name_from_django_field(field): enum
    for field, enum in (
        (Content.document_status, DocumentStatusTypeEnum),
        (Content.document_type, DocumentTypeEnum),
    )
}
