import factory
from factory.django import DjangoModelFactory

from content.models import Content, Tag
from user.factories import UserFactory


class TagFactory(DjangoModelFactory):
    class Meta:  # type: ignore[override]
        model = Tag

    name = factory.Sequence(lambda n: f"Tag {n}")  # type: ignore[override]
    description = factory.Faker("sentence", nb_words=5)  # type: ignore[override]


class ContentFactory(DjangoModelFactory):
    class Meta:  # type: ignore[override]
        model = Content

    document_file = factory.django.FileField(filename="test.txt")
    extracted_file = factory.django.FileField(filename="extracted.txt")
    created_by = factory.SubFactory(UserFactory)  # type: ignore[override]
    modified_by = factory.SubFactory(UserFactory)  # type: ignore[override]

    @factory.post_generation  # type: ignore[override]
    def tag(self, create, extracted, **kwargs):
        if not create:
            return
        # If tags are passed, add them to the M2M field
        if extracted:
            if isinstance(extracted, list):
                for tag in extracted:
                    self.tag.add(tag)  # type: ignore[override]
            else:
                self.tag.add(extracted)  # type: ignore[override]
