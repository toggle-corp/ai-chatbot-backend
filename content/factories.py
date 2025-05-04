import factory
from factory.django import DjangoModelFactory
from content.models import Tag, Content 
from user.factories import UserFactory  

class TagFactory(DjangoModelFactory):
    class Meta:
        model = Tag

    name = factory.Sequence(lambda n: f"Tag {n}")
    description = factory.Faker("sentence", nb_words=5)


class ContentFactory(DjangoModelFactory):
    class Meta:
        model = Content

    document_file = factory.django.FileField(filename="test.pdf")
    extracted_file = factory.django.FileField(filename="extracted.txt")
    created_by = factory.SubFactory(UserFactory)
    modified_by = factory.SubFactory(UserFactory)
    
    @factory.post_generation
    def tag(self, create, extracted, **kwargs):
        if not create:
            return

        # If tags are passed, add them to the M2M field
        if extracted:
            if isinstance(extracted, list):
                for tag in extracted:
                    self.tag.add(tag)
            else:
                self.tag.add(extracted)


 