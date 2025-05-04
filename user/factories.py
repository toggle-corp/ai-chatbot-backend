import factory
from factory.django import DjangoModelFactory

from user.models import User,Member


class UserFactory(DjangoModelFactory):
    class Meta:  # type: ignore[override]
        model = User

    email = factory.Sequence(lambda n: f"user{n}@example.com")  # type: ignore[override]
    display_name = factory.Faker("name")  # type: ignore[override]
    profile_picture = None
    department = User.Department.HR


class MemberFactory(DjangoModelFactory):
    user = factory.SubFactory(UserFactory) # type: ignore[override]
    class Meta:  # type: ignore[override]
        model = Member
