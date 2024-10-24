from django.utils.functional import cached_property
from user.dataloaders import UserDataLoader

class GlobalDataLoader:

    @cached_property
    def user(self):
        return UserDataLoader()