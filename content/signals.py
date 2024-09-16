from django.db.models.signals import post_save
from django.dispatch import receiver

from .models import Content
from .tasks import create_embedding_for_content_task


@receiver(post_save, sender=Content)
def content_handler(sender, instance, created, **kwargs):
    print(sender, instance, created, kwargs)
    create_embedding_for_content_task.delay(instance.id)
