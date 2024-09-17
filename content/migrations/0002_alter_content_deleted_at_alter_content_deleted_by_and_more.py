# Generated by Django 5.1.1 on 2024-09-13 11:15

import django.db.models.deletion
import django.utils.timezone
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("content", "0001_initial"),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.AlterField(
            model_name="content",
            name="deleted_at",
            field=models.DateTimeField(blank=True, default=django.utils.timezone.now),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name="content",
            name="deleted_by",
            field=models.ForeignKey(
                blank=True, null=True, on_delete=django.db.models.deletion.PROTECT, to=settings.AUTH_USER_MODEL
            ),
        ),
        migrations.AlterField(
            model_name="content",
            name="description",
            field=models.TextField(help_text="Content text"),
        ),
        migrations.AlterField(
            model_name="content",
            name="document_file",
            field=models.FileField(blank=True, upload_to="documents"),
        ),
        migrations.AlterField(
            model_name="content",
            name="extracted_file",
            field=models.FileField(blank=True, upload_to="documents"),
        ),
        migrations.AlterField(
            model_name="content",
            name="tag",
            field=models.ManyToManyField(blank=True, to="content.tag"),
        ),
    ]