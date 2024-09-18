# Generated by Django 5.1.1 on 2024-09-18 05:28

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("content", "0001_initial"),
    ]

    operations = [
        migrations.AlterField(
            model_name="content",
            name="document_type",
            field=models.IntegerField(choices=[(1, "Word"), (2, "PDF"), (3, "Text")], default=3),
        ),
    ]