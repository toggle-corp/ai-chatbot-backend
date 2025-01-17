from django.contrib import admin

from .models import UserChatMessage, UserChatSession

# Register your models here.


@admin.register(UserChatSession)
class UserChatSessionAdmin(admin.ModelAdmin):
    list_display = ["user_uuid", "platform"]


@admin.register(UserChatMessage)
class UserChatMessageAdmin(admin.ModelAdmin):
    list_display = ["session", "type", "question", "answer", "status"]
    list_filter = ["status", "type", "session__platform"]
