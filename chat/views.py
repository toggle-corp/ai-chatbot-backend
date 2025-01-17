from chat.models import UserChatMessage, UserChatSession


class UserSession:
    def create_chat_session(self, data):
        obj, _ = UserChatSession.objects.get_or_create(
            user_uuid=data.get("user_id"), defaults={"platform": data.get("platform")}
        )
        return obj

    def create_chat_message(self, data, chat_session):
        user_chat_message = UserChatMessage.objects.create(
            session=chat_session, query=data.get("query"), status=UserChatMessage.Status.STARTED
        )
        return user_chat_message

    def update_chat_message(self, response, user_message):
        user_message.response = response
        user_message.status = UserChatMessage.Status.SUCCESS
        user_message.save(update_fields=["response", "status"])
        return user_message
