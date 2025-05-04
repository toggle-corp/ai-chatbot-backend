from unittest import mock

from django.core.files.uploadedfile import SimpleUploadedFile

from content.factories import ContentFactory, TagFactory
from content.models import Content
from main.test import TestCase
from user.factories import UserFactory


class TestContentMutations(TestCase):
    class Mutation:
        CREATE_CONTENT = """
            mutation MyMutation($data: ContentCreateInput!) {
              createContent(data: $data) {
                ok
                errors
                result {
                  id
                  title
                  documentType
                  documentStatus
                }
              }
            }
        """

        UPDATE_CONTENT_TITLE = """
            mutation MyMutation($data: UpdateContentTitleInput!) {
                updateContentTitle(data: $data) {
                  ok
                  errors
                  result {
                    title
                  }
                }
            }
        """

        ARCHIVE_CONTENT = """
            mutation MyMutation($data: ArchiveContentInput!) {
                archiveContent(data: $data) {
                  ok
                  errors
              }
            }
        """

        RETRIGGER_CONTENT = """
            mutation MyMutation($data: RetriggerContentInput!) {
                retriggerContent(data: $data) {
                ok
                errors
                result {
                    documentStatus
                    id
                    title
                }
                }
            }
        """

    def setUp(self):
        super().setUp()
        self.user1 = UserFactory.create(email="normaluser@gmail.com")
        self.tag1 = TagFactory.create(name="test tag", description="test tag description")
        self.content1 = ContentFactory.create(
            title="Test Document 1", tag=self.tag1, document_status=Content.DocumentStatus.FAILURE
        )
        self.content2 = ContentFactory.create(document_status=Content.DocumentStatus.TEXT_EXTRACTED, is_deleted=False)
        self.tags = TagFactory.create_batch(3)

    def _query_create_content(self, data, **kwargs):
        return self.query_check(
            self.Mutation.CREATE_CONTENT,
            variables={"data": data},
            **kwargs,
        )

    def _query_update_content(self, id: int, data, **kwargs):
        return self.query_check(
            self.Mutation.UPDATE_CONTENT_TITLE,
            variables={"data": {"content": self.gID(id), **data}},
            **kwargs,
        )

    def _query_archive_content(self, id: int, **kwargs):
        variables = {
            "data": {
                "content": id,
            }
        }
        return self.query_check(
            self.Mutation.ARCHIVE_CONTENT,
            variables=variables,
            **kwargs,
        )

    def _query_retrigger_content(self, id: int, **kwargs):
        variables = {
            "data": {
                "content": id,
            }
        }
        return self.query_check(
            self.Mutation.RETRIGGER_CONTENT,
            variables=variables,
            **kwargs,
        )

    @mock.patch("content.tasks.create_embedding_for_content_task.delay")
    def test_create_content(self, mock_create_embedding):
        file = SimpleUploadedFile("test.txt", b"This is a test file.", content_type="text/plain")
        self.force_login(self.user1)

        with self.captureOnCommitCallbacks(execute=True):
            response = self.query_check(
                self.Mutation.CREATE_CONTENT,
                variables={
                    "data": {
                        "title": "Test Content",
                        "documentFile": None,
                        "documentType": "TEXT",
                    }
                },
                files={"0": file},
                map={"0": ["variables.data.documentFile"]},
            )

        # Assert mutation success
        data = response["data"]["createContent"]
        assert data["ok"] is True
        assert data["result"]["title"] == "Test Content"
        assert data["result"]["documentType"] == "TEXT"
        assert data["result"]["documentStatus"] == "TEXT_EXTRACTED"
        # DB validation
        content_obj = Content.objects.get(pk=data["result"]["id"])
        assert content_obj.title == "Test Content"
        mock_create_embedding.assert_called_once_with(content_obj.id)

    def test_update_content_title(self):
        content_id = self.content1.id
        data = {"title": "Updated Title"}
        # Without authentication
        content = self._query_update_content(content_id, data, assert_errors=True)
        assert content["data"] is None
        # With authentication
        self.force_login(self.user1)
        content = self._query_update_content(content_id, data)
        assert content["data"]["updateContentTitle"]["ok"] is True
        assert content["data"]["updateContentTitle"]["result"]["title"] == "Updated Title"

    @mock.patch("content.serializers.delete_content_from_qdrant_task")
    def test_archive_content_mutation(self, mock_delete_task):
        # Without authentication
        response = self._query_archive_content(self.content2.pk, assert_errors=True)
        assert response["data"] is None
        self.force_login(self.user1)
        # Execute the mutation and ensure it triggers commit callbacks
        with self.captureOnCommitCallbacks(execute=True):
            response = self._query_archive_content(self.content2.pk)
            resp_data = response["data"]["archiveContent"]

        assert resp_data["ok"] is True
        assert resp_data["errors"] is None
        self.content2.refresh_from_db()
        assert self.content2.is_deleted is True
        assert self.content2.document_status == Content.DocumentStatus.DELETED_FROM_VECTOR
        mock_delete_task.delay.assert_called_once_with(self.content2.content_id)

    @mock.patch("content.tasks.create_embedding_for_content_task.delay")
    def test_retrigger_content(self, mock_create_embedding_for_content_task):
        # Without authentication
        response = self._query_retrigger_content(self.content1.pk, assert_errors=True)
        assert response["data"] is None
        # With authentication
        self.force_login(self.user1)
        # Trigger the retriggerContent mutation
        with self.captureOnCommitCallbacks(execute=True):
            response = self._query_retrigger_content(self.content1.pk)
            resp_data = response["data"]["retriggerContent"]

        assert resp_data["ok"] is True
        assert resp_data["errors"] is None
        assert resp_data["result"]["id"] == str(self.content1.id)
        mock_create_embedding_for_content_task.assert_called_once_with(self.content1.id)


class TestTagMutations(TestCase):
    class Mutation:
        CREATE_TAG = """
            mutation MyMutation($data: CreateTagCreateInput!) {
                createTag(data: $data) {
                  ok
                  errors
                  result {
                    id
                    name
                    description
                  }
                }
            }
        """

    def setUp(self):
        super().setUp()
        self.user1 = UserFactory.create()

    def _query_create_tag(self, data, **kwargs):
        return self.query_check(
            self.Mutation.CREATE_TAG,
            variables={"data": data},
            **kwargs,
        )

    def test_create_tag(self):
        input_data = {"name": "test tag", "description": " test tag description"}
        # Without authentication
        response = self._query_create_tag(input_data, assert_errors=True)
        assert response["data"] is None
        # With authentication
        self.force_login(self.user1)
        response = self._query_create_tag(input_data)
        assert response["data"]["createTag"]["ok"] is True
        assert response["data"]["createTag"]["result"]["name"] == "test tag"
