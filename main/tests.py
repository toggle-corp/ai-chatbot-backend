from datetime import datetime
from enum import Enum
import typing
from django.db import models
from django.test import TestCase as BaseTestCase
from django.test import override_settings



@override_settings(
    DEBUG=True,
    EMAIL_BACKEND="django.core.mail.backends.console.EmailBackend",
    MEDIA_ROOT="rest-media-temp",
    CELERY_TASK_ALWAYS_EAGER=True,
)
class TestCase(BaseTestCase):
    def setUp(self):
        from django.core.cache import cache

        # Clear all test cache
        cache.clear()
        super().setUp()

    def force_login(self, user):
        self.client.force_login(user)

    def logout(self):
        self.client.logout()

    def query_check(
        self,
        query: str,
        assert_errors: bool = False,
        variables: dict | None = None,
        files: dict | None = None,
        **kwargs,
    ) -> dict:
        import json

        if files:
            # Request type: form data
            response = self.client.post(
                "/graphql/",
                data={
                    "operations": json.dumps(
                        {
                            "query": query,
                            "variables": variables,
                        },
                    ),
                    **files,
                    "map": json.dumps(kwargs.pop("map")),
                },
                **kwargs,
            )
        else:
            # Request type: json
            response = self.client.post(
                "/graphql/",
                data={
                    "query": query,
                    "variables": variables,
                },
                content_type="application/json",
                **kwargs,
            )
        if assert_errors:
            self.assertResponseHasErrors(response)
        else:
            self.assertResponseNoErrors(response)
        return response.json()

    def assertResponseNoErrors(self, resp, msg=None):
        """
        Assert that the call went through correctly. 200 means the syntax is ok,
        if there are no `errors`,
        the call was fine.
        :resp HttpResponse: Response
        """
        content = resp.json()
        assert resp.status_code == 200, msg or content
        assert "errors" not in content, msg or content

    def assertResponseHasErrors(self, resp, msg=None):
        """
        Assert that the call was failing. Take care: Even with errors,
        GraphQL returns status 200!
        :resp HttpResponse: Response
        """
        content = resp.json()
        assert "errors" in content, msg or content

    def genum(self, _enum: models.TextChoices | models.IntegerChoices | Enum) -> str | None:
        """
        Return appropriate enum value.
        """
        if _enum:
            return _enum.name
        return None

    def gdatetime(self, _datetime: datetime | None):
        if _datetime:
            return _datetime.isoformat()
        return None

    def gID(self, pk):
        if pk:
            return str(pk)
        return None
    
    
    def g_pagination(self, *, offset: int, limit: int, total_count: int, results: list[typing.Any]):
        return {
            "totalCount": total_count,
            "pageInfo": {"offset": offset, "limit": limit},
            "results": results,
        }

    def g_mutation_response(self, *, errors: typing.Any = None, ok: bool, result: typing.Any):
        return {
            "errors": errors,
            "ok": ok,
            "result": result,
        }


    def get_media_url(self, path):
        return f"http://testserver/media/{path}"

    def _dict_with_keys(
        self,
        data: dict,
        include_keys=None,
        ignore_keys=None,
    ):
        # TODO: Use self.assertDictEqual instead?
        if all([ignore_keys, include_keys]):
            raise Exception("Please use one of the options among include_keys, ignore_keys")
        return {
            key: value
            for key, value in data.items()
            if ((ignore_keys is not None and key not in ignore_keys) or (include_keys is not None and key in include_keys))
        }

    def assertListDictEqual(
        self,
        left,
        right,
        messages=None,
        ignore_keys: list[str] | None = None,
        include_keys: list[str] | None = None,
    ):
        _left = [self._dict_with_keys(item, ignore_keys=ignore_keys, include_keys=include_keys) for item in left]
        _right = [self._dict_with_keys(item, ignore_keys=ignore_keys, include_keys=include_keys) for item in right]
        assert _left == _right, messages

    def no_op(*args, **_): ...

