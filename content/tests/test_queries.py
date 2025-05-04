from main.tests import TestCase
from content.factories import ContentFactory, TagFactory
from user.factories import UserFactory


class TestContentQueries(TestCase):
    class Query:
        CONTENTS = """
            query GetContents($limit: Int!, $offset: Int!) {
                contents(pagination: {limit: $limit, offset: $offset}) {
                  totalCount
                  pageInfo {
                    limit
                    offset
                  }
                  results {
                    id
                    title
                    createdAt
                    documentStatus
                    documentType
                    tag {
                      id
                      name
                  }
                }
              }
            }
        """

        CONTENT_BY_ID = """
            query GetContent($id: ID!) {
                content(pk: $id) {
                  id
                  title
                  createdAt
                  documentStatus
                  documentType
                  tag {
                    id
                    name
                }
              }
            }
        """


    def setUp(self):
        super().setUp()
        self.user = UserFactory.create()
        self.tags = TagFactory.create_batch(3)
        self.contents = ContentFactory.create_batch(5, tag=self.tags)

    def test_contents_query(self):
          # Without authentication
          content = self.query_check(
              self.Query.CONTENTS,
              variables={"limit": 10, "offset": 0},
          )
          assert content["data"]["contents"]["totalCount"] == 0

          # With authentication        
          self.force_login(self.user)
          content = self.query_check(
              self.Query.CONTENTS,
              variables={"limit": 10, "offset": 0}
          )
          content_data = content["data"]["contents"]  
          assert content_data["totalCount"] == len(self.contents)



    def test_content_by_id_query(self):
        self.content = ContentFactory.create(title="Test Content", tag=self.tags)
        
        # Without authentication
        content = self.query_check(
            self.Query.CONTENT_BY_ID,
            variables={"id": self.gID(self.content.id)},
            assert_errors=True,
        )
        # With authentication
        self.force_login(self.user)

        content = self.query_check(
            self.Query.CONTENT_BY_ID,
            variables={"id": self.gID(self.content.id)},
        )
        content_data = content["data"]["content"]
        assert content_data["id"] == self.gID(self.content.id)
        assert content_data["title"] == self.content.title

      

class TestTagQueries(TestCase):
    class Query:
        TAGS = """
        query GetTags($pagination: OffsetPaginationInput!) {
            tags(pagination: $pagination) {
            totalCount
            results {
            id
            name
            description
            }
          }
        }
        """

        TAG_BY_ID = """
            query GetTag($id: ID!) {
                tag(pk: $id) {
                  id
                  name
                  description
                }
            }
        """

    def setUp(self):
        super().setUp()
        self.user = UserFactory.create()
        self.tags = TagFactory.create_batch(3)


    def test_tag_by_id_query(self):
        # Without authentication
        content = self.query_check(
            self.Query.TAG_BY_ID,
            variables={"id": self.gID(self.tags[0].id)},
            assert_errors=True,
        )
        # with authentication
        self.force_login(self.user)
        tag = TagFactory.create(name="Test Tag", description="Test Description")

        content = self.query_check(
            self.Query.TAG_BY_ID,
            variables={"id": self.gID(tag.id)},
        )
        tag_data = content["data"]["tag"]
        assert tag_data["id"] == self.gID(tag.id)
        assert tag_data["name"] == tag.name
        assert tag_data["description"] == tag.description
    
    def test_tags_query(self):
      # Without authentication
      response = self.query_check(
          self.Query.TAGS,
          variables={"pagination": {"limit": 10, "offset": 0}},
      )

      assert response["data"]["tags"]["totalCount"] == 0

      # With authentication
      self.force_login(self.user)
      response = self.query_check(
          self.Query.TAGS,
          variables={"pagination": {"limit": 10, "offset": 0}},
      )
      tags_data = response["data"]["tags"]
      assert tags_data["totalCount"] == len(self.tags)

    