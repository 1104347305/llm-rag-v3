import unittest

from src.main.python.steps.retrieval.bm25 import bm25_search
from src.main.python.models import Chunk, GraphEdge, Page
from src.main.python.db.elasticsearch import ElasticsearchClient, ElasticsearchUnavailable


class FakeElasticsearchClient:
    def __init__(self):
        self.calls = []

    def search_chunks(self, project_id, query, top_k):
        self.calls.append({"project_id": project_id, "query": query, "top_k": top_k})
        return [("chunk-a", 3.14)]


class TestBM25Elasticsearch(unittest.TestCase):
    def test_bm25_delegates_to_elasticsearch(self):
        client = FakeElasticsearchClient()
        results = bm25_search("家庭医生", "pingan", 10, client=client)
        assert results == [("chunk-a", 3.14)]
        assert client.calls == [{"project_id": "pingan", "query": "家庭医生", "top_k": 10}]

    def test_elasticsearch_chunk_query_uses_should_recall(self):
        client = ElasticsearchClient.__new__(ElasticsearchClient)
        client.config = type("Config", (), {"es_chunks_index": "chunks"})()
        captured = {}

        def fake_request(method, path, body):
            captured["method"] = method
            captured["path"] = path
            captured["body"] = body
            return {"hits": {"hits": [{"_source": {"chunk_id": "chunk-a"}, "_score": 2.0}]}}

        client.request = fake_request
        results = client.search_chunks("pingan", "家庭医生", 5)

        assert results == [("chunk-a", 2.0)]
        bool_query = captured["body"]["query"]["bool"]
        assert bool_query["minimum_should_match"] == 1
        assert "must" not in bool_query
        assert any("term" in clause and "title.keyword" in clause["term"] for clause in bool_query["should"])
        assert any("match_phrase" in clause and "heading_path" in clause["match_phrase"] for clause in bool_query["should"])

    def test_elasticsearch_incremental_write_indexes_only_changed_paths(self):
        client = ElasticsearchClient.__new__(ElasticsearchClient)
        client.config = type(
            "Config",
            (),
            {
                "es_pages_index": "pages",
                "es_chunks_index": "chunks",
                "es_graph_edges_index": "edges",
            },
        )()
        requests = []
        bulks = []
        client.ensure_indexes = lambda: None
        client.request = lambda method, path, body=None, **kwargs: requests.append({"method": method, "path": path, "body": body}) or {}
        client.bulk_index = lambda index, docs: bulks.append({"index": index, "docs": docs})

        pages = [
            Page("p", "a", "a.md", "A", "entity", [], [], "A body", {}, "sha-a", 1),
            Page("p", "b", "b.md", "B", "entity", [], [], "B body", {}, "sha-b", 1),
        ]
        chunks = [
            Chunk("p", "a", "a#0000", "a.md", "A", "A", "entity", [], "A body", 0),
            Chunk("p", "b", "b#0000", "b.md", "B", "B", "entity", [], "B body", 0),
        ]
        edges = [GraphEdge("p", "a", "b", "same_type", 1.0)]

        client.write_indexes("p", pages, chunks, edges, changed_paths={"b.md"}, deleted_paths={"c.md"}, rebuild=False)

        delete_paths_request = next(item for item in requests if item["path"].startswith("pages/_delete_by_query"))
        should = delete_paths_request["body"]["query"]["bool"]["should"]
        assert {"terms": {"path": ["b.md", "c.md"]}} in should
        assert bulks[0]["index"] == "pages"
        assert [doc["_id"] for doc in bulks[0]["docs"]] == ["b"]
        assert bulks[1]["index"] == "chunks"
        assert [doc["_id"] for doc in bulks[1]["docs"]] == ["b#0000"]
        assert bulks[2]["index"] == "edges"
        assert [doc["_id"] for doc in bulks[2]["docs"]] == ["a->b:same_type"]

    def test_bulk_index_reports_item_error_details(self):
        client = ElasticsearchClient.__new__(ElasticsearchClient)
        client.request = lambda *args, **kwargs: {
            "errors": True,
            "items": [
                {
                    "index": {
                        "_id": "chunk-a",
                        "error": {
                            "type": "mapper_parsing_exception",
                            "reason": "failed to parse field [path]",
                            "caused_by": {"type": "illegal_argument_exception", "reason": "mapper conflict"},
                        },
                    }
                }
            ],
        }

        with self.assertRaisesRegex(ElasticsearchUnavailable, "chunk-a.*mapper_parsing_exception.*mapper conflict"):
            client.bulk_index("chunks", [{"_id": "chunk-a", "path": "a.md"}])
