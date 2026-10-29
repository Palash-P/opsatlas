from uuid import uuid4

import pytest

from app.services import retrieval


def test_semantic_search_rejects_blank_query() -> None:
    with pytest.raises(ValueError, match="query must not be empty"):
        retrieval.semantic_search(tenant_id=uuid4(), query="   ")


@pytest.mark.parametrize("limit", [0, -1, 21])
def test_semantic_search_rejects_invalid_limit(limit: int) -> None:
    with pytest.raises(ValueError, match="limit must be between 1 and 20"):
        retrieval.semantic_search(
            tenant_id=uuid4(),
            query="payments failing after deployment",
            limit=limit,
        )


def test_semantic_search_scopes_query_to_tenant_and_current_version(
    monkeypatch,
) -> None:
    tenant_id = uuid4()
    captured = {}

    def fake_embed_texts(texts):
        assert texts == ["payments failing after deployment"]
        return [[0.0] * 384]

    class FakeResult:
        def mappings(self):
            return self

        def all(self):
            return [
                {
                    "chunk_id": uuid4(),
                    "document_id": uuid4(),
                    "version_id": uuid4(),
                    "title": "Payment API Runbook",
                    "source_key": "runbooks/payment-api.md",
                    "version_number": 2,
                    "authority_level": 80,
                    "chunk_index": 0,
                    "content": "Check upstream timeouts and database pool saturation.",
                    "distance": 0.2,
                }
            ]

    class FakeConnection:
        def execute(self, statement, params):
            captured["sql"] = str(statement)
            captured["params"] = params
            return FakeResult()

    class FakeEngine:
        def connect(self):
            class Context:
                def __enter__(self):
                    return FakeConnection()

                def __exit__(self, *args):
                    return False

            return Context()

    monkeypatch.setattr(retrieval, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(retrieval, "get_engine", lambda: FakeEngine())

    results = retrieval.semantic_search(
        tenant_id=tenant_id,
        query="payments failing after deployment",
        limit=5,
    )

    assert len(results) == 1
    assert results[0]["similarity"] == pytest.approx(0.8)
    assert results[0]["cosine_distance"] == pytest.approx(0.2)

    sql = captured["sql"].lower()
    assert "c.tenant_id = :tenant_id" in sql
    assert "d.tenant_id = :tenant_id" in sql
    assert "dv.is_current = true" in sql
    assert "d.status = 'active'" in sql
    assert "c.embedding_model = :model" in sql

    assert captured["params"]["tenant_id"] == tenant_id
    assert captured["params"]["result_limit"] == 5


def test_semantic_search_breaks_distance_ties_by_chunk_id(monkeypatch) -> None:
    captured = {}

    class FakeResult:
        def mappings(self):
            return self

        def all(self):
            return []

    class FakeConnection:
        def execute(self, statement, params):
            captured["sql"] = str(statement)
            return FakeResult()

    class FakeEngine:
        def connect(self):
            class Context:
                def __enter__(self):
                    return FakeConnection()

                def __exit__(self, *args):
                    return False

            return Context()

    monkeypatch.setattr(retrieval, "embed_texts", lambda texts: [[0.0] * 384])
    monkeypatch.setattr(retrieval, "get_engine", lambda: FakeEngine())

    retrieval.semantic_search(tenant_id=uuid4(), query="payment errors", limit=5)

    sql = " ".join(captured["sql"].lower().split())
    assert (
        "order by c.embedding <=> cast(:query_embedding as vector(384)), c.id"
        in sql
    )


def test_keyword_search_scopes_query_to_tenant_and_current_version(monkeypatch) -> None:
    tenant_id = uuid4()
    captured = {}

    class FakeResult:
        def mappings(self):
            return self

        def all(self):
            return [
                {
                    "chunk_id": uuid4(),
                    "document_id": uuid4(),
                    "version_id": uuid4(),
                    "title": "Payment API Runbook",
                    "source_key": "runbooks/payment-api.md",
                    "version_number": 2,
                    "authority_level": 80,
                    "chunk_index": 0,
                    "content": "Check upstream timeouts and database pool saturation.",
                    "keyword_score": 0.42,
                }
            ]

    class FakeConnection:
        def execute(self, statement, params):
            captured["sql"] = str(statement)
            captured["params"] = params
            return FakeResult()

    class FakeEngine:
        def connect(self):
            class Context:
                def __enter__(self):
                    return FakeConnection()

                def __exit__(self, *args):
                    return False

            return Context()

    monkeypatch.setattr(retrieval, "get_engine", lambda: FakeEngine())

    results = retrieval.keyword_search(
        tenant_id=tenant_id,
        query="payment",
        limit=5,
    )

    assert results[0]["keyword_score"] == pytest.approx(0.42)

    sql = " ".join(captured["sql"].lower().split())
    assert "c.tenant_id = :tenant_id" in sql
    assert "d.tenant_id = :tenant_id" in sql
    assert "dv.is_current = true" in sql
    assert "d.status = 'active'" in sql
    assert "to_tsvector('english', c.content) @@ search_query.tsq" in sql

    assert captured["params"]["tenant_id"] == tenant_id
    assert captured["params"]["query"] == "payment"
    assert captured["params"]["result_limit"] == 5