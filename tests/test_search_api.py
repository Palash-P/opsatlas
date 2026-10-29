from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_search_endpoint_returns_results(monkeypatch) -> None:
    from app.api import search

    tenant_id = uuid4()
    expected = [{
        "chunk_id": uuid4(),
        "document_id": uuid4(),
        "version_id": uuid4(),
        "title": "Payment API Runbook",
        "source_key": "runbooks/payment-api.md",
        "version_number": 1,
        "authority_level": 80,
        "chunk_index": 0,
        "content": "Investigate upstream timeouts.",
        "cosine_distance": 0.2,
        "similarity": 0.8,
    }]

    def fake_semantic_search(**kwargs):
        assert kwargs["tenant_id"] == tenant_id
        assert kwargs["query"] == "payments failing after release"
        assert kwargs["limit"] == 5
        return expected

    monkeypatch.setattr(search, "semantic_search", fake_semantic_search)

    response = client.post(
        "/api/v1/search",
        json={
            "tenant_id": str(tenant_id),
            "query": "payments failing after release",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_count"] == 1
    assert body["results"][0]["title"] == "Payment API Runbook"
    assert body["results"][0]["similarity"] == 0.8


def test_search_endpoint_rejects_blank_query() -> None:
    response = client.post(
        "/api/v1/search",
        json={"tenant_id": str(uuid4()), "query": "   "},
    )

    assert response.status_code == 422


def test_search_endpoint_rejects_limit_above_maximum() -> None:
    response = client.post(
        "/api/v1/search",
        json={
            "tenant_id": str(uuid4()),
            "query": "payment errors",
            "limit": 21,
        },
    )

    assert response.status_code == 422


def test_search_endpoint_maps_embedding_service_failure(monkeypatch) -> None:
    from app.api import search
    from app.services.embedding_client import EmbeddingServiceError

    def fail_search(**kwargs):
        raise EmbeddingServiceError("internal connection details")

    monkeypatch.setattr(search, "semantic_search", fail_search)

    response = client.post(
        "/api/v1/search",
        json={"tenant_id": str(uuid4()), "query": "payment errors"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Embedding service is unavailable"


def test_hybrid_search_endpoint_returns_fused_results(monkeypatch) -> None:
    from app.api import search

    tenant_id = uuid4()
    expected = [{
        "chunk_id": uuid4(),
        "document_id": uuid4(),
        "version_id": uuid4(),
        "title": "Payment API Runbook",
        "source_key": "runbooks/payment-api.md",
        "version_number": 1,
        "authority_level": 80,
        "chunk_index": 0,
        "content": "Investigate upstream timeouts.",
        "hybrid_score": 0.0325,
        "semantic_rank": 1,
        "keyword_rank": 2,
        "semantic_similarity": 0.8,
        "keyword_score": 0.4,
    }]

    def fake_hybrid_search(**kwargs):
        assert kwargs["tenant_id"] == tenant_id
        assert kwargs["query"] == "payments failing after release"
        assert kwargs["limit"] == 5
        return expected

    monkeypatch.setattr(search, "hybrid_search", fake_hybrid_search)

    response = client.post(
        "/api/v1/search/hybrid",
        json={
            "tenant_id": str(tenant_id),
            "query": "payments failing after release",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_count"] == 1
    assert body["rrf_k"] == 60
    assert body["results"][0]["title"] == "Payment API Runbook"
    assert body["results"][0]["hybrid_score"] == 0.0325
    assert body["results"][0]["semantic_rank"] == 1
    assert body["results"][0]["keyword_rank"] == 2


def test_hybrid_search_endpoint_rejects_blank_query() -> None:
    response = client.post(
        "/api/v1/search/hybrid",
        json={"tenant_id": str(uuid4()), "query": "   "},
    )

    assert response.status_code == 422


def test_hybrid_search_endpoint_rejects_limit_above_maximum() -> None:
    response = client.post(
        "/api/v1/search/hybrid",
        json={
            "tenant_id": str(uuid4()),
            "query": "payment errors",
            "limit": 21,
        },
    )

    assert response.status_code == 422


def test_hybrid_search_endpoint_maps_embedding_service_failure(monkeypatch) -> None:
    from app.api import search
    from app.services.embedding_client import EmbeddingServiceError

    def fail_search(**kwargs):
        raise EmbeddingServiceError("internal connection details")

    monkeypatch.setattr(search, "hybrid_search", fail_search)

    response = client.post(
        "/api/v1/search/hybrid",
        json={"tenant_id": str(uuid4()), "query": "payment errors"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Embedding service is unavailable"