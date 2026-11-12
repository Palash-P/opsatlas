from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
TEST_TENANT_ID = UUID("11111111-1111-4111-8111-111111111111")


def test_ingest_endpoint_returns_ingestion_result(monkeypatch) -> None:
    from app.api import documents

    tenant_id = TEST_TENANT_ID
    expected = {
        "document_id": str(uuid4()),
        "version_id": str(uuid4()),
        "version_number": 1,
        "content_hash": "a" * 64,
        "chunk_count": 1,
        "deduplicated": False,
        "is_current": True,
        "embedding_status": "pending",
    }

    def fake_ingest_document(**kwargs):
        assert kwargs["tenant_id"] == tenant_id
        assert kwargs["source_key"] == "runbooks/payment-api.md"
        return expected

    monkeypatch.setattr(documents, "ingest_document", fake_ingest_document)
    response = client.post(
        "/api/v1/documents/ingest",
        json={
            "tenant_id": str(tenant_id),
            "source_key": "runbooks/payment-api.md",
            "title": "Payment API Runbook",
            "source_type": "runbook",
            "content": "Investigate 503 responses.",
        },
    )

    assert response.status_code == 201
    assert response.json() == expected


def test_ingest_endpoint_rejects_empty_content() -> None:
    response = client.post(
        "/api/v1/documents/ingest",
        json={
            "tenant_id": str(TEST_TENANT_ID),
            "source_key": "empty.md",
            "title": "Empty document",
            "source_type": "markdown",
            "content": "",
        },
    )
    assert response.status_code == 422
