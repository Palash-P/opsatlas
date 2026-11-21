"""Contract tests for the document ingestion endpoint.

These tests freeze the existing request fields, response keys, status codes,
tenant resolution, and error mapping. They intentionally mock the service so
they stay fast and offline; DB-backed endpoint behavior is covered in
``tests/test_ingestion_service.py``.
"""

from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from app.db.embedding_schema import EmbeddingSchemaError
from app.main import app
from app.services.embedding_client import (
    EmbeddingServiceError,
    EmbeddingUnavailableError,
    EmbeddingValidationError,
)
from app.services.ingestion import (
    EmbeddingCompletenessError,
    EmptyDocumentError,
    IdempotencyKeyCollisionError,
    TenantNotFoundError,
)
from tests.conftest import TEST_PRINCIPAL

client = TestClient(app)
TEST_TENANT_ID = UUID("11111111-1111-4111-8111-111111111111")
INGEST_URL = "/api/v1/documents/ingest"
RESPONSE_KEYS = {
    "document_id",
    "version_id",
    "version_number",
    "content_hash",
    "chunk_count",
    "deduplicated",
    "is_current",
    "embedding_status",
}


def _ingestion_result(**overrides) -> dict:
    result = {
        "document_id": uuid4(),
        "version_id": uuid4(),
        "version_number": 1,
        "content_hash": "a" * 64,
        "chunk_count": 1,
        "deduplicated": False,
        "is_current": True,
        "embedding_status": "complete",
    }
    result.update(overrides)
    return result


def _payload(**overrides) -> dict:
    payload = {
        "tenant_id": str(TEST_TENANT_ID),
        "source_key": "runbooks/payment-api.md",
        "title": "Payment API Runbook",
        "source_type": "runbook",
        "content": "Investigate 503 responses.",
    }
    payload.update(overrides)
    return payload


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
        "embedding_status": "complete",
    }

    def fake_ingest_document(**kwargs):
        assert kwargs["tenant_id"] == tenant_id
        assert kwargs["source_key"] == "runbooks/payment-api.md"
        return expected

    monkeypatch.setattr(documents, "ingest_document", fake_ingest_document)
    response = client.post(INGEST_URL, json=_payload())

    assert response.status_code == 201
    assert set(response.json()) == RESPONSE_KEYS
    assert response.json() == expected


def test_ingest_endpoint_rejects_empty_content() -> None:
    response = client.post(INGEST_URL, json=_payload(content=""))
    assert response.status_code == 422


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_key", ""),
        ("title", ""),
        ("source_type", "pdf"),
        ("authority_level", 101),
        ("source_uri", "x" * 2001),
    ],
)
def test_ingest_endpoint_enforces_existing_field_bounds(field: str, value) -> None:
    response = client.post(INGEST_URL, json=_payload(**{field: value}))
    assert response.status_code == 422


def test_ingest_endpoint_rejects_content_over_two_million_characters() -> None:
    # Built inside the test body rather than parametrized: embedding 2,000,001
    # characters in a test ID makes pytest's PYTEST_CURRENT_TEST env var exceed
    # the 32,767-character limit on Windows.
    response = client.post(INGEST_URL, json=_payload(content="x" * 2_000_001))
    assert response.status_code == 422


def test_ingest_endpoint_forwards_defaults(monkeypatch) -> None:
    from app.api import documents

    captured: dict = {}

    def fake_ingest_document(**kwargs):
        captured.update(kwargs)
        return _ingestion_result()

    monkeypatch.setattr(documents, "ingest_document", fake_ingest_document)
    response = client.post(
        INGEST_URL,
        json={
            "source_key": "runbooks/minimal.md",
            "title": "Minimal runbook",
            "content": "Minimal content.",
        },
    )

    assert response.status_code == 201
    assert captured == {
        "principal": TEST_PRINCIPAL,
        "tenant_id": TEST_TENANT_ID,
        "source_key": "runbooks/minimal.md",
        "title": "Minimal runbook",
        "source_type": "markdown",
        "content": "Minimal content.",
        "source_uri": None,
        "authority_level": 50,
    }


def test_ingest_endpoint_forwards_explicit_optional_fields(monkeypatch) -> None:
    from app.api import documents

    captured: dict = {}

    def fake_ingest_document(**kwargs):
        captured.update(kwargs)
        return _ingestion_result()

    monkeypatch.setattr(documents, "ingest_document", fake_ingest_document)
    response = client.post(
        INGEST_URL,
        json=_payload(
            tenant_id=str(TEST_TENANT_ID),
            source_type="api_spec",
            source_uri="https://example.test/spec",
            authority_level=80,
        ),
    )

    assert response.status_code == 201
    assert captured["tenant_id"] == TEST_TENANT_ID
    assert captured["source_type"] == "api_spec"
    assert captured["source_uri"] == "https://example.test/spec"
    assert captured["authority_level"] == 80


def test_ingest_endpoint_uses_authenticated_tenant_when_body_tenant_is_omitted(
    monkeypatch,
) -> None:
    from app.api import documents

    captured: dict = {}

    def fake_ingest_document(**kwargs):
        captured.update(kwargs)
        return _ingestion_result()

    monkeypatch.setattr(documents, "ingest_document", fake_ingest_document)
    payload = _payload()
    payload.pop("tenant_id")
    response = client.post(INGEST_URL, json=payload)

    assert response.status_code == 201
    assert captured["tenant_id"] == TEST_TENANT_ID


def test_ingest_endpoint_returns_200_for_duplicate_submission(monkeypatch) -> None:
    from app.api import documents

    expected = _ingestion_result(deduplicated=True)

    def fake_ingest_document(**kwargs):
        return expected

    monkeypatch.setattr(documents, "ingest_document", fake_ingest_document)
    response = client.post(INGEST_URL, json=_payload())

    assert response.status_code == 200
    assert response.json()["deduplicated"] is True
    assert response.json()["document_id"] == str(expected["document_id"])


def test_ingest_endpoint_maps_unknown_tenant_to_404(monkeypatch) -> None:
    from app.api import documents

    def fake_ingest_document(**kwargs):
        raise TenantNotFoundError

    monkeypatch.setattr(documents, "ingest_document", fake_ingest_document)
    response = client.post(INGEST_URL, json=_payload())

    assert response.status_code == 404
    assert response.json() == {"detail": "Tenant not found"}


def test_ingest_endpoint_maps_service_empty_content_to_422(monkeypatch) -> None:
    from app.api import documents

    def fake_ingest_document(**kwargs):
        raise EmptyDocumentError("Document content must contain non-whitespace text")

    monkeypatch.setattr(documents, "ingest_document", fake_ingest_document)
    response = client.post(INGEST_URL, json=_payload(content=" \n\t "))

    assert response.status_code == 422
    assert response.json() == {"detail": "Document content must contain non-whitespace text"}


def test_ingest_endpoint_maps_idempotency_collision_to_409(monkeypatch) -> None:
    from app.api import documents

    def fake_ingest_document(**kwargs):
        raise IdempotencyKeyCollisionError

    monkeypatch.setattr(documents, "ingest_document", fake_ingest_document)
    response = client.post(INGEST_URL, json=_payload())

    assert response.status_code == 409
    assert response.json() == {"detail": "Ingestion idempotency key conflict"}


@pytest.mark.parametrize(
    ("error", "status_code", "detail"),
    [
        (
            EmbeddingUnavailableError("token=SECRET dsn=postgresql://user:pass@host/db timeout"),
            503,
            "Embedding service is unavailable",
        ),
        (
            EmbeddingValidationError("provider returned CONFIDENTIAL_RESPONSE"),
            502,
            "Embedding generation failed validation",
        ),
        (
            EmbeddingCompletenessError("chunk 7 embedding is NULL"),
            502,
            "Embedding generation failed validation",
        ),
        (
            EmbeddingServiceError("provider said SECRET"),
            502,
            "Embedding generation failed",
        ),
        (
            EmbeddingSchemaError("column is vector(768) dsn=postgresql://user:pass@host/db"),
            500,
            "Embedding configuration is incompatible with the database schema",
        ),
    ],
)
def test_ingest_endpoint_maps_embedding_errors_without_leaking_details(
    monkeypatch,
    error,
    status_code: int,
    detail: str,
) -> None:
    from app.api import documents

    def fake_ingest_document(**kwargs):
        raise error

    monkeypatch.setattr(documents, "ingest_document", fake_ingest_document)
    response = client.post(INGEST_URL, json=_payload())

    assert response.status_code == status_code
    assert response.json() == {"detail": detail}
    assert "SECRET" not in response.text
    assert "CONFIDENTIAL" not in response.text
    assert "postgresql://" not in response.text
