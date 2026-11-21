"""Contract tests for the asynchronous ingestion endpoint and status lookup.

The service and publisher are mocked so these tests stay offline. Durable
PostgreSQL behavior is covered separately in ``test_ingestion_worker.py``.
"""

from __future__ import annotations

from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from app.main import app
from app.services.ingestion import (
    EmptyDocumentError,
    IngestionJobNotFoundError,
    IngestionQueueUnavailableError,
    SubmittedIngestion,
    TenantNotFoundError,
)

client = TestClient(app)
ASYNC_INGEST_URL = "/api/v1/documents/ingest/async"
SYNC_INGEST_URL = "/api/v1/documents/ingest"
TEST_TENANT_ID = UUID("11111111-1111-4111-8111-111111111111")


def _payload(**overrides) -> dict:
    payload = {
        "source_key": "runbooks/async-payment.md",
        "title": "Async payment runbook",
        "source_type": "runbook",
        "content": "Investigate PAY-503 before escalation.",
    }
    payload.update(overrides)
    return payload


def _submission(**overrides) -> SubmittedIngestion:
    values = {
        "job_id": uuid4(),
        "status": "running",
        "document_id": uuid4(),
        "version_id": uuid4(),
        "created": True,
        "needs_publish": True,
    }
    values.update(overrides)
    return SubmittedIngestion(**values)


def test_async_endpoint_returns_202_with_stable_job_and_status_url(monkeypatch) -> None:
    from app.api import documents

    submission = _submission()
    published: list[tuple[UUID, UUID | None]] = []
    captured: dict = {}

    def fake_submit(**kwargs):
        captured.update(kwargs)
        return submission

    def fake_publish(job_id, *, tenant_id=None):
        published.append((job_id, tenant_id))
        return "task-123"

    monkeypatch.setattr(documents, "submit_ingestion_job", fake_submit)
    monkeypatch.setattr(documents, "publish_ingestion_job", fake_publish)

    response = client.post(ASYNC_INGEST_URL, json=_payload())

    assert response.status_code == 202
    body = response.json()
    assert body == {
        "job_id": str(submission.job_id),
        "status": "running",
        "status_url": f"/api/v1/documents/jobs/{submission.job_id}",
        "deduplicated": False,
    }
    assert captured["tenant_id"] == TEST_TENANT_ID
    assert published == [(submission.job_id, TEST_TENANT_ID)]


def test_async_endpoint_does_not_republish_succeeded_or_running_job(monkeypatch) -> None:
    from app.api import documents

    submission = _submission(status="succeeded", needs_publish=False, created=False)
    monkeypatch.setattr(documents, "submit_ingestion_job", lambda **kwargs: submission)
    monkeypatch.setattr(
        documents,
        "publish_ingestion_job",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not publish")),
    )

    response = client.post(ASYNC_INGEST_URL, json=_payload())

    assert response.status_code == 202
    assert response.json()["job_id"] == str(submission.job_id)
    assert response.json()["status"] == "succeeded"
    assert response.json()["deduplicated"] is True


def test_async_endpoint_maps_broker_failure_to_503(monkeypatch) -> None:
    from app.api import documents

    monkeypatch.setattr(documents, "submit_ingestion_job", lambda **kwargs: _submission())
    monkeypatch.setattr(
        documents,
        "publish_ingestion_job",
        lambda *args, **kwargs: (_ for _ in ()).throw(IngestionQueueUnavailableError()),
    )

    response = client.post(ASYNC_INGEST_URL, json=_payload())

    assert response.status_code == 503
    assert response.json() == {"detail": "Background ingestion queue is unavailable"}


def test_async_endpoint_maps_service_errors(monkeypatch) -> None:
    from app.api import documents

    cases = [
        (TenantNotFoundError(), 404, "Tenant not found"),
        (EmptyDocumentError("empty"), 422, "empty"),
    ]
    for error, status_code, detail in cases:
        monkeypatch.setattr(
            documents,
            "submit_ingestion_job",
            lambda _error=error, **kwargs: (_ for _ in ()).throw(_error),
        )
        response = client.post(ASYNC_INGEST_URL, json=_payload())
        assert response.status_code == status_code
        assert response.json()["detail"] == detail


def test_async_endpoint_rejects_body_tenant_mismatch(monkeypatch) -> None:
    from app.api import documents

    monkeypatch.setattr(
        documents,
        "submit_ingestion_job",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("must not run")),
    )

    response = client.post(
        ASYNC_INGEST_URL,
        json=_payload(tenant_id=str(uuid4())),
    )

    assert response.status_code == 403


def test_job_status_is_tenant_scoped(monkeypatch) -> None:
    from app.api import documents

    job_id = uuid4()
    seen: list[tuple[UUID, UUID]] = []

    def fake_status(value, *, tenant_id, principal):
        seen.append((value, tenant_id))
        assert principal.tenant_id == tenant_id
        raise IngestionJobNotFoundError

    monkeypatch.setattr(documents, "get_ingestion_job_status", fake_status)

    response = client.get(f"/api/v1/documents/jobs/{job_id}")

    assert response.status_code == 404
    assert response.json() == {"detail": "Ingestion job not found"}
    assert seen == [(job_id, TEST_TENANT_ID)]


def test_job_status_returns_sanitized_state(monkeypatch) -> None:
    from app.api import documents

    job_id = uuid4()
    expected = {
        "job_id": str(job_id),
        "status": "failed",
        "attempts": 2,
        "document_id": str(uuid4()),
        "version_id": str(uuid4()),
        "error_code": "embedding_service_unavailable",
        "error_summary": "Embedding service was unavailable",
        "created_at": "2026-01-01T00:00:00+00:00",
        "started_at": "2026-01-01T00:00:01+00:00",
        "finished_at": "2026-01-01T00:00:02+00:00",
    }
    monkeypatch.setattr(
        documents,
        "get_ingestion_job_status",
        lambda value, *, tenant_id, principal: expected,
    )

    response = client.get(f"/api/v1/documents/jobs/{job_id}")

    assert response.status_code == 200
    assert response.json() == expected


def test_synchronous_endpoint_contract_is_unchanged(monkeypatch) -> None:
    from app.api import documents

    expected = {
        "document_id": uuid4(),
        "version_id": uuid4(),
        "version_number": 1,
        "content_hash": "a" * 64,
        "chunk_count": 1,
        "deduplicated": False,
        "is_current": True,
        "embedding_status": "complete",
    }
    monkeypatch.setattr(documents, "ingest_document", lambda **kwargs: expected)
    monkeypatch.setattr(
        documents,
        "submit_ingestion_job",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("async path must not run")),
    )

    response = client.post(SYNC_INGEST_URL, json=_payload())

    assert response.status_code == 201
    assert set(response.json()) == {
        "document_id",
        "version_id",
        "version_number",
        "content_hash",
        "chunk_count",
        "deduplicated",
        "is_current",
        "embedding_status",
    }


def test_submission_event_is_correlated_and_never_contains_content_or_tokens(
    monkeypatch,
) -> None:
    import json
    import logging

    from app.api import documents
    from app.core.observability import logger as request_logger

    submission = _submission()
    monkeypatch.setattr(documents, "submit_ingestion_job", lambda **kwargs: submission)
    monkeypatch.setattr(
        documents, "publish_ingestion_job", lambda job_id, *, tenant_id=None: "task-1"
    )

    events: list[dict] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            data = getattr(record, "event_data", None)
            if isinstance(data, dict):
                events.append(data)

    handler = Capture()
    request_logger.addHandler(handler)
    try:
        response = client.post(
            ASYNC_INGEST_URL,
            headers={
                "Authorization": "Bearer super-secret-bearer-token",
                "X-Request-ID": "req-abc12345",
            },
            json=_payload(content="The launch code is hunter2 and must not be logged."),
        )
    finally:
        request_logger.removeHandler(handler)

    assert response.status_code == 202
    submissions = [event for event in events if event.get("event") == "ingestion_submission"]
    assert len(submissions) == 1
    event = submissions[0]
    assert event["job_id"] == str(submission.job_id)
    assert event["status"] == submission.status
    assert event["deduplicated"] is False
    assert event["published"] is True
    assert event["request_id"] == "req-abc12345"
    assert event["tenant_ref"].startswith("t_")

    serialized = json.dumps(event)
    assert "hunter2" not in serialized
    assert "super-secret-bearer-token" not in serialized
    assert str(TEST_TENANT_ID) not in serialized
