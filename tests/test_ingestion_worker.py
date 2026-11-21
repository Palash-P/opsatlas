"""Real-PostgreSQL tests for background ingestion resumption.

The Celery broker is not required: these tests call the worker service entry
point directly. Embeddings are deterministic local values. Every test uses a
throwaway tenant and deletes it afterwards.
"""

from __future__ import annotations

import time
from uuid import UUID, uuid4

import pytest
from sqlalchemy import inspect, text

from app.db.connections import check_postgres, get_engine
from app.db.embedding_schema import EmbeddingSchemaError
from app.services import ingestion as ingestion_module
from app.services.embedding_client import EmbeddingUnavailableError, EmbeddingValidationError
from app.services.ingestion import (
    IngestionJobNotFoundError,
    IngestionJobStateError,
    IngestionJobTerminalError,
    get_ingestion_job_status,
    mark_ingestion_job_enqueue_failed,
    resume_ingestion_job,
    submit_ingestion_job,
)
from tests.conftest import deterministic_embeddings, principal_for


@pytest.fixture(scope="module")
def db_engine():
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            check_postgres()
        except Exception as exc:  # noqa: BLE001 - absence must skip, not error
            last_error = exc
            if attempt < 2:
                time.sleep(0.25)
            continue

        engine = get_engine()
        job_columns = {column["name"] for column in inspect(engine).get_columns("ingestion_jobs")}
        document_columns = {column["name"] for column in inspect(engine).get_columns("documents")}
        missing = {"version_id", "activation_generation"} - job_columns
        missing |= {
            "next_dispatch_at",
            "dispatch_claimed_until",
            "dispatch_attempts",
            "execution_token",
            "execution_lease_expires_at",
        } - job_columns
        if "last_activation_generation" not in document_columns:
            missing.add("documents.last_activation_generation")
        if missing:
            pytest.fail(
                "Worker tests require sql/003_ingestion_jobs_version.sql, "
                "sql/004_activation_generation.sql, and "
                "sql/005_job_reconciliation.sql; missing: " + ", ".join(sorted(missing)),
                pytrace=False,
            )
        return engine

    assert last_error is not None
    pytest.skip(f"PostgreSQL is not available for worker tests: {type(last_error).__name__}")


@pytest.fixture(autouse=True)
def _deterministic_embedding(deterministic_embedding):
    return deterministic_embedding


@pytest.fixture
def tenant_id(db_engine) -> UUID:
    tenant = uuid4()
    with db_engine.begin() as connection:
        connection.execute(
            text("INSERT INTO tenants (id, name) VALUES (:id, :name)"),
            {"id": tenant, "name": f"phase-f-worker-{tenant}"},
        )
    try:
        yield tenant
    finally:
        with db_engine.begin() as connection:
            connection.execute(text("DELETE FROM tenants WHERE id = :id"), {"id": tenant})


def _content(prefix: str, lines: int = 120) -> str:
    return "\n".join(
        f"{prefix} step {index}: verify dependency {index} before escalation."
        for index in range(lines)
    )


def _submit(
    tenant_id: UUID,
    *,
    source_key: str = "runbooks/worker.md",
    content: str | None = None,
):
    return submit_ingestion_job(
        principal=principal_for(tenant_id),
        tenant_id=tenant_id,
        source_key=source_key,
        title="Worker runbook",
        source_type="runbook",
        content=content or _content("Worker"),
    )


def _job(tenant_id: UUID, job_id: UUID) -> dict:
    with get_engine().connect() as connection:
        row = (
            connection.execute(
                text(
                    """
                    SELECT id, status, attempts, document_id, version_id,
                           activation_generation, error_code, error_summary
                    FROM ingestion_jobs
                    WHERE id = :job_id AND tenant_id = :tenant_id
                    """
                ),
                {"job_id": job_id, "tenant_id": tenant_id},
            )
            .mappings()
            .one()
        )
    return dict(row)


def _document(tenant_id: UUID, source_key: str) -> dict | None:
    with get_engine().connect() as connection:
        row = (
            connection.execute(
                text(
                    """
                    SELECT id, current_version_id, status
                    FROM documents
                    WHERE tenant_id = :tenant_id AND source_key = :source_key
                    """
                ),
                {"tenant_id": tenant_id, "source_key": source_key},
            )
            .mappings()
            .first()
        )
    return dict(row) if row is not None else None


def _versions(document_id: UUID) -> list[dict]:
    with get_engine().connect() as connection:
        rows = (
            connection.execute(
                text(
                    """
                    SELECT id, version_number, is_current
                    FROM document_versions
                    WHERE document_id = :document_id
                    ORDER BY version_number
                    """
                ),
                {"document_id": document_id},
            )
            .mappings()
            .all()
        )
    return [dict(row) for row in rows]


def _chunks(version_id: UUID) -> list[dict]:
    with get_engine().connect() as connection:
        rows = (
            connection.execute(
                text(
                    """
                    SELECT chunk_index, content, embedding, embedding_model, embedded_at
                    FROM chunks
                    WHERE document_version_id = :version_id
                    ORDER BY chunk_index
                    """
                ),
                {"version_id": version_id},
            )
            .mappings()
            .all()
        )
    return [dict(row) for row in rows]


def test_async_submission_persists_non_current_target_without_embedding(
    tenant_id: UUID, monkeypatch
) -> None:
    def exploding_embedder(texts):
        raise AssertionError("submission must not call the embedding service")

    monkeypatch.setattr(ingestion_module, "embed_texts", exploding_embedder)

    submission = _submit(tenant_id, source_key="runbooks/worker-submit.md")
    job = _job(tenant_id, submission.job_id)
    document = _document(tenant_id, "runbooks/worker-submit.md")

    assert submission.needs_publish is True
    assert submission.status == "running"
    assert job["status"] == "running"
    assert job["attempts"] == 1
    assert job["version_id"] == submission.version_id
    assert document is not None
    assert document["current_version_id"] is None
    versions = _versions(document["id"])
    assert len(versions) == 1
    assert versions[0]["is_current"] is False
    assert all(chunk["embedding"] is None for chunk in _chunks(submission.version_id))
    assert all(chunk["content"] for chunk in _chunks(submission.version_id))


def test_worker_resume_embeds_and_activates_durable_job(tenant_id: UUID) -> None:
    submission = _submit(tenant_id, source_key="runbooks/worker-success.md")

    result = resume_ingestion_job(submission.job_id)

    assert result.outcome == "succeeded"
    assert result.status == "succeeded"
    assert result.is_current is True
    job = _job(tenant_id, submission.job_id)
    assert job["status"] == "succeeded"
    assert job["attempts"] == 1
    document = _document(tenant_id, "runbooks/worker-success.md")
    assert document is not None
    assert document["current_version_id"] == submission.version_id
    chunks = _chunks(submission.version_id)
    assert all(chunk["embedding"] is not None for chunk in chunks)
    assert all(chunk["embedded_at"] is not None for chunk in chunks)


def test_duplicate_delivery_after_success_is_a_pure_read(tenant_id: UUID, monkeypatch) -> None:
    submission = _submit(tenant_id, source_key="runbooks/worker-duplicate.md")
    resume_ingestion_job(submission.job_id)
    job_before = _job(tenant_id, submission.job_id)
    chunks_before = _chunks(submission.version_id)

    monkeypatch.setattr(
        ingestion_module,
        "embed_texts",
        lambda texts: (_ for _ in ()).throw(AssertionError("duplicate must not embed")),
    )

    second = resume_ingestion_job(submission.job_id)

    assert second.outcome == "skipped"
    assert second.is_current is True
    job_after = _job(tenant_id, submission.job_id)
    assert job_after["attempts"] == job_before["attempts"] == 1
    assert job_after["status"] == "succeeded"
    assert _chunks(submission.version_id) == chunks_before


def test_transient_embedding_outage_records_failure_then_retry_succeeds(
    tenant_id: UUID, monkeypatch
) -> None:
    submission = _submit(tenant_id, source_key="runbooks/worker-transient.md")

    def unavailable(texts):
        raise EmbeddingUnavailableError("provider timeout")

    monkeypatch.setattr(ingestion_module, "embed_texts", unavailable)

    with pytest.raises(EmbeddingUnavailableError):
        resume_ingestion_job(submission.job_id)

    failed = _job(tenant_id, submission.job_id)
    assert failed["status"] == "failed"
    assert failed["error_code"] == "embedding_service_unavailable"
    assert failed["error_summary"] == "Embedding service was unavailable"
    assert failed["attempts"] == 1

    monkeypatch.setattr(ingestion_module, "embed_texts", deterministic_embeddings)
    result = resume_ingestion_job(submission.job_id)

    assert result.outcome == "succeeded"
    succeeded = _job(tenant_id, submission.job_id)
    assert succeeded["status"] == "succeeded"
    assert succeeded["attempts"] == 2
    assert succeeded["error_code"] is None


def test_permanent_validation_failure_is_not_retried(tenant_id: UUID, monkeypatch) -> None:
    submission = _submit(tenant_id, source_key="runbooks/worker-invalid.md")

    def invalid(texts):
        return deterministic_embeddings(texts)[:-1]

    monkeypatch.setattr(ingestion_module, "embed_texts", invalid)

    with pytest.raises(EmbeddingValidationError):
        resume_ingestion_job(submission.job_id)

    failed = _job(tenant_id, submission.job_id)
    assert failed["status"] == "failed"
    assert failed["error_code"] == "embedding_invalid_response"
    attempts = failed["attempts"]

    with pytest.raises(IngestionJobTerminalError):
        resume_ingestion_job(submission.job_id)

    assert _job(tenant_id, submission.job_id)["attempts"] == attempts


def test_schema_mismatch_is_a_permanent_operational_failure(tenant_id: UUID, monkeypatch) -> None:
    submission = _submit(tenant_id, source_key="runbooks/worker-schema.md")

    def mismatch():
        raise EmbeddingSchemaError("configured 384 but column is vector(768)")

    monkeypatch.setattr(ingestion_module, "check_embedding_schema", mismatch)

    with pytest.raises(EmbeddingSchemaError):
        resume_ingestion_job(submission.job_id)

    failed = _job(tenant_id, submission.job_id)
    assert failed["status"] == "failed"
    assert failed["error_code"] == "embedding_schema_mismatch"

    with pytest.raises(IngestionJobTerminalError):
        resume_ingestion_job(submission.job_id)


def test_missing_job_is_permanent(tenant_id: UUID) -> None:
    with pytest.raises(IngestionJobNotFoundError):
        resume_ingestion_job(uuid4())


def test_job_without_prepared_target_is_invalid_and_recorded(tenant_id: UUID) -> None:
    job_id = uuid4()
    with get_engine().begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO ingestion_jobs (id, tenant_id, idempotency_key, status)
                VALUES (:id, :tenant_id, :key, 'queued')
                """
            ),
            {"id": job_id, "tenant_id": tenant_id, "key": f"v1:{uuid4().hex}"},
        )

    with pytest.raises(IngestionJobStateError):
        resume_ingestion_job(job_id)

    failed = _job(tenant_id, job_id)
    assert failed["status"] == "failed"
    assert failed["error_code"] == "ingestion_job_invalid"
    assert failed["version_id"] is None

    with pytest.raises(IngestionJobTerminalError):
        resume_ingestion_job(job_id)


def test_unexpected_exception_is_recorded_without_secret_text(tenant_id: UUID, monkeypatch) -> None:
    submission = _submit(tenant_id, source_key="runbooks/worker-unexpected.md")

    def explode(*args, **kwargs):
        raise RuntimeError(
            "token=SUPER_SECRET_123 dsn=postgresql://user:pass@host/db "
            "payload=CONFIDENTIAL_PAYLOAD_42"
        )

    monkeypatch.setattr(ingestion_module, "_ensure_embeddings", explode)

    with pytest.raises(RuntimeError):
        resume_ingestion_job(submission.job_id)

    failed = _job(tenant_id, submission.job_id)
    assert failed["status"] == "failed"
    assert failed["error_code"] == "ingestion_failed"
    haystack = f"{failed['error_code']} {failed['error_summary']}".lower()
    for marker in (
        "super_secret_123",
        "confidential_payload_42",
        "postgresql://",
        "user:pass",
    ):
        assert marker not in haystack


def test_duplicate_async_submission_while_running_does_not_republish(tenant_id: UUID) -> None:
    submission = _submit(tenant_id, source_key="runbooks/worker-running.md")
    second = _submit(tenant_id, source_key="runbooks/worker-running.md")

    assert second.job_id == submission.job_id
    assert second.created is False
    assert second.needs_publish is False
    assert _job(tenant_id, submission.job_id)["attempts"] == 1


def test_resubmission_after_success_reuses_job_and_version(tenant_id: UUID) -> None:
    submission = _submit(tenant_id, source_key="runbooks/worker-resubmit.md")
    resume_ingestion_job(submission.job_id)

    second = _submit(tenant_id, source_key="runbooks/worker-resubmit.md")

    assert second.job_id == submission.job_id
    assert second.version_id == submission.version_id
    assert second.status == "succeeded"
    assert second.needs_publish is False
    job = _job(tenant_id, submission.job_id)
    assert job["attempts"] == 1
    assert job["status"] == "succeeded"


def test_broker_publication_failure_is_observable_and_resumable(tenant_id: UUID) -> None:
    submission = _submit(tenant_id, source_key="runbooks/worker-queue-failure.md")

    mark_ingestion_job_enqueue_failed(submission.job_id)

    failed = _job(tenant_id, submission.job_id)
    assert failed["status"] == "failed"
    assert failed["error_code"] == "queue_unavailable"

    result = resume_ingestion_job(submission.job_id)

    assert result.outcome == "succeeded"
    assert _job(tenant_id, submission.job_id)["status"] == "succeeded"


def test_job_status_lookup_is_tenant_scoped(tenant_id: UUID) -> None:
    submission = _submit(tenant_id, source_key="runbooks/worker-status.md")

    status = get_ingestion_job_status(
        submission.job_id,
        principal=principal_for(tenant_id),
        tenant_id=tenant_id,
    )
    assert status["job_id"] == str(submission.job_id)
    assert status["status"] == "running"

    with pytest.raises(IngestionJobNotFoundError):
        get_ingestion_job_status(
            submission.job_id,
            principal=principal_for(uuid4()),
            tenant_id=uuid4(),
        )
