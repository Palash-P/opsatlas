"""Durable ingestion job lifecycle tests (Phase F, Step 2).

These tests exercise the real PostgreSQL job table and the staged service. They
pin the durable lifecycle (queued -> running -> succeeded, or failed), the
derived idempotency key, attempt/timestamp accounting, collision safety, and
rollback behavior.

Every test uses a throwaway tenant; deleting it cascades to jobs, documents,
versions, and chunks. The seeded local corpus is never touched.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import inspect, text

from app.db.connections import check_postgres, get_engine
from app.main import app
from app.services.ingestion import (
    EmptyDocumentError,
    IdempotencyKeyCollisionError,
    TenantNotFoundError,
    derive_idempotency_key,
    ingest_document,
)
from tests.conftest import deterministic_embeddings, principal_for

client = TestClient(app)
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


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def db_engine():
    """Skip DB-backed tests when local PostgreSQL is unavailable."""

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
                "Ingestion job tests require sql/003_ingestion_jobs_version.sql, "
                "sql/004_activation_generation.sql, and "
                "sql/005_job_reconciliation.sql to be applied; missing: "
                + ", ".join(sorted(missing)),
                pytrace=False,
            )
        return engine

    assert last_error is not None
    pytest.skip(f"PostgreSQL is not available for ingestion job tests: {type(last_error).__name__}")


def _insert_tenant(engine) -> UUID:
    tenant = uuid4()
    with engine.begin() as connection:
        connection.execute(
            text("INSERT INTO tenants (id, name) VALUES (:id, :name)"),
            {"id": tenant, "name": f"phase-f-jobs-{tenant}"},
        )
    return tenant


def _delete_tenant(engine, tenant: UUID) -> None:
    with engine.begin() as connection:
        connection.execute(text("DELETE FROM tenants WHERE id = :id"), {"id": tenant})


@pytest.fixture(autouse=True)
def _deterministic_embedding(deterministic_embedding):
    return deterministic_embedding


@pytest.fixture
def tenant_id(db_engine) -> UUID:
    tenant = _insert_tenant(db_engine)
    try:
        yield tenant
    finally:
        _delete_tenant(db_engine, tenant)


def _long_content(prefix: str, lines: int = 120) -> str:
    return "\n".join(
        f"{prefix} step {index}: verify dependency {index} before escalation."
        for index in range(lines)
    )


def _ingest(
    tenant_id: UUID,
    *,
    source_key: str = "runbooks/phase-f-jobs.md",
    content: str = "Investigate PAY-503 before escalation.",
    title: str = "Phase F job runbook",
    source_type: str = "runbook",
    **overrides,
) -> dict:
    kwargs = {
        "principal": principal_for(tenant_id),
        "tenant_id": tenant_id,
        "source_key": source_key,
        "title": title,
        "source_type": source_type,
        "content": content,
    }
    kwargs.update(overrides)
    return ingest_document(**kwargs)


def _jobs(tenant_id: UUID) -> list[dict]:
    with get_engine().connect() as connection:
        rows = (
            connection.execute(
                text(
                    """
                SELECT id, tenant_id, document_id, version_id, idempotency_key,
                       status, attempts, error_code, error_summary,
                       created_at, started_at, finished_at
                FROM ingestion_jobs
                WHERE tenant_id = :tenant_id
                ORDER BY created_at, id
                """
                ),
                {"tenant_id": tenant_id},
            )
            .mappings()
            .all()
        )
    return [dict(row) for row in rows]


def _document(tenant_id: UUID, source_key: str) -> dict | None:
    with get_engine().connect() as connection:
        row = (
            connection.execute(
                text(
                    """
                SELECT id, current_version_id, title, source_type, status
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
                SELECT id, version_number, content_hash, is_current
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
                SELECT chunk_index, content, content_hash
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


def _fail_on_chunk_hash(monkeypatch, *, fail_at_call: int, message: str) -> None:
    """Fail while hashing a chunk, after earlier chunks may have been inserted."""

    from app.services import ingestion as ingestion_module

    real_sha256 = ingestion_module.sha256_text
    calls = 0

    def flaky_sha256(value: str) -> str:
        nonlocal calls
        calls += 1
        if calls >= fail_at_call:
            raise RuntimeError(message)
        return real_sha256(value)

    monkeypatch.setattr(ingestion_module, "sha256_text", flaky_sha256)


def _run_concurrently(*submissions):
    barrier = threading.Barrier(len(submissions))

    def run(submission):
        barrier.wait(timeout=10)
        return submission()

    with ThreadPoolExecutor(max_workers=len(submissions)) as pool:
        futures = [pool.submit(run, submission) for submission in submissions]
        return [future.result(timeout=30) for future in futures]


# ---------------------------------------------------------------------------
# 1. Successful job creation and completion
# ---------------------------------------------------------------------------


def test_successful_ingestion_persists_succeeded_job(tenant_id: UUID) -> None:
    source_key = "runbooks/jobs-success.md"
    result = _ingest(tenant_id, source_key=source_key, content=_long_content("Success"))

    jobs = _jobs(tenant_id)
    assert len(jobs) == 1
    job = jobs[0]
    assert job["tenant_id"] == tenant_id
    assert job["idempotency_key"] == derive_idempotency_key(
        tenant_id, source_key, result["content_hash"]
    )
    assert job["status"] == "succeeded"
    assert job["attempts"] == 1
    assert job["document_id"] == result["document_id"]
    assert job["version_id"] == result["version_id"]
    assert job["error_code"] is None
    assert job["error_summary"] is None
    assert job["created_at"] <= job["started_at"] <= job["finished_at"]


# ---------------------------------------------------------------------------
# 2. Duplicate submission reuses the logical job
# ---------------------------------------------------------------------------


def test_duplicate_submission_reuses_existing_logical_job(tenant_id: UUID) -> None:
    source_key = "runbooks/jobs-duplicate.md"
    content = _long_content("Duplicate")
    first = _ingest(tenant_id, source_key=source_key, content=content)
    first_job = _jobs(tenant_id)[0]

    second = _ingest(tenant_id, source_key=source_key, content=content, title="Renamed")
    jobs = _jobs(tenant_id)

    assert len(jobs) == 1
    job = jobs[0]
    assert job["id"] == first_job["id"]
    assert job["status"] == "succeeded"
    # Phase F, Step 3: a duplicate of the still-current, fully embedded version
    # is a pure read. It must not manufacture a redundant execution attempt.
    assert job["attempts"] == 1
    assert job["version_id"] == first["version_id"]
    assert second["deduplicated"] is True
    assert second["version_id"] == first["version_id"]
    assert len(_versions(first["document_id"])) == 1
    assert len(_chunks(first["version_id"])) == first["chunk_count"]


# ---------------------------------------------------------------------------
# 3. Different content creates a distinct job
# ---------------------------------------------------------------------------


def test_different_content_creates_distinct_jobs(tenant_id: UUID) -> None:
    source_key = "runbooks/jobs-distinct.md"
    first = _ingest(tenant_id, source_key=source_key, content=_long_content("Variant A"))
    second = _ingest(tenant_id, source_key=source_key, content=_long_content("Variant B"))

    jobs = _jobs(tenant_id)
    assert len(jobs) == 2
    assert {job["status"] for job in jobs} == {"succeeded"}
    assert {job["attempts"] for job in jobs} == {1}
    assert len({job["idempotency_key"] for job in jobs}) == 2
    assert {job["version_id"] for job in jobs} == {
        first["version_id"],
        second["version_id"],
    }
    assert {job["document_id"] for job in jobs} == {first["document_id"]}
    assert first["version_number"] == 1
    assert second["version_number"] == 2


# ---------------------------------------------------------------------------
# 4. Job attempts and timestamps
# ---------------------------------------------------------------------------


def test_job_attempts_and_timestamps_track_each_execution(tenant_id: UUID) -> None:
    source_key = "runbooks/jobs-attempts.md"
    content = _long_content("Attempts")
    _ingest(tenant_id, source_key=source_key, content=content)
    first_job = _jobs(tenant_id)[0]

    assert first_job["attempts"] == 1
    assert first_job["created_at"] <= first_job["started_at"] <= first_job["finished_at"]

    _ingest(tenant_id, source_key=source_key, content=content)
    second_job = _jobs(tenant_id)[0]

    assert second_job["id"] == first_job["id"]
    assert second_job["attempts"] == 1
    # started_at records the first start; a redundant success leaves the
    # completion timestamp of the truthful earlier ingestion intact.
    assert second_job["started_at"] == first_job["started_at"]
    assert second_job["finished_at"] == first_job["finished_at"]
    assert second_job["created_at"] == first_job["created_at"]


# ---------------------------------------------------------------------------
# 5. Failed ingestion does not report success
# ---------------------------------------------------------------------------


def test_failed_ingestion_records_failed_job_without_document(tenant_id: UUID, monkeypatch) -> None:
    source_key = "runbooks/jobs-failure.md"
    _fail_on_chunk_hash(
        monkeypatch,
        fail_at_call=2,
        message="simulated chunk persistence failure",
    )

    with pytest.raises(RuntimeError, match="simulated chunk persistence failure"):
        _ingest(tenant_id, source_key=source_key, content=_long_content("Never persisted"))

    jobs = _jobs(tenant_id)
    assert len(jobs) == 1
    job = jobs[0]
    assert job["status"] == "failed"
    assert job["attempts"] == 1
    assert job["error_code"] == "ingestion_failed"
    assert job["error_summary"] == "Ingestion failed during document persistence"
    assert job["document_id"] is None
    assert job["version_id"] is None
    assert job["created_at"] <= job["started_at"] <= job["finished_at"]
    assert _document(tenant_id, source_key) is None


def test_failed_job_can_be_retried_and_transitions_to_succeeded(
    tenant_id: UUID, monkeypatch
) -> None:
    source_key = "runbooks/jobs-retry.md"
    content = _long_content("Retryable")
    _fail_on_chunk_hash(
        monkeypatch,
        fail_at_call=2,
        message="simulated retryable failure",
    )

    with pytest.raises(RuntimeError, match="simulated retryable failure"):
        _ingest(tenant_id, source_key=source_key, content=content)
    failed_job = _jobs(tenant_id)[0]
    assert failed_job["status"] == "failed"
    assert failed_job["attempts"] == 1
    assert failed_job["error_code"] == "ingestion_failed"

    # ``undo`` also removes the module's deterministic embedding patch, so
    # restore it explicitly and keep this retry independent of the local model
    # server.
    from app.services import ingestion as ingestion_module

    monkeypatch.undo()
    monkeypatch.setattr(ingestion_module, "embed_texts", deterministic_embeddings)
    result = _ingest(tenant_id, source_key=source_key, content=content)

    job = _jobs(tenant_id)[0]
    assert job["id"] == failed_job["id"]
    assert job["status"] == "succeeded"
    assert job["attempts"] == 2
    assert job["error_code"] is None
    assert job["error_summary"] is None
    assert job["document_id"] == result["document_id"]
    assert job["version_id"] == result["version_id"]
    assert _document(tenant_id, source_key) is not None


# ---------------------------------------------------------------------------
# 6. Transaction rollback preserves the last known good version
# ---------------------------------------------------------------------------


def test_failed_replacement_rolls_back_and_links_job_to_document(
    tenant_id: UUID, monkeypatch
) -> None:
    source_key = "runbooks/jobs-rollback.md"
    first = _ingest(
        tenant_id,
        source_key=source_key,
        title="Last known good",
        content=_long_content("Known good"),
    )
    document_before = _document(tenant_id, source_key)
    chunks_before = _chunks(first["version_id"])
    assert document_before is not None

    _fail_on_chunk_hash(
        monkeypatch,
        fail_at_call=2,
        message="simulated replacement failure",
    )

    with pytest.raises(RuntimeError, match="simulated replacement failure"):
        _ingest(
            tenant_id,
            source_key=source_key,
            title="Must roll back",
            content=_long_content("Broken replacement"),
        )

    document = _document(tenant_id, source_key)
    assert document is not None
    assert document["current_version_id"] == first["version_id"]
    assert document["title"] == "Last known good"
    versions = _versions(first["document_id"])
    assert len(versions) == 1
    assert versions[0]["is_current"] is True
    assert _chunks(first["version_id"]) == chunks_before

    jobs = _jobs(tenant_id)
    assert len(jobs) == 2
    succeeded = next(job for job in jobs if job["status"] == "succeeded")
    failed = next(job for job in jobs if job["status"] == "failed")
    assert succeeded["version_id"] == first["version_id"]
    assert failed["document_id"] == first["document_id"]
    assert failed["version_id"] is None
    assert failed["attempts"] == 1


# ---------------------------------------------------------------------------
# 7. Concurrent identical submissions
# ---------------------------------------------------------------------------


def test_concurrent_identical_requests_share_one_job_and_version(tenant_id: UUID) -> None:
    source_key = "runbooks/jobs-concurrent-identical.md"
    content = _long_content("Concurrent identical", lines=80)

    results = _run_concurrently(
        lambda: _ingest(tenant_id, source_key=source_key, content=content),
        lambda: _ingest(tenant_id, source_key=source_key, content=content),
    )

    assert len({result["version_id"] for result in results}) == 1
    assert sorted(result["deduplicated"] for result in results) == [False, True]

    jobs = _jobs(tenant_id)
    assert len(jobs) == 1
    job = jobs[0]
    assert job["status"] == "succeeded"
    # One request creates the version; the overlapping request either observes
    # the running job or the finished one. A finished duplicate adds no attempt,
    # while an overlapping execution legitimately records its own attempt.
    assert 1 <= job["attempts"] <= 2
    assert job["version_id"] == results[0]["version_id"]
    document = _document(tenant_id, source_key)
    assert document is not None
    assert len(_versions(document["id"])) == 1


# ---------------------------------------------------------------------------
# 8. Concurrent different-content submissions
# ---------------------------------------------------------------------------


def test_concurrent_different_requests_create_distinct_jobs(tenant_id: UUID) -> None:
    source_key = "runbooks/jobs-concurrent-different.md"
    first_content = _long_content("Concurrent A", lines=80)
    second_content = _long_content("Concurrent B", lines=80)

    results = _run_concurrently(
        lambda: _ingest(tenant_id, source_key=source_key, content=first_content),
        lambda: _ingest(tenant_id, source_key=source_key, content=second_content),
    )

    assert len({result["version_id"] for result in results}) == 2
    assert {result["deduplicated"] for result in results} == {False}

    jobs = _jobs(tenant_id)
    assert len(jobs) == 2
    assert {job["status"] for job in jobs} == {"succeeded"}
    assert {job["attempts"] for job in jobs} == {1}
    assert {job["version_id"] for job in jobs} == {
        results[0]["version_id"],
        results[1]["version_id"],
    }

    document = _document(tenant_id, source_key)
    assert document is not None
    versions = _versions(document["id"])
    assert len(versions) == 2
    assert sum(1 for version in versions if version["is_current"]) == 1


# ---------------------------------------------------------------------------
# 9. Cross-tenant idempotency isolation
# ---------------------------------------------------------------------------


def test_cross_tenant_idempotency_is_isolated(tenant_id: UUID, db_engine) -> None:
    other_tenant = _insert_tenant(db_engine)
    source_key = "runbooks/jobs-cross-tenant.md"
    content = _long_content("Shared content")
    try:
        first = _ingest(tenant_id, source_key=source_key, content=content)
        second = _ingest(other_tenant, source_key=source_key, content=content)

        assert first["document_id"] != second["document_id"]
        assert first["version_id"] != second["version_id"]

        first_jobs = _jobs(tenant_id)
        second_jobs = _jobs(other_tenant)
        assert len(first_jobs) == 1
        assert len(second_jobs) == 1
        assert first_jobs[0]["tenant_id"] == tenant_id
        assert second_jobs[0]["tenant_id"] == other_tenant
        assert first_jobs[0]["id"] != second_jobs[0]["id"]
        assert first_jobs[0]["idempotency_key"] != second_jobs[0]["idempotency_key"]
        assert first_jobs[0]["version_id"] == first["version_id"]
        assert second_jobs[0]["version_id"] == second["version_id"]
    finally:
        _delete_tenant(db_engine, other_tenant)


# ---------------------------------------------------------------------------
# 10. Idempotency-key collision handling
# ---------------------------------------------------------------------------


def test_idempotency_key_collision_never_reuses_unrelated_content(
    tenant_id: UUID, monkeypatch
) -> None:
    from app.services import ingestion as ingestion_module

    monkeypatch.setattr(
        ingestion_module,
        "derive_idempotency_key",
        lambda *args, **kwargs: "v1:forced-collision",
    )

    source_key = "runbooks/jobs-collision.md"
    first = _ingest(tenant_id, source_key=source_key, content=_long_content("Collision A"))
    job_before = _jobs(tenant_id)[0]

    with pytest.raises(IdempotencyKeyCollisionError):
        _ingest(tenant_id, source_key=source_key, content=_long_content("Collision B"))

    jobs = _jobs(tenant_id)
    assert len(jobs) == 1
    job = jobs[0]
    assert job["id"] == job_before["id"]
    assert job["status"] == "succeeded"
    assert job["attempts"] == 1
    assert job["version_id"] == first["version_id"]
    assert job["error_code"] is None
    document = _document(tenant_id, source_key)
    assert document is not None
    assert document["current_version_id"] == first["version_id"]
    assert len(_versions(first["document_id"])) == 1

    # A colliding key must not associate the job with a different source either.
    other_source = "runbooks/jobs-collision-other.md"
    with pytest.raises(IdempotencyKeyCollisionError):
        _ingest(tenant_id, source_key=other_source, content=_long_content("Collision A"))
    assert _document(tenant_id, other_source) is None
    assert len(_jobs(tenant_id)) == 1


def test_idempotency_key_is_deterministic_and_tenant_scoped() -> None:
    tenant = uuid4()
    other_tenant = uuid4()
    content_hash = "a" * 64
    key = derive_idempotency_key(tenant, "runbooks/key.md", content_hash)

    assert key == derive_idempotency_key(tenant, "runbooks/key.md", content_hash)
    version, embedded_hash, digest = key.split(":")
    assert version == "v1"
    assert embedded_hash == content_hash
    assert len(digest) == 64
    assert derive_idempotency_key(other_tenant, "runbooks/key.md", content_hash) != key
    assert derive_idempotency_key(tenant, "runbooks/other.md", content_hash) != key
    assert derive_idempotency_key(tenant, "runbooks/key.md", "b" * 64) != key
    assert "runbooks/key.md" not in key


def test_collision_rejected_for_queued_job_without_links(tenant_id: UUID, monkeypatch) -> None:
    from app.services import ingestion as ingestion_module

    monkeypatch.setattr(
        ingestion_module,
        "derive_idempotency_key",
        lambda *args, **kwargs: "v1:forced-collision",
    )
    with get_engine().begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO ingestion_jobs (tenant_id, idempotency_key, status)
                VALUES (:tenant_id, 'v1:forced-collision', 'queued')
                """
            ),
            {"tenant_id": tenant_id},
        )

    source_key = "runbooks/jobs-queued-collision.md"
    with pytest.raises(IdempotencyKeyCollisionError):
        _ingest(tenant_id, source_key=source_key, content=_long_content("Queued collision"))

    job = _jobs(tenant_id)[0]
    assert job["status"] == "queued"
    assert job["attempts"] == 0
    assert job["document_id"] is None
    assert job["version_id"] is None
    assert _document(tenant_id, source_key) is None


# ---------------------------------------------------------------------------
# 11. Existing API contract remains unchanged
# ---------------------------------------------------------------------------


def test_ingest_endpoint_contract_remains_unchanged(tenant_id: UUID, set_identity) -> None:
    set_identity(tenant_id=tenant_id)
    payload = {
        "source_key": "runbooks/jobs-endpoint.md",
        "title": "Job-backed endpoint",
        "source_type": "runbook",
        "content": _long_content("Job-backed endpoint"),
        "source_uri": "https://example.test/jobs-endpoint",
        "authority_level": 65,
    }

    first = client.post(INGEST_URL, json=payload)
    assert first.status_code == 201
    first_body = first.json()
    assert set(first_body) == RESPONSE_KEYS
    assert first_body["deduplicated"] is False
    assert first_body["embedding_status"] == "complete"

    job_after_first = _jobs(tenant_id)[0]
    assert job_after_first["status"] == "succeeded"
    assert job_after_first["document_id"] == UUID(first_body["document_id"])
    assert job_after_first["version_id"] == UUID(first_body["version_id"])

    second = client.post(INGEST_URL, json=payload)
    assert second.status_code == 200
    second_body = second.json()
    assert set(second_body) == RESPONSE_KEYS
    assert second_body["document_id"] == first_body["document_id"]
    assert second_body["version_id"] == first_body["version_id"]
    assert second_body["deduplicated"] is True

    job_after_second = _jobs(tenant_id)[0]
    assert job_after_second["id"] == job_after_first["id"]
    assert job_after_second["attempts"] == 1
    assert job_after_second["status"] == "succeeded"


# ---------------------------------------------------------------------------
# 12. No secrets or document content in job errors
# ---------------------------------------------------------------------------


def test_failure_summary_never_contains_content_or_secrets(tenant_id: UUID, monkeypatch) -> None:
    from app.services import ingestion as ingestion_module

    real_sha256 = ingestion_module.sha256_text
    calls = 0

    def explosive_sha256(value: str) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_sha256(value)
        raise RuntimeError(
            "token=SUPER_SECRET_123 dsn=postgresql://user:pass@host/db "
            "payload=CONFIDENTIAL_PAYLOAD_42"
        )

    monkeypatch.setattr(ingestion_module, "sha256_text", explosive_sha256)

    with pytest.raises(RuntimeError):
        _ingest(
            tenant_id,
            source_key="runbooks/jobs-secrets.md",
            content="CONFIDENTIAL_PAYLOAD_42 " + _long_content("Secret handling"),
            source_uri="https://example.test/doc?token=QUERY_SECRET_99",
        )

    job = _jobs(tenant_id)[0]
    assert job["status"] == "failed"
    assert job["error_code"] == "ingestion_failed"
    assert job["error_summary"] == "Ingestion failed during document persistence"

    haystack = f"{job['error_code']} {job['error_summary']}".lower()
    for marker in (
        "confidential_payload_42",
        "super_secret_123",
        "query_secret_99",
        "postgresql://",
        "user:pass",
        "password",
    ):
        assert marker not in haystack


# ---------------------------------------------------------------------------
# Rejected requests create no durable job
# ---------------------------------------------------------------------------


def test_rejected_requests_create_no_job(tenant_id: UUID) -> None:
    with pytest.raises(EmptyDocumentError):
        _ingest(tenant_id, content="   \n\t ")
    assert _jobs(tenant_id) == []

    missing_tenant = uuid4()
    with pytest.raises(TenantNotFoundError):
        _ingest(missing_tenant)
    assert _jobs(missing_tenant) == []
