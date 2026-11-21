"""Synchronous/async execution-ownership and fencing tests (F-02).

These tests use the real PostgreSQL job row because the contract depends on
``SELECT ... FOR UPDATE``, leases, and fencing predicates. Celery/Redis is never
required: the worker entry point is called directly. Every test uses a throwaway
tenant whose deletion cascades to its documents, versions, chunks, and jobs.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from sqlalchemy import inspect, text

from app.db.connections import check_postgres, get_engine
from app.services import ingestion as ingestion_module
from app.services.embedding_client import EmbeddingUnavailableError
from app.services.ingestion import (
    IngestionExecutionLeaseLostError,
    IngestionJobBusyError,
    _activate_version,
    _ensure_embeddings,
    _prepare_target,
    _record_job_failure,
    _resolve_or_create_job,
    derive_idempotency_key,
    ingest_document,
    prepare_document,
    resume_ingestion_job,
    submit_ingestion_job,
)
from app.services.reconciliation import run_reconciliation
from tests.conftest import deterministic_embeddings, principal_for

REQUIRED_JOB_COLUMNS = {
    "version_id",
    "activation_generation",
    "next_dispatch_at",
    "dispatch_claimed_until",
    "execution_token",
    "execution_lease_expires_at",
}


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
        missing = REQUIRED_JOB_COLUMNS - job_columns
        if "last_activation_generation" not in document_columns:
            missing.add("documents.last_activation_generation")
        if missing:
            pytest.fail(
                "Sync fencing tests require sql/003, sql/004, and sql/005; missing: "
                + ", ".join(sorted(missing)),
                pytrace=False,
            )
        return engine

    assert last_error is not None
    pytest.skip(f"PostgreSQL is not available for sync fencing tests: {type(last_error).__name__}")


@pytest.fixture(autouse=True)
def _deterministic_embedding(deterministic_embedding):
    return deterministic_embedding


@pytest.fixture
def tenant_id(db_engine) -> UUID:
    tenant = uuid4()
    with db_engine.begin() as connection:
        connection.execute(
            text("INSERT INTO tenants (id, name) VALUES (:id, :name)"),
            {"id": tenant, "name": f"phase-f-sync-fencing-{tenant}"},
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


def _ingest(
    tenant_id: UUID,
    *,
    source_key: str,
    content: str,
    title: str = "Sync fencing runbook",
    **overrides,
) -> dict:
    kwargs = {
        "principal": principal_for(tenant_id),
        "tenant_id": tenant_id,
        "source_key": source_key,
        "title": title,
        "source_type": "runbook",
        "content": content,
    }
    kwargs.update(overrides)
    return ingest_document(**kwargs)


def _submit(
    tenant_id: UUID,
    *,
    source_key: str,
    content: str,
):
    return submit_ingestion_job(
        principal=principal_for(tenant_id),
        tenant_id=tenant_id,
        source_key=source_key,
        title="Sync fencing runbook",
        source_type="runbook",
        content=content,
    )


def _job(job_id: UUID) -> dict:
    with get_engine().connect() as connection:
        row = (
            connection.execute(
                text(
                    """
                    SELECT id, status, attempts, document_id, version_id,
                           activation_generation, error_code,
                           execution_token, execution_lease_expires_at,
                           (
                               execution_lease_expires_at IS NOT NULL
                               AND execution_lease_expires_at > now()
                           ) AS execution_lease_active
                    FROM ingestion_jobs
                    WHERE id = :job_id
                    """
                ),
                {"job_id": job_id},
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
                    SELECT id, current_version_id
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


def _expire_lease(job_id: UUID) -> None:
    with get_engine().begin() as connection:
        connection.execute(
            text(
                """
                UPDATE ingestion_jobs
                SET execution_lease_expires_at = now() - interval '1 second'
                WHERE id = :job_id
                """
            ),
            {"job_id": job_id},
        )


def _prepare_sync_target(
    tenant_id: UUID,
    *,
    source_key: str,
    content: str,
):
    """Create the exact TX2 state of a synchronous request before embedding."""

    principal = principal_for(tenant_id)
    prepared = prepare_document(content)
    idempotency_key = derive_idempotency_key(tenant_id, source_key, prepared.content_hash)
    job, created = _resolve_or_create_job(
        principal=principal,
        tenant_id=tenant_id,
        idempotency_key=idempotency_key,
        source_key=source_key,
        content_hash=prepared.content_hash,
    )
    target = _prepare_target(
        job.id,
        principal=principal,
        tenant_id=tenant_id,
        source_key=source_key,
        title="Sync fencing runbook",
        source_type="runbook",
        source_uri=None,
        authority_level=50,
        prepared=prepared,
        started_at=datetime.now(UTC),
        created=created,
        claim_execution=True,
    )
    return job.id, target


def _job_for_source(tenant_id: UUID, source_key: str) -> dict:
    with get_engine().connect() as connection:
        row = (
            connection.execute(
                text(
                    """
                    SELECT j.id
                    FROM ingestion_jobs AS j
                    JOIN documents AS d ON d.id = j.document_id
                    WHERE d.tenant_id = :tenant_id AND d.source_key = :source_key
                    ORDER BY j.created_at DESC
                    LIMIT 1
                    """
                ),
                {"tenant_id": tenant_id, "source_key": source_key},
            )
            .mappings()
            .one()
        )
    return _job(row["id"])


# ---------------------------------------------------------------------------
# Ownership acquisition
# ---------------------------------------------------------------------------


def test_synchronous_ingestion_holds_a_live_execution_lease(
    tenant_id: UUID,
    monkeypatch,
) -> None:
    source_key = "runbooks/sync-fencing-ownership.md"
    content = _content("Ownership", lines=200)
    entered = threading.Event()
    release = threading.Event()

    def gated_embedder(texts: list[str]) -> list[list[float]]:
        entered.set()
        if not release.wait(timeout=20):
            raise AssertionError("timed out waiting to release synchronous ingestion")
        return deterministic_embeddings(texts)

    monkeypatch.setattr(ingestion_module, "embed_texts", gated_embedder)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_ingest, tenant_id, source_key=source_key, content=content)
        assert entered.wait(timeout=20)

        job = _job_for_source(tenant_id, source_key)
        assert job["status"] == "running"
        assert job["execution_token"] is not None
        assert job["execution_lease_expires_at"] is not None
        assert job["execution_lease_active"] is True

        release.set()
        result = future.result(timeout=30)

    assert result["is_current"] is True
    finished = _job(job["id"])
    assert finished["status"] == "succeeded"
    assert finished["execution_token"] is None
    assert finished["execution_lease_expires_at"] is None


def test_synchronous_target_claims_token_and_lease_same_transaction(tenant_id: UUID) -> None:
    job_id, target = _prepare_sync_target(
        tenant_id,
        source_key="runbooks/sync-fencing-tx2.md",
        content=_content("TX2 ownership"),
    )

    job = _job(job_id)
    assert target.execution_token is not None
    assert job["execution_token"] == target.execution_token
    assert job["execution_lease_active"] is True
    assert job["status"] == "running"
    assert job["attempts"] == 1


# ---------------------------------------------------------------------------
# Reconciliation and crash recovery
# ---------------------------------------------------------------------------


def test_reconciliation_does_not_dispatch_a_live_synchronous_job(tenant_id: UUID) -> None:
    job_id, target = _prepare_sync_target(
        tenant_id,
        source_key="runbooks/sync-fencing-live.md",
        content=_content("Live sync"),
    )

    published: list[UUID] = []
    report = run_reconciliation(batch_size=50, publisher=published.append)

    assert job_id not in published
    assert report.locked is False or report.locked is True
    row = _job(job_id)
    assert row["status"] == "running"
    assert row["execution_token"] == target.execution_token
    assert row["execution_lease_active"] is True


def test_crashed_synchronous_execution_is_recoverable_by_reconciliation(tenant_id: UUID) -> None:
    job_id, target = _prepare_sync_target(
        tenant_id,
        source_key="runbooks/sync-fencing-crash.md",
        content=_content("Crashed sync"),
    )
    _expire_lease(job_id)

    published: list[UUID] = []
    report = run_reconciliation(batch_size=50, publisher=published.append)
    assert report.dispatched >= 1
    assert job_id in published

    result = resume_ingestion_job(job_id)

    assert result.status == "succeeded"
    assert result.version_id == target.version_id
    row = _job(job_id)
    assert row["status"] == "succeeded"
    assert row["execution_token"] is None
    document = _document(tenant_id, "runbooks/sync-fencing-crash.md")
    assert document is not None
    assert document["current_version_id"] == target.version_id
    assert _versions(document["id"])[-1]["is_current"] is True


# ---------------------------------------------------------------------------
# Fencing and stale writers
# ---------------------------------------------------------------------------


def test_synchronous_execution_cannot_activate_after_lease_expiry(tenant_id: UUID) -> None:
    _job_id, target = _prepare_sync_target(
        tenant_id,
        source_key="runbooks/sync-fencing-expired.md",
        content=_content("Expired sync"),
    )
    _expire_lease(_job_id)

    with pytest.raises(IngestionExecutionLeaseLostError):
        _activate_version(
            job_id=_job_id,
            tenant_id=tenant_id,
            document_id=target.document_id,
            version_id=target.version_id,
            version_number=target.version_number,
            content_hash=target.content_hash,
            expected_chunk_count=target.chunk_count,
            execution_token=target.execution_token,
        )

    document = _document(tenant_id, "runbooks/sync-fencing-expired.md")
    assert document is not None
    assert document["current_version_id"] is None


def test_stale_synchronous_failure_cannot_overwrite_a_worker_execution(tenant_id: UUID) -> None:
    source_key = "runbooks/sync-fencing-stale-failure.md"
    job_id, target = _prepare_sync_target(
        tenant_id,
        source_key=source_key,
        content=_content("Stale failure"),
    )
    stale_token = target.execution_token
    assert stale_token is not None
    _expire_lease(job_id)

    result = resume_ingestion_job(job_id)
    assert result.status == "succeeded"
    assert _job(job_id)["status"] == "succeeded"

    recorded = _record_job_failure(
        job_id,
        tenant_id=tenant_id,
        source_key=source_key,
        started_at=datetime.now(UTC),
        error=RuntimeError("late synchronous failure"),
        preserve_succeeded=False,
        execution_token=stale_token,
    )
    assert recorded is False

    # An ownerless writer must also not clobber the newer execution.
    recorded_without_token = _record_job_failure(
        job_id,
        tenant_id=tenant_id,
        source_key=source_key,
        started_at=datetime.now(UTC),
        error=RuntimeError("ownerless late failure"),
        preserve_succeeded=False,
    )
    assert recorded_without_token is False

    after = _job(job_id)
    assert after["status"] == "succeeded"
    assert after["error_code"] is None


def test_worker_cannot_claim_or_overwrite_a_live_synchronous_execution(tenant_id: UUID) -> None:
    source_key = "runbooks/sync-fencing-worker-busy.md"
    job_id, target = _prepare_sync_target(
        tenant_id,
        source_key=source_key,
        content=_content("Worker versus sync"),
    )
    live_token = _job(job_id)["execution_token"]

    with pytest.raises(IngestionJobBusyError):
        resume_ingestion_job(job_id)

    row = _job(job_id)
    assert row["status"] == "running"
    assert row["execution_token"] == live_token

    # The live synchronous owner can still finish and activate.
    _ensure_embeddings(
        version_id=target.version_id,
        tenant_id=tenant_id,
        expected_chunk_count=target.chunk_count,
        heartbeat=lambda: ingestion_module._require_execution_lease(
            job_id,
            target.execution_token,
        ),
    )
    activation = _activate_version(
        job_id=job_id,
        tenant_id=tenant_id,
        document_id=target.document_id,
        version_id=target.version_id,
        version_number=target.version_number,
        content_hash=target.content_hash,
        expected_chunk_count=target.chunk_count,
        execution_token=target.execution_token,
    )
    assert activation.is_current is True
    assert _document(tenant_id, source_key)["current_version_id"] == target.version_id


# ---------------------------------------------------------------------------
# Idempotency and cross-path activation
# ---------------------------------------------------------------------------


def test_duplicate_synchronous_requests_remain_idempotent(tenant_id: UUID) -> None:
    source_key = "runbooks/sync-fencing-duplicate.md"
    content = _content("Duplicate sync")

    first = _ingest(tenant_id, source_key=source_key, content=content)
    second = _ingest(tenant_id, source_key=source_key, content=content)

    assert second["version_id"] == first["version_id"]
    assert second["deduplicated"] is True
    assert second["is_current"] is True
    job = _job_for_source(tenant_id, source_key)
    assert job["status"] == "succeeded"
    assert job["attempts"] == 1


def test_concurrent_identical_synchronous_requests_do_not_steal_ownership(
    tenant_id: UUID,
    monkeypatch,
) -> None:
    source_key = "runbooks/sync-fencing-concurrent.md"
    content = _content("Concurrent duplicate", lines=200)
    entered = threading.Event()
    release = threading.Event()
    batches: list[list[str]] = []

    def gated_embedder(texts: list[str]) -> list[list[float]]:
        batches.append(list(texts))
        entered.set()
        if not release.wait(timeout=20):
            raise AssertionError("timed out waiting to release concurrent ingestion")
        return deterministic_embeddings(texts)

    monkeypatch.setattr(ingestion_module, "embed_texts", gated_embedder)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(_ingest, tenant_id, source_key=source_key, content=content)
        assert entered.wait(timeout=20)
        second = pool.submit(_ingest, tenant_id, source_key=source_key, content=content)
        time.sleep(0.2)
        release.set()
        results = [first.result(timeout=30), second.result(timeout=30)]

    assert len({result["version_id"] for result in results}) == 1
    assert sorted(result["deduplicated"] for result in results) == [False, True]
    # The duplicate waited for the owner instead of calling the provider.
    assert len(batches) >= 1
    document = _document(tenant_id, source_key)
    assert document is not None
    assert sum(1 for version in _versions(document["id"]) if version["is_current"]) == 1


def test_sync_and_async_overlap_preserves_activation_order(
    tenant_id: UUID,
    monkeypatch,
) -> None:
    source_key = "runbooks/sync-fencing-overlap.md"
    older_content = _content("OVERLAP-MARKER-OLDER", lines=200)
    newer_content = _content("OVERLAP-MARKER-NEWER", lines=200)
    older_started = threading.Event()
    release_older = threading.Event()

    def gated_embedder(texts: list[str]) -> list[list[float]]:
        if any("OVERLAP-MARKER-OLDER" in text for text in texts):
            older_started.set()
            if not release_older.wait(timeout=20):
                raise AssertionError("timed out waiting to release the older ingestion")
        return deterministic_embeddings(texts)

    monkeypatch.setattr(ingestion_module, "embed_texts", gated_embedder)

    with ThreadPoolExecutor(max_workers=1) as pool:
        older_future = pool.submit(
            _ingest,
            tenant_id,
            source_key=source_key,
            content=older_content,
        )
        assert older_started.wait(timeout=20)

        newer_submission = _submit(tenant_id, source_key=source_key, content=newer_content)
        newer_result = resume_ingestion_job(newer_submission.job_id)
        assert newer_result.status == "succeeded"

        release_older.set()
        older = older_future.result(timeout=30)

    assert older["version_id"] != newer_result.version_id
    assert older["is_current"] is False
    assert older["embedding_status"] == "complete"

    document = _document(tenant_id, source_key)
    assert document is not None
    assert document["current_version_id"] == newer_result.version_id
    versions = _versions(document["id"])
    assert sum(1 for version in versions if version["is_current"]) == 1
    assert next(version for version in versions if version["is_current"])["id"] == (
        newer_result.version_id
    )


def test_last_known_good_version_survives_a_failed_synchronous_replacement(
    tenant_id: UUID,
    monkeypatch,
) -> None:
    source_key = "runbooks/sync-fencing-lkg.md"
    good = _ingest(tenant_id, source_key=source_key, content=_content("Stable version"))

    def unavailable(texts: list[str]) -> list[list[float]]:
        raise EmbeddingUnavailableError("provider timeout")

    monkeypatch.setattr(ingestion_module, "embed_texts", unavailable)

    with pytest.raises(EmbeddingUnavailableError):
        _ingest(tenant_id, source_key=source_key, content=_content("Broken version"))

    document = _document(tenant_id, source_key)
    assert document is not None
    assert document["current_version_id"] == good["version_id"]
    versions = _versions(document["id"])
    assert len(versions) == 2
    assert versions[0]["is_current"] is True
    assert versions[1]["is_current"] is False

    failed_job = _job_for_source(tenant_id, source_key)
    assert failed_job["status"] == "failed"
    assert failed_job["error_code"] == "embedding_service_unavailable"
    assert failed_job["execution_token"] is None
