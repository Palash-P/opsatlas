"""Durable reconciliation and orphan-recovery tests (Phase F, Step 6).

Most tests use the real PostgreSQL job table because correctness depends on
``SELECT ... FOR UPDATE SKIP LOCKED``, dispatch claims, execution leases, and
fencing-token predicates. Celery/Redis publication is replaced by an injected
publisher, so no broker is required.

Every test uses a throwaway tenant; deleting it cascades to jobs, documents,
versions, and chunks. Tests are skipped only when PostgreSQL is unreachable.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from uuid import UUID, uuid4

import pytest
from sqlalchemy import inspect, text

from app.core.config import Settings
from app.core.observability import (
    RECONCILIATION_LOGGER_NAME,
    tenant_reference,
)
from app.db.connections import check_postgres, get_engine
from app.services import ingestion as ingestion_module
from app.services.ingestion import (
    IngestionExecutionLeaseLostError,
    IngestionJobBusyError,
    _activate_version,
    _claim_resumable_job,
    _record_job_failure,
    ingest_document,
    resume_ingestion_job,
    submit_ingestion_job,
)
from app.services.reconciliation import (
    MAX_RECONCILIATION_BATCH_SIZE,
    RECONCILIATION_ADVISORY_LOCK_KEY,
    ReconciliationReport,
    claim_ingestion_jobs_for_dispatch,
    dispatch_backoff_seconds,
    run_reconciliation,
)
from tests.conftest import principal_for

RECONCILED_COLUMNS = {
    "next_dispatch_at",
    "dispatch_claimed_until",
    "last_dispatch_at",
    "dispatch_attempts",
    "dispatch_error_code",
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
        missing = RECONCILED_COLUMNS - job_columns
        if missing:
            pytest.fail(
                "Reconciliation tests require sql/005_job_reconciliation.sql; "
                "missing: " + ", ".join(sorted(missing)),
                pytrace=False,
            )
        return engine

    assert last_error is not None
    pytest.skip(
        f"PostgreSQL is not available for reconciliation tests: {type(last_error).__name__}"
    )


@pytest.fixture(autouse=True)
def _deterministic_embedding(deterministic_embedding):
    return deterministic_embedding


@pytest.fixture
def tenants(db_engine) -> list[UUID]:
    created: list[UUID] = []

    def _create() -> UUID:
        tenant = uuid4()
        with db_engine.begin() as connection:
            connection.execute(
                text("INSERT INTO tenants (id, name) VALUES (:id, :name)"),
                {"id": tenant, "name": f"phase-f-reconcile-{tenant}"},
            )
        created.append(tenant)
        return tenant

    try:
        yield _create
    finally:
        for tenant in created:
            with db_engine.begin() as connection:
                connection.execute(text("DELETE FROM tenants WHERE id = :id"), {"id": tenant})


def _content(prefix: str, lines: int = 60) -> str:
    return "\n".join(
        f"{prefix} step {index}: verify dependency {index} before escalation."
        for index in range(lines)
    )


def _submit(
    tenant_id: UUID,
    *,
    source_key: str | None = None,
    content: str | None = None,
):
    key = source_key or f"runbooks/reconcile-{uuid4()}.md"
    return submit_ingestion_job(
        principal=principal_for(tenant_id),
        tenant_id=tenant_id,
        source_key=key,
        title="Reconciliation runbook",
        source_type="runbook",
        content=content or _content(f"Reconcile {key}"),
    )


def _job(job_id: UUID) -> dict:
    with get_engine().connect() as connection:
        row = (
            connection.execute(
                text(
                    """
                    SELECT id, tenant_id, status, attempts, document_id, version_id,
                           activation_generation, error_code, error_summary,
                           next_dispatch_at, dispatch_claimed_until, last_dispatch_at,
                           dispatch_attempts, dispatch_error_code, execution_token,
                           execution_lease_expires_at,
                           (next_dispatch_at IS NULL OR next_dispatch_at <= now())
                               AS dispatch_due
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


def _version_is_current(version_id: UUID) -> bool:
    with get_engine().connect() as connection:
        return bool(
            connection.execute(
                text("SELECT is_current FROM document_versions WHERE id = :id"),
                {"id": version_id},
            ).scalar_one()
        )


def _assign(job_id: UUID, clause: str, parameters: dict | None = None) -> None:
    """Test setup helper: apply an explicit, non-production column assignment."""

    base = {"job_id": job_id}
    if parameters:
        base.update(parameters)
    with get_engine().begin() as connection:
        connection.execute(text(f"UPDATE ingestion_jobs SET {clause} WHERE id = :job_id"), base)


def _set_live_lease(job_id: UUID) -> UUID:
    token = uuid4()
    _assign(
        job_id,
        "execution_token = :token, execution_lease_expires_at = now() + interval '300 seconds'",
        {"token": token},
    )
    return token


def _set_expired_lease(job_id: UUID) -> UUID:
    token = uuid4()
    _assign(
        job_id,
        "execution_token = :token, execution_lease_expires_at = now() - interval '5 seconds'",
        {"token": token},
    )
    return token


# ---------------------------------------------------------------------------
# Dispatch eligibility and recovery
# ---------------------------------------------------------------------------


def test_queued_job_that_was_never_published_is_dispatched(tenants) -> None:
    tenant = tenants()
    submission = _submit(tenant)
    _assign(
        submission.job_id,
        "status = 'queued', attempts = 0, started_at = NULL, finished_at = NULL, "
        "next_dispatch_at = now()",
    )

    published: list[UUID] = []
    report = run_reconciliation(batch_size=10, publisher=published.append)

    assert report.scanned == 1
    assert report.dispatched == 1
    assert report.failures == 0
    assert published == [submission.job_id]

    row = _job(submission.job_id)
    assert row["status"] == "queued"
    assert row["dispatch_attempts"] == 1
    assert row["last_dispatch_at"] is not None
    assert row["dispatch_claimed_until"] is not None


def test_crash_after_commit_before_initial_publication_is_recovered(tenants) -> None:
    tenant = tenants()
    submission = _submit(tenant)

    # ``submit_ingestion_job`` committed a running job with a target version; in
    # this scenario the API process died before ``publish_ingestion_job`` ran.
    committed = _job(submission.job_id)
    assert committed["status"] == "running"
    assert committed["execution_token"] is None
    assert committed["version_id"] is not None

    published: list[UUID] = []
    report = run_reconciliation(batch_size=10, publisher=published.append)

    assert report.dispatched == 1
    assert published == [submission.job_id]


def test_broker_outage_does_not_fail_or_dead_letter_the_job(tenants) -> None:
    tenant = tenants()
    submission = _submit(tenant)

    def unavailable(_job_id: UUID) -> None:
        raise RuntimeError("redis is unavailable: secret-looking detail")

    report = run_reconciliation(batch_size=10, publisher=unavailable)

    assert report.scanned == 1
    assert report.dispatched == 0
    assert report.failures == 1

    row = _job(submission.job_id)
    assert row["status"] == "running"
    assert row["error_code"] is None
    assert row["dispatch_error_code"] == "queue_unavailable"
    assert row["next_dispatch_at"] is not None
    assert row["dispatch_due"] is False
    # The bounded backoff is retried only after it elapses.
    assert run_reconciliation(batch_size=10, publisher=unavailable).scanned == 0


def test_duplicate_reconciliation_does_not_republish_a_claimed_job(tenants) -> None:
    tenant = tenants()
    submission = _submit(tenant)

    published: list[UUID] = []
    first = run_reconciliation(batch_size=10, publisher=published.append)
    second = run_reconciliation(batch_size=10, publisher=published.append)

    assert first.dispatched == 1
    assert second.scanned == 0
    assert published == [submission.job_id]
    assert _job(submission.job_id)["dispatch_attempts"] == 1


def test_concurrent_dispatch_claims_are_disjoint(tenants) -> None:
    tenant = tenants()
    submissions = [_submit(tenant) for _ in range(6)]
    expected = {submission.job_id for submission in submissions}
    barrier = threading.Barrier(2)

    def claim() -> list:
        barrier.wait(timeout=10)
        return claim_ingestion_jobs_for_dispatch(batch_size=6, claim_seconds=60)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(claim) for _ in range(2)]
        batches = [future.result(timeout=30) for future in futures]

    claimed = [candidate.job_id for batch in batches for candidate in batch]
    assert len(claimed) == len(set(claimed))
    assert set(claimed) == expected
    # Each row reflects exactly one claim, so no job was counted twice.
    for submission in submissions:
        assert _job(submission.job_id)["dispatch_attempts"] == 1


def test_running_job_with_a_live_lease_is_not_dispatched(tenants) -> None:
    tenant = tenants()
    submission = _submit(tenant)
    token = _set_live_lease(submission.job_id)

    published: list[UUID] = []
    report = run_reconciliation(batch_size=10, publisher=published.append)

    assert report.scanned == 0
    assert published == []
    row = _job(submission.job_id)
    assert row["status"] == "running"
    assert row["execution_token"] == token


def test_running_job_with_an_expired_lease_is_reclaimed(tenants) -> None:
    tenant = tenants()
    submission = _submit(tenant)
    stale_token = _set_expired_lease(submission.job_id)

    published: list[UUID] = []
    report = run_reconciliation(batch_size=10, publisher=published.append)

    assert report.dispatched == 1
    assert published == [submission.job_id]
    row = _job(submission.job_id)
    assert row["status"] == "running"
    # Reconciliation only republishes; ownership changes when the new worker
    # claims the job.
    assert row["execution_token"] == stale_token

    result = resume_ingestion_job(submission.job_id)

    assert result.status == "succeeded"
    assert result.outcome == "succeeded"
    finished = _job(submission.job_id)
    assert finished["status"] == "succeeded"
    assert finished["attempts"] == 2
    assert finished["execution_token"] is None
    assert finished["execution_lease_expires_at"] is None
    assert _version_is_current(submission.version_id)


def test_late_task_cannot_change_a_newer_execution_outcome(tenants) -> None:
    tenant = tenants()
    submission = _submit(tenant)
    stale_token = _set_expired_lease(submission.job_id)

    # A newer execution claims and completes the job.
    result = resume_ingestion_job(submission.job_id)
    assert result.status == "succeeded"
    succeeded = _job(submission.job_id)
    assert succeeded["status"] == "succeeded"

    # The old worker now wakes up and tries to activate with its stale token.
    with pytest.raises(IngestionExecutionLeaseLostError):
        _activate_version(
            job_id=submission.job_id,
            tenant_id=tenant,
            document_id=submission.document_id,
            version_id=submission.version_id,
            version_number=1,
            content_hash="0" * 64,
            expected_chunk_count=1,
            execution_token=stale_token,
        )

    # A stale failure write is also refused and the durable success is intact.
    recorded = _record_job_failure(
        submission.job_id,
        tenant_id=tenant,
        source_key="runbooks/late.md",
        started_at=ingestion_module.datetime.now(ingestion_module.UTC),
        error=RuntimeError("late worker failed"),
        preserve_succeeded=False,
        execution_token=stale_token,
    )
    assert recorded is False
    after = _job(submission.job_id)
    assert after["status"] == "succeeded"
    assert after["error_code"] is None
    assert _version_is_current(submission.version_id)


def test_worker_crash_after_embeddings_before_activation_resumes_safely(tenants) -> None:
    tenant = tenants()
    source_key = f"runbooks/reconcile-crash-{uuid4()}.md"

    first = ingest_document(
        principal=principal_for(tenant),
        tenant_id=tenant,
        source_key=source_key,
        title="Crash runbook",
        source_type="runbook",
        content=_content("Version one"),
    )
    assert first["is_current"] is True

    second = _submit(tenant, source_key=source_key, content=_content("Version two"))
    # The crashed worker persisted every embedding for the new version but died
    # before activation and before renewing its lease.
    ingestion_module._ensure_embeddings(
        version_id=second.version_id,
        tenant_id=tenant,
        expected_chunk_count=_chunk_count(second.version_id),
    )
    _set_expired_lease(second.job_id)

    published: list[UUID] = []
    report = run_reconciliation(batch_size=10, publisher=published.append)
    assert report.dispatched == 1

    result = resume_ingestion_job(second.job_id)

    assert result.status == "succeeded"
    document = _document(tenant, source_key)
    assert document is not None
    assert document["current_version_id"] == second.version_id
    assert _version_is_current(first["version_id"]) is False


def _chunk_count(version_id: UUID) -> int:
    with get_engine().connect() as connection:
        return int(
            connection.execute(
                text("SELECT count(*) FROM chunks WHERE document_version_id = :id"),
                {"id": version_id},
            ).scalar_one()
        )


def test_duplicate_celery_delivery_is_busy_and_preserves_attempts(tenants) -> None:
    tenant = tenants()
    submission = _submit(tenant)

    claim = _claim_resumable_job(
        submission.job_id, started_at=ingestion_module.datetime.now(ingestion_module.UTC)
    )
    assert claim.execution_token  # the first delivery owns a live lease
    attempts_after_claim = _job(submission.job_id)["attempts"]

    with pytest.raises(IngestionJobBusyError):
        resume_ingestion_job(submission.job_id)

    row = _job(submission.job_id)
    assert row["status"] == "running"
    assert row["attempts"] == attempts_after_claim
    assert row["execution_token"] == claim.execution_token


def test_succeeded_job_is_never_redispatched(tenants) -> None:
    tenant = tenants()
    source_key = f"runbooks/reconcile-done-{uuid4()}.md"
    result = ingest_document(
        principal=principal_for(tenant),
        tenant_id=tenant,
        source_key=source_key,
        title="Done runbook",
        source_type="runbook",
        content=_content("Completed"),
    )
    job_id = _job_for_version(result["version_id"])

    published: list[UUID] = []
    report = run_reconciliation(batch_size=10, publisher=published.append)

    assert report.scanned == 0
    assert published == []
    assert _job(job_id)["dispatch_attempts"] == 0


def _job_for_version(version_id: UUID) -> UUID:
    with get_engine().connect() as connection:
        return connection.execute(
            text("SELECT id FROM ingestion_jobs WHERE version_id = :id"),
            {"id": version_id},
        ).scalar_one()


def test_repeatedly_lost_transient_task_is_recovered(tenants) -> None:
    tenant = tenants()
    submission = _submit(tenant)
    _assign(
        submission.job_id,
        "status = 'failed', error_code = 'embedding_service_unavailable', "
        "error_summary = 'Embedding service was unavailable', next_dispatch_at = now()",
    )

    published: list[UUID] = []
    first = run_reconciliation(batch_size=10, publisher=published.append)
    assert first.dispatched == 1

    # The republished task is lost: no worker claims it and the dispatch claim
    # expires. The durable failed state must become eligible again.
    _assign(submission.job_id, "dispatch_claimed_until = now() - interval '1 second'")
    second = run_reconciliation(batch_size=10, publisher=published.append)

    assert second.dispatched == 1
    assert published == [submission.job_id, submission.job_id]
    row = _job(submission.job_id)
    assert row["status"] == "failed"
    assert row["dispatch_attempts"] == 2


def test_ambiguous_publish_does_not_downgrade_a_claimed_job(tenants) -> None:
    tenant = tenants()
    submission = _submit(tenant)
    token = _set_live_lease(submission.job_id)

    # The API saw a broker error even though the message was delivered and a
    # worker claimed the job. The failure marker must not clear the live lease.
    ingestion_module.mark_ingestion_job_enqueue_failed(submission.job_id)

    row = _job(submission.job_id)
    assert row["status"] == "running"
    assert row["execution_token"] == token
    assert row["execution_lease_expires_at"] is not None


def test_permanent_failure_and_dead_letter_are_never_redispatched(tenants) -> None:
    tenant = tenants()
    permanent = _submit(tenant)
    dead_letter = _submit(tenant)
    _assign(
        permanent.job_id,
        "status = 'failed', error_code = 'embedding_schema_mismatch', "
        "error_summary = 'schema mismatch', next_dispatch_at = now()",
    )
    _assign(
        dead_letter.job_id,
        "status = 'dead_letter', error_code = 'task_failed', "
        "error_summary = 'retries exhausted', next_dispatch_at = now()",
    )

    published: list[UUID] = []
    report = run_reconciliation(batch_size=10, publisher=published.append)

    assert report.scanned == 0
    assert published == []
    assert _job(permanent.job_id)["dispatch_attempts"] == 0
    assert _job(dead_letter.job_id)["dispatch_attempts"] == 0


def test_batch_processing_is_bounded(tenants) -> None:
    tenant = tenants()
    submissions = [_submit(tenant) for _ in range(5)]
    expected = {submission.job_id for submission in submissions}

    published: list[UUID] = []
    reports = [run_reconciliation(batch_size=2, publisher=published.append) for _ in range(3)]

    assert [report.scanned for report in reports] == [2, 2, 1]
    assert [report.dispatched for report in reports] == [2, 2, 1]
    assert set(published) == expected
    assert len(published) == len(set(published)) == 5


def test_tenant_isolation_and_sanitized_logs(tenants, reconciliation_logs) -> None:
    tenant_a = tenants()
    tenant_b = tenants()
    submission_a = _submit(tenant_a)
    submission_b = _submit(tenant_b)

    published: list[UUID] = []
    run_reconciliation(batch_size=10, publisher=published.append)

    assert set(published) == {submission_a.job_id, submission_b.job_id}
    rows = {
        submission_a.job_id: _job(submission_a.job_id),
        submission_b.job_id: _job(submission_b.job_id),
    }
    assert rows[submission_a.job_id]["tenant_id"] == tenant_a
    assert rows[submission_b.job_id]["tenant_id"] == tenant_b

    job_events = [
        record.event_data
        for record in reconciliation_logs
        if record.event_data.get("event") == "ingestion_reconciliation_job"
    ]
    assert {event["job_id"] for event in job_events} == {
        str(submission_a.job_id),
        str(submission_b.job_id),
    }
    for event in job_events:
        assert event["tenant_ref"].startswith("t_")
        assert str(tenant_a) not in event["tenant_ref"]
        assert str(tenant_b) not in event["tenant_ref"]
        assert event["batch_id"]
        assert event["recovery_reason"]
        assert event["dispatch_attempts"] == 1

    summary = [
        record.event_data
        for record in reconciliation_logs
        if record.event_data.get("event") == "ingestion_reconciliation"
    ]
    assert summary
    assert summary[-1]["scanned"] == 2
    assert summary[-1]["dispatched"] == 2
    assert summary[-1]["failures"] == 0
    assert summary[-1]["duration_ms"] >= 0
    # No document content, embeddings, or raw error text is ever emitted.
    rendered = repr([record.event_data for record in reconciliation_logs])
    assert "step 0" not in rendered
    assert "redis is unavailable" not in rendered


def test_last_known_good_version_survives_a_failed_replacement(tenants) -> None:
    tenant = tenants()
    source_key = f"runbooks/reconcile-lkg-{uuid4()}.md"
    good = ingest_document(
        principal=principal_for(tenant),
        tenant_id=tenant,
        source_key=source_key,
        title="Stable runbook",
        source_type="runbook",
        content=_content("Stable version"),
    )
    replacement = _submit(tenant, source_key=source_key, content=_content("Broken version"))

    # The replacement execution fails permanently; the current version must not
    # move and the failed job must not be redispatched.
    _assign(
        replacement.job_id,
        "status = 'failed', error_code = 'embedding_incomplete', "
        "error_summary = 'Embedding completeness verification failed', "
        "next_dispatch_at = now()",
    )
    published: list[UUID] = []
    report = run_reconciliation(batch_size=10, publisher=published.append)

    assert report.scanned == 0
    assert published == []
    document = _document(tenant, source_key)
    assert document is not None
    assert document["current_version_id"] == good["version_id"]
    assert _version_is_current(replacement.version_id) is False


def test_overlapping_reconciliation_runs_are_serialized(tenants) -> None:
    tenant = tenants()
    submission = _submit(tenant)
    engine = get_engine()
    holder = engine.connect()
    try:
        holder.execute(
            text("SELECT pg_advisory_lock(:key)"),
            {"key": RECONCILIATION_ADVISORY_LOCK_KEY},
        )
        published: list[UUID] = []
        report = run_reconciliation(batch_size=10, publisher=published.append)
    finally:
        holder.execute(
            text("SELECT pg_advisory_unlock(:key)"),
            {"key": RECONCILIATION_ADVISORY_LOCK_KEY},
        )
        holder.close()

    assert report.locked is True
    assert report.scanned == 0
    assert published == []
    assert _job(submission.job_id)["dispatch_attempts"] == 0


# ---------------------------------------------------------------------------
# Configuration and pure helpers (no database)
# ---------------------------------------------------------------------------


def test_report_serializes_only_sanitized_counters() -> None:
    report = ReconciliationReport(
        batch_id="batch-1", scanned=3, dispatched=2, skipped=1, failures=0, duration_ms=7
    )
    assert report.as_dict() == {
        "batch_id": "batch-1",
        "locked": False,
        "scanned": 3,
        "dispatched": 2,
        "skipped": 1,
        "failures": 0,
        "duration_ms": 7,
    }


def test_dispatch_backoff_is_bounded() -> None:
    settings = Settings(
        _env_file=None,
        ingestion_reconcile_dispatch_backoff_seconds=15,
        ingestion_reconcile_dispatch_backoff_max_seconds=60,
        ingestion_execution_lease_seconds=30,
    )
    assert dispatch_backoff_seconds(1, settings=settings) == 15
    assert dispatch_backoff_seconds(2, settings=settings) == 30
    assert dispatch_backoff_seconds(10, settings=settings) == 60


@pytest.mark.parametrize(
    "overrides",
    [
        {"ingestion_reconcile_batch_size": 0},
        {"ingestion_reconcile_batch_size": MAX_RECONCILIATION_BATCH_SIZE + 1},
        {"ingestion_reconcile_interval_seconds": 4},
        {"ingestion_execution_lease_seconds": 0},
        {"ingestion_reconcile_dispatch_lease_seconds": 0},
        {"ingestion_reconcile_dispatch_backoff_seconds": -1},
    ],
)
def test_reconciliation_settings_reject_unsafe_values(overrides) -> None:
    with pytest.raises(ValueError):
        Settings(_env_file=None, **overrides)


def test_execution_lease_must_be_shorter_than_broker_visibility() -> None:
    with pytest.raises(ValueError, match="ingestion_execution_lease_seconds"):
        Settings(
            _env_file=None,
            celery_task_soft_time_limit_seconds=60,
            celery_task_time_limit_seconds=90,
            celery_visibility_timeout_seconds=120,
            ingestion_execution_lease_seconds=120,
        )


def test_dispatch_backoff_ceiling_must_not_be_below_floor() -> None:
    with pytest.raises(ValueError, match="backoff"):
        Settings(
            _env_file=None,
            ingestion_reconcile_dispatch_backoff_seconds=30,
            ingestion_reconcile_dispatch_backoff_max_seconds=10,
        )


def test_tenant_reference_is_salted_and_not_the_raw_identifier() -> None:
    tenant = uuid4()
    reference = tenant_reference(tenant)
    assert reference.startswith("t_")
    assert str(tenant) not in reference
    assert reference == tenant_reference(tenant)
    assert reference != tenant_reference(uuid4())


# ---------------------------------------------------------------------------
# Log capture
# ---------------------------------------------------------------------------


class _CapturingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list = []

    def emit(self, record: logging.LogRecord) -> None:
        data = getattr(record, "event_data", None)
        if isinstance(data, dict):
            self.records.append(_Record(data))


class _Record:
    def __init__(self, event_data: dict) -> None:
        self.event_data = event_data


@pytest.fixture
def reconciliation_logs():
    logger = logging.getLogger(RECONCILIATION_LOGGER_NAME)
    handler = _CapturingHandler()
    previous_level = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        yield handler.records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
