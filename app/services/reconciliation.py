"""Bounded, concurrency-safe recovery for orphaned ingestion jobs.

The durable ``ingestion_jobs`` row is the only source of truth. Redis and Celery
are treated as replaceable publication infrastructure, so this module never
assumes a publish and a PostgreSQL update are atomic.

Recovery has two independent safety layers:

1. **Dispatch claim.** A bounded ``SELECT ... FOR UPDATE SKIP LOCKED`` batch sets
   ``dispatch_claimed_until`` before anything is published. Concurrent
   reconcilers therefore claim disjoint jobs, and a job cannot be republished
   until its claim expires. A process crash after the claim but before publish is
   safe: the claim expires and the next run retries.
2. **Execution lease and fencing token.** A worker mints ``execution_token`` and
   an ``execution_lease_expires_at`` when it claims a job. Activation, success,
   and failure writes require the same token and a live lease. Reclaiming an
   abandoned execution mints a new token, so the old worker can no longer change
   the job's outcome.

Only non-terminal jobs with a committed target version and activation generation
are eligible. Succeeded, permanent failed, and dead-letter jobs are never
redispatched.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import text

from app.core.config import get_settings
from app.core.observability import (
    RECONCILIATION_LOGGER_NAME,
    tenant_reference,
)
from app.db.connections import get_engine
from app.services.ingestion import TRANSIENT_JOB_ERROR_CODES

logger = logging.getLogger(RECONCILIATION_LOGGER_NAME)

# One cluster-wide advisory lock prevents overlapping reconciliation runs from
# doing duplicate work. It is a scheduling convenience only: correctness against
# concurrent publishers still comes from the PostgreSQL dispatch claim.
RECONCILIATION_ADVISORY_LOCK_KEY = 2_026_111_401

# ``queue_unavailable`` is the durable state left by a failed initial publish.
# The transient codes are written by a fenced worker after an execution failure;
# they are eligible only once the worker-recorded grace has elapsed.
DISPATCHABLE_FAILED_ERROR_CODES = frozenset({"queue_unavailable"}) | TRANSIENT_JOB_ERROR_CODES

MAX_RECONCILIATION_BATCH_SIZE = 1000


@dataclass(frozen=True)
class DispatchCandidate:
    """One job claimed for publication, with sanitized log metadata."""

    job_id: UUID
    tenant_id: UUID
    status: str
    recovery_reason: str
    dispatch_attempts: int


@dataclass(frozen=True)
class ReconciliationReport:
    """Bounded summary of one reconciliation run."""

    batch_id: str
    scanned: int = 0
    dispatched: int = 0
    skipped: int = 0
    failures: int = 0
    duration_ms: int = 0
    locked: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "locked": self.locked,
            "scanned": self.scanned,
            "dispatched": self.dispatched,
            "skipped": self.skipped,
            "failures": self.failures,
            "duration_ms": self.duration_ms,
        }


def claim_ingestion_jobs_for_dispatch(
    *,
    batch_size: int,
    claim_seconds: int,
) -> list[DispatchCandidate]:
    """Claim a bounded batch of eligible jobs with ``SKIP LOCKED``.

    The claim is committed before publication. If the caller crashes, the lease
    expires and the job becomes eligible again; if the caller publishes but
    crashes before recording success, the duplicate delivery is harmless.
    """

    if not 1 <= batch_size <= MAX_RECONCILIATION_BATCH_SIZE:
        raise ValueError(f"batch_size must be between 1 and {MAX_RECONCILIATION_BATCH_SIZE}")
    if claim_seconds < 1:
        raise ValueError("claim_seconds must be at least 1")

    with get_engine().begin() as connection:
        rows = (
            connection.execute(
                text(
                    """
                    SELECT id, tenant_id, status, dispatch_attempts,
                           CASE
                               WHEN status = 'running' AND execution_token IS NULL
                                   THEN 'committed_never_dispatched'
                               WHEN status = 'running'
                                   THEN 'abandoned_execution'
                               WHEN status = 'failed' AND error_code = 'queue_unavailable'
                                   THEN 'publication_failed'
                               WHEN status = 'failed'
                                   THEN 'lost_retry'
                               ELSE 'queued_never_dispatched'
                           END AS recovery_reason
                    FROM ingestion_jobs
                    WHERE version_id IS NOT NULL
                      AND activation_generation IS NOT NULL
                      AND (next_dispatch_at IS NULL OR next_dispatch_at <= now())
                      AND (
                          dispatch_claimed_until IS NULL
                          OR dispatch_claimed_until <= now()
                      )
                      AND (
                          status = 'queued'
                          OR (
                              status = 'running'
                              AND execution_token IS NULL
                          )
                          OR (
                              status = 'running'
                              AND execution_token IS NOT NULL
                              AND execution_lease_expires_at IS NOT NULL
                              AND execution_lease_expires_at <= now()
                          )
                          OR (
                              status = 'failed'
                              AND error_code = 'queue_unavailable'
                          )
                          OR (
                              status = 'failed'
                              AND error_code = ANY(:failed_error_codes)
                              AND next_dispatch_at IS NOT NULL
                              AND next_dispatch_at <= now()
                          )
                          OR (
                              -- A republished transient failure may lose its new
                              -- task too. Once the dispatch claim expires, the
                              -- durable failed state is eligible again.
                              status = 'failed'
                              AND error_code = ANY(:failed_error_codes)
                              AND next_dispatch_at IS NULL
                              AND dispatch_claimed_until IS NOT NULL
                              AND dispatch_claimed_until <= now()
                          )
                      )
                    ORDER BY created_at, id
                    LIMIT :batch_size
                    FOR UPDATE SKIP LOCKED
                    """
                ),
                {
                    "batch_size": batch_size,
                    "failed_error_codes": sorted(DISPATCHABLE_FAILED_ERROR_CODES),
                },
            )
            .mappings()
            .all()
        )
        if not rows:
            return []

        job_ids = [row["id"] for row in rows]
        connection.execute(
            text(
                """
                UPDATE ingestion_jobs
                SET dispatch_claimed_until = now() + make_interval(secs => :claim_seconds),
                    last_dispatch_at = now(),
                    dispatch_attempts = dispatch_attempts + 1,
                    dispatch_error_code = NULL
                WHERE id = ANY(:job_ids)
                """
            ),
            {"job_ids": job_ids, "claim_seconds": claim_seconds},
        )

    return [
        DispatchCandidate(
            job_id=row["id"],
            tenant_id=row["tenant_id"],
            status=row["status"],
            recovery_reason=row["recovery_reason"],
            dispatch_attempts=int(row["dispatch_attempts"]) + 1,
        )
        for row in rows
    ]


def mark_ingestion_job_dispatched(
    job_id: UUID,
    *,
    claim_seconds: int,
    increment_attempts: bool = True,
) -> bool:
    """Record a successful publication and hold the next dispatch for a lease.

    Best-effort from the API path and transactional from the reconciler. A
    failure here does not undo a successful publish; at worst the job is
    republished after the claim expires.
    """

    if claim_seconds < 1:
        raise ValueError("claim_seconds must be at least 1")

    attempts_assignment = ""
    if increment_attempts:
        attempts_assignment = ", dispatch_attempts = dispatch_attempts + 1"

    with get_engine().begin() as connection:
        result = connection.execute(
            text(
                f"""
                UPDATE ingestion_jobs
                SET last_dispatch_at = now(),
                    dispatch_claimed_until = now() + make_interval(secs => :claim_seconds),
                    dispatch_error_code = NULL,
                    next_dispatch_at = NULL
                    {attempts_assignment}
                WHERE id = :job_id AND status <> 'succeeded'
                """
            ),
            {"job_id": job_id, "claim_seconds": claim_seconds},
        )
    return result.rowcount == 1


def mark_ingestion_job_dispatch_failed(job_id: UUID, *, backoff_seconds: float) -> bool:
    """Record a broker failure without changing the durable job status.

    The job stays ``queued``/``running``/``failed`` with a bounded
    ``next_dispatch_at`` in the future. A Redis outage must never turn a valid
    committed job into a permanent failure or dead letter.
    """

    with get_engine().begin() as connection:
        result = connection.execute(
            text(
                """
                UPDATE ingestion_jobs
                SET dispatch_claimed_until = NULL,
                    next_dispatch_at = now() + make_interval(secs => :backoff_seconds),
                    dispatch_error_code = 'queue_unavailable'
                WHERE id = :job_id
                  AND status IN ('queued', 'running', 'failed')
                """
            ),
            {"job_id": job_id, "backoff_seconds": max(0.0, backoff_seconds)},
        )
    return result.rowcount == 1


def dispatch_backoff_seconds(attempts: int, *, settings: Any | None = None) -> float:
    """Return capped exponential backoff for a failed dispatch."""

    settings = settings or get_settings()
    base = max(0.0, settings.ingestion_reconcile_dispatch_backoff_seconds)
    maximum = max(base, settings.ingestion_reconcile_dispatch_backoff_max_seconds)
    exponent = max(0, attempts - 1)
    return min(base * (2**exponent), maximum)


def _log_job_event(
    *,
    batch_id: str,
    candidate: DispatchCandidate,
    outcome: str,
    level: int = logging.INFO,
    error_category: str | None = None,
) -> None:
    logger.log(
        level,
        "ingestion reconciliation job",
        extra={
            "event_data": {
                "event": "ingestion_reconciliation_job",
                "batch_id": batch_id,
                "job_id": str(candidate.job_id),
                "tenant_ref": tenant_reference(candidate.tenant_id),
                "status": candidate.status,
                "recovery_reason": candidate.recovery_reason,
                "dispatch_attempts": candidate.dispatch_attempts,
                "outcome": outcome,
                "error_category": error_category,
            }
        },
    )


def _log_summary(report: ReconciliationReport) -> None:
    logger.log(
        logging.WARNING if report.failures else logging.INFO,
        "ingestion reconciliation batch",
        extra={
            "event_data": {
                "event": "ingestion_reconciliation",
                "batch_id": report.batch_id,
                "locked": report.locked,
                "scanned": report.scanned,
                "dispatched": report.dispatched,
                "skipped": report.skipped,
                "failures": report.failures,
                "duration_ms": report.duration_ms,
            }
        },
    )


def run_reconciliation(
    *,
    batch_size: int | None = None,
    publisher: Any | None = None,
) -> ReconciliationReport:
    """Dispatch one bounded batch of orphaned jobs.

    ``publisher`` exists for deterministic tests. Production uses the Celery
    publication helper, which sends only the opaque job identifier.
    """

    settings = get_settings()
    effective_batch_size = (
        settings.ingestion_reconcile_batch_size if batch_size is None else batch_size
    )
    if not 1 <= effective_batch_size <= MAX_RECONCILIATION_BATCH_SIZE:
        raise ValueError(f"batch_size must be between 1 and {MAX_RECONCILIATION_BATCH_SIZE}")

    batch_id = uuid4().hex
    started = time.perf_counter()
    lock_connection = get_engine().connect().execution_options(isolation_level="AUTOCOMMIT")

    try:
        acquired = bool(
            lock_connection.execute(
                text("SELECT pg_try_advisory_lock(:key)"),
                {"key": RECONCILIATION_ADVISORY_LOCK_KEY},
            ).scalar()
        )
        if not acquired:
            report = ReconciliationReport(
                batch_id=batch_id,
                duration_ms=int((time.perf_counter() - started) * 1000),
                locked=True,
            )
            _log_summary(report)
            return report

        try:
            candidates = claim_ingestion_jobs_for_dispatch(
                batch_size=effective_batch_size,
                claim_seconds=settings.ingestion_reconcile_dispatch_lease_seconds,
            )
            dispatched = 0
            failures = 0

            for candidate in candidates:
                try:
                    if publisher is None:
                        # Imported lazily so this service can be imported by the
                        # task module without a circular import.
                        from app.tasks.ingestion import send_ingestion_job

                        send_ingestion_job(candidate.job_id)
                    else:
                        publisher(candidate.job_id)
                except Exception:  # noqa: BLE001 - broker errors are sanitized
                    failures += 1
                    backoff = dispatch_backoff_seconds(
                        candidate.dispatch_attempts,
                        settings=settings,
                    )
                    try:
                        mark_ingestion_job_dispatch_failed(
                            candidate.job_id,
                            backoff_seconds=backoff,
                        )
                    except Exception:  # noqa: BLE001 - claim expiry retries later
                        logger.debug(
                            "could not record dispatch failure; claim will expire",
                            extra={
                                "event_data": {
                                    "event": "ingestion_reconciliation_job",
                                    "batch_id": batch_id,
                                    "job_id": str(candidate.job_id),
                                    "outcome": "dispatch_state_unrecorded",
                                    "error_category": "database_error",
                                }
                            },
                        )
                    _log_job_event(
                        batch_id=batch_id,
                        candidate=candidate,
                        outcome="dispatch_failed",
                        level=logging.WARNING,
                        error_category="queue_unavailable",
                    )
                    continue

                dispatched += 1
                _log_job_event(
                    batch_id=batch_id,
                    candidate=candidate,
                    outcome="dispatched",
                )

            report = ReconciliationReport(
                batch_id=batch_id,
                scanned=len(candidates),
                dispatched=dispatched,
                skipped=0,
                failures=failures,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )
            _log_summary(report)
            return report
        finally:
            lock_connection.execute(
                text("SELECT pg_advisory_unlock(:key)"),
                {"key": RECONCILIATION_ADVISORY_LOCK_KEY},
            )
    finally:
        lock_connection.close()
