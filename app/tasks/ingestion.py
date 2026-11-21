"""Celery task orchestration for durable ingestion jobs.

The task is deliberately thin: it accepts one opaque job identifier, resolves
the already-committed target from PostgreSQL, and delegates every embedding,
validation, persistence, and activation step to
:func:`app.services.ingestion.resume_ingestion_job`. Document content is never
placed on the broker, in task arguments, or in task results.

Retry policy
------------
* ``EmbeddingUnavailableError`` is transient: bounded exponential backoff.
* Invalid provider output, schema mismatch, activation-integrity failure, and a
  missing/invalid job are permanent: no retry.
* Unexpected exceptions receive a small, separately bounded retry budget because
  the service is idempotent and every failed attempt is recorded in PostgreSQL.
* Exhausted retries move the durable job to ``dead_letter``. Duplicate delivery
  of a succeeded or terminal job is a pure read/no-op.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from celery.exceptions import SoftTimeLimitExceeded
from sqlalchemy.exc import SQLAlchemyError

from app.celery_app import OpsAtlasTask, celery_app
from app.core.config import get_settings
from app.core.observability import WORKER_LOGGER_NAME, tenant_reference
from app.db.embedding_schema import EmbeddingSchemaError
from app.db.schema_validation import SchemaValidationError, validate_database_schema
from app.services.embedding_client import (
    EmbeddingServiceError,
    EmbeddingUnavailableError,
    EmbeddingValidationError,
)
from app.services.ingestion import (
    ActivationIntegrityError,
    EmbeddingCompletenessError,
    EmptyDocumentError,
    IdempotencyKeyCollisionError,
    IngestionExecutionLeaseLostError,
    IngestionExecutionResult,
    IngestionJobBusyError,
    IngestionJobNotFoundError,
    IngestionJobStateError,
    IngestionJobTerminalError,
    IngestionQueueUnavailableError,
    TenantNotFoundError,
    get_ingestion_job_snapshot,
    mark_ingestion_job_enqueue_failed,
    mark_ingestion_job_retries_exhausted,
    resume_ingestion_job,
)
from app.services.reconciliation import mark_ingestion_job_dispatched

logger = logging.getLogger(WORKER_LOGGER_NAME)
TASK_NAME = "opsatlas.ingestion.resume_job"


@dataclass(frozen=True)
class TaskFailurePolicy:
    """How one exception class maps onto Celery retry behavior."""

    retry: bool
    max_retries: int
    error_category: str


def classify_task_failure(error: BaseException, settings: Any | None = None) -> TaskFailurePolicy:
    """Return a bounded, non-sensitive retry policy for an exception.

    Provider and schema errors are classified explicitly so a generic retry
    decorator can never accidentally retry a deterministic contract failure.
    """

    settings = settings or get_settings()

    if isinstance(error, EmbeddingUnavailableError):
        return TaskFailurePolicy(
            retry=True,
            max_retries=max(0, settings.ingestion_task_max_retries),
            error_category="embedding_service_unavailable",
        )

    if isinstance(
        error,
        (
            EmbeddingValidationError,
            EmbeddingCompletenessError,
            EmbeddingSchemaError,
            EmbeddingServiceError,
            IdempotencyKeyCollisionError,
            ActivationIntegrityError,
            IngestionJobNotFoundError,
            IngestionJobStateError,
            IngestionJobBusyError,
            IngestionExecutionLeaseLostError,
            EmptyDocumentError,
            TenantNotFoundError,
        ),
    ):
        return TaskFailurePolicy(
            retry=False,
            max_retries=0,
            error_category=_permanent_error_category(error),
        )

    # Timeouts and database errors may be transient. They are still bounded and
    # safe because the ingestion service never holds a transaction across an
    # embedding call and reuses the same immutable target on retry.
    if isinstance(error, SoftTimeLimitExceeded):
        category = "task_timeout"
    elif isinstance(error, SQLAlchemyError):
        category = "database_error"
    else:
        category = "task_failed"
    return TaskFailurePolicy(
        retry=True,
        max_retries=max(0, settings.ingestion_task_unexpected_max_retries),
        error_category=category,
    )


def _permanent_error_category(error: BaseException) -> str:
    if isinstance(error, EmbeddingValidationError):
        return "embedding_invalid_response"
    if isinstance(error, EmbeddingCompletenessError):
        return "embedding_incomplete"
    if isinstance(error, EmbeddingSchemaError):
        return "embedding_schema_mismatch"
    if isinstance(error, IdempotencyKeyCollisionError):
        return "idempotency_conflict"
    if isinstance(error, ActivationIntegrityError):
        return "activation_rejected"
    if isinstance(error, IngestionJobNotFoundError):
        return "job_not_found"
    if isinstance(error, IngestionJobStateError):
        return "ingestion_job_invalid"
    if isinstance(error, IngestionJobBusyError):
        return "job_busy"
    if isinstance(error, IngestionExecutionLeaseLostError):
        return "execution_lease_lost"
    if isinstance(error, EmptyDocumentError):
        return "empty_document"
    if isinstance(error, TenantNotFoundError):
        return "tenant_not_found"
    return "embedding_failed"


def retry_countdown(retries: int, settings: Any | None = None) -> float:
    """Return capped exponential backoff for the next retry."""

    settings = settings or get_settings()
    base = max(0.0, settings.ingestion_task_retry_backoff_seconds)
    maximum = max(base, settings.ingestion_task_retry_backoff_max_seconds)
    return min(base * (2 ** max(0, retries)), maximum)


def _safe_snapshot(job_id: UUID) -> dict[str, Any]:
    try:
        return get_ingestion_job_snapshot(job_id) or {}
    except Exception:  # noqa: BLE001 - logging must never mask the task error
        return {}


def _log_task_event(
    *,
    job_id: str,
    task_id: str | None,
    attempt: int,
    outcome: str,
    started: float,
    level: int = logging.INFO,
    tenant_id: UUID | None = None,
    status: str | None = None,
    job_attempts: int | None = None,
    error_category: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "event": "ingestion_task",
        "job_id": job_id,
        "task_id": task_id,
        "attempt": attempt,
        "job_attempts": job_attempts,
        "status": status,
        "outcome": outcome,
        "error_category": error_category,
        "duration_ms": int((time.perf_counter() - started) * 1000),
    }
    if tenant_id is not None:
        payload["tenant_ref"] = tenant_reference(tenant_id)
    logger.log(level, "ingestion task", extra={"event_data": payload})


def _safe_mark_exhausted(
    job_id: UUID,
    *,
    error_category: str,
    expected_attempts: int | None = None,
    expected_error_code: str | None = None,
) -> None:
    try:
        mark_ingestion_job_retries_exhausted(
            job_id,
            error_code=error_category,
            expected_attempts=expected_attempts,
            expected_error_code=expected_error_code,
        )
    except Exception:  # noqa: BLE001 - the sanitized task event still records failure
        logger.error(
            "could not mark ingestion job terminal",
            extra={
                "event_data": {
                    "event": "ingestion_task",
                    "job_id": str(job_id),
                    "outcome": "state_update_failed",
                    "error_category": "database_error",
                }
            },
        )


def execute_ingestion_task(task: Any, job_id: str) -> dict[str, Any]:
    """Execute one task delivery using only a durable job identifier.

    ``task`` is the bound Celery task (or a small fake in tests). Keeping the
    body separate makes retry-policy tests deterministic without a broker.
    """

    settings = get_settings()
    started = time.perf_counter()
    retries = int(getattr(getattr(task, "request", None), "retries", 0) or 0)
    task_id = getattr(getattr(task, "request", None), "id", None)

    try:
        job_uuid = UUID(str(job_id))
    except (TypeError, ValueError):
        _log_task_event(
            job_id=str(job_id),
            task_id=task_id,
            attempt=retries + 1,
            outcome="failed",
            started=started,
            level=logging.ERROR,
            error_category="invalid_job_id",
        )
        return {
            "job_id": str(job_id),
            "status": "failed",
            "outcome": "failed",
            "error_category": "invalid_job_id",
        }

    try:
        result: IngestionExecutionResult = resume_ingestion_job(job_uuid)
    except IngestionJobTerminalError:
        snapshot = _safe_snapshot(job_uuid)
        _log_task_event(
            job_id=str(job_uuid),
            task_id=task_id,
            attempt=retries + 1,
            outcome="skipped",
            started=started,
            tenant_id=snapshot.get("tenant_id"),
            status=snapshot.get("status"),
            job_attempts=snapshot.get("attempts"),
            error_category=snapshot.get("error_code") or "job_terminal",
        )
        return {
            "job_id": str(job_uuid),
            "status": snapshot.get("status", "failed"),
            "outcome": "skipped",
            "error_category": snapshot.get("error_code") or "job_terminal",
        }
    except IngestionJobBusyError:
        snapshot = _safe_snapshot(job_uuid)
        _log_task_event(
            job_id=str(job_uuid),
            task_id=task_id,
            attempt=retries + 1,
            outcome="skipped",
            started=started,
            tenant_id=snapshot.get("tenant_id"),
            status=snapshot.get("status"),
            job_attempts=snapshot.get("attempts"),
            error_category="job_busy",
        )
        return {
            "job_id": str(job_uuid),
            "status": snapshot.get("status", "running"),
            "outcome": "skipped",
            "error_category": "job_busy",
        }
    except IngestionExecutionLeaseLostError:
        snapshot = _safe_snapshot(job_uuid)
        _log_task_event(
            job_id=str(job_uuid),
            task_id=task_id,
            attempt=retries + 1,
            outcome="skipped",
            started=started,
            level=logging.WARNING,
            tenant_id=snapshot.get("tenant_id"),
            status=snapshot.get("status"),
            job_attempts=snapshot.get("attempts"),
            error_category="execution_lease_lost",
        )
        return {
            "job_id": str(job_uuid),
            "status": snapshot.get("status", "running"),
            "outcome": "skipped",
            "error_category": "execution_lease_lost",
        }
    except Exception as error:  # noqa: BLE001 - mapped deliberately below
        policy = classify_task_failure(error, settings)
        snapshot = _safe_snapshot(job_uuid)

        if policy.retry and retries < policy.max_retries:
            countdown = retry_countdown(retries, settings)
            _log_task_event(
                job_id=str(job_uuid),
                task_id=task_id,
                attempt=retries + 1,
                outcome="retry",
                started=started,
                level=logging.WARNING,
                tenant_id=snapshot.get("tenant_id"),
                status=snapshot.get("status"),
                job_attempts=snapshot.get("attempts"),
                error_category=policy.error_category,
            )
            # Do not pass ``exc`` to Celery: its retry logging can render the
            # exception message, which may contain provider or database payloads.
            # The sanitized category is recorded in the structured event and in
            # the PostgreSQL job row instead.
            raise task.retry(
                countdown=countdown,
                max_retries=policy.max_retries,
            )

        if policy.retry:
            _safe_mark_exhausted(
                job_uuid,
                error_category=policy.error_category,
                expected_attempts=snapshot.get("attempts"),
                expected_error_code=snapshot.get("error_code"),
            )
            outcome = "exhausted"
        else:
            outcome = "failed"

        _log_task_event(
            job_id=str(job_uuid),
            task_id=task_id,
            attempt=retries + 1,
            outcome=outcome,
            started=started,
            level=logging.ERROR,
            tenant_id=snapshot.get("tenant_id"),
            status=snapshot.get("status"),
            job_attempts=snapshot.get("attempts"),
            error_category=policy.error_category,
        )
        return {
            "job_id": str(job_uuid),
            "status": snapshot.get("status", "failed"),
            "outcome": outcome,
            "error_category": policy.error_category,
        }

    snapshot = _safe_snapshot(job_uuid)
    _log_task_event(
        job_id=str(result.job_id),
        task_id=task_id,
        attempt=retries + 1,
        outcome=result.outcome,
        started=started,
        tenant_id=result.tenant_id,
        status=result.status,
        job_attempts=result.attempts,
        error_category=snapshot.get("error_code"),
    )
    return {
        "job_id": str(result.job_id),
        "status": result.status,
        "outcome": result.outcome,
        "attempts": result.attempts,
        "is_current": result.is_current,
    }


def run_ingestion_task(task: Any, job_id: str) -> dict[str, Any]:
    """Worker entry guard: never resume a job against an incompatible schema.

    A task delivery is acknowledged with a sanitized ``blocked`` result instead
    of being marked failed. The durable job remains in PostgreSQL for the
    reconciler to republish after migrations are restored, so a schema incident
    cannot silently destroy queued work.
    """

    started = time.perf_counter()
    task_id = getattr(getattr(task, "request", None), "id", None)
    try:
        validate_database_schema()
    except SchemaValidationError as exc:
        logger.error(
            "worker refused to consume an ingestion job: database schema is not ready",
            extra={
                "event_data": {
                    "event": "ingestion_task",
                    "job_id": str(job_id),
                    "task_id": task_id,
                    "outcome": "blocked",
                    "error_category": "schema_incompatible",
                    "detail": str(exc),
                    "duration_ms": int((time.perf_counter() - started) * 1000),
                }
            },
        )
        return {
            "job_id": str(job_id),
            "status": "blocked",
            "outcome": "blocked",
            "error_category": "schema_incompatible",
        }
    return execute_ingestion_task(task, job_id)


@celery_app.task(
    bind=True,
    base=OpsAtlasTask,
    name=TASK_NAME,
    ignore_result=True,
)
def resume_ingestion_job_task(self, job_id: str) -> dict[str, Any]:
    """Celery entry point: resume one durable job by identifier only."""

    return run_ingestion_task(self, job_id)


def send_ingestion_job(job_id: UUID) -> str:
    """Send one task containing only an opaque job identifier.

    This is the shared low-level publication helper for both the API and the
    reconciler. It deliberately does not update PostgreSQL; each caller records
    success or failure with the policy appropriate to its path.
    """

    settings = get_settings()
    async_result = celery_app.send_task(
        TASK_NAME,
        args=[str(job_id)],
        queue=settings.celery_task_default_queue,
    )
    return str(getattr(async_result, "id", "") or "")


def publish_ingestion_job(job_id: UUID, *, tenant_id: UUID | None = None) -> str:
    """Publish an already-committed job to the ingestion queue.

    Publication happens only after the API transaction has committed the job and
    its non-current target. If the broker is unavailable, the durable job is
    marked failed with a sanitized code so it remains observable; the reconciler
    can also republish it later. A successful publish records a dispatch claim
    so a concurrent reconciler does not immediately duplicate it.
    """

    settings = get_settings()
    started = time.perf_counter()
    try:
        task_id = send_ingestion_job(job_id)
    except Exception as error:
        try:
            mark_ingestion_job_enqueue_failed(job_id)
        except Exception:  # noqa: BLE001 - still raise the publication failure
            logger.error(
                "could not record queue publication failure",
                extra={
                    "event_data": {
                        "event": "ingestion_task",
                        "job_id": str(job_id),
                        "outcome": "state_update_failed",
                        "error_category": "database_error",
                    }
                },
            )
        _log_task_event(
            job_id=str(job_id),
            task_id=None,
            attempt=1,
            outcome="publish_failed",
            started=started,
            level=logging.ERROR,
            tenant_id=tenant_id,
            error_category="queue_unavailable",
        )
        raise IngestionQueueUnavailableError from error

    try:
        mark_ingestion_job_dispatched(
            job_id,
            claim_seconds=settings.ingestion_reconcile_dispatch_lease_seconds,
        )
    except Exception:  # noqa: BLE001 - publication already succeeded
        # Worst case the reconciler republishes after the job's claim expires;
        # duplicate delivery is safe. Never log broker/database details.
        logger.warning(
            "could not record successful dispatch",
            extra={
                "event_data": {
                    "event": "ingestion_task",
                    "job_id": str(job_id),
                    "outcome": "dispatch_record_failed",
                    "error_category": "database_error",
                }
            },
        )

    _log_task_event(
        job_id=str(job_id),
        task_id=task_id,
        attempt=1,
        outcome="queued",
        started=started,
        tenant_id=tenant_id,
    )
    return task_id
