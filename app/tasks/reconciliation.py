"""Periodic Celery task for bounded ingestion-job reconciliation.

The task carries no arguments and returns only a sanitized counter summary. A
worker may receive it from Celery beat or from an operator's manual ``celery
call``; overlapping runs are made safe by a PostgreSQL advisory lock plus the
per-job dispatch claim in :mod:`app.services.reconciliation`.

The application routes this task to ``Settings.celery_reconciliation_queue`` so
it is consumed by the dedicated reconciliation worker rather than the ingestion
worker.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from app.celery_app import RECONCILE_TASK_NAME, OpsAtlasTask, celery_app
from app.core.observability import WORKER_LOGGER_NAME
from app.db.schema_validation import SchemaValidationError, validate_database_schema
from app.services.reconciliation import run_reconciliation

logger = logging.getLogger(WORKER_LOGGER_NAME)


def run_reconciliation_task() -> dict[str, Any]:
    """Worker entry guard: never mutate job state against an incompatible schema.

    Worker startup already validates the schema. This repeats the cached check
    on every delivery so a schema change after startup cannot let reconciliation
    claim jobs, update dispatch state, or publish work against the wrong schema.
    A failure returns a sanitized ``blocked`` summary and leaves every durable job
    untouched for a later run after the schema is repaired.
    """

    started = time.perf_counter()
    try:
        validate_database_schema()
    except SchemaValidationError as exc:
        logger.error(
            "worker refused to run reconciliation: database schema is not ready",
            extra={
                "event_data": {
                    "event": "ingestion_reconciliation",
                    "outcome": "blocked",
                    "error_category": "schema_incompatible",
                    "detail": str(exc),
                    "duration_ms": int((time.perf_counter() - started) * 1000),
                }
            },
        )
        return {
            "status": "blocked",
            "outcome": "blocked",
            "error_category": "schema_incompatible",
        }
    return run_reconciliation().as_dict()


@celery_app.task(
    bind=True,
    base=OpsAtlasTask,
    name=RECONCILE_TASK_NAME,
    ignore_result=True,
)
def reconcile_ingestion_jobs_task(self) -> dict[str, Any]:
    """Run one bounded reconciliation batch and return sanitized counters."""

    return run_reconciliation_task()
