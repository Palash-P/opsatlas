"""Celery application for background ingestion.

Redis is the broker only. The durable authority for a job is the PostgreSQL
``ingestion_jobs`` row, so this application deliberately has **no result
backend**: task return values are ignored and no caller polls Celery for state.
A missing broker therefore cannot erase the fact that a job was requested; it
can only delay or fail publication, which the API records in PostgreSQL.

Serialization is JSON, task arguments are restricted to opaque job IDs, and
late acknowledgement plus worker-loss rejection let a crashed or evicted worker
redeliver the same task. Duplicate delivery is safe because the ingestion
service is idempotent and only resumes an already-prepared target.

Ingestion and reconciliation use separate queues. The default queue carries
document ingestion; ``RECONCILE_TASK_NAME`` is routed to
``Settings.celery_reconciliation_queue`` and consumed by a dedicated worker, so
periodic recovery cannot be starved by a busy ingestion backlog.
"""

from __future__ import annotations

import logging

from celery import Celery, Task
from celery.signals import celeryd_after_setup

from app.core.config import Settings, get_settings
from app.core.observability import WORKER_LOGGER_NAME, configure_logging

logger = logging.getLogger(WORKER_LOGGER_NAME)

RECONCILE_TASK_NAME = "opsatlas.ingestion.reconcile_jobs"


def validate_worker_schema_on_startup(*_args: object, **_kwargs: object) -> None:
    """Fail worker setup before the consumer starts on an incompatible schema.

    The task entry point repeats a cached validation on every delivery, so even
    a schema change after startup cannot cause ingestion work to run against the
    wrong database. Disposing the engine here avoids inheriting pre-fork pooled
    connections into prefork children.

    Celery's ``Signal.send`` catches ``Exception`` from receivers and only logs
    it, so raising ``SchemaValidationError`` here would not stop the worker from
    consuming. Raising ``SystemExit`` (a ``BaseException``) propagates through
    the signal dispatch and aborts worker startup instead.
    """

    from app.db.connections import get_engine
    from app.db.schema_validation import SchemaValidationError, validate_worker_startup_schema

    try:
        validate_worker_startup_schema()
    except SchemaValidationError as exc:
        logger.error(
            "worker startup refused: database schema is not ready",
            extra={
                "event_data": {
                    "event": "schema_validation",
                    "outcome": "failed",
                    "error_category": "schema_incompatible",
                    "detail": str(exc),
                }
            },
        )
        raise SystemExit(1) from exc
    finally:
        get_engine().dispose()


celeryd_after_setup.connect(validate_worker_schema_on_startup)


class OpsAtlasTask(Task):
    """Task base that never logs arguments, content, or exception payloads.

    Celery's default ``on_failure`` logs the task arguments and a traceback. The
    traceback can contain provider responses, SQL parameters, or document text
    from a library exception. The worker records a sanitized failure in
    PostgreSQL and emits a structured event separately, so the default logging
    is replaced rather than supplemented.
    """

    def on_failure(self, exc, task_id, args, kwargs, einfo) -> None:
        logger.error(
            "ingestion task failed",
            extra={
                "event_data": {
                    "event": "ingestion_task",
                    "task_id": task_id,
                    "outcome": "failed",
                    "error_category": type(exc).__name__,
                }
            },
        )

    def on_retry(self, exc, task_id, args, kwargs, einfo) -> None:
        # Retry decisions are logged with job/attempt correlation by the task
        # itself. Avoid Celery's default traceback here.
        return None


def create_celery_app(settings: Settings | None = None) -> Celery:
    """Build the Celery application from environment-backed settings.

    Kept as a factory so tests can assert configuration without connecting to
    Redis and so multiple processes cannot accidentally share mutable state.
    """

    settings = settings or get_settings()
    configure_logging(settings)

    app = Celery(
        "opsatlas",
        broker=settings.celery_broker_url,
        include=["app.tasks.ingestion", "app.tasks.reconciliation"],
    )

    app.conf.update(
        broker_url=settings.celery_broker_url,
        result_backend=None,
        task_ignore_result=True,
        task_store_errors_even_if_ignored=False,
        task_serializer=settings.celery_task_serializer,
        result_serializer=settings.celery_task_serializer,
        event_serializer=settings.celery_task_serializer,
        accept_content=settings.celery_accept_content,
        task_acks_late=settings.celery_task_acks_late,
        task_reject_on_worker_lost=settings.celery_task_reject_on_worker_lost,
        worker_prefetch_multiplier=settings.celery_worker_prefetch_multiplier,
        worker_concurrency=settings.celery_worker_concurrency,
        task_soft_time_limit=settings.celery_task_soft_time_limit_seconds,
        task_time_limit=settings.celery_task_time_limit_seconds,
        task_default_queue=settings.celery_task_default_queue,
        task_default_exchange=settings.celery_task_default_queue,
        task_default_routing_key=settings.celery_task_default_queue,
        # The reconciliation task is the only routed task. Ingestion publishes
        # explicitly to the default queue, so every other task keeps its existing
        # destination.
        task_routes={
            RECONCILE_TASK_NAME: {"queue": settings.celery_reconciliation_queue},
        },
        broker_connection_retry_on_startup=settings.celery_broker_connection_retry_on_startup,
        broker_transport_options={
            "visibility_timeout": settings.celery_visibility_timeout_seconds,
        },
        # Our JSON handler is already installed on the named loggers. Do not let
        # Celery replace the root logger or redirect stdout in worker processes.
        worker_hijack_root_logger=False,
        worker_redirect_stdouts=False,
    )

    beat_schedule: dict[str, dict[str, object]] = {}
    if settings.ingestion_reconcile_enabled:
        beat_schedule = {
            "opsatlas-reconcile-ingestion-jobs": {
                "task": RECONCILE_TASK_NAME,
                "schedule": float(settings.ingestion_reconcile_interval_seconds),
                "options": {"queue": settings.celery_reconciliation_queue},
            }
        }
    app.conf.beat_schedule = beat_schedule

    app.Task = OpsAtlasTask
    return app


celery_app = create_celery_app()

# Celery's CLI convention is ``-A app.celery_app:celery_app``. The alias also
# lets ``-A app.celery_app`` auto-discover the module-level application.
app = celery_app
