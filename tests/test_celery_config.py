"""Celery application configuration and serialization tests.

These tests never connect to Redis. They construct the application from explicit
settings so the assertions do not depend on a developer's local ``.env``.
"""

from __future__ import annotations

from uuid import uuid4

from app.celery_app import create_celery_app
from app.core.config import Settings


def _settings(**overrides) -> Settings:
    values = {
        "celery_broker_url": "redis://broker.example.test:6379/9",
        "celery_task_default_queue": "ingestion-test",
        "celery_reconciliation_queue": "reconciliation-test",
        "celery_task_soft_time_limit_seconds": 60,
        "celery_task_time_limit_seconds": 90,
        "celery_visibility_timeout_seconds": 120,
        "ingestion_execution_lease_seconds": 30,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_celery_uses_json_and_has_no_result_backend(monkeypatch) -> None:
    # Celery gives the CELERY_BROKER_URL environment variable precedence over
    # constructor/conf values, and CI sets it for the integration job. Remove it
    # so this test asserts the factory mapping from explicit settings.
    monkeypatch.delenv("CELERY_BROKER_URL", raising=False)
    app = create_celery_app(_settings())

    assert app.conf.broker_url == "redis://broker.example.test:6379/9"
    assert app.conf.task_serializer == "json"
    assert app.conf.result_serializer == "json"
    assert app.conf.event_serializer == "json"
    assert app.conf.accept_content == ["json"]
    assert app.conf.result_backend is None
    assert app.conf.task_ignore_result is True
    assert app.conf.task_store_errors_even_if_ignored is False


def test_celery_acknowledgement_and_worker_defaults_are_deliberate() -> None:
    app = create_celery_app(_settings())

    assert app.conf.task_acks_late is True
    assert app.conf.task_reject_on_worker_lost is True
    assert app.conf.worker_prefetch_multiplier == 1
    assert app.conf.worker_concurrency == 2
    assert app.conf.task_default_queue == "ingestion-test"
    assert app.conf.task_soft_time_limit == 60
    assert app.conf.task_time_limit == 90
    assert app.conf.broker_transport_options["visibility_timeout"] == 120
    assert app.conf.broker_connection_retry_on_startup is True


def test_celery_task_name_is_stable() -> None:
    from app.tasks.ingestion import TASK_NAME, resume_ingestion_job_task

    assert TASK_NAME == "opsatlas.ingestion.resume_job"
    assert resume_ingestion_job_task.name == TASK_NAME


def test_reconciliation_task_is_included_and_scheduled() -> None:
    from app.celery_app import RECONCILE_TASK_NAME

    app = create_celery_app(_settings())
    entry = app.conf.beat_schedule["opsatlas-reconcile-ingestion-jobs"]

    assert entry["task"] == RECONCILE_TASK_NAME
    assert entry["schedule"] == 60.0
    assert entry["options"]["queue"] == "reconciliation-test"
    assert "app.tasks.reconciliation" in app.conf.include


def test_reconciliation_task_routes_to_the_reconciliation_queue() -> None:
    from app.celery_app import RECONCILE_TASK_NAME

    app = create_celery_app(_settings())
    route = app.amqp.router.route({}, RECONCILE_TASK_NAME)

    assert app.conf.task_routes == {
        RECONCILE_TASK_NAME: {"queue": "reconciliation-test"},
    }
    assert route["queue"].name == "reconciliation-test"


def test_ingestion_task_keeps_routing_to_the_ingestion_queue() -> None:
    from app.tasks.ingestion import TASK_NAME

    app = create_celery_app(_settings())
    route = app.amqp.router.route({}, TASK_NAME)

    assert TASK_NAME == "opsatlas.ingestion.resume_job"
    assert route["queue"].name == "ingestion-test"


def test_reconciliation_queue_must_differ_from_ingestion_queue() -> None:
    import pytest

    with pytest.raises(ValueError, match="celery_reconciliation_queue"):
        _settings(celery_reconciliation_queue="ingestion-test")


def test_beat_schedule_can_be_disabled() -> None:
    app = create_celery_app(_settings(ingestion_reconcile_enabled=False))

    assert app.conf.beat_schedule == {}


def test_publish_sends_only_the_job_identifier(monkeypatch) -> None:
    from app.tasks import ingestion as task_module

    captured: dict = {}
    job_id = uuid4()

    class FakeResult:
        id = "task-123"

    def fake_send_task(name, *, args, queue):
        captured.update({"name": name, "args": args, "queue": queue})
        return FakeResult()

    monkeypatch.setattr(task_module.celery_app, "send_task", fake_send_task)
    monkeypatch.setattr(task_module, "mark_ingestion_job_dispatched", lambda *args, **kwargs: True)
    task_id = task_module.publish_ingestion_job(job_id)

    assert task_id == "task-123"
    assert captured["name"] == task_module.TASK_NAME
    assert captured["args"] == [str(job_id)]
    assert captured["queue"] == task_module.get_settings().celery_task_default_queue
    assert "content" not in repr(captured).lower()


def test_publish_failure_records_a_resumable_job_state(monkeypatch) -> None:
    import pytest

    from app.services.ingestion import IngestionQueueUnavailableError
    from app.tasks import ingestion as task_module

    job_id = uuid4()
    marked: list = []

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("broker credential and content must not leak")

    monkeypatch.setattr(task_module.celery_app, "send_task", unavailable)
    monkeypatch.setattr(
        task_module, "mark_ingestion_job_enqueue_failed", lambda value: marked.append(value)
    )

    with pytest.raises(IngestionQueueUnavailableError):
        task_module.publish_ingestion_job(job_id)

    assert marked == [job_id]


def test_settings_reject_hard_limit_below_soft_limit() -> None:
    import pytest

    with pytest.raises(ValueError, match="greater than"):
        _settings(
            celery_task_soft_time_limit_seconds=100,
            celery_task_time_limit_seconds=100,
        )


def test_settings_reject_visibility_timeout_below_hard_limit() -> None:
    import pytest

    with pytest.raises(ValueError, match="visibility"):
        _settings(
            celery_task_soft_time_limit_seconds=60,
            celery_task_time_limit_seconds=90,
            celery_visibility_timeout_seconds=90,
        )
