"""Deterministic retry and orchestration tests for the ingestion Celery task.

The task body is exercised with a fake bound task, so no broker, worker, or
PostgreSQL connection is required. The real-PostgreSQL behavior is covered in
``tests/test_ingestion_worker.py``.
"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from app.core.config import Settings
from app.db.embedding_schema import EmbeddingSchemaError
from app.services.embedding_client import (
    EmbeddingServiceError,
    EmbeddingUnavailableError,
    EmbeddingValidationError,
)
from app.services.ingestion import (
    IngestionExecutionLeaseLostError,
    IngestionExecutionResult,
    IngestionJobBusyError,
    IngestionJobNotFoundError,
    IngestionJobStateError,
    IngestionJobTerminalError,
)
from app.tasks import ingestion as task_module


@pytest.fixture
def worker_settings(monkeypatch):
    settings = Settings(
        _env_file=None,
        ingestion_task_max_retries=3,
        ingestion_task_retry_backoff_seconds=5.0,
        ingestion_task_retry_backoff_max_seconds=120.0,
        ingestion_task_unexpected_max_retries=1,
        ingestion_execution_lease_seconds=30,
    )
    monkeypatch.setattr(task_module, "get_settings", lambda: settings)
    return settings


class FakeRetry(Exception):
    def __init__(self, kwargs: dict) -> None:
        super().__init__("retry requested")
        self.kwargs = kwargs


class FakeTask:
    def __init__(self, retries: int = 0) -> None:
        self.request = SimpleNamespace(retries=retries, id=f"task-{retries}")

    def retry(self, **kwargs):
        raise FakeRetry(kwargs)


def _snapshot(*, status: str = "failed", attempts: int = 1, error_code=None) -> dict:
    return {
        "status": status,
        "attempts": attempts,
        "error_code": error_code,
        "tenant_id": None,
    }


def _execution_result(job_id: UUID, *, outcome: str = "succeeded") -> IngestionExecutionResult:
    return IngestionExecutionResult(
        job_id=job_id,
        tenant_id=UUID("11111111-1111-4111-8111-111111111111"),
        status="succeeded",
        attempts=1,
        document_id=uuid4(),
        version_id=uuid4(),
        version_number=1,
        content_hash="a" * 64,
        chunk_count=1,
        is_current=True,
        outcome=outcome,
    )


def test_task_signature_accepts_only_a_job_identifier() -> None:
    import inspect

    parameters = list(inspect.signature(task_module.resume_ingestion_job_task.run).parameters)
    assert parameters == ["job_id"]


def test_task_invokes_the_service_with_the_job_uuid(monkeypatch, worker_settings) -> None:
    job_id = uuid4()
    calls: list[UUID] = []

    def fake_resume(value: UUID) -> IngestionExecutionResult:
        calls.append(value)
        return _execution_result(value)

    monkeypatch.setattr(task_module, "resume_ingestion_job", fake_resume)
    monkeypatch.setattr(task_module, "get_ingestion_job_snapshot", lambda _job: _snapshot())

    result = task_module.execute_ingestion_task(FakeTask(), str(job_id))

    assert calls == [job_id]
    assert result["job_id"] == str(job_id)
    assert result["status"] == "succeeded"
    assert result["outcome"] == "succeeded"


def test_transient_embedding_failure_retries_with_backoff(monkeypatch, worker_settings) -> None:
    job_id = uuid4()
    monkeypatch.setattr(
        task_module,
        "resume_ingestion_job",
        lambda _job: (_ for _ in ()).throw(EmbeddingUnavailableError("timeout")),
    )
    monkeypatch.setattr(
        task_module,
        "get_ingestion_job_snapshot",
        lambda _job: _snapshot(error_code="embedding_service_unavailable"),
    )

    with pytest.raises(FakeRetry) as excinfo:
        task_module.execute_ingestion_task(FakeTask(retries=0), str(job_id))

    assert excinfo.value.kwargs["countdown"] == 5.0
    assert excinfo.value.kwargs["max_retries"] == 3


def test_transient_retries_are_bounded_and_become_terminal(monkeypatch, worker_settings) -> None:
    job_id = uuid4()
    marked: list[tuple[UUID, str]] = []
    monkeypatch.setattr(
        task_module,
        "resume_ingestion_job",
        lambda _job: (_ for _ in ()).throw(EmbeddingUnavailableError("timeout")),
    )
    monkeypatch.setattr(
        task_module,
        "get_ingestion_job_snapshot",
        lambda _job: _snapshot(error_code="embedding_service_unavailable"),
    )
    monkeypatch.setattr(
        task_module,
        "mark_ingestion_job_retries_exhausted",
        lambda value, *, error_code, expected_attempts=None, expected_error_code=None: (
            marked.append((value, error_code))
        ),
    )

    result = task_module.execute_ingestion_task(FakeTask(retries=3), str(job_id))

    assert result["outcome"] == "exhausted"
    assert marked == [(job_id, "embedding_service_unavailable")]


@pytest.mark.parametrize(
    ("error", "category"),
    [
        (EmbeddingValidationError("bad response"), "embedding_invalid_response"),
        (EmbeddingSchemaError("vector(768)"), "embedding_schema_mismatch"),
        (EmbeddingServiceError("provider rejected request"), "embedding_failed"),
        (IngestionJobNotFoundError(), "job_not_found"),
        (IngestionJobStateError("no target"), "ingestion_job_invalid"),
    ],
)
def test_permanent_failures_are_not_retried(monkeypatch, worker_settings, error, category) -> None:
    job_id = uuid4()
    marked: list = []
    monkeypatch.setattr(
        task_module,
        "resume_ingestion_job",
        lambda _job: (_ for _ in ()).throw(error),
    )
    monkeypatch.setattr(task_module, "get_ingestion_job_snapshot", lambda _job: _snapshot())
    monkeypatch.setattr(
        task_module,
        "mark_ingestion_job_retries_exhausted",
        lambda *args, **kwargs: marked.append(args),
    )

    result = task_module.execute_ingestion_task(FakeTask(retries=0), str(job_id))

    assert result["outcome"] == "failed"
    assert result["error_category"] == category
    assert marked == []


def test_unexpected_failure_has_a_small_bounded_retry(monkeypatch, worker_settings) -> None:
    job_id = uuid4()
    monkeypatch.setattr(
        task_module,
        "resume_ingestion_job",
        lambda _job: (_ for _ in ()).throw(RuntimeError("secret-looking text")),
    )
    monkeypatch.setattr(task_module, "get_ingestion_job_snapshot", lambda _job: _snapshot())

    with pytest.raises(FakeRetry) as excinfo:
        task_module.execute_ingestion_task(FakeTask(retries=0), str(job_id))

    assert excinfo.value.kwargs["max_retries"] == 1
    assert "secret-looking" not in repr(excinfo.value.kwargs)


def test_exhausted_unexpected_failure_becomes_terminal(monkeypatch, worker_settings) -> None:
    job_id = uuid4()
    marked: list = []
    monkeypatch.setattr(
        task_module,
        "resume_ingestion_job",
        lambda _job: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(
        task_module,
        "get_ingestion_job_snapshot",
        lambda _job: _snapshot(error_code="ingestion_failed"),
    )
    monkeypatch.setattr(
        task_module,
        "mark_ingestion_job_retries_exhausted",
        lambda value, *, error_code, expected_attempts=None, expected_error_code=None: (
            marked.append((value, error_code, expected_attempts, expected_error_code))
        ),
    )

    result = task_module.execute_ingestion_task(FakeTask(retries=1), str(job_id))

    assert result["outcome"] == "exhausted"
    assert marked == [(job_id, "task_failed", 1, "ingestion_failed")]


def test_terminal_job_delivery_is_skipped(monkeypatch, worker_settings) -> None:
    job_id = uuid4()
    monkeypatch.setattr(
        task_module,
        "resume_ingestion_job",
        lambda _job: (_ for _ in ()).throw(IngestionJobTerminalError("terminal")),
    )
    monkeypatch.setattr(
        task_module,
        "get_ingestion_job_snapshot",
        lambda _job: _snapshot(status="dead_letter", error_code="embedding_service_unavailable"),
    )

    result = task_module.execute_ingestion_task(FakeTask(), str(job_id))

    assert result["outcome"] == "skipped"
    assert result["status"] == "dead_letter"


@pytest.mark.parametrize(
    ("error", "category"),
    [
        (IngestionJobBusyError("live lease"), "job_busy"),
        (IngestionExecutionLeaseLostError("lease lost"), "execution_lease_lost"),
    ],
)
def test_busy_or_fenced_out_delivery_is_skipped_without_retry(
    monkeypatch, worker_settings, error, category
) -> None:
    job_id = uuid4()
    monkeypatch.setattr(
        task_module,
        "resume_ingestion_job",
        lambda _job: (_ for _ in ()).throw(error),
    )
    monkeypatch.setattr(
        task_module,
        "get_ingestion_job_snapshot",
        lambda _job: _snapshot(status="running", error_code=None),
    )

    result = task_module.execute_ingestion_task(FakeTask(), str(job_id))

    assert result["outcome"] == "skipped"
    assert result["status"] == "running"
    assert result["error_category"] == category


def test_succeeded_job_delivery_reports_skipped_service_result(
    monkeypatch, worker_settings
) -> None:
    job_id = uuid4()
    monkeypatch.setattr(
        task_module,
        "resume_ingestion_job",
        lambda value: _execution_result(value, outcome="skipped"),
    )
    monkeypatch.setattr(task_module, "get_ingestion_job_snapshot", lambda _job: _snapshot())

    result = task_module.execute_ingestion_task(FakeTask(), str(job_id))

    assert result["outcome"] == "skipped"
    assert result["status"] == "succeeded"


def test_invalid_job_identifier_is_permanent_and_never_hits_the_database(
    monkeypatch, worker_settings
) -> None:
    called = False

    def fake_resume(_job):
        nonlocal called
        called = True
        raise AssertionError("must not run")

    monkeypatch.setattr(task_module, "resume_ingestion_job", fake_resume)

    result = task_module.execute_ingestion_task(FakeTask(), "not-a-uuid")

    assert result["outcome"] == "failed"
    assert result["error_category"] == "invalid_job_id"
    assert called is False


def test_retry_countdown_is_capped(worker_settings) -> None:
    assert task_module.retry_countdown(0, worker_settings) == 5.0
    assert task_module.retry_countdown(1, worker_settings) == 10.0
    assert task_module.retry_countdown(10, worker_settings) == 120.0
