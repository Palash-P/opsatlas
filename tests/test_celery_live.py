"""Live Redis + Celery worker integration.

These tests exercise the parts the offline suite stubs out:

* a real Celery worker consumes a real Redis broker message and ingests into a
  disposable PostgreSQL database using a deterministic HTTP embedding service;
* the reconciler republishes an orphaned durable job through the same broker and
  the worker completes it;
* a worker pointed at an empty database refuses to start instead of consuming
  work against an incompatible schema.

They require ``OPSATLAS_LIVE_TESTS=1``. PostgreSQL and Redis availability are
checked separately; ``OPSATLAS_REQUIRE_POSTGRES=1`` and
``OPSATLAS_REQUIRE_REDIS=1`` turn an unavailable dependency into a failure (CI
sets both). Every database is uniquely named ``opsatlas_test_<uuid>`` and every
queue/key is unique to the test, so nothing shared is modified.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from contextlib import contextmanager
from pathlib import Path
from uuid import UUID

import pytest
from redis import Redis
from sqlalchemy import text

from app.services import ingestion as ingestion_module
from app.services import reconciliation as reconciliation_module
from app.services.ingestion import submit_ingestion_job
from app.tasks import ingestion as ingestion_tasks
from tests.conftest import principal_for

ROOT = Path(__file__).resolve().parents[1]
LIVE = os.environ.get("OPSATLAS_LIVE_TESTS") == "1"
REDIS_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1")
FAKE_MODEL = "opsatlas-fake-all-MiniLM-L6-v2"

pytestmark = pytest.mark.skipif(
    not LIVE,
    reason="OPSATLAS_LIVE_TESTS is not 1",
)


def _required(name: str) -> bool:
    return os.getenv(name, "").lower() in {"1", "true", "yes"}


def _redis_or_skip() -> None:
    client = Redis.from_url(REDIS_URL, socket_connect_timeout=0.5, socket_timeout=0.5)
    try:
        client.ping()
    except Exception as exc:  # noqa: BLE001 - absence skips, requirement fails
        reason = f"Redis is not available: {type(exc).__name__}"
        if _required("OPSATLAS_REQUIRE_REDIS"):
            pytest.fail(f"Redis is required but unavailable: {reason}", pytrace=False)
        pytest.skip(reason)
    finally:
        client.close()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for(predicate, *, timeout: float, interval: float = 0.2) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


@contextmanager
def _fake_embedding_server(model: str):
    port = _free_port()
    process = subprocess.Popen(
        [
            sys.executable,
            str(ROOT / "scripts" / "ops" / "fake_embedding_service.py"),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--model",
            model,
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    url = f"http://127.0.0.1:{port}"

    def healthy() -> bool:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=0.5) as response:
                return response.status == 200
        except (urllib.error.URLError, TimeoutError, OSError):
            return False

    try:
        if not _wait_for(healthy, timeout=20):
            output = ""
            if process.poll() is not None and process.stdout is not None:
                output = process.stdout.read()
            pytest.fail(
                "fake embedding service did not become healthy\n" + output[-2000:],
                pytrace=False,
            )
        yield url
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def _worker_log_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


@contextmanager
def _celery_worker(
    *,
    database_url: str,
    queue: str,
    embedding_url: str,
    model: str,
    log_path: Path,
    wait_ready: bool = True,
):
    env = os.environ.copy()
    env.update(
        {
            "APP_ENV": "local",
            "DATABASE_URL": database_url,
            "CELERY_BROKER_URL": REDIS_URL,
            "CELERY_TASK_DEFAULT_QUEUE": queue,
            "CELERY_RECONCILIATION_QUEUE": f"{queue}-reconcile",
            "CELERY_WORKER_CONCURRENCY": "1",
            "EMBEDDING_URL": embedding_url,
            "EMBEDDING_MODEL": model,
            "EMBEDDING_DIMENSIONS": "384",
            "EMBEDDING_MAX_RETRIES": "0",
            "EMBEDDING_RETRY_BACKOFF_SECONDS": "0",
            "INGESTION_RECONCILE_ENABLED": "false",
            "INGESTION_TASK_RETRY_BACKOFF_SECONDS": "0.1",
            "INGESTION_TASK_RETRY_BACKOFF_MAX_SECONDS": "0.5",
            "RATE_LIMIT_BACKEND": "memory",
            "AUTH_TOKENS": "{}",
            "LOG_LEVEL": "INFO",
        }
    )
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "celery",
                "-A",
                "app.celery_app:celery_app",
                "worker",
                "--loglevel=INFO",
                "--pool=solo",
                "--concurrency=1",
                f"--queues={queue}",
                "--hostname=live-test@%h",
                "--without-gossip",
                "--without-mingle",
                "--without-heartbeat",
            ],
            cwd=ROOT,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            if wait_ready:
                def ready() -> bool:
                    if process.poll() is not None:
                        return True
                    return "ready." in _worker_log_text(log_path)

                if not _wait_for(ready, timeout=60):
                    pytest.fail(
                        "Celery worker did not become ready\n"
                        + _worker_log_text(log_path)[-4000:],
                        pytrace=False,
                    )
                if process.poll() is not None:
                    pytest.fail(
                        "Celery worker exited during startup\n"
                        + _worker_log_text(log_path)[-4000:],
                        pytrace=False,
                    )
            yield process, log_path
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=20)


def _insert_tenant(engine) -> UUID:
    tenant_id = uuid.uuid4()
    with engine.begin() as connection:
        connection.execute(
            text("INSERT INTO tenants (id, name) VALUES (:id, :name)"),
            {"id": tenant_id, "name": f"live-celery-{tenant_id}"},
        )
    return tenant_id


def _job_row(engine, job_id: UUID) -> dict:
    with engine.connect() as connection:
        row = (
            connection.execute(
                text(
                    """
                    SELECT status, document_id, version_id, attempts,
                           error_code, execution_token
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


def _wait_for_job(engine, job_id: UUID, *, timeout: float = 90) -> dict:
    state = {"row": None}

    def finished() -> bool:
        row = _job_row(engine, job_id)
        state["row"] = row
        return row["status"] in {"succeeded", "failed", "dead_letter"}

    if not _wait_for(finished, timeout=timeout, interval=0.25):
        pytest.fail(
            f"job {job_id} did not reach a terminal state; last={state['row']}",
            pytrace=False,
        )
    return state["row"]


def _chunk_counts(engine, version_id: UUID) -> tuple[int, int, int]:
    with engine.connect() as connection:
        row = connection.execute(
            text(
                """
                SELECT count(*) AS total,
                       count(embedding) AS embedded,
                       count(embedded_at) AS stamped
                FROM chunks
                WHERE document_version_id = :version_id
                """
            ),
            {"version_id": version_id},
        ).one()
    return int(row[0]), int(row[1]), int(row[2])


def _patch_test_database(monkeypatch, engine) -> None:
    monkeypatch.setattr(ingestion_module, "get_engine", lambda: engine)
    monkeypatch.setattr(reconciliation_module, "get_engine", lambda: engine)


def _publisher_settings(queue: str):
    from app.core.config import Settings

    return Settings(
        _env_file=None,
        celery_task_default_queue=queue,
        ingestion_reconcile_dispatch_lease_seconds=300,
    )


def test_real_worker_ingests_a_published_job(
    disposable_postgres, migrated_postgres, monkeypatch, tmp_path
) -> None:
    _redis_or_skip()
    engine = migrated_postgres.engine
    _patch_test_database(monkeypatch, engine)
    tenant_id = _insert_tenant(engine)
    queue = f"opsatlas-live-{uuid.uuid4().hex[:12]}"
    monkeypatch.setattr(
        ingestion_tasks, "get_settings", lambda: _publisher_settings(queue)
    )

    with _fake_embedding_server(FAKE_MODEL) as embedding_url, _celery_worker(
        database_url=migrated_postgres.url,
        queue=queue,
        embedding_url=embedding_url,
        model=FAKE_MODEL,
        log_path=tmp_path / "worker.log",
    ):
        content = "\n".join(
            f"PAY-503 live step {index}: check dependency {index} before escalation."
            for index in range(80)
        )
        submission = submit_ingestion_job(
            principal=principal_for(tenant_id),
            tenant_id=tenant_id,
            source_key=f"live/worker-{uuid.uuid4().hex}.md",
            title="Live worker runbook",
            source_type="runbook",
            content=content,
        )
        assert submission.needs_publish is True
        ingestion_tasks.publish_ingestion_job(
            submission.job_id, tenant_id=tenant_id
        )

        row = _wait_for_job(engine, submission.job_id)

    assert row["status"] == "succeeded", row
    assert row["error_code"] is None
    assert row["version_id"] == submission.version_id
    assert row["execution_token"] is None
    total, embedded, stamped = _chunk_counts(engine, submission.version_id)
    assert total > 0
    assert embedded == stamped == total
    with engine.connect() as connection:
        document = (
            connection.execute(
                text(
                    """
                    SELECT current_version_id, last_activation_generation
                    FROM documents
                    WHERE id = :document_id
                    """
                ),
                {"document_id": submission.document_id},
            )
            .mappings()
            .one()
        )
    assert document["current_version_id"] == submission.version_id
    assert document["last_activation_generation"] is not None


def test_reconciler_republishes_an_orphaned_job_to_a_real_worker(
    disposable_postgres, migrated_postgres, monkeypatch, tmp_path
) -> None:
    _redis_or_skip()
    engine = migrated_postgres.engine
    _patch_test_database(monkeypatch, engine)
    tenant_id = _insert_tenant(engine)
    queue = f"opsatlas-live-reconcile-{uuid.uuid4().hex[:12]}"
    monkeypatch.setattr(
        ingestion_tasks, "get_settings", lambda: _publisher_settings(queue)
    )

    submission = submit_ingestion_job(
        principal=principal_for(tenant_id),
        tenant_id=tenant_id,
        source_key=f"live/orphan-{uuid.uuid4().hex}.md",
        title="Orphaned live runbook",
        source_type="runbook",
        content="PAY-504 orphan recovery step: verify the queue before escalation.",
    )
    assert submission.needs_publish is True

    with _fake_embedding_server(FAKE_MODEL) as embedding_url, _celery_worker(
        database_url=migrated_postgres.url,
        queue=queue,
        embedding_url=embedding_url,
        model=FAKE_MODEL,
        log_path=tmp_path / "worker.log",
    ):
        report = reconciliation_module.run_reconciliation(batch_size=10)

        assert report.locked is False
        assert report.dispatched == 1, report.as_dict()
        assert report.failures == 0

        row = _wait_for_job(engine, submission.job_id)

    assert row["status"] == "succeeded", row
    assert row["error_code"] is None


def test_worker_refuses_to_start_against_an_empty_database(
    disposable_postgres, tmp_path
) -> None:
    _redis_or_skip()
    queue = f"opsatlas-live-empty-{uuid.uuid4().hex[:12]}"
    log_path = tmp_path / "worker-empty.log"

    with _celery_worker(
        database_url=disposable_postgres.url,
        queue=queue,
        embedding_url="http://127.0.0.1:9",
        model=FAKE_MODEL,
        log_path=log_path,
        wait_ready=False,
    ) as (process, path):
        assert _wait_for(lambda: process.poll() is not None, timeout=60), (
            "worker did not refuse an incompatible schema\n"
            + _worker_log_text(path)[-4000:]
        )

    assert process.returncode != 0
    log = _worker_log_text(log_path)
    assert "schema_validation" in log
    assert '"outcome": "failed"' in log
    assert "Database schema is not ready" in log
