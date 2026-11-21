"""Live API ingestion over real HTTP embeddings and a disposable database.

This is the API-level counterpart to the Celery live tests: a real authenticated
request enters ``/api/v1/documents/ingest``, the application calls a real local
HTTP embedding service (the deterministic fake), writes to a unique disposable
PostgreSQL database, and the document becomes searchable through the API.

Requires ``OPSATLAS_LIVE_TESTS=1``. The fake service is started from
``scripts/ops/fake_embedding_service.py``; no paid provider or model download is
involved.
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
from fastapi.testclient import TestClient
from sqlalchemy import text

from app.core import auth as auth_module
from app.core.config import Settings
from app.db import embedding_schema as embedding_schema_module
from app.main import app
from app.services import embedding_client
from app.services import ingestion as ingestion_module
from app.services import retrieval as retrieval_module

ROOT = Path(__file__).resolve().parents[1]
LIVE = os.environ.get("OPSATLAS_LIVE_TESTS") == "1"
FAKE_MODEL = "opsatlas-fake-all-MiniLM-L6-v2"
TOKEN = "live-api-ingestion-development-token"
TENANT_ID = UUID("55555555-5555-4555-8555-555555555555")

pytestmark = pytest.mark.skipif(
    not LIVE,
    reason="OPSATLAS_LIVE_TESTS is not 1",
)

client = TestClient(app)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


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

    deadline = time.monotonic() + 20
    try:
        while time.monotonic() < deadline:
            if healthy():
                break
            time.sleep(0.1)
        else:
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


def _insert_tenant(engine) -> None:
    with engine.begin() as connection:
        connection.execute(
            text("INSERT INTO tenants (id, name) VALUES (:id, :name)"),
            {"id": TENANT_ID, "name": f"live-api-{TENANT_ID}"},
        )


def _settings(embedding_url: str) -> Settings:
    return Settings(
        _env_file=None,
        app_env="local",
        embedding_url=embedding_url,
        embedding_model=FAKE_MODEL,
        embedding_dimensions=384,
        embedding_timeout_seconds=5,
        embedding_max_retries=0,
        embedding_retry_backoff_seconds=0,
        auth_tokens={
            TOKEN: {
                "tenant_id": str(TENANT_ID),
                "subject": "live-api-user",
                "roles": ["member", "document_writer"],
                "groups": [],
            }
        },
        auth_log_tenant_salt="live-api-test-salt",
        rate_limit_backend="memory",
    )


def test_live_api_ingest_and_hybrid_search(
    migrated_postgres, monkeypatch, real_auth
) -> None:
    engine = migrated_postgres.engine
    _insert_tenant(engine)
    monkeypatch.setattr(ingestion_module, "get_engine", lambda: engine)
    monkeypatch.setattr(retrieval_module, "get_engine", lambda: engine)
    monkeypatch.setattr(embedding_schema_module, "get_engine", lambda: engine)

    canary = f"zqlive{uuid.uuid4().hex[:12]}"
    with _fake_embedding_server(FAKE_MODEL) as embedding_url:
        settings = _settings(embedding_url)
        monkeypatch.setattr(auth_module, "get_settings", lambda: settings)
        monkeypatch.setattr(embedding_client, "get_settings", lambda: settings)
        monkeypatch.setattr(ingestion_module, "get_settings", lambda: settings)

        response = client.post(
            "/api/v1/documents/ingest",
            headers={"Authorization": f"Bearer {TOKEN}"},
            json={
                "tenant_id": str(TENANT_ID),
                "source_key": f"live/api-{canary}.md",
                "title": "Live API runbook",
                "source_type": "runbook",
                "content": (
                    f"{canary} PAY-505 live API ingestion check. "
                    "Verify the dependency and escalate to the on-call engineer."
                ),
            },
        )
        assert response.status_code == 201, response.text
        body = response.json()
        assert body["version_number"] == 1
        assert body["deduplicated"] is False
        assert body["is_current"] is True

        search = client.post(
            "/api/v1/search/hybrid",
            headers={"Authorization": f"Bearer {TOKEN}"},
            json={
                "tenant_id": str(TENANT_ID),
                "query": canary,
                "limit": 10,
            },
        )

    assert search.status_code == 200, search.text
    results = search.json()["results"]
    assert any(result["source_key"] == f"live/api-{canary}.md" for result in results)
    assert any(canary in result["content"] for result in results)


def test_live_api_ingest_requires_a_real_bearer_token(
    migrated_postgres, monkeypatch, real_auth
) -> None:
    engine = migrated_postgres.engine
    monkeypatch.setattr(ingestion_module, "get_engine", lambda: engine)
    settings = _settings("http://127.0.0.1:9")
    monkeypatch.setattr(auth_module, "get_settings", lambda: settings)

    response = client.post(
        "/api/v1/documents/ingest",
        json={
            "tenant_id": str(TENANT_ID),
            "source_key": "live/unauthorized.md",
            "title": "Unauthorized",
            "content": "must not be ingested",
        },
    )

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert TOKEN not in response.text


def test_live_api_rejects_a_body_tenant_mismatch(
    migrated_postgres, monkeypatch, real_auth
) -> None:
    engine = migrated_postgres.engine
    _insert_tenant(engine)
    monkeypatch.setattr(ingestion_module, "get_engine", lambda: engine)
    settings = _settings("http://127.0.0.1:9")
    monkeypatch.setattr(auth_module, "get_settings", lambda: settings)

    response = client.post(
        "/api/v1/documents/ingest",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json={
            "tenant_id": "66666666-6666-4666-8666-666666666666",
            "source_key": "live/mismatch.md",
            "title": "Mismatch",
            "content": "must not be ingested",
        },
    )

    assert response.status_code == 403
    assert response.json() == {"detail": "Forbidden"}
