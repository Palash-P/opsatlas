"""Orchestrate the Phase E operational validation on the local stack.

The script is intentionally conservative:

- one uvicorn worker at a time;
- bounded request counts and concurrency;
- a temporary tenant B used for isolation checks and deleted in a ``finally``;
- every stopped container is restarted and readiness is re-checked;
- no schema migration and no writes to tenant A data.

It writes only measurements and redacted configuration to the output directory.
Bearer tokens, raw tenant UUIDs for the temporary tenant, and document content
are never written to the result artifacts.
"""

from __future__ import annotations

import argparse
import asyncio  # noqa: F401 - kept for documentation of the async load harness
import concurrent.futures
import contextlib
import json
import os
import platform
import secrets
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
OPS_DIR = Path(__file__).resolve().parent
if str(OPS_DIR) not in sys.path:
    sys.path.insert(0, str(OPS_DIR))

import httpx
from live_failure_injection import (
    docker_container_status,
    run_failure_injection,
    wait_for_container,
)
from load_test import _summary
from resource_sampler import ResourceSampler
from sqlalchemy import create_engine, text
from validate_logs import index_events, load_events, validate

LOAD_TEST = OPS_DIR / "load_test.py"
FAKE_PROVIDER = OPS_DIR / "fake_llm_provider.py"
EMBEDDING_SCRIPT = REPO_ROOT / "scripts" / "embedding_service.py"

POSTGRES_CONTAINER = "opsatlas-v1-postgres-1"
REDIS_CONTAINER = "opsatlas-v1-redis-1"
EMBEDDING_URL = "http://127.0.0.1:8091"
FAKE_PROVIDER_URL = "http://127.0.0.1:8099"

TENANT_A_ID = UUID("96e52951-46e1-4547-9103-b82de97e6822")
TOKEN_A = "val-a-" + secrets.token_urlsafe(24)
TOKEN_B1 = "val-b1-" + secrets.token_urlsafe(24)
TOKEN_B2 = "val-b2-" + secrets.token_urlsafe(24)
FAKE_API_KEY = "validation-fake-key-not-a-real-secret"

QUESTIONS = [
    "When PAY-503 is raised, which initial checks should happen before escalation?",
    "What should I check before rolling back the payment API?",
    "How do I recover from Redis failover problems?",
    "What does the database pool postmortem say about connection exhaustion?",
    "How do I diagnose Kafka consumer lag?",
]


def now() -> str:
    return datetime.now(UTC).isoformat()


def run_command(
    command: list[str],
    *,
    timeout: float = 60.0,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(cwd) if cwd else None,
        check=False,
    )


def wait_for_url(
    url: str,
    *,
    predicate,
    timeout: float = 60.0,
    interval: float = 0.5,
) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = httpx.get(url, timeout=5.0)
            if predicate(response.status_code, response.text):
                return True
        except httpx.HTTPError:
            pass
        time.sleep(interval)
    return False


def find_embedding_pid() -> int | None:
    result = run_command(
        [
            "powershell",
            "-NoLogo",
            "-NoProfile",
            "-Command",
            (
                "Get-CimInstance Win32_Process | "
                "Where-Object { $_.CommandLine -like '*embedding_service.py*' } | "
                "Select-Object -First 1 -ExpandProperty ProcessId"
            ),
        ],
        timeout=20,
    )
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return int(value) if value.isdigit() else None


def ensure_containers() -> dict[str, str]:
    result = run_command(
        ["docker", "compose", "up", "-d", "--no-recreate", "postgres", "redis"],
        timeout=120,
        cwd=REPO_ROOT,
    )
    status: dict[str, str] = {"compose_exit": str(result.returncode)}
    for container in (POSTGRES_CONTAINER, REDIS_CONTAINER):
        if not wait_for_container(container, timeout=90):
            raise RuntimeError(f"container did not become healthy: {container}")
        status[container] = docker_container_status(container) or "unknown"
    if result.returncode != 0:
        status["compose_stderr"] = result.stderr[-500:]
    return status


def start_embedding(log_path: Path) -> tuple[int | None, Any]:
    handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, str(EMBEDDING_SCRIPT)],
        cwd=str(REPO_ROOT),
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    ready = wait_for_url(
        f"{EMBEDDING_URL}/health",
        predicate=lambda status, _: status == 200,
        timeout=180.0,
    )
    if not ready:
        process.terminate()
        raise RuntimeError("embedding service did not become healthy")
    return process.pid, handle


def ensure_embedding(log_path: Path) -> tuple[int | None, bool, Any | None]:
    if wait_for_url(
        f"{EMBEDDING_URL}/health",
        predicate=lambda status, _: status == 200,
        timeout=2.0,
    ):
        return find_embedding_pid(), False, None
    pid, handle = start_embedding(log_path)
    return pid, True, handle


def collect_environment() -> dict[str, Any]:
    from importlib.metadata import PackageNotFoundError, version

    packages = {}
    for name in (
        "fastapi",
        "uvicorn",
        "sqlalchemy",
        "psycopg",
        "redis",
        "pydantic",
        "pytest",
        "ruff",
        "httpx",
        "sentence-transformers",
        "torch",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None

    docker = run_command(["docker", "version", "--format", "{{.Server.Version}}"], timeout=20)
    compose = run_command(["docker", "compose", "version", "--short"], timeout=20)
    psql = run_command(
        [
            "docker",
            "exec",
            POSTGRES_CONTAINER,
            "psql",
            "-U",
            "opsatlas",
            "-d",
            "opsatlas",
            "-At",
            "-c",
            "SELECT version();",
            "-c",
            "SHOW max_connections;",
            "-c",
            "SHOW shared_buffers;",
            "-c",
            "SHOW work_mem;",
            "-c",
            "SHOW effective_cache_size;",
        ],
        timeout=30,
    )
    redis = run_command(
        [
            "docker",
            "exec",
            REDIS_CONTAINER,
            "redis-cli",
            "info",
            "server",
        ],
        timeout=30,
    )
    total_memory = None
    try:
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        total_memory = status.ullTotalPhys
    except Exception:  # noqa: BLE001 - environment detail is best effort
        total_memory = None

    from app.core.config import get_settings

    settings = get_settings()
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version,
        "cpu_count": os.cpu_count(),
        "total_memory_bytes": total_memory,
        "packages": packages,
        "docker_server": docker.stdout.strip() if docker.returncode == 0 else None,
        "docker_compose": compose.stdout.strip() if compose.returncode == 0 else None,
        "postgres": psql.stdout.strip().splitlines() if psql.returncode == 0 else psql.stderr,
        "redis": redis.stdout.strip().splitlines() if redis.returncode == 0 else redis.stderr,
        "settings": {
            "rate_limit_per_minute": settings.rate_limit_per_minute,
            "rate_limit_window_seconds": settings.rate_limit_window_seconds,
            "max_concurrent_requests": settings.max_concurrent_requests,
            "concurrency_acquire_timeout_seconds": settings.concurrency_acquire_timeout_seconds,
            "db_pool_size": settings.db_pool_size,
            "db_max_overflow": settings.db_max_overflow,
            "db_pool_timeout_seconds": settings.db_pool_timeout_seconds,
            "db_pool_recycle_seconds": settings.db_pool_recycle_seconds,
            "db_connect_timeout_seconds": settings.db_connect_timeout_seconds,
            "db_statement_timeout_ms": settings.db_statement_timeout_ms,
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
            "llm_temperature": settings.llm_temperature,
            "llm_max_output_tokens": settings.llm_max_output_tokens,
            "llm_timeout_seconds": settings.llm_timeout_seconds,
            "llm_max_retries": settings.llm_max_retries,
            "embedding_url": settings.embedding_url,
            "embedding_timeout_seconds": settings.embedding_timeout_seconds,
            "embedding_max_retries": settings.embedding_max_retries,
            "answer_context_token_budget": settings.answer_context_token_budget,
            "answer_max_evidence": settings.answer_max_evidence,
        },
    }


def db_counts() -> dict[str, int]:
    from app.db.connections import get_engine

    with get_engine().connect() as connection:
        tenants = connection.execute(text("SELECT count(*) FROM tenants")).scalar_one()
        documents = connection.execute(text("SELECT count(*) FROM documents")).scalar_one()
    return {"tenants": int(tenants), "documents": int(documents)}


@dataclass
class TempTenant:
    tenant_id: UUID
    canary_b: str
    restricted_canary: str
    restricted_document_id: UUID
    created: bool = True


def create_temp_tenant() -> TempTenant:
    from app.db.connections import get_engine
    from app.services.ingestion import ingest_document

    tenant_b = uuid4()
    canary_b = "zqx" + secrets.token_hex(10)
    restricted_canary = "zqx" + secrets.token_hex(10)
    engine = get_engine()

    with engine.begin() as connection:
        connection.execute(
            text("INSERT INTO tenants (id, name) VALUES (:id, :name)"),
            {"id": tenant_b, "name": f"phase-e-validation-{tenant_b}"},
        )

    ingest_document(
        tenant_id=tenant_b,
        source_key="validation/canary-b.md",
        title="Tenant B validation canary",
        source_type="runbook",
        content=f"{canary_b} is visible only to tenant B members.",
    )
    restricted = ingest_document(
        tenant_id=tenant_b,
        source_key="validation/restricted-b.md",
        title="Tenant B restricted validation canary",
        source_type="runbook",
        content=f"{restricted_canary} requires the validation-allowed group.",
    )

    with engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO document_acl (
                    document_id, principal_type, principal_id, permission
                )
                VALUES (
                    :document_id, 'group', 'validation-allowed', 'read'
                )
                """
            ),
            {"document_id": restricted["document_id"]},
        )

    return TempTenant(
        tenant_id=tenant_b,
        canary_b=canary_b,
        restricted_canary=restricted_canary,
        restricted_document_id=restricted["document_id"],
    )


def cleanup_temp_tenant(temp: TempTenant) -> None:
    from app.db.connections import get_engine

    engine = get_engine()
    with engine.begin() as connection:
        connection.execute(
            text("DELETE FROM tenants WHERE id = :id"),
            {"id": temp.tenant_id},
        )


def auth_tokens_json(temp: TempTenant) -> str:
    return json.dumps(
        {
            TOKEN_A: {
                "tenant_id": str(TENANT_A_ID),
                "subject": "phase-e-a",
                "roles": ["member"],
                "groups": ["on-call"],
            },
            TOKEN_B1: {
                "tenant_id": str(temp.tenant_id),
                "subject": "phase-e-b1",
                "roles": ["member"],
                "groups": ["validation-allowed"],
            },
            TOKEN_B2: {
                "tenant_id": str(temp.tenant_id),
                "subject": "phase-e-b2",
                "roles": ["member"],
                "groups": ["validation-other"],
            },
        },
        separators=(",", ":"),
    )


@dataclass
class ApiInstance:
    name: str
    port: int
    process: subprocess.Popen
    log_path: Path
    handle: Any
    base_url: str = field(init=False)

    def __post_init__(self) -> None:
        self.base_url = f"http://127.0.0.1:{self.port}"


def start_api(
    *,
    name: str,
    port: int,
    out_dir: Path,
    overrides: dict[str, str],
    temp: TempTenant,
) -> ApiInstance:
    log_path = out_dir / f"api-{name}.log"
    handle = log_path.open("w", encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "APP_ENV": "local",
            "AUTH_TOKENS": auth_tokens_json(temp),
            "DATABASE_URL": os.environ.get(
                "DATABASE_URL",
                "postgresql+psycopg://opsatlas:opsatlas_dev_password@localhost:5433/opsatlas",
            ),
            "REDIS_URL": os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
            "EMBEDDING_URL": os.environ.get("EMBEDDING_URL", EMBEDDING_URL),
            "RATE_LIMIT_PER_MINUTE": "240",
            "RATE_LIMIT_WINDOW_SECONDS": "60",
            "MAX_CONCURRENT_REQUESTS": "16",
            "CONCURRENCY_ACQUIRE_TIMEOUT_SECONDS": "0.25",
            "DB_POOL_SIZE": "5",
            "DB_MAX_OVERFLOW": "5",
            "DB_POOL_TIMEOUT_SECONDS": "5",
            "DB_STATEMENT_TIMEOUT_MS": "15000",
            "LLM_PROVIDER": "extractive",
            "LLM_MODEL": "extractive-local-v1",
            "LLM_TIMEOUT_SECONDS": "20",
            "LLM_MAX_RETRIES": "1",
            "LLM_RETRY_BACKOFF_SECONDS": "0.25",
            "LOG_LEVEL": "INFO",
        }
    )
    env.update(overrides)
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--no-access-log",
            "--log-level",
            "warning",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    instance = ApiInstance(name=name, port=port, process=process, log_path=log_path, handle=handle)
    ready = wait_for_url(
        f"{instance.base_url}/ready",
        predicate=lambda status, _: status in (200, 503),
        timeout=30.0,
    )
    if not ready:
        stop_api(instance)
        raise RuntimeError(f"API instance {name} did not start; see {log_path}")
    return instance


def stop_api(instance: ApiInstance) -> None:
    if instance.process.poll() is None:
        instance.process.terminate()
        try:
            instance.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            instance.process.kill()
            instance.process.wait(timeout=5)
    try:
        instance.handle.close()
    except Exception:  # noqa: BLE001 - closing an already-closed handle is harmless
        return


def api_call(
    client: httpx.Client,
    *,
    method: str,
    path: str,
    token: str,
    request_id: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    headers = {"X-Request-ID": request_id, "Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    started = time.perf_counter()
    status: int | None = None
    body: Any = None
    text_body = ""
    error: str | None = None
    retry_after: str | None = None
    try:
        response = client.request(
            method,
            path,
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        status = response.status_code
        text_body = response.text
        retry_after = response.headers.get("Retry-After")
        try:
            body = response.json()
        except ValueError:
            body = text_body[:500]
    except httpx.HTTPError as exc:
        error = f"{type(exc).__name__}: {exc}"
    return {
        "request_id": request_id,
        "method": method,
        "path": path,
        "status": status,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "body": body,
        "text_body": text_body,
        "retry_after": retry_after,
        "client_error": error,
        "recorded_at": now(),
    }


def run_load(
    *,
    name: str,
    base_url: str,
    scenario: str,
    requests: int,
    concurrency: int,
    labels: list[str],
    tokens: dict[str, str],
    out_dir: Path,
    inner_scenario: str = "answer_lexical",
    questions: list[str] | None = None,
    canaries: dict[str, str] | None = None,
    timeout: float = 30.0,
    warmup: int = 2,
) -> dict[str, Any]:
    output = out_dir / f"load-{name}.json"
    prefix = f"ov-{name.replace('_', '-')[:18]}-"
    args = [
        sys.executable,
        str(LOAD_TEST),
        "--base-url",
        base_url,
        "--scenario",
        scenario,
        "--inner-scenario",
        inner_scenario,
        "--requests",
        str(requests),
        "--concurrency",
        str(concurrency),
        "--warmup",
        str(warmup),
        "--timeout",
        str(timeout),
        "--request-prefix",
        prefix,
        "--output",
        str(output),
        "--quiet",
    ]
    for label in labels:
        args.extend(["--token", f"{label}={tokens[label]}"])
    for label, value in (canaries or {}).items():
        args.extend(["--canary", f"{label}={value}"])
    for question in questions or []:
        args.extend(["--question", question])
    result = subprocess.run(
        args,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=max(60.0, timeout * requests),
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"load test {name} failed: {result.returncode}\n{result.stderr[-2000:]}"
        )
    summary = json.loads(output.read_text(encoding="utf-8"))
    records_path = output.with_suffix(".records.jsonl")
    return {
        "summary": summary,
        "summary_path": output,
        "records_path": records_path,
    }


def augment_stage_latency(
    *,
    summary_path: Path,
    records_path: Path,
    log_path: Path,
) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    events, non_json = load_events(log_path)
    indexed = index_events(events)
    fields = (
        "retrieval_latency_ms",
        "evidence_latency_ms",
        "generation_latency_ms",
        "total_latency_ms",
    )
    stage_values: dict[str, list[float]] = {field: [] for field in fields}
    matched = 0
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for record in records:
        matches = indexed.get(record["request_id"]) or []
        if len(matches) != 1:
            continue
        matched += 1
        event = matches[0]
        for latency_field in fields:
            value = event.get(latency_field)
            if isinstance(value, int) and value >= 0:
                stage_values[latency_field].append(float(value))

    summary["stage_latency_ms"] = {
        "matched_requests": matched,
        "recorded_requests": len(records),
        "non_json_log_lines": non_json,
        **{
            field.replace("_latency_ms", ""): _summary(values)
            for field, values in stage_values.items()
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def records(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_expectations(
    *,
    records_path: Path,
    scenario: str,
    provider: str = "extractive",
) -> list[dict[str, Any]]:
    expectations: list[dict[str, Any]] = []
    for record in records(records_path):
        if record.get("status") is None:
            continue
        path = record.get("path")
        expectation: dict[str, Any] = {
            "request_id": record["request_id"],
            "status_code": record["status"],
            "path": path,
        }
        if scenario in ("invalid_auth", "missing_auth"):
            expectation["error_category"] = "unauthorized"
            expectation["tenant_ref"] = None
        elif path == "/health" or path == "/ready":
            expectation["tenant_ref"] = None
            expectation["__non_null__"] = ["total_latency_ms"]
        elif record["status"] == 200 and path.startswith("/api/v1/search"):
            expectation["retrieval_strategy"] = (
                "hybrid" if "hybrid" in (path or "") else "semantic"
            )
            expectation["error_category"] = None
            expectation["__non_null__"] = [
                "retrieval_latency_ms",
                "total_latency_ms",
                "tenant_ref",
            ]
        elif record["status"] == 200 and path == "/api/v1/answer":
            expectation["provider"] = provider
            expectation["error_category"] = None
            expectation["retry_count"] = 0
            expectation["__non_null__"] = [
                "retrieval_latency_ms",
                "evidence_latency_ms",
                "generation_latency_ms",
                "total_latency_ms",
                "input_tokens",
                "output_tokens",
                "token_source",
                "tenant_ref",
            ]
            if provider == "extractive":
                expectation["observed_model"] = "extractive-local-v1"
                expectation["token_source"] = "estimated"
        elif record["status"] == 429:
            expectation["error_category"] = "rate_limited"
            expectation["__non_null__"] = ["total_latency_ms"]
        elif record["status"] == 503:
            expectation["error_category"] = "database_unavailable"
            expectation["__non_null__"] = ["total_latency_ms"]
        expectations.append(expectation)
    return expectations


def forbidden_values(temp: TempTenant, questions: list[str]) -> list[str]:
    return [
        TOKEN_A,
        TOKEN_B1,
        TOKEN_B2,
        str(TENANT_A_ID),
        str(temp.tenant_id),
        temp.canary_b,
        temp.restricted_canary,
        FAKE_API_KEY,
        "Authorization",
        "Bearer ",
        *questions,
    ]


def validate_log_file(
    *,
    name: str,
    log_path: Path,
    expectations: list[dict[str, Any]],
    temp: TempTenant,
    questions: list[str],
    out_dir: Path,
) -> dict[str, Any]:
    report = validate(
        log_path=log_path,
        expectations=expectations,
        forbidden=forbidden_values(temp, questions),
    )
    (out_dir / f"logs-{name}.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def probe_auth(base_url: str) -> dict[str, Any]:
    def call(index: int, token: str) -> dict[str, Any]:
        with httpx.Client(base_url=base_url, timeout=15.0) as client:
            return api_call(
                client,
                method="POST",
                path="/api/v1/answer",
                token=token,
                request_id=f"ov-auth-{index:04d}",
                payload={"question": "What does PAY-503 mean?"},
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        invalid = list(pool.map(lambda i: call(i, "invalid-token"), range(30)))
        missing = list(pool.map(lambda i: call(i + 100, ""), range(15)))

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "count": len(items),
            "status_counts": {
                str(status): sum(1 for item in items if item["status"] == status)
                for status in {item["status"] for item in items}
            },
            "consistent_body": len({json.dumps(item["body"], sort_keys=True) for item in items}) == 1,
            "request_ids": [item["request_id"] for item in items],
            "latency_ms": _summary([item["latency_ms"] for item in items]),
        }

    return {"invalid_token": summarize(invalid), "missing_token": summarize(missing)}


def probe_tenant_isolation(
    base_url: str,
    temp: TempTenant,
    *,
    count: int = 20,
) -> dict[str, Any]:
    question = f"What does {temp.canary_b} mean?"
    payload = {"question": question, "retrieval": {"strategy": "lexical"}}

    def call(label: str, token: str, index: int) -> dict[str, Any]:
        with httpx.Client(base_url=base_url, timeout=30.0) as client:
            return api_call(
                client,
                method="POST",
                path="/api/v1/answer",
                token=token,
                request_id=f"ov-iso-{label}-{index:04d}",
                payload=payload,
            )

    def contains(result: dict[str, Any]) -> bool:
        body = result.get("body")
        if not isinstance(body, dict):
            return False
        evidence = body.get("evidence") or []
        text = " ".join(str(item.get("content", "")) for item in evidence)
        citations = body.get("citations") or []
        text += " " + " ".join(str(item.get("excerpt", "")) for item in citations)
        return temp.canary_b in text

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        tenant_a = list(
            pool.map(lambda i: call("a", TOKEN_A, i), range(count))
        )
        tenant_b = list(
            pool.map(lambda i: call("b1", TOKEN_B1, i), range(count))
        )

    return {
        "tenant_a_requests": len(tenant_a),
        "tenant_a_contains_tenant_b_canary": sum(1 for item in tenant_a if contains(item)),
        "tenant_b_requests": len(tenant_b),
        "tenant_b_contains_own_canary": sum(1 for item in tenant_b if contains(item)),
        "tenant_a_status_counts": {
            str(status): sum(1 for item in tenant_a if item["status"] == status)
            for status in {item["status"] for item in tenant_a}
        },
        "tenant_b_status_counts": {
            str(status): sum(1 for item in tenant_b if item["status"] == status)
            for status in {item["status"] for item in tenant_b}
        },
        "request_ids": [item["request_id"] for item in tenant_a + tenant_b],
    }


def probe_acl(
    base_url: str,
    temp: TempTenant,
    *,
    count: int = 20,
) -> dict[str, Any]:
    question = f"What does {temp.restricted_canary} mean?"
    payload = {"question": question, "retrieval": {"strategy": "lexical"}}

    def call(label: str, token: str, index: int) -> dict[str, Any]:
        with httpx.Client(base_url=base_url, timeout=30.0) as client:
            return api_call(
                client,
                method="POST",
                path="/api/v1/answer",
                token=token,
                request_id=f"ov-acl-{label}-{index:04d}",
                payload=payload,
            )

    def contains(result: dict[str, Any]) -> bool:
        body = result.get("body")
        if not isinstance(body, dict):
            return False
        text = " ".join(str(item.get("content", "")) for item in body.get("evidence") or [])
        text += " " + " ".join(
            str(item.get("excerpt", "")) for item in body.get("citations") or []
        )
        return temp.restricted_canary in text

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        allowed = list(
            pool.map(lambda i: call("b1", TOKEN_B1, i), range(count))
        )
        denied = list(
            pool.map(lambda i: call("b2", TOKEN_B2, i), range(count))
        )

    return {
        "allowed_requests": len(allowed),
        "allowed_saw_restricted_document": sum(1 for item in allowed if contains(item)),
        "denied_requests": len(denied),
        "denied_saw_restricted_document": sum(1 for item in denied if contains(item)),
        "allowed_status_counts": {
            str(status): sum(1 for item in allowed if item["status"] == status)
            for status in {item["status"] for item in allowed}
        },
        "denied_status_counts": {
            str(status): sum(1 for item in denied if item["status"] == status)
            for status in {item["status"] for item in denied}
        },
        "request_ids": [item["request_id"] for item in allowed + denied],
    }


def probe_rate_limit(
    base_url: str,
    *,
    limit: int = 20,
    extra: int = 5,
) -> dict[str, Any]:
    payload = {"query": "payment error", "limit": 5}

    def burst(label: str, token: str, count: int, offset: int = 0) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        with httpx.Client(base_url=base_url, timeout=30.0) as client:
            for index in range(count):
                results.append(
                    api_call(
                        client,
                        method="POST",
                        path="/api/v1/search",
                        token=token,
                        request_id=f"ov-rate-{label}-{offset + index:04d}",
                        payload=payload,
                    )
                )
        return results

    tenant_a = burst("a", TOKEN_A, limit + extra)
    tenant_b = burst("b1", TOKEN_B1, limit + extra)
    final_a = burst("a-final", TOKEN_A, 1, offset=100)
    final_b = burst("b1-final", TOKEN_B1, 1, offset=100)

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "status_counts": {
                str(status): sum(1 for item in items if item["status"] == status)
                for status in {item["status"] for item in items}
            },
            "retry_after": [
                item.get("retry_after")
                for item in items
                if item.get("retry_after") is not None
            ][:3],
            "request_ids": [item["request_id"] for item in items],
        }

    return {
        "configured_limit_per_minute": limit,
        "tenant_a": summarize(tenant_a),
        "tenant_b_same_limit_after_tenant_a": summarize(tenant_b),
        "tenant_a_after_both": summarize(final_a),
        "tenant_b_after_both": summarize(final_b),
        "statuses": {
            "a": [item["status"] for item in tenant_a],
            "b": [item["status"] for item in tenant_b],
            "a_final": [item["status"] for item in final_a],
            "b_final": [item["status"] for item in final_b],
        },
        "all_request_ids": [
            item["request_id"]
            for item in tenant_a + tenant_b + final_a + final_b
        ],
    }


def probe_concurrency(base_url: str, *, count: int = 8) -> dict[str, Any]:
    question = "[[mode=slow]] What should I check before rolling back?"

    def call(index: int) -> dict[str, Any]:
        with httpx.Client(base_url=base_url, timeout=20.0) as client:
            return api_call(
                client,
                method="POST",
                path="/api/v1/answer",
                token=TOKEN_A,
                request_id=f"ov-conc-{index:04d}",
                payload={"question": question, "retrieval": {"strategy": "lexical"}},
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
        results = list(pool.map(call, range(count)))

    with httpx.Client(base_url=base_url, timeout=20.0) as client:
        follow_up = api_call(
            client,
            method="POST",
            path="/api/v1/answer",
            token=TOKEN_A,
            request_id="ov-conc-followup-0001",
            payload={
                "question": "[[mode=ok]] What should I check before rolling back?",
                "retrieval": {"strategy": "lexical"},
            },
        )

    return {
        "concurrent_requests": len(results),
        "status_counts": {
            str(status): sum(1 for item in results if item["status"] == status)
            for status in {item["status"] for item in results}
        },
        "statuses": [item["status"] for item in results],
        "latency_ms": _summary([item["latency_ms"] for item in results]),
        "follow_up_status": follow_up["status"],
        "request_ids": [item["request_id"] for item in results]
        + [follow_up["request_id"]],
    }


PROVIDER_MODES = {
    "ok": (200, None, 0, "actual"),
    "timeout": (504, "provider_timeout", 1, None),
    "rate_limit": (429, "provider_rate_limited", 1, None),
    "invalid_envelope": (502, "invalid_model_output", 0, None),
    "invalid_output": (502, "invalid_model_output", 0, None),
    "bad_citation": (502, "citation_validation_error", 0, None),
    "server_error": (502, "provider_error", 1, None),
}


def probe_provider_modes(base_url: str) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    with httpx.Client(base_url=base_url, timeout=30.0) as client:
        for mode, (expected_status, expected_category, expected_retries, token_source) in (
            PROVIDER_MODES.items()
        ):
            request_id = f"ov-provider-{mode.replace('_', '-')[:20]}-0001"
            result = api_call(
                client,
                method="POST",
                path="/api/v1/answer",
                token=TOKEN_A,
                request_id=request_id,
                payload={
                    "question": (
                        f"[[mode={mode}]] What should I check before rolling back?"
                    ),
                    "retrieval": {"strategy": "lexical"},
                },
                timeout=30.0,
            )
            results.append(
                {
                    "mode": mode,
                    "expected_status": expected_status,
                    "expected_error_category": expected_category,
                    "expected_retry_count": expected_retries,
                    "expected_token_source": token_source,
                    **result,
                }
            )
    return {"results": results}


def probe_cancellation(base_url: str) -> dict[str, Any]:
    cancelled_id = "ov-cancel-request-0001"
    follow_up_id = "ov-cancel-followup-0001"
    cancelled: dict[str, Any] = {
        "request_id": cancelled_id,
        "client_error": None,
        "status": None,
        "latency_ms": None,
    }
    try:
        with httpx.Client(base_url=base_url, timeout=0.25) as client:
            response = client.post(
                "/api/v1/answer",
                headers={
                    "Authorization": f"Bearer {TOKEN_A}",
                    "X-Request-ID": cancelled_id,
                    "Content-Type": "application/json",
                },
                json={
                    "question": "[[mode=slow]] What should I check before rolling back?",
                    "retrieval": {"strategy": "lexical"},
                },
            )
            cancelled["status"] = response.status_code
    except httpx.HTTPError as exc:
        cancelled["client_error"] = f"{type(exc).__name__}: {exc}"

    time.sleep(2.0)
    with httpx.Client(base_url=base_url, timeout=20.0) as client:
        follow_up = api_call(
            client,
            method="POST",
            path="/api/v1/answer",
            token=TOKEN_A,
            request_id=follow_up_id,
            payload={
                "question": "[[mode=ok]] What should I check before rolling back?",
                "retrieval": {"strategy": "lexical"},
            },
        )
    return {"cancelled": cancelled, "follow_up": follow_up}


def probe_pool_exhaustion(base_url: str, dsn: str) -> dict[str, Any]:
    locked = threading.Event()
    lock_error: list[str] = []

    def hold_lock() -> None:
        engine = create_engine(dsn, pool_pre_ping=False)
        try:
            with engine.connect() as connection:
                connection.execute(text("LOCK TABLE chunks IN ACCESS EXCLUSIVE MODE"))
                locked.set()
                time.sleep(3.0)
                connection.rollback()
        except Exception as exc:  # noqa: BLE001 - record and continue
            lock_error.append(f"{type(exc).__name__}: {exc}")
            locked.set()
        finally:
            engine.dispose()

    thread = threading.Thread(target=hold_lock, daemon=True)
    thread.start()
    if not locked.wait(10.0):
        return {"error": "could not acquire the table lock", "lock_error": lock_error}

    def call(index: int) -> dict[str, Any]:
        time.sleep(index * 0.2)
        with httpx.Client(base_url=base_url, timeout=20.0) as client:
            return api_call(
                client,
                method="POST",
                path="/api/v1/answer",
                token=TOKEN_A,
                request_id=f"ov-pool-{index:04d}",
                payload={
                    "question": "What does PAY-503 mean?",
                    "retrieval": {"strategy": "lexical"},
                },
                timeout=20.0,
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(call, range(2)))

    thread.join(timeout=15.0)
    with httpx.Client(base_url=base_url, timeout=20.0) as client:
        recovery = api_call(
            client,
            method="POST",
            path="/api/v1/answer",
            token=TOKEN_A,
            request_id="ov-pool-recovery-0001",
            payload={
                "question": "What does PAY-503 mean?",
                "retrieval": {"strategy": "lexical"},
            },
        )
    return {
        "requests": results,
        "statuses": [item["status"] for item in results],
        "recovery_status": recovery["status"],
        "request_ids": [item["request_id"] for item in results]
        + [recovery["request_id"]],
        "lock_error": lock_error,
    }


def provider_expectations(probe: dict[str, Any]) -> list[dict[str, Any]]:
    expectations = []
    for result in probe["results"]:
        expectation = {
            "request_id": result["request_id"],
            "status_code": result["expected_status"],
            "path": "/api/v1/answer",
            "provider": "openai_compatible",
            "error_category": result["expected_error_category"],
            "retry_count": result["expected_retry_count"],
            "__non_null__": ["generation_latency_ms", "total_latency_ms", "tenant_ref"],
        }
        if result["mode"] == "ok":
            expectation["observed_model"] = "fake-model"
            expectation["token_source"] = result["expected_token_source"]
            expectation["__non_null__"].extend(["input_tokens", "output_tokens"])
        expectations.append(expectation)
    return expectations


def cancellation_expectations(
    probe: dict[str, Any],
) -> list[dict[str, Any]]:
    expectations = [
        {
            "request_id": probe["follow_up"]["request_id"],
            "status_code": 200,
            "path": "/api/v1/answer",
            "provider": "openai_compatible",
            "error_category": None,
            "retry_count": 0,
            "token_source": "actual",
            "__non_null__": [
                "retrieval_latency_ms",
                "evidence_latency_ms",
                "generation_latency_ms",
                "total_latency_ms",
                "input_tokens",
                "output_tokens",
                "tenant_ref",
            ],
        }
    ]
    if probe["cancelled"].get("status") is None:
        # The client timed out. The server-side outcome is read from the log
        # after the fact; require a completed, correlated event regardless of
        # whether uvicorn chose 200 or 500.
        expectations.append(
            {
                "request_id": probe["cancelled"]["request_id"],
                "__non_null__": ["total_latency_ms"],
            }
        )
    return expectations


def pool_expectations(probe: dict[str, Any]) -> list[dict[str, Any]]:
    expectations = []
    for result in probe["requests"]:
        expectation = {
            "request_id": result["request_id"],
            "status_code": result["status"],
            "path": "/api/v1/answer",
            "__non_null__": ["total_latency_ms"],
        }
        if result["status"] == 503:
            expectation["error_category"] = "database_unavailable"
        expectations.append(expectation)
    expectations.append(
        {
            "request_id": probe["request_ids"][-1],
            "status_code": 200,
            "path": "/api/v1/answer",
            "provider": "extractive",
            "error_category": None,
            "retry_count": 0,
            "token_source": "estimated",
            "__non_null__": ["total_latency_ms", "tenant_ref"],
        }
    )
    return expectations


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--skip-failure", action="store_true")
    parser.add_argument("--skip-load", action="store_true")
    parser.add_argument("--keep-embedding", action="store_true")
    parser.add_argument("--no-temp-tenant", action="store_true")
    args = parser.parse_args(argv)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out_dir or (REPO_ROOT / "evals" / "ops" / stamp)
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "phase": "E",
        "started_at": now(),
        "out_dir": str(out_dir),
        "tokens_written": False,
    }
    api_instances: list[ApiInstance] = []
    extra_processes: list[subprocess.Popen] = []
    extra_handles: list[Any] = []
    temp: TempTenant | None = None
    embedding_pid: int | None = None
    embedding_owned = False
    embedding_preexisting = True
    embedding_log = out_dir / "embedding.log"

    try:
        report["environment"] = collect_environment()
        report["containers"] = ensure_containers()
        report["db_counts_before"] = db_counts()

        embedding_pid, embedding_owned, embedding_handle = ensure_embedding(embedding_log)
        embedding_preexisting = not embedding_owned
        if embedding_handle is not None:
            extra_handles.append(embedding_handle)
        report["embedding"] = {
            "pid_found": embedding_pid is not None,
            "started_by_harness": embedding_owned,
        }

        if args.no_temp_tenant:
            raise RuntimeError("--no-temp-tenant is not supported for this validation run")
        temp = create_temp_tenant()
        report["temp_tenant_created"] = True

        tokens = {"a": TOKEN_A, "b1": TOKEN_B1, "b2": TOKEN_B2}
        forbidden_questions = QUESTIONS

        baseline = start_api(
            name="baseline",
            port=8010,
            out_dir=out_dir,
            overrides={"RATE_LIMIT_PER_MINUTE": "0"},
            temp=temp,
        )
        api_instances.append(baseline)

        if not args.skip_load:
            load_runs: dict[str, Any] = {}
            load_specs = [
                ("health", "health", 100, 1, ["a"], None, 5.0),
                ("ready", "ready", 50, 1, ["a"], None, 10.0),
                ("answer_single", "answer", 30, 1, ["a"], None, 30.0),
                ("answer_concurrent", "answer", 100, 8, ["a"], None, 30.0),
                ("answer_lexical", "answer_lexical", 100, 8, ["a"], None, 30.0),
                ("search_semantic", "search", 80, 8, ["a"], None, 30.0),
                ("search_hybrid", "search_hybrid", 80, 8, ["a"], None, 30.0),
                (
                    "multi_tenant",
                    "multi_tenant",
                    100,
                    8,
                    ["a", "b1"],
                    {"b1": temp.canary_b},
                    30.0,
                ),
                ("mixed", "mixed", 100, 8, ["a"], None, 30.0),
            ]
            sampler = ResourceSampler(
                processes={
                    "api-baseline": baseline.process.pid,
                    **(
                        {"embedding": embedding_pid}
                        if embedding_pid is not None
                        else {}
                    ),
                },
                containers=[POSTGRES_CONTAINER, REDIS_CONTAINER],
                interval_seconds=1.0,
            )
            sampler.start()
            for name, scenario, count, concurrency, labels, canaries, timeout in load_specs:
                result = run_load(
                    name=name,
                    base_url=baseline.base_url,
                    scenario=scenario,
                    requests=count,
                    concurrency=concurrency,
                    labels=labels,
                    tokens=tokens,
                    out_dir=out_dir,
                    inner_scenario="answer_lexical" if scenario == "multi_tenant" else "answer",
                    questions=QUESTIONS,
                    canaries=canaries,
                    timeout=timeout,
                )
                summary = augment_stage_latency(
                    summary_path=result["summary_path"],
                    records_path=result["records_path"],
                    log_path=baseline.log_path,
                )
                expectations = build_expectations(
                    records_path=result["records_path"],
                    scenario=scenario,
                    provider="extractive",
                )
                log_report = validate_log_file(
                    name=name,
                    log_path=baseline.log_path,
                    expectations=expectations,
                    temp=temp,
                    questions=forbidden_questions,
                    out_dir=out_dir,
                )
                summary["log_validation"] = {
                    "passed": log_report["passed"],
                    "expectations_checked": log_report["expectations_checked"],
                    "failures": len(log_report["expectation_failures"]),
                    "forbidden_hits": len(log_report["forbidden_hits"]),
                }
                result["summary_path"].write_text(
                    json.dumps(summary, indent=2) + "\n",
                    encoding="utf-8",
                )
                load_runs[name] = summary
            report["resources"] = sampler.stop().__dict__
            report["load"] = load_runs

            auth_probe = probe_auth(baseline.base_url)
            report["auth_probe"] = auth_probe
            auth_expectations = [
                {
                    "request_id": request_id,
                    "status_code": 401,
                    "error_category": "unauthorized",
                    "tenant_ref": None,
                }
                for request_id in auth_probe["invalid_token"]["request_ids"]
                + auth_probe["missing_token"]["request_ids"]
            ]
            report["auth_log_validation"] = validate_log_file(
                name="auth",
                log_path=baseline.log_path,
                expectations=auth_expectations,
                temp=temp,
                questions=forbidden_questions,
                out_dir=out_dir,
            )

            isolation = probe_tenant_isolation(baseline.base_url, temp)
            report["tenant_isolation_probe"] = isolation
            isolation_expectations = [
                {
                    "request_id": request_id,
                    "status_code": 200,
                    "path": "/api/v1/answer",
                    "provider": "extractive",
                    "error_category": None,
                    "retry_count": 0,
                    "token_source": "estimated",
                    "__non_null__": ["total_latency_ms", "tenant_ref"],
                }
                for request_id in isolation["request_ids"]
            ]
            report["tenant_isolation_log_validation"] = validate_log_file(
                name="tenant-isolation",
                log_path=baseline.log_path,
                expectations=isolation_expectations,
                temp=temp,
                questions=forbidden_questions,
                out_dir=out_dir,
            )

            acl_probe = probe_acl(baseline.base_url, temp)
            report["acl_probe"] = acl_probe
            acl_expectations = [
                {
                    "request_id": request_id,
                    "status_code": 200,
                    "path": "/api/v1/answer",
                    "provider": "extractive",
                    "error_category": None,
                    "retry_count": 0,
                    "token_source": "estimated",
                    "__non_null__": ["total_latency_ms", "tenant_ref"],
                }
                for request_id in acl_probe["request_ids"]
            ]
            report["acl_log_validation"] = validate_log_file(
                name="acl",
                log_path=baseline.log_path,
                expectations=acl_expectations,
                temp=temp,
                questions=forbidden_questions,
                out_dir=out_dir,
            )

            report["observability"] = {
                "request_correlation": "checked per scenario",
                "redaction": "forbidden-value scan per scenario",
                "baseline_log": str(baseline.log_path),
            }

        # Rate limit isolation on a fresh process.
        rate_api = start_api(
            name="rate",
            port=8012,
            out_dir=out_dir,
            overrides={
                "RATE_LIMIT_PER_MINUTE": "20",
                "MAX_CONCURRENT_REQUESTS": "0",
            },
            temp=temp,
        )
        api_instances.append(rate_api)
        rate_probe = probe_rate_limit(rate_api.base_url)
        report["rate_limit_probe"] = rate_probe
        rate_expectations = [
            {
                "request_id": request_id,
                "status_code": status,
                "path": "/api/v1/search",
                "error_category": "rate_limited" if status == 429 else None,
                "__non_null__": ["total_latency_ms"],
            }
            for request_id, status in zip(
                rate_probe["all_request_ids"],
                rate_probe["statuses"]["a"]
                + rate_probe["statuses"]["b"]
                + rate_probe["statuses"]["a_final"]
                + rate_probe["statuses"]["b_final"],
            )
        ]
        report["rate_limit_log_validation"] = validate_log_file(
            name="rate",
            log_path=rate_api.log_path,
            expectations=rate_expectations,
            temp=temp,
            questions=forbidden_questions,
            out_dir=out_dir,
        )
        stop_api(rate_api)
        api_instances.remove(rate_api)

        # Fake provider for provider failures and concurrency.
        fake_log = out_dir / "fake-provider.log"
        fake_handle = fake_log.open("w", encoding="utf-8")
        fake_process = subprocess.Popen(
            [sys.executable, str(FAKE_PROVIDER), "--port", "8099"],
            cwd=str(REPO_ROOT),
            stdout=fake_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        extra_processes.append(fake_process)
        extra_handles.append(fake_handle)
        if not wait_for_url(
            f"{FAKE_PROVIDER_URL}/health",
            predicate=lambda status, _: status == 200,
            timeout=15.0,
        ):
            raise RuntimeError("fake provider did not start")

        fake_env = {
            "LLM_PROVIDER": "openai_compatible",
            "LLM_API_BASE": f"{FAKE_PROVIDER_URL}/v1",
            "LLM_API_KEY": FAKE_API_KEY,
            "LLM_MODEL": "fake-model",
            "LLM_TIMEOUT_SECONDS": "1",
            "LLM_MAX_RETRIES": "1",
            "LLM_RETRY_BACKOFF_SECONDS": "0.05",
            "RATE_LIMIT_PER_MINUTE": "0",
        }

        concurrency_api = start_api(
            name="concurrency",
            port=8013,
            out_dir=out_dir,
            overrides={
                **fake_env,
                "LLM_TIMEOUT_SECONDS": "8",
                "LLM_MAX_RETRIES": "0",
                "MAX_CONCURRENT_REQUESTS": "1",
                "CONCURRENCY_ACQUIRE_TIMEOUT_SECONDS": "0.0",
            },
            temp=temp,
        )
        api_instances.append(concurrency_api)
        concurrency_probe = probe_concurrency(concurrency_api.base_url)
        report["concurrency_probe"] = concurrency_probe
        concurrency_expectations = [
            {
                "request_id": request_id,
                "status_code": status,
                "path": "/api/v1/answer",
                "error_category": "concurrency_limited" if status == 429 else None,
                "__non_null__": ["total_latency_ms"],
            }
            for request_id, status in zip(
                concurrency_probe["request_ids"][:-1],
                concurrency_probe["statuses"],
                strict=True,
            )
        ]
        concurrency_expectations.append(
            {
                "request_id": concurrency_probe["request_ids"][-1],
                "status_code": 200,
                "path": "/api/v1/answer",
                "provider": "openai_compatible",
                "error_category": None,
                "retry_count": 0,
                "token_source": "actual",
                "__non_null__": [
                    "generation_latency_ms",
                    "total_latency_ms",
                    "input_tokens",
                    "output_tokens",
                    "tenant_ref",
                ],
            }
        )
        report["concurrency_log_validation"] = validate_log_file(
            name="concurrency",
            log_path=concurrency_api.log_path,
            expectations=concurrency_expectations,
            temp=temp,
            questions=forbidden_questions,
            out_dir=out_dir,
        )
        cancellation_probe = probe_cancellation(concurrency_api.base_url)
        report["cancellation_probe"] = cancellation_probe
        cancellation_log = validate_log_file(
            name="cancellation",
            log_path=concurrency_api.log_path,
            expectations=cancellation_expectations(cancellation_probe),
            temp=temp,
            questions=forbidden_questions,
            out_dir=out_dir,
        )
        report["cancellation_log_validation"] = cancellation_log
        events, _ = load_events(concurrency_api.log_path)
        indexed = index_events(events)
        cancelled_events = indexed.get(cancellation_probe["cancelled"]["request_id"]) or []
        report["cancellation_server_side"] = (
            {
                key: cancelled_events[0].get(key)
                for key in (
                    "status_code",
                    "error_category",
                    "total_latency_ms",
                    "retrieval_latency_ms",
                    "generation_latency_ms",
                )
            }
            if cancelled_events
            else None
        )
        stop_api(concurrency_api)
        api_instances.remove(concurrency_api)

        provider_api = start_api(
            name="provider",
            port=8014,
            out_dir=out_dir,
            overrides=fake_env,
            temp=temp,
        )
        api_instances.append(provider_api)
        provider_probe = probe_provider_modes(provider_api.base_url)
        report["provider_probe"] = provider_probe
        report["provider_log_validation"] = validate_log_file(
            name="provider",
            log_path=provider_api.log_path,
            expectations=provider_expectations(provider_probe),
            temp=temp,
            questions=forbidden_questions,
            out_dir=out_dir,
        )
        stop_api(provider_api)
        api_instances.remove(provider_api)

        pool_api = start_api(
            name="pool",
            port=8015,
            out_dir=out_dir,
            overrides={
                "DB_POOL_SIZE": "1",
                "DB_MAX_OVERFLOW": "0",
                "DB_POOL_TIMEOUT_SECONDS": "1",
                "RATE_LIMIT_PER_MINUTE": "0",
                "MAX_CONCURRENT_REQUESTS": "16",
            },
            temp=temp,
        )
        api_instances.append(pool_api)
        dsn = os.environ.get(
            "DATABASE_URL",
            "postgresql+psycopg://opsatlas:opsatlas_dev_password@localhost:5433/opsatlas",
        )
        pool_probe = probe_pool_exhaustion(pool_api.base_url, dsn)
        report["pool_exhaustion_probe"] = pool_probe
        report["pool_log_validation"] = validate_log_file(
            name="pool",
            log_path=pool_api.log_path,
            expectations=pool_expectations(pool_probe),
            temp=temp,
            questions=forbidden_questions,
            out_dir=out_dir,
        )
        stop_api(pool_api)
        api_instances.remove(pool_api)

        if not args.skip_failure:
            injection = run_failure_injection(
                base_url=baseline.base_url,
                token=TOKEN_A,
                postgres_container=POSTGRES_CONTAINER,
                redis_container=REDIS_CONTAINER,
                embedding_pid=embedding_pid,
                embedding_python=Path(sys.executable),
                embedding_script=EMBEDDING_SCRIPT,
                embedding_log=out_dir / "embedding-restart.log",
                output=out_dir / "failure-injection.json",
            )
            report["failure_injection"] = injection
            new_pid = None
            for line in injection.get("notes", []):
                if "embedding restarted pid=" in line:
                    try:
                        new_pid = int(line.rsplit("=", 1)[1])
                    except ValueError:
                        new_pid = None
            if new_pid is not None:
                embedding_pid = new_pid
                embedding_owned = True
            injection_expectations = []
            for item in injection.get("results", []):
                expectation = {
                    "request_id": item["request_id"],
                    "status_code": item["status"],
                    "path": item["path"],
                    "__non_null__": ["total_latency_ms"],
                }
                if "postgres" in item["case"] and item["status"] == 503:
                    expectation["error_category"] = "database_unavailable"
                if (
                    "postgres" in item["case"]
                    and item["path"] == "/ready"
                    and item["status"] == 503
                ):
                    expectation["error_category"] = "dependency_unavailable"
                if "embedding_down_search" in item["case"]:
                    expectation["error_category"] = "embedding_unavailable"
                injection_expectations.append(expectation)
            report["failure_injection_log_validation"] = validate_log_file(
                name="failure-injection",
                log_path=baseline.log_path,
                expectations=injection_expectations,
                temp=temp,
                questions=forbidden_questions,
                out_dir=out_dir,
            )
            report["containers_after_failure"] = {
                POSTGRES_CONTAINER: docker_container_status(POSTGRES_CONTAINER),
                REDIS_CONTAINER: docker_container_status(REDIS_CONTAINER),
            }

        report["db_counts_after"] = db_counts()
        report["finished_at"] = now()
        save_json(out_dir / "results.json", report)

    finally:
        for instance in api_instances:
            stop_api(instance)
        for process in extra_processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
        for handle in extra_handles:
            with contextlib.suppress(Exception):
                handle.close()
        if temp is not None:
            try:
                cleanup_temp_tenant(temp)
                report["temp_tenant_cleaned"] = True
            except Exception as exc:  # noqa: BLE001 - report cleanup failure
                report["temp_tenant_cleaned"] = False
                report["temp_tenant_cleanup_error"] = f"{type(exc).__name__}: {exc}"
        if embedding_owned and not embedding_preexisting and not args.keep_embedding and embedding_pid is not None:
            run_command(
                [
                    "powershell",
                    "-NoLogo",
                    "-NoProfile",
                    "-Command",
                    f"Stop-Process -Id {embedding_pid} -Force",
                ],
                timeout=15,
            )
            report["embedding_stopped"] = True
        try:
            report["db_counts_final"] = db_counts()
        except Exception as exc:  # noqa: BLE001 - final count is best effort
            report["db_counts_final_error"] = f"{type(exc).__name__}: {exc}"
        save_json(out_dir / "results.json", report)

    print(json.dumps({k: v for k, v in report.items() if k != "environment"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
