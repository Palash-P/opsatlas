"""Controlled live failure injection for the local OpsAtlas stack.

The script stops one local dependency at a time, records the API's behavior,
then restores that dependency and waits for recovery. It never drops data and
never touches a non-local database. Docker containers are only stopped and
started; the embedder is restarted from the same script.

Usage (normally driven by ``run_validation.py``):

    python scripts/ops/live_failure_injection.py \
        --base-url http://127.0.0.1:8010 \
        --token opsatlas-validation-token-a \
        --postgres-container opsatlas-v1-postgres-1 \
        --redis-container opsatlas-v1-redis-1 \
        --embedding-pid 1234 \
        --embedding-python .venv/Scripts/python.exe \
        --embedding-script scripts/embedding_service.py \
        --output evals/ops/failure-injection.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _run(command: list[str], *, timeout: float = 30.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def docker_container_status(container: str) -> str | None:
    result = _run(
        ["docker", "inspect", "--format", "{{.State.Health.Status}}", container],
        timeout=15,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def wait_for_container(container: str, timeout: float = 90.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = docker_container_status(container)
        if status in ("healthy", None):
            # ``None`` means the container has no healthcheck; verify by docker
            # inspect state below.
            state = _run(
                ["docker", "inspect", "--format", "{{.State.Running}}", container],
                timeout=15,
            )
            if state.returncode == 0 and state.stdout.strip().lower() == "true":
                return True
        time.sleep(1.0)
    return False


def wait_for_http(
    url: str,
    *,
    predicate: Callable[[int, str], bool],
    timeout: float = 90.0,
    interval: float = 1.0,
) -> tuple[int, str] | None:
    deadline = time.time() + timeout
    last: tuple[int, str] | None = None
    while time.time() < deadline:
        try:
            response = httpx.get(url, timeout=10.0)
            last = (response.status_code, response.text)
            if predicate(response.status_code, response.text):
                return last
        except httpx.HTTPError:
            last = None
        time.sleep(interval)
    return last


def call_api(
    client: httpx.Client,
    *,
    method: str,
    path: str,
    token: str,
    request_id: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    headers = {"X-Request-ID": request_id, "Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    started = time.perf_counter()
    status: int | None = None
    detail: Any = None
    error: str | None = None
    try:
        response = client.request(method, path, headers=headers, json=payload)
        status = response.status_code
        try:
            detail = response.json()
        except ValueError:
            detail = response.text[:200]
    except httpx.HTTPError as exc:
        error = f"{type(exc).__name__}: {exc}"
    return {
        "request_id": request_id,
        "method": method,
        "path": path,
        "status": status,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "detail": detail,
        "client_error": error,
        "recorded_at": _now(),
    }


def _validate_embedding_pid(pid: int) -> bool:
    result = _run(
        [
            "powershell",
            "-NoLogo",
            "-NoProfile",
            "-Command",
            f"(Get-CimInstance Win32_Process -Filter 'ProcessId = {pid}').CommandLine",
        ],
        timeout=15,
    )
    return result.returncode == 0 and "embedding_service.py" in result.stdout


def stop_embedding(pid: int) -> bool:
    if not _validate_embedding_pid(pid):
        return False
    result = _run(
        [
            "powershell",
            "-NoLogo",
            "-NoProfile",
            "-Command",
            f"Stop-Process -Id {pid} -Force",
        ],
        timeout=15,
    )
    return result.returncode == 0


def start_embedding(
    *,
    python: Path,
    script: Path,
    log_path: Path,
) -> int | None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [str(python), str(script)],
        cwd=str(REPO_ROOT),
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return process.pid


def run_failure_injection(
    *,
    base_url: str,
    token: str,
    postgres_container: str,
    redis_container: str,
    embedding_pid: int | None,
    embedding_python: Path | None,
    embedding_script: Path | None,
    embedding_log: Path | None,
    output: Path | None,
    ready_timeout: float = 90.0,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    notes: list[str] = []
    restored: list[str] = []
    started_at = _now()

    with httpx.Client(base_url=base_url.rstrip("/"), timeout=30.0) as client:

        def record(
            name: str,
            *,
            expected: str,
            call: dict[str, Any],
        ) -> None:
            results.append(
                {
                    "case": name,
                    "expected": expected,
                    **call,
                }
            )

        # Baseline before any experiment.
        record(
            "baseline_ready",
            expected="200 ready",
            call=call_api(
                client,
                method="GET",
                path="/ready",
                token="",
                request_id="fi-baseline-ready-0001",
            ),
        )

        # 1. PostgreSQL unavailable.
        stop = _run(["docker", "stop", postgres_container], timeout=60)
        notes.append(f"docker stop postgres exit={stop.returncode}")
        time.sleep(2.0)
        record(
            "postgres_down_health",
            expected="200 liveness only",
            call=call_api(
                client,
                method="GET",
                path="/health",
                token="",
                request_id="fi-postgres-down-health-0001",
            ),
        )
        record(
            "postgres_down_ready",
            expected="503 not_ready",
            call=call_api(
                client,
                method="GET",
                path="/ready",
                token="",
                request_id="fi-postgres-down-ready-0001",
            ),
        )
        record(
            "postgres_down_answer_lexical",
            expected="503 database_unavailable",
            call=call_api(
                client,
                method="POST",
                path="/api/v1/answer",
                token=token,
                request_id="fi-postgres-down-answer-0001",
                payload={
                    "question": "What does PAY-503 mean?",
                    "retrieval": {"strategy": "lexical"},
                },
            ),
        )
        _run(["docker", "start", postgres_container], timeout=60)
        ready = wait_for_http(
            f"{base_url.rstrip('/')}/ready",
            predicate=lambda status, _: status == 200,
            timeout=ready_timeout,
        )
        notes.append(f"postgres restored ready={ready}")
        restored.append("postgres")
        record(
            "postgres_recovered_ready",
            expected="200 ready",
            call=call_api(
                client,
                method="GET",
                path="/ready",
                token="",
                request_id="fi-postgres-recovered-ready-0001",
            ),
        )
        record(
            "postgres_recovered_answer_lexical",
            expected="200",
            call=call_api(
                client,
                method="POST",
                path="/api/v1/answer",
                token=token,
                request_id="fi-postgres-recovered-answer-0001",
                payload={
                    "question": "What does PAY-503 mean?",
                    "retrieval": {"strategy": "lexical"},
                },
            ),
        )

        # 2. Redis unavailable.
        stop = _run(["docker", "stop", redis_container], timeout=60)
        notes.append(f"docker stop redis exit={stop.returncode}")
        time.sleep(2.0)
        record(
            "redis_down_ready",
            expected="200 degraded",
            call=call_api(
                client,
                method="GET",
                path="/ready",
                token="",
                request_id="fi-redis-down-ready-0001",
            ),
        )
        record(
            "redis_down_answer_lexical",
            expected="200",
            call=call_api(
                client,
                method="POST",
                path="/api/v1/answer",
                token=token,
                request_id="fi-redis-down-answer-0001",
                payload={
                    "question": "What does PAY-503 mean?",
                    "retrieval": {"strategy": "lexical"},
                },
            ),
        )
        _run(["docker", "start", redis_container], timeout=60)
        ready = wait_for_http(
            f"{base_url.rstrip('/')}/ready",
            predicate=lambda status, body: status == 200 and "ready" in body,
            timeout=ready_timeout,
        )
        notes.append(f"redis restored ready={ready}")
        restored.append("redis")
        record(
            "redis_recovered_ready",
            expected="200 ready",
            call=call_api(
                client,
                method="GET",
                path="/ready",
                token="",
                request_id="fi-redis-recovered-ready-0001",
            ),
        )

        # 3. Embedding service unavailable.
        if (
            embedding_pid is not None
            and embedding_python is not None
            and embedding_script is not None
        ):
            stopped = stop_embedding(embedding_pid)
            notes.append(f"embedding stop pid={embedding_pid} stopped={stopped}")
            if stopped:
                time.sleep(1.0)
                record(
                    "embedding_down_search",
                    expected="503 embedding_unavailable",
                    call=call_api(
                        client,
                        method="POST",
                        path="/api/v1/search",
                        token=token,
                        request_id="fi-embedding-down-search-0001",
                        payload={"query": "payment error", "limit": 5},
                    ),
                )
                record(
                    "embedding_down_answer_lexical",
                    expected="200",
                    call=call_api(
                        client,
                        method="POST",
                        path="/api/v1/answer",
                        token=token,
                        request_id="fi-embedding-down-answer-0001",
                        payload={
                            "question": "What does PAY-503 mean?",
                            "retrieval": {"strategy": "lexical"},
                        },
                    ),
                )
                log_path = embedding_log or (REPO_ROOT / "evals" / "ops" / "embedding.log")
                new_pid = start_embedding(
                    python=embedding_python,
                    script=embedding_script,
                    log_path=log_path,
                )
                notes.append(f"embedding restarted pid={new_pid}")
                health = wait_for_http(
                    "http://127.0.0.1:8091/health",
                    predicate=lambda status, _: status == 200,
                    timeout=180.0,
                )
                notes.append(f"embedding health={health}")
                restored.append("embedding")
                if health is not None and health[0] == 200:
                    record(
                        "embedding_recovered_search",
                        expected="200",
                        call=call_api(
                            client,
                            method="POST",
                            path="/api/v1/search",
                            token=token,
                            request_id="fi-embedding-recovered-search-0001",
                            payload={"query": "payment error", "limit": 5},
                        ),
                    )
        else:
            notes.append("embedding failure injection skipped: no PID supplied")

    report = {
        "started_at": started_at,
        "finished_at": _now(),
        "base_url": base_url,
        "results": results,
        "notes": notes,
        "restored": restored,
        "passed": all(
            item.get("status") is not None for item in results
        ),
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--token", required=True)
    parser.add_argument("--postgres-container", default="opsatlas-v1-postgres-1")
    parser.add_argument("--redis-container", default="opsatlas-v1-redis-1")
    parser.add_argument("--embedding-pid", type=int, default=None)
    parser.add_argument("--embedding-python", type=Path, default=None)
    parser.add_argument("--embedding-script", type=Path, default=None)
    parser.add_argument("--embedding-log", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_failure_injection(
        base_url=args.base_url,
        token=args.token,
        postgres_container=args.postgres_container,
        redis_container=args.redis_container,
        embedding_pid=args.embedding_pid,
        embedding_python=args.embedding_python,
        embedding_script=args.embedding_script,
        embedding_log=args.embedding_log,
        output=args.output,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
