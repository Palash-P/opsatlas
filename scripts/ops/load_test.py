"""Reproducible HTTP load harness for the local OpsAtlas API.

The harness is dependency-free beyond the project's existing ``httpx`` dev
dependency. It deliberately runs a bounded number of requests at a bounded
concurrency so a local laptop and the synthetic corpus are not overwhelmed.

Examples (PowerShell):

    python scripts/ops/load_test.py --scenario health --requests 100 \
        --concurrency 1 --output evals/ops/health.json

    python scripts/ops/load_test.py --scenario answer --requests 120 \
        --concurrency 8 --token a=$TOKEN_A --canary b=CANARY_B \
        --request-prefix lt-run-answer- --output evals/ops/answer.json

The output JSON contains measurements only. Token values are never written to
the result or the records file.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

DEFAULT_QUESTIONS = [
    "When PAY-503 is raised, which initial checks should happen before escalation?",
    "What should I check before rolling back the payment API?",
    "How do I recover from Redis failover problems?",
    "What does the database pool postmortem say about connection exhaustion?",
    (
        "Which revision of the payment troubleshooting runbook adds a "
        "database-connectivity check before retrying?"
    ),
    "How do I diagnose Kafka consumer lag?",
]

SCENARIOS = (
    "health",
    "ready",
    "search",
    "search_hybrid",
    "answer",
    "answer_lexical",
    "mixed",
    "multi_tenant",
    "invalid_auth",
    "missing_auth",
)


@dataclass
class Record:
    index: int
    label: str
    request_id: str
    scenario: str
    method: str
    path: str
    status: int | None
    ok: bool
    latency_ms: float
    error: str | None
    response_bytes: int
    canary_leak: bool
    started_at: str


@dataclass
class RunConfig:
    base_url: str
    scenario: str
    inner_scenario: str
    requests: int
    concurrency: int
    warmup: int
    timeout_s: float
    request_prefix: str
    labels: list[str]


def parse_pairs(values: list[str], *, name: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        label, sep, item = value.partition("=")
        if not sep or not label or not item:
            raise SystemExit(f"--{name} must use LABEL=VALUE, got: {value}")
        parsed[label] = item
    return parsed


def percentile(values: list[float], fraction: float) -> float | None:
    """Linear-interpolated percentile over an already sorted list."""

    if not values:
        return None
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * fraction
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (rank - lower)


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "mean_ms": None,
            "min_ms": None,
            "max_ms": None,
        }
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "p50_ms": round(percentile(ordered, 0.50) or 0.0, 2),
        "p95_ms": round(percentile(ordered, 0.95) or 0.0, 2),
        "p99_ms": round(percentile(ordered, 0.99) or 0.0, 2),
        "mean_ms": round(sum(ordered) / len(ordered), 2),
        "min_ms": round(ordered[0], 2),
        "max_ms": round(ordered[-1], 2),
    }


def _request_spec(
    scenario: str,
    inner_scenario: str,
    index: int,
    questions: list[str],
    labels: list[str],
) -> tuple[str, str, dict[str, Any] | None, str]:
    question = questions[index % len(questions)]
    label = labels[index % len(labels)]
    effective = inner_scenario if scenario == "multi_tenant" else scenario

    if effective == "health":
        return "GET", "/health", None, label
    if effective == "ready":
        return "GET", "/ready", None, label
    if effective == "search":
        return "POST", "/api/v1/search", {"query": question, "limit": 5}, label
    if effective == "search_hybrid":
        return "POST", "/api/v1/search/hybrid", {"query": question, "limit": 5}, label
    if effective == "answer":
        return (
            "POST",
            "/api/v1/answer",
            {"question": question, "retrieval": {"strategy": "hybrid", "limit": 5}},
            label,
        )
    if effective == "answer_lexical":
        return (
            "POST",
            "/api/v1/answer",
            {"question": question, "retrieval": {"strategy": "lexical", "limit": 5}},
            label,
        )
    if effective == "invalid_auth":
        return (
            "POST",
            "/api/v1/answer",
            {"question": question},
            label,
        )
    if effective == "missing_auth":
        return (
            "POST",
            "/api/v1/answer",
            {"question": question},
            label,
        )
    if effective == "mixed":
        rotation = index % 4
        if rotation == 0:
            return "POST", "/api/v1/answer", {"question": question}, label
        if rotation == 1:
            return (
                "POST",
                "/api/v1/search",
                {"query": question, "limit": 5},
                label,
            )
        if rotation == 2:
            return (
                "POST",
                "/api/v1/search/hybrid",
                {"query": question, "limit": 5},
                label,
            )
        return "GET", "/ready", None, label
    raise SystemExit(f"Unsupported scenario: {scenario}")


class LoadRun:
    def __init__(
        self,
        *,
        config: RunConfig,
        tokens: dict[str, str],
        canaries: dict[str, str],
        questions: list[str] | None = None,
    ) -> None:
        self.config = config
        self.tokens = tokens
        self.canaries = canaries
        self.questions = questions or DEFAULT_QUESTIONS
        self.records: list[Record] = []
        self.started_at = 0.0
        self.finished_at = 0.0

    def _headers(self, label: str, request_id: str) -> dict[str, str]:
        headers = {"X-Request-ID": request_id, "Accept": "application/json"}
        token = self.tokens.get(label)
        if token and self.config.scenario not in ("invalid_auth", "missing_auth"):
            headers["Authorization"] = f"Bearer {token}"
        if self.config.scenario == "invalid_auth":
            headers["Authorization"] = "Bearer not-a-configured-validation-token"
        return headers

    async def _one(
        self,
        client: httpx.AsyncClient,
        index: int,
        *,
        warmup: bool = False,
    ) -> Record:
        method, path, payload, label = _request_spec(
            self.config.scenario,
            self.config.inner_scenario,
            index,
            self.questions,
            self.config.labels,
        )
        request_id = (
            f"{self.config.request_prefix}warmup{index:04d}"
            if warmup
            else f"{self.config.request_prefix}{index:06d}"
        )
        headers = self._headers(label, request_id)
        started_wall = datetime.now(UTC).isoformat()
        started = time.perf_counter()
        status: int | None = None
        error: str | None = None
        body = b""
        try:
            response = await client.request(
                method,
                path,
                headers=headers,
                json=payload,
            )
            status = response.status_code
            body = response.content
        except httpx.TimeoutException as exc:
            error = f"timeout:{type(exc).__name__}"
        except httpx.HTTPError as exc:
            error = f"http_error:{type(exc).__name__}"
        except Exception as exc:  # noqa: BLE001 - a load client must keep going
            error = f"client_error:{type(exc).__name__}"
        latency_ms = (time.perf_counter() - started) * 1000

        canary_leak = False
        if body and self.canaries:
            text = body.decode("utf-8", errors="replace")
            for other_label, canary in self.canaries.items():
                if other_label != label and canary and canary in text:
                    canary_leak = True
                    break

        return Record(
            index=index,
            label=label,
            request_id=request_id,
            scenario=self.config.scenario,
            method=method,
            path=path,
            status=status,
            ok=status is not None and 200 <= status < 300,
            latency_ms=round(latency_ms, 2),
            error=error,
            response_bytes=len(body),
            canary_leak=canary_leak,
            started_at=started_wall,
        )

    async def _execute(self) -> None:
        limits = httpx.Limits(
            max_connections=max(1, self.config.concurrency),
            max_keepalive_connections=max(1, self.config.concurrency),
        )
        timeout = httpx.Timeout(self.config.timeout_s)
        async with httpx.AsyncClient(
            base_url=self.config.base_url,
            timeout=timeout,
            limits=limits,
        ) as client:
            for index in range(self.config.warmup):
                await self._one(client, index, warmup=True)

            queue: asyncio.Queue[int] = asyncio.Queue()
            for index in range(self.config.requests):
                queue.put_nowait(index)

            async def worker() -> None:
                while True:
                    try:
                        index = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    self.records.append(await self._one(client, index))

            self.started_at = time.time()
            await asyncio.gather(
                *(
                    worker()
                    for _ in range(max(1, self.config.concurrency))
                )
            )
            self.finished_at = time.time()

    def run(self) -> dict[str, Any]:
        asyncio.run(self._execute())
        duration = max(0.0, self.finished_at - self.started_at)
        latencies = [record.latency_ms for record in self.records]
        ok_latencies = [
            record.latency_ms for record in self.records if record.ok
        ]
        status_counts: dict[str, int] = {}
        for record in self.records:
            key = str(record.status) if record.status is not None else "error"
            status_counts[key] = status_counts.get(key, 0) + 1

        by_label: dict[str, Any] = {}
        for label in sorted({record.label for record in self.records}):
            label_records = [r for r in self.records if r.label == label]
            by_label[label] = {
                "count": len(label_records),
                "ok": sum(1 for r in label_records if r.ok),
                "error_rate": round(
                    sum(1 for r in label_records if not r.ok) / len(label_records),
                    4,
                )
                if label_records
                else None,
                "latency": _summary([r.latency_ms for r in label_records]),
            }

        summary = {
            "generated_at": datetime.now(UTC).isoformat(),
            "duration_seconds": round(duration, 3),
            "throughput_rps": round(len(self.records) / duration, 2)
            if duration
            else None,
            "successful_throughput_rps": round(
                sum(1 for record in self.records if record.ok) / duration,
                2,
            )
            if duration
            else None,
            "counts": {
                "total": len(self.records),
                "ok": sum(1 for record in self.records if record.ok),
                "errors": sum(1 for record in self.records if not record.ok),
                "client_errors": sum(
                    1 for record in self.records if record.error is not None
                ),
                "canary_violations": sum(
                    1 for record in self.records if record.canary_leak
                ),
            },
            "error_rate": round(
                sum(1 for record in self.records if not record.ok)
                / len(self.records),
                4,
            )
            if self.records
            else None,
            "status_counts": status_counts,
            "latency_ms": _summary(latencies),
            "successful_latency_ms": _summary(ok_latencies),
            "by_label": by_label,
            "config": asdict(self.config),
            "error_samples": [
                {
                    "request_id": record.request_id,
                    "status": record.status,
                    "error": record.error,
                    "label": record.label,
                }
                for record in self.records
                if not record.ok
            ][:10],
        }
        return summary


def write_outputs(
    *,
    summary: dict[str, Any],
    records: list[Record],
    output: Path,
    records_path: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with records_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record)) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--scenario", choices=SCENARIOS, default="answer")
    parser.add_argument(
        "--inner-scenario",
        choices=SCENARIOS,
        default="answer_lexical",
        help="Scenario used per request when --scenario multi_tenant is selected.",
    )
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--token",
        action="append",
        default=[],
        metavar="LABEL=VALUE",
        help="Bearer token per tenant label. The value is never written to output.",
    )
    parser.add_argument(
        "--canary",
        action="append",
        default=[],
        metavar="LABEL=VALUE",
        help="Unique string owned by one tenant label; checked in other labels' bodies.",
    )
    parser.add_argument(
        "--question",
        action="append",
        default=[],
        help="Override the built-in question rotation. Repeatable.",
    )
    parser.add_argument("--request-prefix", default="lt-")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.requests < 1:
        raise SystemExit("--requests must be at least 1")
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be at least 1")
    if len(args.request_prefix) < 3 or len(args.request_prefix) > 48:
        raise SystemExit("--request-prefix must be 3-48 characters")

    tokens = parse_pairs(args.token, name="token")
    canaries = parse_pairs(args.canary, name="canary")
    labels = list(tokens) or ["anonymous"]

    config = RunConfig(
        base_url=args.base_url.rstrip("/"),
        scenario=args.scenario,
        inner_scenario=args.inner_scenario,
        requests=args.requests,
        concurrency=args.concurrency,
        warmup=args.warmup,
        timeout_s=args.timeout,
        request_prefix=args.request_prefix,
        labels=labels,
    )
    run = LoadRun(
        config=config,
        tokens=tokens,
        canaries=canaries,
        questions=args.question or None,
    )
    summary = run.run()
    records_path = args.records or args.output.with_suffix(".records.jsonl")
    write_outputs(
        summary=summary,
        records=run.records,
        output=args.output,
        records_path=records_path,
    )
    if not args.quiet:
        printable = {
            key: value
            for key, value in summary.items()
            if key not in ("error_samples",)
        }
        print(json.dumps(printable, indent=2))
        if summary["error_samples"]:
            print(f"error samples written to {records_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
