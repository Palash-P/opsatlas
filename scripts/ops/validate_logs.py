"""Validate OpsAtlas structured request logs against load-test request IDs.

This is intentionally a test helper, not a log shipper. It checks correlation
and redaction only; it does not store or forward the log stream.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_events(path: Path) -> tuple[list[dict[str, Any]], int]:
    events: list[dict[str, Any]] = []
    non_json = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                non_json += 1
                continue
            if isinstance(payload, dict) and payload.get("event") == "request":
                events.append(payload)
    return events, non_json


def index_events(events: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    indexed: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        request_id = event.get("request_id")
        if isinstance(request_id, str):
            indexed.setdefault(request_id, []).append(event)
    return indexed


def check_expectation(
    expectation: dict[str, Any],
    indexed: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    request_id = expectation.get("request_id")
    matches = indexed.get(request_id, [])
    failures: list[str] = []
    if not request_id:
        return {"request_id": None, "passed": False, "failures": ["missing request_id"]}
    if len(matches) != 1:
        return {
            "request_id": request_id,
            "passed": False,
            "failures": [f"expected exactly one log event, found {len(matches)}"],
        }
    event = matches[0]
    for field, expected in expectation.items():
        if field == "request_id" or field.startswith("__"):
            continue
        actual = event.get(field)
        if actual != expected:
            failures.append(f"{field}: expected {expected!r}, got {actual!r}")
    non_null = expectation.get("__non_null__", [])
    if isinstance(non_null, list):
        for field in non_null:
            if event.get(field) is None:
                failures.append(f"{field}: expected a non-null value")
    return {"request_id": request_id, "passed": not failures, "failures": failures}


def scan_forbidden(
    events: list[dict[str, Any]],
    forbidden: list[str],
) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for event in events:
        serialized = json.dumps(event, sort_keys=True, default=str)
        for value in forbidden:
            if value and value in serialized:
                hits.append(
                    {
                        "request_id": event.get("request_id"),
                        "forbidden_value_length": len(value),
                        "field": "serialized_event",
                    }
                )
    return hits


def validate(
    *,
    log_path: Path,
    expectations: list[dict[str, Any]],
    forbidden: list[str],
    require_unique: bool = True,
) -> dict[str, Any]:
    events, non_json = load_events(log_path)
    indexed = index_events(events)
    checks = [check_expectation(item, indexed) for item in expectations]
    forbidden_hits = scan_forbidden(events, forbidden)

    requested_ids = {
        item["request_id"]
        for item in expectations
        if isinstance(item.get("request_id"), str)
    }
    duplicates = {
        request_id: len(matches)
        for request_id, matches in indexed.items()
        if require_unique and request_id in requested_ids and len(matches) != 1
    }
    missing = sorted(
        request_id for request_id in requested_ids if request_id not in indexed
    )

    return {
        "log_file": str(log_path),
        "event_count": len(events),
        "non_json_lines": non_json,
        "expectations_checked": len(checks),
        "expectations_passed": sum(1 for check in checks if check["passed"]),
        "expectation_failures": [check for check in checks if not check["passed"]],
        "forbidden_hits": forbidden_hits,
        "missing_request_ids": missing,
        "duplicate_request_ids": duplicates,
        "passed": (
            all(check["passed"] for check in checks)
            and not forbidden_hits
            and not missing
            and not duplicates
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument(
        "--expectations",
        type=Path,
        required=True,
        help="JSON array of expected request-log fields.",
    )
    parser.add_argument(
        "--forbid",
        action="append",
        default=[],
        help="String that must never appear in a structured request event. Repeatable.",
    )
    parser.add_argument("--forbid-file", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    expectations = json.loads(args.expectations.read_text(encoding="utf-8"))
    if not isinstance(expectations, list):
        raise SystemExit("--expectations must contain a JSON array")
    forbidden = list(args.forbid)
    if args.forbid_file is not None:
        forbidden.extend(
            line.strip()
            for line in args.forbid_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    report = validate(
        log_path=args.log,
        expectations=expectations,
        forbidden=forbidden,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
