"""Inspect or apply bounded retention for terminal ingestion jobs.

The default is a dry run: eligible rows are counted and nothing is deleted. Pass
``--apply`` to delete them. Failed and dead-letter jobs are excluded unless
``--include-failed`` / ``--include-dead-letter`` are supplied. Documents,
versions, chunks, tenants, and ACL rows are never touched.
"""

from __future__ import annotations

import argparse
import json
from uuid import UUID

from app.core.config import get_settings
from app.core.observability import configure_logging
from app.services.retention import (
    MAX_RETENTION_LIMIT,
    RetentionPolicy,
    run_retention,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete eligible rows (default is a read-only dry run)",
    )
    parser.add_argument("--succeeded-days", type=int, default=30)
    parser.add_argument("--failed-days", type=int, default=90)
    parser.add_argument("--dead-letter-days", type=int, default=365)
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Also consider permanently failed jobs",
    )
    parser.add_argument(
        "--include-dead-letter",
        action="store_true",
        help="Also consider dead-letter jobs (forensic record; usually keep)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help=f"Maximum rows per status per run (1-{MAX_RETENTION_LIMIT})",
    )
    parser.add_argument(
        "--tenant-id",
        type=UUID,
        default=None,
        help="Restrict to one tenant UUID (salted references are logged)",
    )
    parser.add_argument("--json", action="store_true", help="Print one JSON object")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not 1 <= args.limit <= MAX_RETENTION_LIMIT:
        parser.error(f"--limit must be between 1 and {MAX_RETENTION_LIMIT}")

    try:
        policy = RetentionPolicy(
            succeeded_days=args.succeeded_days,
            failed_days=args.failed_days,
            dead_letter_days=args.dead_letter_days,
            include_failed=args.include_failed,
            include_dead_letter=args.include_dead_letter,
            limit=args.limit,
            tenant_id=args.tenant_id,
        )
    except ValueError as exc:
        parser.error(str(exc))

    configure_logging(get_settings())
    report = run_retention(policy, apply=args.apply).as_dict()

    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        mode = "deleted" if report["applied"] else "eligible (dry run)"
        print(
            f"retention {mode}: "
            + ", ".join(f"{status}={count}" for status, count in report["counts"].items())
            + f", total={report['total']}, limit={report['limit']}, "
            f"tenant_scoped={report['tenant_scoped']}"
        )
        if not report["applied"]:
            print("nothing was deleted; pass --apply to execute")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
