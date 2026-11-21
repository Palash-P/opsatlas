"""List dead-letter ingestion jobs without changing durable state.

This command is read-only. Output uses salted tenant references rather than raw
tenant UUIDs and never includes document content or embeddings. See
``app/services/dead_letters.py`` for why requeueing is intentionally out of
scope for this release.
"""

from __future__ import annotations

import argparse
import json
from uuid import UUID

from app.core.config import get_settings
from app.core.observability import configure_logging
from app.services.dead_letters import MAX_INSPECTION_LIMIT, list_dead_letter_jobs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument(
        "--tenant-id",
        type=UUID,
        default=None,
        help="Restrict to one tenant UUID (output uses a salted reference)",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON lines")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 1 <= args.limit <= MAX_INSPECTION_LIMIT:
        parser.error(f"--limit must be between 1 and {MAX_INSPECTION_LIMIT}")

    configure_logging(get_settings())
    jobs = list_dead_letter_jobs(limit=args.limit, tenant_id=args.tenant_id)

    if args.json:
        for job in jobs:
            print(json.dumps(job.as_dict(), sort_keys=True))
        return 0

    if not jobs:
        print("no dead-letter ingestion jobs")
        return 0

    print(f"{len(jobs)} dead-letter ingestion job(s):")
    for job in jobs:
        print(
            f"  job={job.job_id} tenant={job.tenant_ref} "
            f"attempts={job.attempts} dispatch_attempts={job.dispatch_attempts} "
            f"error={job.error_code} finished={job.finished_at} "
            f"summary={job.error_summary!r}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
