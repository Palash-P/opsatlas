"""Run one bounded ingestion-job reconciliation batch from an operator shell.

The same service is scheduled by Celery beat. This command is for controlled
manual recovery when beat is paused or for inspecting a single batch. It prints
only sanitized counters; document content, embeddings, secrets, and exception
text are never emitted.
"""

from __future__ import annotations

import argparse
import json

from app.core.config import get_settings
from app.core.observability import configure_logging
from app.services.reconciliation import (
    MAX_RECONCILIATION_BATCH_SIZE,
    run_reconciliation,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Maximum jobs to claim (defaults to INGESTION_RECONCILE_BATCH_SIZE)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the sanitized report as one JSON object",
    )
    args = parser.parse_args()

    if args.batch_size is not None and not (1 <= args.batch_size <= MAX_RECONCILIATION_BATCH_SIZE):
        parser.error(f"--batch-size must be between 1 and {MAX_RECONCILIATION_BATCH_SIZE}")

    settings = get_settings()
    configure_logging(settings)

    report = run_reconciliation(batch_size=args.batch_size).as_dict()
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(
            "ingestion reconciliation: "
            f"scanned={report['scanned']} dispatched={report['dispatched']} "
            f"skipped={report['skipped']} failures={report['failures']} "
            f"locked={report['locked']} duration_ms={report['duration_ms']}"
        )

    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
