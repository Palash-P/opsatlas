"""Bounded, opt-in retention for terminal ``ingestion_jobs`` rows.

Retention never touches documents, document versions, chunks, tenants, or ACL
rows. Only terminal job audit rows are eligible, and only when an operator asks
for them:

* ``succeeded`` jobs may be removed after a configurable age.
* ``failed`` jobs are removed only when ``include_failed`` is set explicitly.
* ``dead_letter`` jobs are removed only when ``include_dead_letter`` is set
  explicitly, because they are the primary forensic record of repeated failure.

Active jobs (``queued``, ``running``) are never eligible. Every operation is
bounded by ``limit`` and can be scoped to one tenant. Counting is a pure read, so
``dry_run`` is guaranteed not to mutate anything.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.core.observability import WORKER_LOGGER_NAME
from app.db.connections import get_engine

MAX_RETENTION_LIMIT = 5000
ACTIVE_JOB_STATUSES = frozenset({"queued", "running"})

logger = logging.getLogger(WORKER_LOGGER_NAME)


@dataclass(frozen=True)
class RetentionPolicy:
    """Which terminal jobs an operator is willing to delete, and how old."""

    succeeded_days: int = 30
    failed_days: int = 90
    dead_letter_days: int = 365
    include_failed: bool = False
    include_dead_letter: bool = False
    limit: int = 500
    tenant_id: UUID | None = None

    def __post_init__(self) -> None:
        for name in ("succeeded_days", "failed_days", "dead_letter_days"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be at least 1")
        if not 1 <= self.limit <= MAX_RETENTION_LIMIT:
            raise ValueError(f"limit must be between 1 and {MAX_RETENTION_LIMIT}")

    def plan(self) -> list[tuple[str, int]]:
        """Return ``(status, age_days)`` pairs to evaluate."""

        plan = [("succeeded", self.succeeded_days)]
        if self.include_failed:
            plan.append(("failed", self.failed_days))
        if self.include_dead_letter:
            plan.append(("dead_letter", self.dead_letter_days))
        return plan


@dataclass(frozen=True)
class RetentionReport:
    counts: dict[str, int]
    applied: bool
    tenant_scoped: bool
    limit: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "event": "ingestion_retention",
            "counts": dict(self.counts),
            "total": sum(self.counts.values()),
            "applied": self.applied,
            "tenant_scoped": self.tenant_scoped,
            "limit": self.limit,
        }


def _eligible_predicate() -> str:
    # ``finished_at`` is set by terminal transitions; fall back to ``created_at``
    # for old rows that predate that convention. Active statuses never match
    # because the caller passes one terminal status explicitly.
    return (
        "status = :status "
        "AND COALESCE(finished_at, created_at) < now() - make_interval(days => :days) "
        "AND (CAST(:tenant_id AS uuid) IS NULL OR tenant_id = CAST(:tenant_id AS uuid))"
    )


def count_eligible_jobs(
    policy: RetentionPolicy,
    *,
    engine: Engine | None = None,
) -> dict[str, int]:
    """Count eligible rows without deleting anything."""

    engine = engine or get_engine()
    counts: dict[str, int] = {}
    with engine.connect() as connection:
        for status, days in policy.plan():
            count = connection.execute(
                text(
                    "SELECT count(*) FROM ("
                    "  SELECT id FROM ingestion_jobs"
                    f"  WHERE {_eligible_predicate()}"
                    "  ORDER BY COALESCE(finished_at, created_at), id"
                    "  LIMIT :limit"
                    ") AS eligible"
                ),
                {
                    "status": status,
                    "days": days,
                    "tenant_id": policy.tenant_id,
                    "limit": policy.limit,
                },
            ).scalar_one()
            counts[status] = int(count)
        connection.rollback()
    return counts


def delete_eligible_jobs(
    policy: RetentionPolicy,
    *,
    engine: Engine | None = None,
) -> dict[str, int]:
    """Delete at most ``policy.limit`` rows per selected terminal status."""

    engine = engine or get_engine()
    deleted: dict[str, int] = {}
    with engine.begin() as connection:
        for status, days in policy.plan():
            result = connection.execute(
                text(
                    "DELETE FROM ingestion_jobs WHERE id IN ("
                    "  SELECT id FROM ingestion_jobs"
                    f"  WHERE {_eligible_predicate()}"
                    "  ORDER BY COALESCE(finished_at, created_at), id"
                    "  LIMIT :limit"
                    ")"
                ),
                {
                    "status": status,
                    "days": days,
                    "tenant_id": policy.tenant_id,
                    "limit": policy.limit,
                },
            )
            deleted[status] = int(result.rowcount or 0)
    return deleted


def run_retention(
    policy: RetentionPolicy,
    *,
    apply: bool = False,
    engine: Engine | None = None,
) -> RetentionReport:
    """Count or delete eligible terminal rows and emit one sanitized event."""

    if apply:
        counts = delete_eligible_jobs(policy, engine=engine)
    else:
        counts = count_eligible_jobs(policy, engine=engine)

    report = RetentionReport(
        counts=counts,
        applied=apply,
        tenant_scoped=policy.tenant_id is not None,
        limit=policy.limit,
    )
    logger.info(
        "ingestion job retention",
        extra={"event_data": report.as_dict()},
    )
    return report
