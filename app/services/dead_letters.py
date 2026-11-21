"""Read-only inspection of dead-letter ingestion jobs.

Dead-letter rows are the durable forensic record of an ingestion that exhausted
its retry budget. They are tenant-scoped, sanitized for display, and never
mutated by this module.

A requeue helper is deliberately **not** provided in this release. Requeueing a
dead-letter job changes durable state and republishes work; it needs an explicit
operator workflow, a fencing transition, and an audit record. Operators can
still recover a job by fixing the underlying cause and resubmitting the source
document through the normal authenticated ingestion API, which creates a new
idempotent target instead of rewriting history.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.core.observability import tenant_reference
from app.db.connections import get_engine

MAX_INSPECTION_LIMIT = 1000


@dataclass(frozen=True)
class DeadLetterJob:
    job_id: str
    tenant_ref: str
    document_id: str | None
    version_id: str | None
    attempts: int
    error_code: str | None
    error_summary: str | None
    dispatch_attempts: int
    created_at: datetime
    finished_at: datetime | None
    last_dispatch_at: datetime | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "tenant_ref": self.tenant_ref,
            "document_id": self.document_id,
            "version_id": self.version_id,
            "attempts": self.attempts,
            "error_code": self.error_code,
            "error_summary": self.error_summary,
            "dispatch_attempts": self.dispatch_attempts,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "last_dispatch_at": (
                self.last_dispatch_at.isoformat() if self.last_dispatch_at else None
            ),
        }


def list_dead_letter_jobs(
    *,
    limit: int = 50,
    tenant_id: UUID | None = None,
    engine: Engine | None = None,
) -> list[DeadLetterJob]:
    """Return at most ``limit`` dead-letter rows without mutating anything."""

    if not 1 <= limit <= MAX_INSPECTION_LIMIT:
        raise ValueError(f"limit must be between 1 and {MAX_INSPECTION_LIMIT}")

    engine = engine or get_engine()
    with engine.connect() as connection:
        # Fail closed if any future edit accidentally tries to write here.
        connection.execute(text("SET TRANSACTION READ ONLY"))
        rows = (
            connection.execute(
                text(
                    """
                    SELECT id, tenant_id, document_id, version_id, attempts,
                           error_code, error_summary, dispatch_attempts,
                           created_at, finished_at, last_dispatch_at
                    FROM ingestion_jobs
                    WHERE status = 'dead_letter'
                      AND (CAST(:tenant_id AS uuid) IS NULL
                           OR tenant_id = CAST(:tenant_id AS uuid))
                    ORDER BY COALESCE(finished_at, created_at) DESC, id
                    LIMIT :limit
                    """
                ),
                {"tenant_id": tenant_id, "limit": limit},
            )
            .mappings()
            .all()
        )
        connection.rollback()

    return [
        DeadLetterJob(
            job_id=str(row["id"]),
            tenant_ref=tenant_reference(row["tenant_id"]),
            document_id=str(row["document_id"]) if row["document_id"] else None,
            version_id=str(row["version_id"]) if row["version_id"] else None,
            attempts=int(row["attempts"]),
            error_code=row["error_code"],
            # Summaries are written by the application, never by a provider or
            # database driver, so they cannot contain request bodies or secrets.
            error_summary=row["error_summary"],
            dispatch_attempts=int(row["dispatch_attempts"]),
            created_at=row["created_at"],
            finished_at=row["finished_at"],
            last_dispatch_at=row["last_dispatch_at"],
        )
        for row in rows
    ]
