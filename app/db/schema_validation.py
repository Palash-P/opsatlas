"""Schema readiness for the API and background workers.

Process liveness is not database readiness. A process can answer HTTP or Celery
health checks while its database is empty, still at the provisional 1536-column
schema, or missing Phase F columns. This module validates both:

1. the recorded migration history (all known migrations applied, checksums
   unchanged, no out-of-order or unknown rows), and
2. the live schema required by the application (tables, Phase F columns,
   constraints, indexes, sequence, data invariants, and ``vector(384)``).

Successful validation is cached briefly; failures are never cached so an
operator sees recovery immediately after repairing the database.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from dataclasses import dataclass

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from app.db.migration_registry import (
    MigrationError,
    MigrationHistoryError,
    discover_migrations,
    read_applied_migrations,
    validate_history,
)
from app.db.schema_contract import (
    EMBEDDING_DIMENSIONS,
    REQUIRED_MIGRATION_VERSIONS,
    expectation_for,
    full_schema_problems,
)

logger = logging.getLogger("opsatlas.schema")


class SchemaValidationError(RuntimeError):
    """The live database is reachable but not compatible with this application."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = tuple(problems)
        super().__init__(
            "Database schema is not ready for this application:\n  - "
            + "\n  - ".join(self.problems)
        )


@dataclass(frozen=True)
class SchemaStatus:
    applied_versions: tuple[int, ...]
    embedding_dimensions: int
    checked_at: float


_cache_lock = threading.Lock()
_status_cache: dict[str, tuple[float, SchemaStatus]] = {}


def reset_schema_validation_cache() -> None:
    """Clear cached successes (used by tests and long-lived maintenance tools)."""

    with _cache_lock:
        _status_cache.clear()


def _cache_key(engine: Engine) -> str:
    return engine.url.render_as_string(hide_password=True)


def _schema_plan_problems() -> list[str]:
    try:
        files = discover_migrations()
    except MigrationError as exc:
        return [str(exc)]

    problems: list[str] = []
    versions = {migration.version for migration in files}
    missing = sorted(set(REQUIRED_MIGRATION_VERSIONS) - versions)
    if missing:
        problems.append(
            "required migration file(s) are missing for version(s) "
            + ", ".join(f"{version:03d}" for version in missing)
        )
    for migration in files:
        if expectation_for(migration.version) is None:
            problems.append(
                f"migration {migration.filename} has no schema contract entry in "
                "app.db.schema_contract"
            )
    return problems


def validate_database_schema(
    *,
    engine: Engine | None = None,
    force: bool = False,
) -> SchemaStatus:
    """Validate migration history and the live schema, or raise.

    ``force=True`` bypasses the short successful-validation cache. A failed
    validation always bypasses it and clears any cached success for the engine.
    """

    if engine is None:
        from app.db.connections import get_engine

        engine = get_engine()

    cache_key = _cache_key(engine)
    if not force:
        from app.core.config import get_settings

        ttl = get_settings().schema_validation_cache_seconds
        if ttl > 0:
            with _cache_lock:
                cached = _status_cache.get(cache_key)
                if cached is not None and (time.monotonic() - cached[0]) < ttl:
                    return cached[1]

    problems = _schema_plan_problems()
    applied_versions: tuple[int, ...] = ()

    try:
        with engine.connect() as connection:
            try:
                files = discover_migrations()
            except MigrationError as exc:
                problems.append(str(exc))
                files = []

            applied = read_applied_migrations(connection)
            connection.commit()
            if applied is None:
                problems.append(
                    "schema_migrations table is missing; run "
                    "python -m app.db.migrations to apply or adopt the database"
                )
            elif files:
                applied_versions = tuple(record.version for record in applied)
                try:
                    validate_history(files, applied)
                except MigrationHistoryError as exc:
                    problems.append(str(exc))

                recorded = {record.version for record in applied}
                missing = [
                    migration.version
                    for migration in files
                    if migration.version not in recorded
                ]
                if missing:
                    problems.append(
                        "migration(s) not recorded as applied: "
                        + ", ".join(f"{version:03d}" for version in missing)
                        + "; run python -m app.db.migrations to apply or adopt them"
                    )

            problems.extend(full_schema_problems(connection))
            connection.rollback()
    except SchemaValidationError:
        raise
    except Exception as exc:  # noqa: BLE001 - readiness must classify any failure
        problems.append(f"could not inspect database schema: {type(exc).__name__}: {exc}")

    if problems:
        with _cache_lock:
            _status_cache.pop(cache_key, None)
        raise SchemaValidationError(problems)

    status = SchemaStatus(
        applied_versions=applied_versions,
        embedding_dimensions=EMBEDDING_DIMENSIONS,
        checked_at=time.time(),
    )
    with _cache_lock:
        _status_cache[cache_key] = (time.monotonic(), status)
    return status


def check_schema_readiness() -> SchemaStatus:
    """Readiness entry point used by the API and worker health checks."""

    return validate_database_schema()


def validate_worker_startup_schema() -> SchemaStatus:
    """Fail worker startup before it can consume an incompatible job queue."""

    status = validate_database_schema(force=True)
    logger.info(
        "worker schema validation passed",
        extra={
            "event_data": {
                "event": "schema_validation",
                "outcome": "ready",
                "applied_versions": list(status.applied_versions),
                "embedding_dimensions": status.embedding_dimensions,
            }
        },
    )
    return status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate OpsAtlas migration history and live database schema.",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="SQLAlchemy database URL. Defaults to Settings.database_url.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore the successful-validation cache.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print nothing on success; still prints failures.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    engine = create_engine(args.database_url, pool_pre_ping=True) if args.database_url else None
    try:
        status = validate_database_schema(engine=engine, force=bool(args.force))
        if not args.quiet:
            versions = ", ".join(f"{version:03d}" for version in status.applied_versions)
            print(f"Schema ready: migrations [{versions}], vector({status.embedding_dimensions}).")
        return 0
    except SchemaValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        if engine is not None:
            engine.dispose()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
