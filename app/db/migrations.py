"""Small, deterministic PostgreSQL migration runner.

Usage::

    python -m app.db.migrations
    python -m app.db.migrations --check
    python -m app.db.migrations --no-wait --lock-timeout-ms 5000

Execution model
---------------
* A fixed PostgreSQL session advisory lock serializes runners across processes
  and hosts.
* ``schema_migrations`` records version, filename, checksum, time, and whether
  the row was executed or adopted from a verified pre-existing schema.
* Each migration file and its tracking row commit in one transaction. Explicit
  outer ``BEGIN``/``COMMIT`` statements in historical files are removed because
  the runner owns the boundary; a failure rolls back both the schema change and
  the history row.
* Historical files are never edited or replayed. A recorded checksum change is
  rejected. Existing schemas with manually applied migrations are adopted only
  when every expected object for that migration is present; a partial migration
  fails closed with the exact missing or mismatched objects.

The runner never drops, truncates, or recreates an application table.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Connection, Engine

from app.db.migration_registry import (
    DEFAULT_MIGRATIONS_DIR,
    MigrationAdoptionRequired,
    MigrationError,
    MigrationExecutionError,
    MigrationFile,
    MigrationHistoryError,
    MigrationLockError,
    MigrationPlanError,
    PartialMigrationError,
    discover_migrations,
    ensure_schema_migrations_table,
    migration_versions_to_apply,
    read_applied_migrations,
    strip_transaction_control,
    validate_history,
)
from app.db.schema_contract import (
    REQUIRED_MIGRATION_VERSIONS,
    MigrationExpectation,
    expectation_for,
    inspect_migration_state,
    verify_full_schema,
    verify_migration,
)

# Cluster-wide key for migration serialization. It is deliberately distinct
# from the document ACL and reconciliation lock families.
MIGRATION_ADVISORY_LOCK_KEY = 2_026_111_402

DEFAULT_LOCK_TIMEOUT_MS = 5_000


@dataclass
class MigrationReport:
    applied: list[int] = field(default_factory=list)
    adopted: list[int] = field(default_factory=list)
    skipped: list[int] = field(default_factory=list)
    duration_ms: int = 0

    @property
    def changed(self) -> bool:
        return bool(self.applied or self.adopted)

    def as_dict(self) -> dict[str, object]:
        return {
            "applied": list(self.applied),
            "adopted": list(self.adopted),
            "skipped": list(self.skipped),
            "duration_ms": self.duration_ms,
        }

    def format_lines(self) -> list[str]:
        lines = []
        for version in self.adopted:
            lines.append(f"adopted  {version:03d} (schema verified, SQL not executed)")
        for version in self.applied:
            lines.append(f"applied  {version:03d}")
        for version in self.skipped:
            lines.append(f"current  {version:03d}")
        if not lines:
            lines.append("no migrations found")
        return lines


def _engine_for(database_url: str | None) -> Engine:
    if database_url:
        return create_engine(database_url, pool_pre_ping=True)
    # Imported lazily so the CLI can be used with an explicit URL without
    # constructing the application settings singleton.
    from app.db.connections import get_engine

    return get_engine()


def _acquire_lock(connection: Connection, *, wait: bool) -> None:
    # The pooled connection may carry a short statement timeout. Advisory-lock
    # waiting is an operator-controlled operation, so remove the limit for this
    # session and restore it before the connection returns to the pool.
    connection.exec_driver_sql("SET statement_timeout = 0")
    if wait:
        connection.exec_driver_sql(
            "SELECT pg_advisory_lock(%s)", (MIGRATION_ADVISORY_LOCK_KEY,)
        )
        connection.commit()
        return

    acquired = bool(
        connection.exec_driver_sql(
            "SELECT pg_try_advisory_lock(%s)", (MIGRATION_ADVISORY_LOCK_KEY,)
        ).scalar()
    )
    connection.commit()
    if not acquired:
        raise MigrationLockError(
            "another migration runner owns the advisory lock; wait for it to finish "
            "or retry without --no-wait"
        )


def _release_lock(connection: Connection) -> None:
    try:
        connection.exec_driver_sql(
            "SELECT pg_advisory_unlock(%s)", (MIGRATION_ADVISORY_LOCK_KEY,)
        )
        connection.commit()
    except Exception:  # noqa: BLE001 - closing the connection also releases the lock
        connection.rollback()
    try:
        connection.exec_driver_sql("RESET statement_timeout")
        connection.commit()
    except Exception:  # noqa: BLE001 - pool recycling restores session defaults
        connection.rollback()


def _plan(files: list[MigrationFile]) -> dict[int, MigrationExpectation]:
    discovered_versions = {migration.version for migration in files}
    missing_files = sorted(set(REQUIRED_MIGRATION_VERSIONS) - discovered_versions)
    if missing_files:
        raise MigrationPlanError(
            "required migration file(s) are missing for version(s) "
            + ", ".join(f"{version:03d}" for version in missing_files)
        )

    expectations: dict[int, MigrationExpectation] = {}
    for migration in files:
        expectation = expectation_for(migration.version)
        if expectation is None:
            raise MigrationPlanError(
                f"migration {migration.filename} has no entry in "
                "app.db.schema_contract; add its expected schema before applying it"
            )
        expectations[migration.version] = expectation
    return expectations


def _insert_history(
    connection: Connection,
    migration: MigrationFile,
    *,
    applied_by: str,
) -> None:
    connection.exec_driver_sql(
        """
        INSERT INTO schema_migrations (version, filename, checksum, applied_by)
        VALUES (%s, %s, %s, %s)
        """,
        (migration.version, migration.filename, migration.checksum, applied_by),
    )


def _execute_migration_sql(connection: Connection, statement: str) -> None:
    """Execute migration text without driver placeholder interpretation.

    psycopg treats a literal ``%`` as an incomplete placeholder even when the
    caller supplies no parameters. Historical migration ``RAISE`` messages use
    ``%``, so execute through the DBAPI cursor, which is the same connection and
    transaction SQLAlchemy manages.
    """

    cursor = connection.connection.cursor()
    try:
        cursor.execute(statement)
    finally:
        cursor.close()


def _apply_migration(
    connection: Connection,
    migration: MigrationFile,
    expectation: MigrationExpectation,
    *,
    lock_timeout_ms: int,
    verify_full_after: bool,
) -> None:
    statements = strip_transaction_control(migration.sql, filename=migration.filename)

    try:
        # Fail fast instead of queuing behind a long-running reader/writer. On a
        # populated database this turns an indefinite ACCESS EXCLUSIVE wait into
        # an explicit, retryable failure.
        _execute_migration_sql(connection, f"SET LOCAL lock_timeout = {int(lock_timeout_ms)}")
        for statement in statements:
            _execute_migration_sql(connection, statement)
        verify_migration(connection, expectation)
        if verify_full_after:
            verify_full_schema(connection)
        _insert_history(connection, migration, applied_by="runner")
        connection.commit()
    except Exception as exc:  # wrapped with migration context
        connection.rollback()
        hint = _execution_hint(migration, exc, lock_timeout_ms=lock_timeout_ms)
        raise MigrationExecutionError(
            f"migration {migration.label} ({migration.filename}) failed and was "
            f"rolled back: {exc}{hint}"
        ) from exc


def _execution_hint(migration: MigrationFile, error: BaseException, *, lock_timeout_ms: int) -> str:
    message = str(error).lower()
    if migration.version == 2 and "chunks already contain embeddings" in message:
        return (
            "\nHint: existing 1536-dimensional vectors cannot be converted to 384. "
            "See docs/DATABASE_MIGRATIONS.md#embedding-dimension-migration for the "
            "backup, explicit NULLing, migration, and re-embedding procedure. Do not "
            "truncate, pad, or reinterpret vectors."
        )
    if "lock timeout" in message or "canceling statement due to lock timeout" in message:
        return (
            "\nHint: a table lock could not be acquired within "
            f"{lock_timeout_ms} ms. Retry during a quieter window or raise "
            "--lock-timeout-ms deliberately."
        )
    return ""


def _adopt_migration(
    connection: Connection,
    migration: MigrationFile,
    expectation: MigrationExpectation,
    *,
    verify_full_after: bool,
) -> None:
    try:
        # Re-verify immediately before writing history; the state probe was made
        # in a previous read transaction.
        verify_migration(connection, expectation)
        if verify_full_after:
            verify_full_schema(connection)
        _insert_history(connection, migration, applied_by="adopted")
        connection.commit()
    except Exception as exc:  # wrapped with migration context
        connection.rollback()
        raise MigrationExecutionError(
            f"could not adopt migration {migration.label} ({migration.filename}): {exc}"
        ) from exc


def migrate(
    *,
    migrations_dir: Path | str | None = None,
    engine: Engine | None = None,
    database_url: str | None = None,
    lock_timeout_ms: int = DEFAULT_LOCK_TIMEOUT_MS,
    wait_for_lock: bool = True,
    adopt_existing: bool = True,
) -> MigrationReport:
    """Apply or adopt migrations and return a deterministic report."""

    if lock_timeout_ms < 0:
        raise ValueError("lock_timeout_ms must be non-negative")

    started = time.perf_counter()
    files = discover_migrations(migrations_dir)
    expectations = _plan(files)
    active_engine = engine or _engine_for(database_url)
    owns_engine = engine is None
    report = MigrationReport()

    try:
        with active_engine.connect() as connection:
            _acquire_lock(connection, wait=wait_for_lock)
            try:
                ensure_schema_migrations_table(connection)
                connection.commit()

                applied = read_applied_migrations(connection)
                connection.commit()
                if applied is None:  # pragma: no cover - ensured above
                    raise MigrationHistoryError("schema_migrations table could not be read")

                validate_history(files, applied)
                report.skipped = [record.version for record in applied]

                pending = migration_versions_to_apply(files, applied)
                if not pending:
                    verify_full_schema(connection)
                    connection.commit()
                    report.duration_ms = int((time.perf_counter() - started) * 1000)
                    return report

                for index, migration in enumerate(pending):
                    expectation = expectations[migration.version]
                    state = inspect_migration_state(connection, expectation)
                    connection.commit()
                    verify_full_after = index == len(pending) - 1

                    if state.all_checks_present:
                        if not adopt_existing:
                            raise MigrationAdoptionRequired(
                                f"migration {migration.label} is fully present in the "
                                "database but has no schema_migrations row; rerun "
                                "without --no-adopt to adopt the verified schema"
                            )
                        _adopt_migration(
                            connection,
                            migration,
                            expectation,
                            verify_full_after=verify_full_after,
                        )
                        report.adopted.append(migration.version)
                        continue

                    if state.any_state_present:
                        details = list(state.problems)
                        if not details:
                            details = list(state.state_problems)
                        raise PartialMigrationError(
                            f"migration {migration.label} is partially present and "
                            "cannot be applied or adopted safely. Refusing to guess. "
                            "Missing or mismatched objects:\n  - "
                            + "\n  - ".join(details)
                            + "\nRepair the schema manually (do not drop tables) or "
                            "restore the database from backup, then rerun "
                            "python -m app.db.migrations."
                        )

                    _apply_migration(
                        connection,
                        migration,
                        expectation,
                        lock_timeout_ms=lock_timeout_ms,
                        verify_full_after=verify_full_after,
                    )
                    report.applied.append(migration.version)

                verify_full_schema(connection)
                connection.commit()
                report.duration_ms = int((time.perf_counter() - started) * 1000)
                return report
            finally:
                _release_lock(connection)
    finally:
        if owns_engine:
            active_engine.dispose()


def check_migrations(
    *,
    migrations_dir: Path | str | None = None,
    engine: Engine | None = None,
    database_url: str | None = None,
) -> MigrationReport:
    """Read-only validation of history and the full application schema."""

    started = time.perf_counter()
    files = discover_migrations(migrations_dir)
    _plan(files)
    active_engine = engine or _engine_for(database_url)
    owns_engine = engine is None

    try:
        with active_engine.connect() as connection:
            applied = read_applied_migrations(connection)
            connection.commit()
            if applied is None:
                raise MigrationHistoryError(
                    "schema_migrations is missing; run python -m app.db.migrations "
                    "to apply or adopt the database"
                )
            validate_history(files, applied)
            verify_full_schema(connection)
            connection.commit()
            report = MigrationReport(
                skipped=[record.version for record in applied],
                duration_ms=int((time.perf_counter() - started) * 1000),
            )
            return report
    finally:
        if owns_engine:
            active_engine.dispose()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply or verify OpsAtlas PostgreSQL migrations.",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="SQLAlchemy database URL. Defaults to Settings.database_url.",
    )
    parser.add_argument(
        "--sql-dir",
        type=Path,
        default=None,
        help=f"Directory containing NNN_*.sql files (default: {DEFAULT_MIGRATIONS_DIR}).",
    )
    parser.add_argument(
        "--lock-timeout-ms",
        type=int,
        default=DEFAULT_LOCK_TIMEOUT_MS,
        help="Per-migration lock acquisition timeout (default: 5000).",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Fail immediately if another runner owns the advisory lock.",
    )
    parser.add_argument(
        "--no-adopt",
        action="store_true",
        help="Refuse to establish a baseline for an already-present schema.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify history and schema without writing anything.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.check:
            report = check_migrations(
                migrations_dir=args.sql_dir,
                database_url=args.database_url,
            )
            print("Migration history and schema are valid.")
            for line in report.format_lines():
                print(line)
            return 0

        report = migrate(
            migrations_dir=args.sql_dir,
            database_url=args.database_url,
            lock_timeout_ms=args.lock_timeout_ms,
            wait_for_lock=not args.no_wait,
            adopt_existing=not args.no_adopt,
        )
        for line in report.format_lines():
            print(line)
        print(f"Migration run complete in {report.duration_ms} ms.")
        return 0
    except MigrationError as exc:
        print(f"MIGRATION FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
