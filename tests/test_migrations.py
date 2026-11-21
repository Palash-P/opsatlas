"""Disposable-PostgreSQL tests for the migration runner.

Every test creates its own uniquely named database through the
``disposable_postgres`` fixture. The shared development database is never
migrated, altered, or dropped.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import psycopg
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from app.db.migration_registry import (
    DEFAULT_MIGRATIONS_DIR,
    MigrationExecutionError,
    MigrationHistoryError,
    MigrationLockError,
    PartialMigrationError,
    discover_migrations,
)
from app.db.migrations import MIGRATION_ADVISORY_LOCK_KEY, migrate
from app.db.schema_contract import full_schema_problems


def _psycopg_dsn(url: str) -> str:
    return make_url(url).set(drivername="postgresql").render_as_string(hide_password=False)


def _apply_manually(database_url: str, count: int) -> None:
    """Apply the first ``count`` files without the runner or tracking table.

    This simulates an operator running ``psql -f`` against a database before the
    migration system existed.
    """

    files = discover_migrations()[:count]
    with psycopg.connect(_psycopg_dsn(database_url), autocommit=True) as connection:
        for migration in files:
            connection.execute(migration.sql)


def _copy_migrations(target: Path) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    for migration in discover_migrations():
        shutil.copy2(migration.path, target / migration.filename)
    return target


def _history(engine) -> list[tuple[int, str, str, str]]:
    with engine.connect() as connection:
        return [
            (int(row[0]), str(row[1]), str(row[2]), str(row[3]))
            for row in connection.execute(
                text(
                    "SELECT version, filename, checksum, applied_by "
                    "FROM schema_migrations ORDER BY version"
                )
            ).all()
        ]


def test_fresh_database_applies_every_migration_and_is_idempotent(
    disposable_postgres,
) -> None:
    files = discover_migrations()
    report = migrate(engine=disposable_postgres.engine)

    assert report.applied == [migration.version for migration in files]
    assert report.adopted == []
    assert report.skipped == []

    history = _history(disposable_postgres.engine)
    assert [row[0] for row in history] == [1, 2, 3, 4, 5]
    assert [row[1] for row in history] == [migration.filename for migration in files]
    assert [row[2] for row in history] == [migration.checksum for migration in files]
    assert {row[3] for row in history} == {"runner"}

    with disposable_postgres.engine.connect() as connection:
        assert full_schema_problems(connection) == []
        embedding_type = connection.execute(
            text(
                """
                SELECT format_type(a.atttypid, a.atttypmod)
                FROM pg_attribute AS a
                WHERE a.attrelid = 'chunks'::regclass
                  AND a.attname = 'embedding'
                  AND NOT a.attisdropped
                """
            )
        ).scalar_one()
        phase_f_columns = connection.execute(
            text(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = 'ingestion_jobs'
                """
            )
        ).scalars().all()

    assert embedding_type == "vector(384)"
    assert {
        "version_id",
        "activation_generation",
        "next_dispatch_at",
        "dispatch_claimed_until",
        "last_dispatch_at",
        "dispatch_attempts",
        "dispatch_error_code",
        "execution_token",
        "execution_lease_expires_at",
    } <= set(phase_f_columns)

    second = migrate(engine=disposable_postgres.engine)
    assert second.applied == []
    assert second.adopted == []
    assert second.skipped == [1, 2, 3, 4, 5]
    assert _history(disposable_postgres.engine) == history


def test_fully_manually_migrated_database_is_adopted(disposable_postgres) -> None:
    _apply_manually(disposable_postgres.url, len(discover_migrations()))

    report = migrate(engine=disposable_postgres.engine)

    assert report.applied == []
    assert report.adopted == [1, 2, 3, 4, 5]
    history = _history(disposable_postgres.engine)
    assert [row[0] for row in history] == [1, 2, 3, 4, 5]
    assert {row[3] for row in history} == {"adopted"}


def test_partially_manually_migrated_database_adopts_then_applies(disposable_postgres) -> None:
    _apply_manually(disposable_postgres.url, 3)

    report = migrate(engine=disposable_postgres.engine)

    assert report.adopted == [1, 2, 3]
    assert report.applied == [4, 5]
    assert [row[0] for row in _history(disposable_postgres.engine)] == [1, 2, 3, 4, 5]


def test_adoption_can_be_refused_explicitly(disposable_postgres) -> None:
    from app.db.migration_registry import MigrationAdoptionRequired

    _apply_manually(disposable_postgres.url, 2)

    with pytest.raises(MigrationAdoptionRequired, match="--no-adopt"):
        migrate(engine=disposable_postgres.engine, adopt_existing=False)

    assert _history(disposable_postgres.engine) == []


def test_partially_applied_migration_fails_without_recording_it(disposable_postgres) -> None:
    _apply_manually(disposable_postgres.url, 2)
    with disposable_postgres.engine.begin() as connection:
        # Simulate a manual half-application: the column exists, its foreign key
        # does not. Migration 003 must not be executed or recorded.
        connection.execute(
            text(
                "ALTER TABLE ingestion_jobs ADD COLUMN version_id UUID"
            )
        )

    with pytest.raises(PartialMigrationError) as excinfo:
        migrate(engine=disposable_postgres.engine)

    message = str(excinfo.value)
    assert "003 ingestion_jobs_version" in message
    assert "ingestion_jobs_version_id_fkey" in message
    assert "Refusing to guess" in message
    assert [row[0] for row in _history(disposable_postgres.engine)] == [1, 2]


def test_populated_1536_schema_blocks_migration_002_without_deleting_data(
    disposable_postgres,
) -> None:
    _apply_manually(disposable_postgres.url, 1)

    with disposable_postgres.engine.begin() as connection:
        connection.execute(
            text("INSERT INTO tenants (id, name) VALUES (gen_random_uuid(), 'legacy')")
        )
        tenant_id = connection.execute(text("SELECT id FROM tenants LIMIT 1")).scalar_one()
        document_id = connection.execute(
            text(
                """
                INSERT INTO documents (tenant_id, source_key, title, source_type)
                VALUES (:tenant_id, 'legacy/runbook', 'Legacy runbook', 'runbook')
                RETURNING id
                """
            ),
            {"tenant_id": tenant_id},
        ).scalar_one()
        version_id = connection.execute(
            text(
                """
                INSERT INTO document_versions
                    (document_id, version_number, content_hash, is_current)
                VALUES (:document_id, 1, 'legacy-hash', TRUE)
                RETURNING id
                """
            ),
            {"document_id": document_id},
        ).scalar_one()
        connection.execute(
            text(
                """
                INSERT INTO chunks
                    (tenant_id, document_id, document_version_id, chunk_index,
                     content, content_hash, embedding)
                VALUES
                    (:tenant_id, :document_id, :version_id, 0,
                     'legacy content', 'legacy-content-hash',
                     array_fill(0.01::real, ARRAY[1536])::vector)
                """
            ),
            {
                "tenant_id": tenant_id,
                "document_id": document_id,
                "version_id": version_id,
            },
        )

    with pytest.raises(MigrationExecutionError) as excinfo:
        migrate(engine=disposable_postgres.engine)

    message = str(excinfo.value)
    assert "002 embedding_384" in message
    assert "chunks already contain embeddings" in message
    assert "docs/DATABASE_MIGRATIONS.md" in message

    assert [row[0] for row in _history(disposable_postgres.engine)] == [1]
    with disposable_postgres.engine.connect() as connection:
        assert connection.execute(text("SELECT count(*) FROM chunks")).scalar_one() == 1
        assert (
            connection.execute(text("SELECT count(*) FROM document_versions")).scalar_one() == 1
        )
        assert connection.execute(text("SELECT count(*) FROM documents")).scalar_one() == 1
        embedding_type = connection.execute(
            text(
                """
                SELECT format_type(a.atttypid, a.atttypmod)
                FROM pg_attribute AS a
                WHERE a.attrelid = 'chunks'::regclass AND a.attname = 'embedding'
                """
            )
        ).scalar_one()
    assert embedding_type == "vector(1536)"


def test_modified_checksum_for_applied_migration_is_rejected(
    migrated_postgres, tmp_path: Path
) -> None:
    modified_dir = _copy_migrations(tmp_path / "sql")
    target = modified_dir / "003_ingestion_jobs_version.sql"
    target.write_text(target.read_text(encoding="utf-8") + "\n-- edited\n", encoding="utf-8")

    with pytest.raises(MigrationHistoryError) as excinfo:
        migrate(engine=migrated_postgres.engine, migrations_dir=modified_dir)

    assert "checksum mismatch" in str(excinfo.value)
    assert len(_history(migrated_postgres.engine)) == 5


def test_out_of_order_history_is_rejected(migrated_postgres) -> None:
    with migrated_postgres.engine.begin() as connection:
        connection.execute(text("DELETE FROM schema_migrations WHERE version = 2"))

    with pytest.raises(MigrationHistoryError, match="out of order"):
        migrate(engine=migrated_postgres.engine)

    history = _history(migrated_postgres.engine)
    assert [row[0] for row in history] == [1, 3, 4, 5]


def test_failed_migration_is_not_recorded_and_a_later_run_recovers(
    disposable_postgres, tmp_path: Path
) -> None:
    broken_dir = _copy_migrations(tmp_path / "sql")
    broken = broken_dir / "003_ingestion_jobs_version.sql"
    broken.write_text(
        broken.read_text(encoding="utf-8")
        + "\nSELECT opsatlas_function_that_does_not_exist();\n",
        encoding="utf-8",
    )

    with pytest.raises(MigrationExecutionError) as excinfo:
        migrate(engine=disposable_postgres.engine, migrations_dir=broken_dir)

    assert "003 ingestion_jobs_version" in str(excinfo.value)
    assert [row[0] for row in _history(disposable_postgres.engine)] == [1, 2]

    recovered = migrate(engine=disposable_postgres.engine)
    assert recovered.applied == [3, 4, 5]
    assert [row[0] for row in _history(disposable_postgres.engine)] == [1, 2, 3, 4, 5]


def test_concurrent_runner_cannot_enter_the_migration_section(disposable_postgres) -> None:
    lock_engine = create_engine(disposable_postgres.url)
    try:
        with lock_engine.connect() as connection:
            connection.exec_driver_sql(
                "SELECT pg_advisory_lock(%s)", (MIGRATION_ADVISORY_LOCK_KEY,)
            )
            connection.commit()

            with pytest.raises(MigrationLockError, match="advisory lock"):
                migrate(engine=disposable_postgres.engine, wait_for_lock=False)

            with disposable_postgres.engine.connect() as probe:
                assert (
                    probe.execute(text("SELECT to_regclass('public.schema_migrations')"))
                    .scalar_one_or_none()
                    is None
                )

            connection.exec_driver_sql(
                "SELECT pg_advisory_unlock(%s)", (MIGRATION_ADVISORY_LOCK_KEY,)
            )
            connection.commit()

        report = migrate(engine=disposable_postgres.engine)
        assert report.applied == [1, 2, 3, 4, 5]
    finally:
        lock_engine.dispose()


def test_default_migration_directory_is_the_repository_sql_directory() -> None:
    assert DEFAULT_MIGRATIONS_DIR.name == "sql"
    assert [migration.version for migration in discover_migrations()] == [1, 2, 3, 4, 5]
