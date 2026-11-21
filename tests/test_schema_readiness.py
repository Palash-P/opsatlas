"""Readiness and worker schema-protection tests.

Schema-shape tests run against disposable PostgreSQL databases. The API and
worker tests monkeypatch the dependency seam so they remain deterministic and do
not touch the shared development database.
"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy import text

from app.db.schema_validation import (
    SchemaValidationError,
    reset_schema_validation_cache,
    validate_database_schema,
)


def _readiness_errors(engine) -> list[str]:
    reset_schema_validation_cache()
    with pytest.raises(SchemaValidationError) as excinfo:
        validate_database_schema(engine=engine, force=True)
    return list(excinfo.value.problems)


def test_complete_schema_is_ready(migrated_postgres) -> None:
    reset_schema_validation_cache()
    status = validate_database_schema(engine=migrated_postgres.engine, force=True)

    assert status.applied_versions == (1, 2, 3, 4, 5)
    assert status.embedding_dimensions == 384


def test_missing_migration_history_prevents_readiness(migrated_postgres) -> None:
    with migrated_postgres.engine.begin() as connection:
        connection.execute(text("DROP TABLE schema_migrations"))

    problems = _readiness_errors(migrated_postgres.engine)
    assert any("schema_migrations" in problem and "python -m app.db.migrations" in problem for problem in problems)


def test_empty_migration_history_prevents_readiness_even_with_complete_schema(
    migrated_postgres,
) -> None:
    with migrated_postgres.engine.begin() as connection:
        connection.execute(text("DELETE FROM schema_migrations"))

    problems = _readiness_errors(migrated_postgres.engine)
    assert any("not recorded as applied" in problem for problem in problems)


def test_out_of_order_migration_history_prevents_readiness(migrated_postgres) -> None:
    with migrated_postgres.engine.begin() as connection:
        connection.execute(text("DELETE FROM schema_migrations WHERE version = 2"))

    problems = _readiness_errors(migrated_postgres.engine)
    assert any("out of order" in problem for problem in problems)


def test_incorrect_vector_dimension_prevents_readiness(migrated_postgres) -> None:
    with migrated_postgres.engine.begin() as connection:
        connection.execute(
            text("ALTER TABLE chunks ALTER COLUMN embedding TYPE vector(768) USING NULL::vector(768)")
        )

    problems = _readiness_errors(migrated_postgres.engine)
    assert any("chunks.embedding" in problem and "vector(768)" in problem for problem in problems)


def test_missing_phase_f_column_prevents_readiness(migrated_postgres) -> None:
    with migrated_postgres.engine.begin() as connection:
        connection.execute(text("ALTER TABLE ingestion_jobs DROP COLUMN execution_token"))

    problems = _readiness_errors(migrated_postgres.engine)
    assert any("execution_token" in problem for problem in problems)


def test_checksum_change_prevents_readiness(migrated_postgres, tmp_path, monkeypatch) -> None:
    import shutil

    from app.db import migration_registry
    from app.db.migration_registry import discover_migrations

    modified = tmp_path / "sql"
    modified.mkdir()
    for migration in discover_migrations():
        shutil.copy2(migration.path, modified / migration.filename)
    target = modified / "004_activation_generation.sql"
    target.write_text(target.read_text(encoding="utf-8") + "\n-- drift\n", encoding="utf-8")

    monkeypatch.setattr(migration_registry, "DEFAULT_MIGRATIONS_DIR", modified)
    # validate_database_schema imports discover_migrations by name.
    monkeypatch.setattr(
        "app.db.schema_validation.discover_migrations",
        lambda: discover_migrations(modified),
    )

    problems = _readiness_errors(migrated_postgres.engine)
    assert any("checksum mismatch" in problem for problem in problems)


def test_successful_validation_is_cached_but_failures_are_not(migrated_postgres, monkeypatch) -> None:
    from app.core.config import Settings

    reset_schema_validation_cache()
    settings = Settings(_env_file=None, schema_validation_cache_seconds=300)
    monkeypatch.setattr("app.core.config.get_settings", lambda: settings)

    first = validate_database_schema(engine=migrated_postgres.engine)
    with migrated_postgres.engine.begin() as connection:
        connection.execute(text("ALTER TABLE ingestion_jobs DROP COLUMN execution_token"))

    # Cached success is still returned within the TTL.
    assert validate_database_schema(engine=migrated_postgres.engine) == first

    # A forced failure is never cached; the next non-forced attempt fails too.
    with pytest.raises(SchemaValidationError):
        validate_database_schema(engine=migrated_postgres.engine, force=True)
    with pytest.raises(SchemaValidationError):
        validate_database_schema(engine=migrated_postgres.engine, force=False)


def test_worker_entry_refuses_to_resume_when_schema_validation_fails(monkeypatch) -> None:
    from app.tasks import ingestion as task_module

    calls: list[object] = []
    monkeypatch.setattr(
        task_module,
        "validate_database_schema",
        lambda: (_ for _ in ()).throw(SchemaValidationError(["missing schema"])),
    )
    monkeypatch.setattr(task_module, "resume_ingestion_job", lambda job_id: calls.append(job_id))

    result = task_module.run_ingestion_task(
        SimpleNamespace(request=SimpleNamespace(id="task-1", retries=0)),
        str(uuid4()),
    )

    assert result["outcome"] == "blocked"
    assert result["error_category"] == "schema_incompatible"
    assert calls == []


def test_reconciliation_entry_refuses_to_mutate_when_schema_validation_fails(
    monkeypatch,
) -> None:
    from app.tasks import reconciliation as task_module

    calls: list[str] = []
    monkeypatch.setattr(
        task_module,
        "validate_database_schema",
        lambda: (_ for _ in ()).throw(SchemaValidationError(["missing schema"])),
    )
    monkeypatch.setattr(
        task_module,
        "run_reconciliation",
        lambda: calls.append("ran"),
    )

    result = task_module.run_reconciliation_task()

    assert result == {
        "status": "blocked",
        "outcome": "blocked",
        "error_category": "schema_incompatible",
    }
    # A schema failure must not claim jobs, update dispatch state, or publish.
    assert calls == []


def test_worker_startup_signal_fails_before_consuming(monkeypatch) -> None:
    from app import celery_app as celery_module
    from app.db import schema_validation

    class FakeEngine:
        def dispose(self) -> None:
            return None

    monkeypatch.setattr(
        schema_validation,
        "validate_worker_startup_schema",
        lambda: (_ for _ in ()).throw(SchemaValidationError(["incompatible"])),
    )
    monkeypatch.setattr("app.db.connections.get_engine", lambda: FakeEngine())

    # Celery's signal.send swallows Exception, so startup must abort with a
    # BaseException to actually prevent the worker from consuming jobs.
    with pytest.raises(SystemExit) as excinfo:
        celery_module.validate_worker_schema_on_startup()
    assert excinfo.value.code == 1
