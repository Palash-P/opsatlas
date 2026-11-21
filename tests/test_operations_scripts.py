"""Tests for operational backup, restore, retention, and dead-letter tooling.

Database-backed tests use disposable PostgreSQL databases created by the shared
fixtures. Backup and restore integration is skipped when PostgreSQL client tools
are not installed. No test targets the shared development database.
"""

from __future__ import annotations

import shutil
from contextlib import contextmanager
from pathlib import Path
from uuid import UUID, uuid4

import psycopg
import pytest
from psycopg import sql
from sqlalchemy import text
from sqlalchemy.engine import make_url

from app.services.dead_letters import list_dead_letter_jobs
from app.services.retention import (
    RetentionPolicy,
    count_eligible_jobs,
    delete_eligible_jobs,
)
from scripts.ops import db_backup, db_restore

PG_TOOLS_AVAILABLE = bool(shutil.which("pg_dump") and shutil.which("pg_restore"))
TENANT_ID = UUID("33333333-3333-4333-8333-333333333333")
ZERO_VECTOR = "[" + ",".join(["0"] * 384) + "]"


def _admin_dsn(database_url: str) -> str:
    return (
        make_url(database_url)
        .set(drivername="postgresql", database="postgres")
        .render_as_string(hide_password=False)
    )


def _target_url(source_url: str, name: str) -> str:
    return (
        make_url(source_url)
        .set(database=name)
        .render_as_string(hide_password=False)
    )


@contextmanager
def _disposable_target(source_url: str):
    name = f"opsatlas_restore_{uuid4().hex}"
    admin_dsn = _admin_dsn(source_url)
    admin = psycopg.connect(admin_dsn, autocommit=True)
    try:
        yield name, _target_url(source_url, name)
    finally:
        try:
            admin.execute(
                """
                SELECT pg_terminate_backend(pid)
                FROM pg_stat_activity
                WHERE datname = %s AND pid <> pg_backend_pid()
                """,
                (name,),
            )
            admin.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(name)))
        finally:
            admin.close()


def _seed_document(engine) -> None:
    document_id = uuid4()
    version_id = uuid4()
    chunk_id = uuid4()
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO tenants (id, name) VALUES (:id, :name) "
                "ON CONFLICT (id) DO NOTHING"
            ),
            {"id": TENANT_ID, "name": "operations-test"},
        )
        connection.execute(
            text(
                """
                INSERT INTO documents (id, tenant_id, source_key, title, source_type)
                VALUES (:id, :tenant, :source_key, :title, 'runbook')
                """
            ),
            {
                "id": document_id,
                "tenant": TENANT_ID,
                "source_key": "runbooks/backup-test.md",
                "title": "Backup test",
            },
        )
        connection.execute(
            text(
                """
                INSERT INTO document_versions
                    (id, document_id, version_number, content_hash, is_current)
                VALUES (:id, :document, 1, :hash, TRUE)
                """
            ),
            {"id": version_id, "document": document_id, "hash": "hash-v1"},
        )
        connection.execute(
            text(
                """
                INSERT INTO chunks
                    (id, tenant_id, document_id, document_version_id, chunk_index,
                     content, content_hash, embedding)
                VALUES (:id, :tenant, :document, :version, 0,
                        'backup test content', 'chunk-hash', CAST(:embedding AS vector))
                """
            ),
            {
                "id": chunk_id,
                "tenant": TENANT_ID,
                "document": document_id,
                "version": version_id,
                "embedding": ZERO_VECTOR,
            },
        )


def _seed_job(
    engine,
    *,
    status: str,
    age_days: int,
    error_code: str | None = None,
    error_summary: str | None = None,
) -> UUID:
    job_id = uuid4()
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO tenants (id, name)
                VALUES (:id, 'operations-test')
                ON CONFLICT (id) DO NOTHING
                """
            ),
            {"id": TENANT_ID},
        )
        connection.execute(
            text(
                """
                INSERT INTO ingestion_jobs
                    (id, tenant_id, idempotency_key, status, error_code,
                     error_summary, created_at, finished_at)
                VALUES
                    (:id, :tenant, :key, :status, :error_code, :error_summary,
                     now() - make_interval(days => :age),
                     now() - make_interval(days => :age))
                """
            ),
            {
                "id": job_id,
                "tenant": TENANT_ID,
                "key": f"retention-{job_id}",
                "status": status,
                "error_code": error_code,
                "error_summary": error_summary,
                "age": age_days,
            },
        )
    return job_id


def _statuses(engine) -> list[str]:
    with engine.connect() as connection:
        return [
            row[0]
            for row in connection.execute(
                text("SELECT status FROM ingestion_jobs ORDER BY status")
            ).all()
        ]


# ---------------------------------------------------------------------------
# Backup and restore
# ---------------------------------------------------------------------------


def test_backup_requires_an_explicit_database_url(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(SystemExit) as excinfo:
        db_backup.main(["--output", str(tmp_path / "backup.dump")])
    assert excinfo.value.code == 2


def test_backup_never_overwrites_an_existing_file(migrated_postgres, tmp_path) -> None:
    output = tmp_path / "existing.dump"
    output.write_bytes(b"not a dump")

    with pytest.raises(FileExistsError):
        db_backup.create_backup(migrated_postgres.url, output)


def test_backup_records_database_history_and_counts(migrated_postgres, tmp_path) -> None:
    if not PG_TOOLS_AVAILABLE:
        pytest.skip("PostgreSQL client tools are not installed")
    _seed_document(migrated_postgres.engine)
    output = tmp_path / "opsatlas.dump"

    manifest = db_backup.create_backup(migrated_postgres.url, output)

    assert output.exists()
    assert manifest["verified"] is True
    assert manifest["database"] == migrated_postgres.name
    assert manifest["applied_versions"] == [1, 2, 3, 4, 5]
    assert manifest["embedding_type"] == "vector(384)"
    assert manifest["row_counts"]["documents"] == 1
    assert manifest["row_counts"]["document_versions"] == 1
    assert manifest["row_counts"]["chunks"] == 1
    assert db_backup.manifest_path(output).exists()
    assert not db_backup.manifest_path(output).read_text(encoding="utf-8").count(
        "opsatlas_dev_password"
    )


def test_restore_rejects_reserved_source_and_unconfirmed_targets() -> None:
    with pytest.raises(SystemExit):
        db_restore.build_parser().parse_args(["--backup", "backup.dump"])
    with pytest.raises(ValueError):
        db_restore.validate_restore_target(
            source_database="opsatlas",
            target_database="opsatlas",
            confirm="opsatlas",
        )
    with pytest.raises(ValueError):
        db_restore.validate_restore_target(
            source_database="opsatlas",
            target_database="opsatlas_restore_1",
            confirm="different-name",
        )
    with pytest.raises(ValueError):
        db_restore.validate_restore_target(
            source_database="opsatlas",
            target_database="opsatlas",
            confirm="opsatlas_restore_1",
        )


def test_restore_refuses_to_reuse_an_existing_database(migrated_postgres, tmp_path) -> None:
    if not PG_TOOLS_AVAILABLE:
        pytest.skip("PostgreSQL client tools are not installed")
    output = tmp_path / "reuse.dump"
    manifest = db_backup.create_backup(migrated_postgres.url, output)

    with pytest.raises(ValueError) as excinfo:
        db_restore.restore_backup(
            output,
            migrated_postgres.url,
            confirm=migrated_postgres.name,
            manifest=manifest,
        )

    # The target name is reserved/reused before any destructive action occurs.
    assert "reserved" in str(excinfo.value) or "source database" in str(excinfo.value)


def test_restore_dry_run_does_not_create_the_target(migrated_postgres, tmp_path) -> None:
    if not PG_TOOLS_AVAILABLE:
        pytest.skip("PostgreSQL client tools are not installed")
    output = tmp_path / "dry-run.dump"
    manifest = db_backup.create_backup(migrated_postgres.url, output)

    with _disposable_target(migrated_postgres.url) as (name, target_url):
        plan = db_restore.restore_backup(
            output,
            target_url,
            confirm=name,
            manifest=manifest,
            dry_run=True,
        )

        assert plan["restored"] is False
        with psycopg.connect(_admin_dsn(migrated_postgres.url)) as admin:
            exists = admin.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s", (name,)
            ).fetchone()
        assert exists is None


def test_backup_restores_into_a_separate_database_with_matching_counts(
    migrated_postgres, tmp_path
) -> None:
    if not PG_TOOLS_AVAILABLE:
        pytest.skip("PostgreSQL client tools are not installed")
    _seed_document(migrated_postgres.engine)
    output = tmp_path / "restore.dump"
    manifest = db_backup.create_backup(migrated_postgres.url, output)

    with _disposable_target(migrated_postgres.url) as (name, target_url):
        result = db_restore.restore_backup(
            output,
            target_url,
            confirm=name,
            manifest=manifest,
        )

        assert result["restored"] is True
        assert result["target_database"] == name
        assert result["row_counts"]["documents"] == 1
        assert result["row_counts"]["document_versions"] == 1
        assert result["row_counts"]["chunks"] == 1
        assert result["embedding_type"] == "vector(384)"
        assert result["applied_versions"] == [1, 2, 3, 4, 5]

        restored = db_backup.inspect_database(target_url)
        assert restored["row_counts"] == manifest["row_counts"]


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------


def test_retention_dry_run_does_not_delete_anything(migrated_postgres) -> None:
    _seed_document(migrated_postgres.engine)
    _seed_job(migrated_postgres.engine, status="succeeded", age_days=90)
    _seed_job(migrated_postgres.engine, status="succeeded", age_days=1)
    _seed_job(migrated_postgres.engine, status="running", age_days=90)
    _seed_job(migrated_postgres.engine, status="dead_letter", age_days=400)

    policy = RetentionPolicy(succeeded_days=30, limit=10)
    before = _statuses(migrated_postgres.engine)

    counts = count_eligible_jobs(policy, engine=migrated_postgres.engine)

    assert counts == {"succeeded": 1}
    assert _statuses(migrated_postgres.engine) == before


def test_retention_apply_only_deletes_selected_terminal_statuses(
    migrated_postgres,
) -> None:
    _seed_document(migrated_postgres.engine)
    _seed_job(migrated_postgres.engine, status="succeeded", age_days=90)
    _seed_job(migrated_postgres.engine, status="running", age_days=90)
    _seed_job(migrated_postgres.engine, status="dead_letter", age_days=400)
    document_count = _count(migrated_postgres.engine, "documents")
    chunk_count = _count(migrated_postgres.engine, "chunks")

    deleted = delete_eligible_jobs(
        RetentionPolicy(succeeded_days=30, limit=10),
        engine=migrated_postgres.engine,
    )

    assert deleted == {"succeeded": 1}
    assert _statuses(migrated_postgres.engine) == ["dead_letter", "running"]
    # Retention must never delete documents, versions, chunks, or ACL rows.
    assert _count(migrated_postgres.engine, "documents") == document_count
    assert _count(migrated_postgres.engine, "chunks") == chunk_count


def test_retention_requires_explicit_opt_in_for_failures_and_dead_letters(
    migrated_postgres,
) -> None:
    _seed_job(migrated_postgres.engine, status="failed", age_days=400, error_code="x")
    _seed_job(
        migrated_postgres.engine,
        status="dead_letter",
        age_days=400,
        error_code="x",
    )

    assert count_eligible_jobs(
        RetentionPolicy(limit=10), engine=migrated_postgres.engine
    ) == {"succeeded": 0}

    assert count_eligible_jobs(
        RetentionPolicy(include_failed=True, include_dead_letter=True, limit=10),
        engine=migrated_postgres.engine,
    ) == {"succeeded": 0, "failed": 1, "dead_letter": 1}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"succeeded_days": 0},
        {"limit": 0},
        {"limit": 5001},
    ],
)
def test_retention_policy_validates_bounds(kwargs) -> None:
    with pytest.raises(ValueError):
        RetentionPolicy(**kwargs)


# ---------------------------------------------------------------------------
# Dead-letter inspection
# ---------------------------------------------------------------------------


def test_dead_letter_inspection_is_read_only_and_tenant_salted(
    migrated_postgres,
) -> None:
    dead_id = _seed_job(
        migrated_postgres.engine,
        status="dead_letter",
        age_days=2,
        error_code="embedding_service_unavailable",
        error_summary="Retries exhausted after transient failures",
    )
    _seed_job(migrated_postgres.engine, status="succeeded", age_days=2)

    jobs = list_dead_letter_jobs(
        limit=10, tenant_id=TENANT_ID, engine=migrated_postgres.engine
    )

    assert len(jobs) == 1
    job = jobs[0]
    assert job.job_id == str(dead_id)
    assert job.tenant_ref.startswith("t_")
    assert str(TENANT_ID) not in job.tenant_ref
    assert job.error_code == "embedding_service_unavailable"

    # Nothing was mutated by the read-only inspection.
    assert _statuses(migrated_postgres.engine) == ["dead_letter", "succeeded"]


def test_dead_letter_inspection_rejects_unbounded_limits(migrated_postgres) -> None:
    with pytest.raises(ValueError):
        list_dead_letter_jobs(limit=0, engine=migrated_postgres.engine)
    with pytest.raises(ValueError):
        list_dead_letter_jobs(limit=1001, engine=migrated_postgres.engine)


def test_dead_letter_inspection_without_a_tenant_filter_is_read_only(
    migrated_postgres,
) -> None:
    dead_id = _seed_job(migrated_postgres.engine, status="dead_letter", age_days=1)
    _seed_job(migrated_postgres.engine, status="succeeded", age_days=1)

    jobs = list_dead_letter_jobs(limit=10, engine=migrated_postgres.engine)

    assert [job.job_id for job in jobs] == [str(dead_id)]
    assert _statuses(migrated_postgres.engine) == ["dead_letter", "succeeded"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count(engine, table: str) -> int:
    with engine.connect() as connection:
        return int(connection.execute(text(f'SELECT count(*) FROM "{table}"')).scalar_one())


def test_backup_manifest_sidecar_path_is_stable(tmp_path: Path) -> None:
    assert db_backup.manifest_path(tmp_path / "x.dump") == tmp_path / "x.dump.manifest.json"
