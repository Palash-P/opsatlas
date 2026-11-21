"""Restore a custom-format backup into a separate, disposable database.

Safety rules enforced before any write:

* The target database must be named explicitly and confirmed with ``--confirm``.
* The target must differ from the database recorded in the backup manifest.
* Reserved names (``opsatlas``, ``postgres``, ``template0``, ``template1``) are
  rejected, so the shared application database can never be the default target.
* An existing database is never dropped or reused; the command fails instead.
* ``--dry-run`` validates every argument and prints the plan without writing.

The restore always creates a new database and restores with
``pg_restore --no-owner --no-privileges``. If the client is newer than the
server, the archive is rendered to SQL and the one unsupported session setting
is removed before applying it with ``psql``/``ON_ERROR_STOP``. Row counts,
migration history, and the embedding column type are compared against the
manifest after the restore.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import psycopg
from psycopg import sql
from sqlalchemy.engine import make_url

if __package__ in (None, ""):
    # Allow ``python scripts/ops/db_restore.py`` as well as ``python -m``.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ops.db_backup import (
    database_name,
    inspect_database,
    manifest_path,
    postgres_dsn,
    verify_archive,
)

RESERVED_TARGET_DATABASES = frozenset({"opsatlas", "postgres", "template0", "template1"})


def _require_tool(name: str) -> str:
    import shutil

    executable = shutil.which(name)
    if executable is None:
        raise RuntimeError(
            f"{name} is not installed or not on PATH; install PostgreSQL client tools"
        )
    return executable


def load_manifest(backup: Path, explicit: Path | None = None) -> dict[str, Any]:
    path = explicit or manifest_path(backup)
    if not path.exists():
        raise FileNotFoundError(f"backup manifest not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("backup manifest must be a JSON object")
    return data


def validate_restore_target(
    *,
    source_database: str | None,
    target_database: str,
    confirm: str,
) -> None:
    """Reject targets that could overwrite the shared or a reserved database."""

    if not target_database:
        raise ValueError("target database name is required")
    if target_database in RESERVED_TARGET_DATABASES:
        raise ValueError(
            f"refusing to restore into reserved database {target_database!r}; "
            "create a separate disposable database"
        )
    if source_database and target_database == source_database:
        raise ValueError(
            "refusing to restore into the source database; restore into a separate database"
        )
    if confirm != target_database:
        raise ValueError(
            "--confirm must exactly match the target database name "
            f"({target_database!r})"
        )


def _database_exists(admin_dsn: str, name: str) -> bool:
    with psycopg.connect(admin_dsn, connect_timeout=10) as connection:
        row = connection.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (name,)
        ).fetchone()
    return row is not None


def _create_database(target_url: str) -> None:
    url = make_url(target_url)
    admin_dsn = (
        url.set(drivername="postgresql", database="postgres").render_as_string(
            hide_password=False
        )
    )
    target_name = database_name(target_url)
    with psycopg.connect(admin_dsn, autocommit=True, connect_timeout=10) as connection:
        connection.execute(
            sql.SQL("CREATE DATABASE {}").format(sql.Identifier(target_name))
        )


def _server_version_num(target_url: str) -> int:
    with psycopg.connect(postgres_dsn(target_url), connect_timeout=10) as connection:
        row = connection.execute("SHOW server_version_num").fetchone()
    return int(row[0]) if row else 0


def _run_restore(backup: Path, target_url: str) -> subprocess.CompletedProcess[str]:
    """Restore the archive, tolerating a newer client than the server.

    ``pg_dump`` 17+ emits ``SET transaction_timeout``, which PostgreSQL 16 and
    older reject. When the client is newer than the server, render the archive
    to SQL, drop only that unsupported session setting, and apply the script
    with ``ON_ERROR_STOP``. Otherwise restore the archive directly.
    """

    pg_restore = _require_tool("pg_restore")
    target_dsn = postgres_dsn(target_url)
    server_version = _server_version_num(target_url)

    if server_version >= 170000:
        return subprocess.run(
            [
                pg_restore,
                "--no-owner",
                "--no-privileges",
                "--exit-on-error",
                f"--dbname={target_dsn}",
                str(backup),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

    psql = _require_tool("psql")
    with tempfile.TemporaryDirectory(prefix="opsatlas-restore-") as tmp:
        script = Path(tmp) / "restore.sql"
        render = subprocess.run(
            [
                pg_restore,
                "--no-owner",
                "--no-privileges",
                f"--file={script}",
                str(backup),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if render.returncode != 0 or not script.exists():
            return render

        lines = script.read_text(encoding="utf-8").splitlines(keepends=True)
        script.write_text(
            "".join(
                line
                for line in lines
                if not line.strip().startswith("SET transaction_timeout = ")
            ),
            encoding="utf-8",
        )
        return subprocess.run(
            [
                psql,
                "--no-psqlrc",
                "--quiet",
                "-v",
                "ON_ERROR_STOP=1",
                f"--dbname={target_dsn}",
                "--file",
                str(script),
            ],
            capture_output=True,
            text=True,
            check=False,
        )


def _compare_manifest(
    restored: dict[str, Any], manifest: dict[str, Any]
) -> list[str]:
    problems: list[str] = []
    expected_counts = manifest.get("row_counts") or {}
    actual_counts = restored.get("row_counts") or {}
    for table, expected in expected_counts.items():
        if expected is None:
            continue
        if actual_counts.get(table) != expected:
            problems.append(
                f"row count mismatch for {table}: expected {expected}, "
                f"found {actual_counts.get(table)}"
            )
    if (
        manifest.get("embedding_type")
        and restored.get("embedding_type") != manifest["embedding_type"]
    ):
        problems.append(
            "embedding column type mismatch: expected "
            f"{manifest['embedding_type']}, found {restored.get('embedding_type')}"
        )
    if manifest.get("applied_versions") != restored.get("applied_versions"):
        problems.append(
            "migration history mismatch: expected "
            f"{manifest.get('applied_versions')}, found {restored.get('applied_versions')}"
        )
    return problems


def restore_backup(
    backup: Path,
    target_url: str,
    *,
    confirm: str,
    manifest: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Create a new database and restore the archive into it."""

    if not backup.exists():
        raise FileNotFoundError(f"backup file not found: {backup}")

    manifest = manifest or load_manifest(backup)
    source_database = manifest.get("database")
    target_name = database_name(target_url)
    validate_restore_target(
        source_database=source_database,
        target_database=target_name,
        confirm=confirm,
    )

    archive_entries = verify_archive(backup)

    url = make_url(target_url)
    admin_dsn = (
        url.set(drivername="postgresql", database="postgres").render_as_string(
            hide_password=False
        )
    )
    exists = _database_exists(admin_dsn, target_name)
    if exists:
        raise RuntimeError(
            f"target database {target_name!r} already exists; "
            "choose a new disposable database name"
        )

    plan = {
        "backup": str(backup),
        "source_database": source_database,
        "target_database": target_name,
        "archive_entries": len(archive_entries),
        "dry_run": dry_run,
    }
    if dry_run:
        plan["restored"] = False
        return plan

    _create_database(target_url)
    result = _run_restore(backup, target_url)
    if result.returncode != 0:
        raise RuntimeError(
            "pg_restore failed; the partially restored target database was left "
            "in place for inspection"
        )

    restored = inspect_database(target_url)
    problems = _compare_manifest(restored, manifest)
    if problems:
        raise RuntimeError("restore verification failed: " + "; ".join(problems))

    plan.update(
        {
            "restored": True,
            "row_counts": restored["row_counts"],
            "embedding_type": restored["embedding_type"],
            "applied_versions": restored["applied_versions"],
        }
    )
    return plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup", required=True, type=Path, help="Source .dump file")
    parser.add_argument(
        "--target-database-url",
        required=True,
        help="URL for a new, separate database (never the source database)",
    )
    parser.add_argument(
        "--confirm",
        required=True,
        help="Exact target database name; guards against a mistyped URL",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Manifest path (defaults to <backup>.manifest.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate arguments and print the plan without creating anything",
    )
    parser.add_argument("--json", action="store_true", help="Print the plan as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        manifest = load_manifest(args.backup, args.manifest)
        result = restore_backup(
            args.backup,
            args.target_database_url,
            confirm=args.confirm,
            manifest=manifest,
            dry_run=args.dry_run,
        )
    except (RuntimeError, FileExistsError, FileNotFoundError, TypeError, ValueError) as exc:
        print(f"restore failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, sort_keys=True))
    elif result["dry_run"]:
        print(
            "restore plan: "
            f"source={result['source_database']} "
            f"target={result['target_database']} "
            f"archive_entries={result['archive_entries']} (dry run, nothing created)"
        )
    else:
        counts = result["row_counts"]
        print(
            "restore complete: "
            f"target={result['target_database']} "
            "counts="
            + ",".join(f"{name}={value}" for name, value in counts.items())
            + f" embedding={result['embedding_type']} "
            f"versions={result['applied_versions']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
