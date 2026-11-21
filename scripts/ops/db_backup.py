"""Create a verified, custom-format PostgreSQL backup plus a manifest.

The dump is produced with ``pg_dump --format=custom`` so it can be inspected and
selectively restored with ``pg_restore``. A sidecar ``*.manifest.json`` records
the source database, server version, migration history, embedding column type,
and application row counts. ``scripts/ops/db_restore.py`` uses the manifest to
verify a restore into a separate database.

This command is read-only against PostgreSQL. It never drops, truncates, or
rewrites data. By default it refuses to overwrite an existing output file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psycopg
from sqlalchemy.engine import make_url

EMBEDDING_DIMENSIONS = 384
MANIFEST_SUFFIX = ".manifest.json"

# Tables whose row counts prove that a restore preserved application data. The
# list deliberately excludes derived caches; there are none in this schema.
COUNTED_TABLES = ("documents", "document_versions", "chunks", "ingestion_jobs")


def postgres_dsn(database_url: str) -> str:
    """Convert a SQLAlchemy URL into a libpq DSN without the driver suffix."""

    return (
        make_url(database_url)
        .set(drivername="postgresql")
        .render_as_string(hide_password=False)
    )


def database_name(database_url: str) -> str:
    name = make_url(database_url).database
    if not name:
        raise ValueError("database URL must include a database name")
    return name


def manifest_path(output: Path) -> Path:
    return output.with_name(output.name + MANIFEST_SUFFIX)


def _table_count(connection: psycopg.Connection, table: str) -> int | None:
    exists = connection.execute(
        "SELECT to_regclass(%s) IS NOT NULL", (f"public.{table}",)
    ).fetchone()[0]
    if not exists:
        return None
    return int(connection.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0])


def inspect_database(database_url: str, *, dsn: str | None = None) -> dict[str, Any]:
    """Collect the manifest fields needed to verify a restore.

    The function tolerates a database that has not been migrated yet: missing
    tables become ``null`` rather than masking the original connection error.
    """

    dsn = dsn or postgres_dsn(database_url)
    with psycopg.connect(dsn, connect_timeout=10) as connection:
        connection.execute("SET TRANSACTION READ ONLY")
        server_version = connection.execute("SHOW server_version").fetchone()[0]
        history_present = bool(
            connection.execute(
                "SELECT to_regclass('public.schema_migrations') IS NOT NULL"
            ).fetchone()[0]
        )
        history: list[dict[str, Any]] = []
        if history_present:
            history = [
                {"version": int(row[0]), "filename": row[1], "checksum": row[2]}
                for row in connection.execute(
                    "SELECT version, filename, checksum FROM schema_migrations ORDER BY version"
                ).fetchall()
            ]
        embedding_type = connection.execute(
            """
            SELECT format_type(atttypid, atttypmod)
            FROM pg_attribute
            WHERE attrelid = to_regclass('public.chunks')
              AND attname = 'embedding'
            """
        ).fetchone()
        counts = {table: _table_count(connection, table) for table in COUNTED_TABLES}
        connection.rollback()

    return {
        "database": database_name(database_url),
        "server_version": str(server_version),
        "schema_migrations_present": history_present,
        "applied_versions": [row["version"] for row in history],
        "migration_history": history,
        "embedding_type": embedding_type[0] if embedding_type else None,
        "embedding_dimensions": EMBEDDING_DIMENSIONS,
        "row_counts": counts,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_tool(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise RuntimeError(
            f"{name} is not installed or not on PATH; install PostgreSQL client tools"
        )
    return executable


def verify_archive(archive: Path) -> list[str]:
    """Return ``pg_restore --list`` lines, raising if the dump is unreadable."""

    result = subprocess.run(
        [_require_tool("pg_restore"), "--list", str(archive)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "backup verification failed: pg_restore could not read the archive"
        )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("backup verification failed: archive contains no entries")
    return lines


def create_backup(
    database_url: str,
    output: Path,
    *,
    verify: bool = True,
    dsn: str | None = None,
) -> dict[str, Any]:
    """Write one custom-format dump and its manifest. Never modifies the source."""

    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing backup: {output}")

    dsn = dsn or postgres_dsn(database_url)
    output.parent.mkdir(parents=True, exist_ok=True)
    inspected = inspect_database(database_url, dsn=dsn)

    result = subprocess.run(
        [
            _require_tool("pg_dump"),
            "--format=custom",
            "--no-owner",
            "--no-privileges",
            f"--file={output}",
            dsn,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "pg_dump failed; check connectivity and PostgreSQL client/server versions"
        )

    manifest: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "backup_file": output.name,
        "format": "custom",
        "sha256": _file_sha256(output),
        "size_bytes": output.stat().st_size,
        **inspected,
    }

    if verify:
        entries = verify_archive(output)
        present = " ".join(entries)
        missing = sorted(
            token
            for token in ("documents", "document_versions", "chunks")
            if token not in present
        )
        if missing:
            raise RuntimeError(
                "backup verification failed: archive is missing expected tables: "
                + ", ".join(missing)
            )
        manifest["verified"] = True
        manifest["archive_entries"] = len(entries)

    sidecar = manifest_path(output)
    sidecar.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL"),
        help="Source PostgreSQL URL (defaults to DATABASE_URL; required)",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination .dump file; an existing file is never overwritten",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip pg_restore --list verification (not recommended)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the manifest as JSON instead of a summary",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.database_url:
        parser.error("--database-url or DATABASE_URL is required")

    try:
        manifest = create_backup(
            args.database_url,
            args.output,
            verify=not args.no_verify,
        )
    except (RuntimeError, FileExistsError, ValueError) as exc:
        print(f"backup failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(manifest, sort_keys=True))
    else:
        counts = manifest["row_counts"]
        print(
            "backup complete: "
            f"database={manifest['database']} "
            f"file={manifest['backup_file']} "
            f"bytes={manifest['size_bytes']} "
            f"verified={manifest.get('verified', False)} "
            "counts="
            + ",".join(f"{name}={value}" for name, value in counts.items())
            + f" embedding={manifest['embedding_type']}"
        )
        print(f"manifest: {manifest_path(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
