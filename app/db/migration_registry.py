"""Discovery, checksums, and history validation for versioned SQL migrations.

The repository intentionally keeps a very small migration mechanism: ordered
``sql/NNN_name.sql`` files, a ``schema_migrations`` table, and PostgreSQL
advisory locks. This module owns the file and history rules that must be shared
by the runner and by application readiness checks.

Historical migrations are treated as immutable. A checksum is the SHA-256 of the
UTF-8 file with line endings normalized to ``\\n`` so a Windows checkout and a
Linux checkout of the same commit hash the same content. Any other byte change
is a checksum change and is rejected once the migration has been recorded.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.engine import Connection

DEFAULT_MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "sql"

MIGRATION_FILENAME_PATTERN = re.compile(r"^(?P<version>\d+)_(?P<name>[a-z0-9_]+)\.sql$")

SCHEMA_MIGRATIONS_TABLE = "schema_migrations"

CREATE_SCHEMA_MIGRATIONS_SQL = f"""
CREATE TABLE IF NOT EXISTS {SCHEMA_MIGRATIONS_TABLE} (
    version INTEGER PRIMARY KEY CHECK (version > 0),
    filename TEXT NOT NULL UNIQUE,
    checksum TEXT NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    applied_by TEXT NOT NULL DEFAULT 'runner'
)
"""


class MigrationError(RuntimeError):
    """Base class for migration plan, history, lock, and execution failures."""


class MigrationPlanError(MigrationError):
    """The SQL directory is missing, malformed, or lacks a schema contract."""


class MigrationHistoryError(MigrationError):
    """Recorded migration history is missing, changed, or out of order."""


class MigrationLockError(MigrationError):
    """Another migration runner currently owns the migration advisory lock."""


class MigrationExecutionError(MigrationError):
    """One migration failed and was rolled back."""


class PartialMigrationError(MigrationError):
    """An unrecorded migration has some, but not all, of its expected schema."""


class MigrationAdoptionRequired(MigrationError):
    """Existing schema matches a migration that has no tracking row."""


@dataclass(frozen=True)
class MigrationFile:
    version: int
    name: str
    filename: str
    path: Path
    checksum: str
    sql: str

    @property
    def label(self) -> str:
        return f"{self.version:03d} {self.name}"


@dataclass(frozen=True)
class AppliedMigration:
    version: int
    filename: str
    checksum: str
    applied_at: object
    applied_by: str


def checksum_for_bytes(raw: bytes) -> str:
    """Return the canonical SHA-256 for a migration file's content."""

    try:
        decoded = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:  # pragma: no cover - malformed repository file
        raise MigrationPlanError("migration file is not valid UTF-8") from exc
    normalized = decoded.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def discover_migrations(directory: Path | str | None = None) -> list[MigrationFile]:
    """Load ``NNN_name.sql`` files in deterministic version order.

    Any ``.sql`` file that does not follow the naming convention is an error:
    silently ignoring a mistyped migration would make the applied schema diverge
    from the repository without an obvious failure.
    """

    root = Path(directory) if directory is not None else DEFAULT_MIGRATIONS_DIR
    if not root.is_dir():
        raise MigrationPlanError(f"migration directory does not exist: {root}")

    migrations: list[MigrationFile] = []
    seen_versions: dict[int, str] = {}

    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.suffix.lower() != ".sql":
            continue

        match = MIGRATION_FILENAME_PATTERN.match(path.name)
        if match is None:
            raise MigrationPlanError(
                f"invalid migration filename {path.name!r}; expected NNN_lower_snake_case.sql"
            )

        version = int(match.group("version"))
        if version <= 0:
            raise MigrationPlanError(f"migration version must be positive: {path.name}")
        if version in seen_versions:
            raise MigrationPlanError(
                "duplicate migration version "
                f"{version}: {seen_versions[version]!r} and {path.name!r}"
            )

        raw = path.read_bytes()
        try:
            sql = raw.decode("utf-8-sig")
        except UnicodeDecodeError as exc:  # pragma: no cover - malformed repository file
            raise MigrationPlanError(f"migration file is not valid UTF-8: {path.name}") from exc

        migrations.append(
            MigrationFile(
                version=version,
                name=match.group("name"),
                filename=path.name,
                path=path,
                checksum=checksum_for_bytes(raw),
                sql=sql,
            )
        )
        seen_versions[version] = path.name

    if not migrations:
        raise MigrationPlanError(f"no migration files found in {root}")

    migrations.sort(key=lambda item: (item.version, item.filename))
    return migrations


def schema_migrations_exists(connection: Connection) -> bool:
    value = connection.execute(
        text("SELECT to_regclass('public.schema_migrations')")
    ).scalar_one_or_none()
    return value is not None


def ensure_schema_migrations_table(connection: Connection) -> None:
    connection.exec_driver_sql(CREATE_SCHEMA_MIGRATIONS_SQL)


def read_applied_migrations(connection: Connection) -> list[AppliedMigration] | None:
    """Return tracked migrations, or ``None`` when the tracking table is absent."""

    if not schema_migrations_exists(connection):
        return None

    rows = (
        connection.execute(
            text(
                """
                SELECT version, filename, checksum, applied_at, applied_by
                FROM schema_migrations
                ORDER BY version
                """
            )
        )
        .mappings()
        .all()
    )
    return [
        AppliedMigration(
            version=int(row["version"]),
            filename=str(row["filename"]),
            checksum=str(row["checksum"]),
            applied_at=row["applied_at"],
            applied_by=str(row["applied_by"]),
        )
        for row in rows
    ]


def validate_history(
    migrations: list[MigrationFile],
    applied: list[AppliedMigration],
) -> None:
    """Reject changed, unknown, missing, or out-of-order migration history.

    Applied history must be exactly the first ``N`` migrations in file order.
    This catches a checksum change, an applied migration whose file disappeared,
    a newly inserted migration older than the newest applied version, and a gap
    caused by deleting or manually editing history.
    """

    problems: list[str] = []
    by_version = {migration.version: migration for migration in migrations}

    expected_prefix = migrations[: len(applied)]
    for index, record in enumerate(applied):
        migration = by_version.get(record.version)
        if migration is None:
            problems.append(
                f"applied version {record.version} ({record.filename}) has no migration file"
            )
            continue

        if index >= len(expected_prefix):
            problems.append(
                f"applied version {record.version} is beyond the available migration list"
            )
            continue

        expected = expected_prefix[index]
        if record.version != expected.version:
            problems.append(
                "migration history is out of order: expected version "
                f"{expected.version} ({expected.filename}) at position {index + 1}, "
                f"found {record.version} ({record.filename})"
            )
            continue

        if record.filename != expected.filename:
            problems.append(
                f"applied version {record.version} is recorded as {record.filename!r} "
                f"but the repository file is {expected.filename!r}"
            )
        if record.checksum != expected.checksum:
            problems.append(
                f"checksum mismatch for applied migration {expected.filename}: "
                f"recorded {record.checksum}, current {expected.checksum}"
            )

    if problems:
        raise MigrationHistoryError(
            "Migration history validation failed:\n  - " + "\n  - ".join(problems)
        )


def migration_versions_to_apply(
    migrations: list[MigrationFile],
    applied: list[AppliedMigration],
) -> list[MigrationFile]:
    applied_versions = {record.version for record in applied}
    return [migration for migration in migrations if migration.version not in applied_versions]


def _skip_single_quoted(sql: str, start: int) -> int:
    i = start + 1
    length = len(sql)
    while i < length:
        if sql[i] == "'":
            if i + 1 < length and sql[i + 1] == "'":
                i += 2
                continue
            return i + 1
        i += 1
    raise MigrationPlanError("unterminated single-quoted string in migration")


def _skip_double_quoted(sql: str, start: int) -> int:
    i = start + 1
    length = len(sql)
    while i < length:
        if sql[i] == '"':
            if i + 1 < length and sql[i + 1] == '"':
                i += 2
                continue
            return i + 1
        i += 1
    raise MigrationPlanError("unterminated double-quoted identifier in migration")


def _skip_line_comment(sql: str, start: int) -> int:
    newline = sql.find("\n", start)
    return len(sql) if newline < 0 else newline + 1


def _skip_block_comment(sql: str, start: int) -> int:
    i = start + 2
    depth = 1
    length = len(sql)
    while i < length:
        if sql.startswith("/*", i):
            depth += 1
            i += 2
        elif sql.startswith("*/", i):
            depth -= 1
            i += 2
            if depth == 0:
                return i
        else:
            i += 1
    raise MigrationPlanError("unterminated block comment in migration")


def _dollar_quote_tag(sql: str, start: int) -> str | None:
    if sql[start] != "$":
        return None
    i = start + 1
    length = len(sql)
    while i < length and (sql[i].isalnum() or sql[i] == "_"):
        i += 1
    if i >= length or sql[i] != "$":
        return None
    tag = sql[start : i + 1]
    # PostgreSQL tags are identifiers; a leading digit is not valid.
    if len(tag) > 2 and tag[1].isdigit():
        return None
    return tag


def split_sql_statements(sql: str) -> list[str]:
    """Split a migration on top-level semicolons.

    Dollar-quoted PL/pgSQL bodies, quoted strings, quoted identifiers, and both
    comment forms are opaque. This is deliberately not a full SQL parser; it is
    narrow enough to be auditable and covers the constructs used by this
    repository's migrations.
    """

    statements: list[str] = []
    start = 0
    i = 0
    length = len(sql)

    while i < length:
        char = sql[i]
        if char == "'":
            i = _skip_single_quoted(sql, i)
        elif char == '"':
            i = _skip_double_quoted(sql, i)
        elif char == "-" and i + 1 < length and sql[i + 1] == "-":
            i = _skip_line_comment(sql, i)
        elif char == "/" and i + 1 < length and sql[i + 1] == "*":
            i = _skip_block_comment(sql, i)
        elif char == "$":
            tag = _dollar_quote_tag(sql, i)
            if tag is None:
                i += 1
            else:
                closing = sql.find(tag, i + len(tag))
                if closing < 0:
                    raise MigrationPlanError(f"unterminated dollar-quoted string {tag}")
                i = closing + len(tag)
        elif char == ";":
            statements.append(sql[start:i])
            start = i + 1
            i += 1
        else:
            i += 1

    statements.append(sql[start:])
    return statements


def _leading_keyword(statement: str) -> str | None:
    """Return the first SQL keyword after leading whitespace and comments."""

    i = 0
    length = len(statement)
    while i < length:
        char = statement[i]
        if char.isspace():
            i += 1
        elif char == "-" and i + 1 < length and statement[i + 1] == "-":
            i = _skip_line_comment(statement, i)
        elif char == "/" and i + 1 < length and statement[i + 1] == "*":
            i = _skip_block_comment(statement, i)
        else:
            break

    start = i
    while i < length and (statement[i].isalpha() or statement[i] == "_"):
        i += 1
    if i == start:
        return None
    keyword = statement[start:i].upper()
    if keyword == "START":
        remainder = statement[i:].lstrip()
        if remainder[:11].upper() == "TRANSACTION":
            return "START TRANSACTION"
    return keyword


def strip_transaction_control(sql: str, *, filename: str = "<migration>") -> list[str]:
    """Return executable statements without outer ``BEGIN``/``COMMIT``.

    The runner owns the transaction boundary so the schema change and the
    ``schema_migrations`` row commit atomically. A file may wrap its statements
    in one explicit transaction, as the historical migrations do; that wrapper
    is removed. Multiple transaction blocks are rejected because flattening them
    would change failure semantics.
    """

    kept: list[str] = []
    begins = 0
    commits = 0

    for statement in split_sql_statements(sql):
        keyword = _leading_keyword(statement)
        if keyword is None:
            continue

        if keyword == "BEGIN" or keyword == "START TRANSACTION":
            begins += 1
            if begins > 1:
                raise MigrationPlanError(
                    f"{filename} contains more than one explicit transaction block; "
                    "the migration runner requires a single transaction per file"
                )
            continue

        if keyword in {"COMMIT", "END"}:
            commits += 1
            if commits > 1:
                raise MigrationPlanError(
                    f"{filename} contains more than one COMMIT; "
                    "the migration runner requires a single transaction per file"
                )
            continue

        if keyword == "ROLLBACK":
            raise MigrationPlanError(
                f"{filename} contains an explicit ROLLBACK; remove transaction "
                "control from the migration and let the runner manage it"
            )

        kept.append(statement)

    if not kept:
        raise MigrationPlanError(f"{filename} contains no executable SQL")
    return kept
