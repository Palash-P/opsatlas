"""Expected database schema for each migration and for application readiness.

This module is the single source of truth for what "migration N is present"
means. The migration runner uses it to decide whether an unrecorded migration
can be safely adopted, and the readiness check uses it to refuse to serve or
consume work against an incomplete schema.

A migration is never considered applied because one table or one column exists.
Each expectation lists every table, column type, constraint, index, sequence, and
data invariant that migration is responsible for. Tables and extensions that a
later migration intentionally changes are not asserted by the earlier
expectation: for example, migration 001 only requires that ``chunks.embedding``
exists, while migration 002 requires that it is exactly ``vector(384)``.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.engine import Connection

PUBLIC_SCHEMA = "public"
EMBEDDING_DIMENSIONS = 384


@dataclass(frozen=True)
class ExtensionRequirement:
    name: str


@dataclass(frozen=True)
class TableRequirement:
    table: str


@dataclass(frozen=True)
class ColumnRequirement:
    table: str
    column: str
    type_sql: str | None = None
    nullable: bool | None = None
    default_contains: str | None = None


@dataclass(frozen=True)
class IndexRequirement:
    name: str
    fragments: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConstraintRequirement:
    table: str
    name: str
    constraint_type: str
    fragments: tuple[str, ...] = ()


@dataclass(frozen=True)
class SequenceRequirement:
    name: str


@dataclass(frozen=True)
class DataRequirement:
    description: str
    sql: str


Requirement = (
    ExtensionRequirement
    | TableRequirement
    | ColumnRequirement
    | IndexRequirement
    | ConstraintRequirement
    | SequenceRequirement
    | DataRequirement
)


@dataclass(frozen=True)
class MigrationExpectation:
    version: int
    name: str
    checks: tuple[Requirement, ...]
    # A small subset used to decide whether a migration is completely absent.
    # If any state check is present without all of them being present, the
    # database is partially migrated and must fail closed.
    state_checks: tuple[Requirement, ...] | None = None

    @property
    def label(self) -> str:
        return f"{self.version:03d} {self.name}"

    def state_requirements(self) -> tuple[Requirement, ...]:
        return self.state_checks if self.state_checks is not None else self.checks


@dataclass(frozen=True)
class MigrationState:
    all_checks_present: bool
    any_state_present: bool
    problems: tuple[str, ...]
    state_problems: tuple[str, ...]


def _column(
    table: str,
    column: str,
    type_sql: str | None = None,
    *,
    nullable: bool | None = None,
    default_contains: str | None = None,
) -> ColumnRequirement:
    return ColumnRequirement(
        table=table,
        column=column,
        type_sql=type_sql,
        nullable=nullable,
        default_contains=default_contains,
    )


def _table(name: str) -> TableRequirement:
    return TableRequirement(table=name)


def _extension(name: str) -> ExtensionRequirement:
    return ExtensionRequirement(name=name)


def _index(name: str, *fragments: str) -> IndexRequirement:
    return IndexRequirement(name=name, fragments=tuple(fragment.lower() for fragment in fragments))


def _constraint(
    table: str, name: str, constraint_type: str, *fragments: str
) -> ConstraintRequirement:
    return ConstraintRequirement(
        table=table,
        name=name,
        constraint_type=constraint_type,
        fragments=tuple(fragment.lower() for fragment in fragments),
    )


def _sequence(name: str) -> SequenceRequirement:
    return SequenceRequirement(name=name)


def _data(description: str, sql: str) -> DataRequirement:
    return DataRequirement(description=description, sql=sql)


_MIGRATION_001_TABLES = (
    _table("tenants"),
    _table("users"),
    _table("documents"),
    _table("document_versions"),
    _table("document_acl"),
    _table("chunks"),
    _table("ingestion_jobs"),
    _table("query_runs"),
)

_MIGRATION_001_COLUMNS = (
    _column("tenants", "id", "uuid", nullable=False, default_contains="gen_random_uuid()"),
    _column("tenants", "name", "text", nullable=False),
    _column("tenants", "created_at", "timestamp with time zone", nullable=False, default_contains="now()"),
    _column("users", "id", "uuid", nullable=False, default_contains="gen_random_uuid()"),
    _column("users", "tenant_id", "uuid", nullable=False),
    _column("users", "external_subject", "text", nullable=False),
    _column("users", "email", "text", nullable=True),
    _column("users", "role", "text", nullable=False, default_contains="member"),
    _column("users", "created_at", "timestamp with time zone", nullable=False, default_contains="now()"),
    _column("documents", "id", "uuid", nullable=False, default_contains="gen_random_uuid()"),
    _column("documents", "tenant_id", "uuid", nullable=False),
    _column("documents", "source_key", "text", nullable=False),
    _column("documents", "title", "text", nullable=False),
    _column("documents", "source_type", "text", nullable=False),
    _column("documents", "current_version_id", "uuid", nullable=True),
    _column("documents", "status", "text", nullable=False, default_contains="active"),
    _column("documents", "created_at", "timestamp with time zone", nullable=False, default_contains="now()"),
    _column("documents", "updated_at", "timestamp with time zone", nullable=False, default_contains="now()"),
    _column("document_versions", "id", "uuid", nullable=False, default_contains="gen_random_uuid()"),
    _column("document_versions", "document_id", "uuid", nullable=False),
    _column("document_versions", "version_number", "integer", nullable=False),
    _column("document_versions", "content_hash", "text", nullable=False),
    _column("document_versions", "source_uri", "text", nullable=True),
    _column("document_versions", "authority_level", "smallint", nullable=False, default_contains="50"),
    _column("document_versions", "published_at", "timestamp with time zone", nullable=True),
    _column("document_versions", "ingested_at", "timestamp with time zone", nullable=False, default_contains="now()"),
    _column("document_versions", "is_current", "boolean", nullable=False, default_contains="false"),
    _column("document_acl", "id", "uuid", nullable=False, default_contains="gen_random_uuid()"),
    _column("document_acl", "document_id", "uuid", nullable=False),
    _column("document_acl", "principal_type", "text", nullable=False),
    _column("document_acl", "principal_id", "text", nullable=False),
    _column("document_acl", "permission", "text", nullable=False),
    _column("document_acl", "created_at", "timestamp with time zone", nullable=False, default_contains="now()"),
    _column("chunks", "id", "uuid", nullable=False, default_contains="gen_random_uuid()"),
    _column("chunks", "tenant_id", "uuid", nullable=False),
    _column("chunks", "document_id", "uuid", nullable=False),
    _column("chunks", "document_version_id", "uuid", nullable=False),
    _column("chunks", "chunk_index", "integer", nullable=False),
    _column("chunks", "content", "text", nullable=False),
    _column("chunks", "content_hash", "text", nullable=False),
    _column("chunks", "token_count", "integer", nullable=True),
    _column("chunks", "section_path", "text", nullable=True),
    _column("chunks", "metadata", "jsonb", nullable=False, default_contains="{}"),
    # The dimension is owned by migration 002, not by the original 1536 proposal.
    _column("chunks", "embedding", None, nullable=True),
    _column("chunks", "created_at", "timestamp with time zone", nullable=False, default_contains="now()"),
    _column("ingestion_jobs", "id", "uuid", nullable=False, default_contains="gen_random_uuid()"),
    _column("ingestion_jobs", "tenant_id", "uuid", nullable=False),
    _column("ingestion_jobs", "document_id", "uuid", nullable=True),
    _column("ingestion_jobs", "idempotency_key", "text", nullable=False),
    _column("ingestion_jobs", "status", "text", nullable=False, default_contains="queued"),
    _column("ingestion_jobs", "attempts", "integer", nullable=False, default_contains="0"),
    _column("ingestion_jobs", "error_code", "text", nullable=True),
    _column("ingestion_jobs", "error_summary", "text", nullable=True),
    _column("ingestion_jobs", "created_at", "timestamp with time zone", nullable=False, default_contains="now()"),
    _column("ingestion_jobs", "started_at", "timestamp with time zone", nullable=True),
    _column("ingestion_jobs", "finished_at", "timestamp with time zone", nullable=True),
    _column("query_runs", "id", "uuid", nullable=False, default_contains="gen_random_uuid()"),
    _column("query_runs", "tenant_id", "uuid", nullable=False),
    _column("query_runs", "user_id", "uuid", nullable=True),
    _column("query_runs", "request_id", "text", nullable=False),
    _column("query_runs", "config_version", "text", nullable=False),
    _column("query_runs", "query_hash", "text", nullable=False),
    _column("query_runs", "retrieved_chunk_ids", "uuid[]", nullable=False, default_contains="{}"),
    _column("query_runs", "latency_ms", "integer", nullable=True),
    _column("query_runs", "input_tokens", "integer", nullable=True),
    _column("query_runs", "output_tokens", "integer", nullable=True),
    _column("query_runs", "estimated_cost_usd", "numeric(12,8)", nullable=True),
    _column("query_runs", "outcome", "text", nullable=False, default_contains="completed"),
    _column("query_runs", "created_at", "timestamp with time zone", nullable=False, default_contains="now()"),
)

_MIGRATION_001_CONSTRAINTS = (
    _constraint(
        "documents",
        "documents_current_version_fk",
        "f",
        "foreign key (current_version_id)",
        "references document_versions",
        "deferrable initially deferred",
    ),
    _constraint("users", "users_tenant_id_fkey", "f", "references tenants", "on delete cascade"),
    _constraint("documents", "documents_tenant_id_fkey", "f", "references tenants", "on delete cascade"),
    _constraint(
        "document_versions",
        "document_versions_document_id_fkey",
        "f",
        "references documents",
        "on delete cascade",
    ),
    _constraint(
        "document_acl",
        "document_acl_document_id_fkey",
        "f",
        "references documents",
        "on delete cascade",
    ),
    _constraint("chunks", "chunks_tenant_id_fkey", "f", "references tenants", "on delete cascade"),
    _constraint(
        "chunks", "chunks_document_id_fkey", "f", "references documents", "on delete cascade"
    ),
    _constraint(
        "chunks",
        "chunks_document_version_id_fkey",
        "f",
        "references document_versions",
        "on delete cascade",
    ),
    _constraint(
        "ingestion_jobs",
        "ingestion_jobs_tenant_id_fkey",
        "f",
        "references tenants",
        "on delete cascade",
    ),
    _constraint(
        "ingestion_jobs",
        "ingestion_jobs_document_id_fkey",
        "f",
        "references documents",
        "on delete set null",
    ),
    _constraint("query_runs", "query_runs_tenant_id_fkey", "f", "references tenants", "on delete cascade"),
    _constraint("query_runs", "query_runs_user_id_fkey", "f", "references users", "on delete set null"),
)

_MIGRATION_001_INDEXES = (
    _index("chunks_tenant_document_idx", "tenant_id", "document_id"),
    _index("chunks_metadata_gin_idx", "gin", "metadata"),
    _index("chunks_content_fts_idx", "gin", "to_tsvector", "content"),
    _index("documents_tenant_status_idx", "tenant_id", "status"),
    _index("versions_document_current_idx", "document_id", "is_current"),
    _index("jobs_status_created_idx", "status", "created_at"),
    _index("query_runs_tenant_created_idx", "tenant_id", "created_at desc"),
)

MIGRATION_001 = MigrationExpectation(
    version=1,
    name="initial_schema",
    checks=(
        _extension("vector"),
        _extension("pgcrypto"),
        *_MIGRATION_001_TABLES,
        *_MIGRATION_001_COLUMNS,
        *_MIGRATION_001_CONSTRAINTS,
        *_MIGRATION_001_INDEXES,
    ),
    state_checks=_MIGRATION_001_TABLES,
)

MIGRATION_002 = MigrationExpectation(
    version=2,
    name="embedding_384",
    checks=(
        _column("chunks", "embedding", f"vector({EMBEDDING_DIMENSIONS})", nullable=True),
        _column("chunks", "embedding_model", "text", nullable=True),
        _column("chunks", "embedded_at", "timestamp with time zone", nullable=True),
    ),
)

MIGRATION_003 = MigrationExpectation(
    version=3,
    name="ingestion_jobs_version",
    checks=(
        _column("ingestion_jobs", "version_id", "uuid", nullable=True),
        _constraint(
            "ingestion_jobs",
            "ingestion_jobs_version_id_fkey",
            "f",
            "foreign key (version_id)",
            "references document_versions",
            "on delete set null",
        ),
    ),
)

MIGRATION_004 = MigrationExpectation(
    version=4,
    name="activation_generation",
    checks=(
        _index(
            "document_versions_one_current_idx",
            "unique index",
            "document_id",
            "where",
            "is_current",
        ),
        _sequence("ingestion_activation_generation_seq"),
        _column(
            "documents",
            "last_activation_generation",
            "bigint",
            nullable=False,
            default_contains="0",
        ),
        _column("ingestion_jobs", "activation_generation", "bigint", nullable=True),
        _data(
            "at most one current document version per document",
            """
            SELECT count(*)
            FROM (
                SELECT document_id
                FROM document_versions
                WHERE is_current
                GROUP BY document_id
                HAVING count(*) > 1
            ) AS conflicting_documents
            """,
        ),
    ),
    state_checks=(
        _index(
            "document_versions_one_current_idx",
            "unique index",
            "document_id",
            "where",
            "is_current",
        ),
        _sequence("ingestion_activation_generation_seq"),
        _column(
            "documents",
            "last_activation_generation",
            "bigint",
            nullable=False,
            default_contains="0",
        ),
        _column("ingestion_jobs", "activation_generation", "bigint", nullable=True),
    ),
)

MIGRATION_005 = MigrationExpectation(
    version=5,
    name="job_reconciliation",
    checks=(
        _column("ingestion_jobs", "next_dispatch_at", "timestamp with time zone", nullable=True),
        _column(
            "ingestion_jobs", "dispatch_claimed_until", "timestamp with time zone", nullable=True
        ),
        _column("ingestion_jobs", "last_dispatch_at", "timestamp with time zone", nullable=True),
        _column(
            "ingestion_jobs",
            "dispatch_attempts",
            "integer",
            nullable=False,
            default_contains="0",
        ),
        _column("ingestion_jobs", "dispatch_error_code", "text", nullable=True),
        _column("ingestion_jobs", "execution_token", "uuid", nullable=True),
        _column(
            "ingestion_jobs",
            "execution_lease_expires_at",
            "timestamp with time zone",
            nullable=True,
        ),
        _constraint(
            "ingestion_jobs",
            "ingestion_jobs_dispatch_attempts_nonnegative",
            "c",
            "dispatch_attempts >= 0",
        ),
        _index(
            "ingestion_jobs_reconcile_idx",
            "status",
            "next_dispatch_at",
            "where",
            "queued",
            "running",
            "failed",
        ),
        _index(
            "ingestion_jobs_execution_lease_idx",
            "execution_lease_expires_at",
            "where",
            "running",
            "execution_token is not null",
        ),
    ),
)

EXPECTATIONS: dict[int, MigrationExpectation] = {
    expectation.version: expectation
    for expectation in (MIGRATION_001, MIGRATION_002, MIGRATION_003, MIGRATION_004, MIGRATION_005)
}

REQUIRED_MIGRATION_VERSIONS = tuple(sorted(EXPECTATIONS))


def expectation_for(version: int) -> MigrationExpectation | None:
    return EXPECTATIONS.get(version)


def _evaluate(connection: Connection, requirement: Requirement) -> str | None:
    if isinstance(requirement, ExtensionRequirement):
        found = connection.execute(
            text("SELECT 1 FROM pg_extension WHERE extname = :name"),
            {"name": requirement.name},
        ).scalar_one_or_none()
        return None if found else f"extension {requirement.name!r} is missing"

    if isinstance(requirement, TableRequirement):
        found = connection.execute(
            text(
                """
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = :schema AND table_name = :table
                """
            ),
            {"schema": PUBLIC_SCHEMA, "table": requirement.table},
        ).scalar_one_or_none()
        return None if found else f"table {requirement.table!r} is missing"

    if isinstance(requirement, ColumnRequirement):
        row = (
            connection.execute(
                text(
                    """
                    SELECT format_type(a.atttypid, a.atttypmod) AS type_sql,
                           a.attnotnull,
                           pg_get_expr(d.adbin, d.adrelid) AS default_sql
                    FROM pg_attribute AS a
                    JOIN pg_class AS c ON c.oid = a.attrelid
                    JOIN pg_namespace AS n ON n.oid = c.relnamespace
                    LEFT JOIN pg_attrdef AS d ON d.adrelid = a.attrelid AND d.adnum = a.attnum
                    WHERE n.nspname = :schema
                      AND c.relname = :table
                      AND a.attname = :column
                      AND a.attnum > 0
                      AND NOT a.attisdropped
                    """
                ),
                {
                    "schema": PUBLIC_SCHEMA,
                    "table": requirement.table,
                    "column": requirement.column,
                },
            )
            .mappings()
            .one_or_none()
        )
        label = f"column {requirement.table}.{requirement.column}"
        if row is None:
            return f"{label} is missing"
        if requirement.type_sql is not None and row["type_sql"] != requirement.type_sql:
            return f"{label} has type {row['type_sql']!r}, expected {requirement.type_sql!r}"
        if requirement.nullable is not None:
            actual_nullable = not bool(row["attnotnull"])
            if actual_nullable is not requirement.nullable:
                expected = "NULL" if requirement.nullable else "NOT NULL"
                return f"{label} must be {expected}"
        if requirement.default_contains is not None:
            default_sql = (row["default_sql"] or "").lower()
            if requirement.default_contains.lower() not in default_sql:
                return (
                    f"{label} default {row['default_sql']!r} does not contain "
                    f"{requirement.default_contains!r}"
                )
        return None

    if isinstance(requirement, IndexRequirement):
        indexdef = connection.execute(
            text(
                """
                SELECT indexdef
                FROM pg_indexes
                WHERE schemaname = :schema AND indexname = :name
                """
            ),
            {"schema": PUBLIC_SCHEMA, "name": requirement.name},
        ).scalar_one_or_none()
        if indexdef is None:
            return f"index {requirement.name!r} is missing"
        normalized = " ".join(indexdef.lower().split())
        missing = [fragment for fragment in requirement.fragments if fragment not in normalized]
        if missing:
            return f"index {requirement.name!r} does not match expected definition (missing {missing})"
        return None

    if isinstance(requirement, ConstraintRequirement):
        row = (
            connection.execute(
                text(
                    """
                    SELECT c.contype, pg_get_constraintdef(c.oid) AS definition
                    FROM pg_constraint AS c
                    JOIN pg_class AS t ON t.oid = c.conrelid
                    JOIN pg_namespace AS n ON n.oid = t.relnamespace
                    WHERE n.nspname = :schema
                      AND t.relname = :table
                      AND c.conname = :name
                    """
                ),
                {
                    "schema": PUBLIC_SCHEMA,
                    "table": requirement.table,
                    "name": requirement.name,
                },
            )
            .mappings()
            .one_or_none()
        )
        if row is None:
            return f"constraint {requirement.name!r} on {requirement.table!r} is missing"
        if str(row["contype"]) != requirement.constraint_type:
            return (
                f"constraint {requirement.name!r} has type {row['contype']!r}, "
                f"expected {requirement.constraint_type!r}"
            )
        definition = " ".join(str(row["definition"]).lower().split())
        missing = [fragment for fragment in requirement.fragments if fragment not in definition]
        if missing:
            return (
                f"constraint {requirement.name!r} does not match expected definition "
                f"(missing {missing})"
            )
        return None

    if isinstance(requirement, SequenceRequirement):
        found = connection.execute(
            text(
                """
                SELECT 1
                FROM information_schema.sequences
                WHERE sequence_schema = :schema AND sequence_name = :name
                """
            ),
            {"schema": PUBLIC_SCHEMA, "name": requirement.name},
        ).scalar_one_or_none()
        return None if found else f"sequence {requirement.name!r} is missing"

    if isinstance(requirement, DataRequirement):
        try:
            violating_rows = int(connection.execute(text(requirement.sql)).scalar_one())
        except Exception as exc:  # noqa: BLE001 - report an actionable schema problem
            return f"could not evaluate invariant {requirement.description!r}: {exc}"
        if violating_rows:
            return (
                f"data invariant {requirement.description!r} is violated "
                f"by {violating_rows} row(s)"
            )
        return None

    raise TypeError(f"unknown schema requirement: {requirement!r}")  # pragma: no cover


def evaluate_requirements(
    connection: Connection, requirements: tuple[Requirement, ...]
) -> list[str]:
    problems: list[str] = []
    for requirement in requirements:
        problem = _evaluate(connection, requirement)
        if problem is not None:
            problems.append(problem)
    return problems


def inspect_migration_state(
    connection: Connection, expectation: MigrationExpectation
) -> MigrationState:
    """Determine whether a migration is fully, partially, or not present.

    ``all_checks_present`` means safe adoption. ``any_state_present`` without a
    complete check set means a partial manual application; the runner must stop
    rather than execute SQL that could mask the discrepancy.
    """

    problems = evaluate_requirements(connection, expectation.checks)
    state_requirements = expectation.state_requirements()
    state_problems = evaluate_requirements(connection, state_requirements)
    return MigrationState(
        all_checks_present=not problems,
        any_state_present=len(state_problems) < len(state_requirements),
        problems=tuple(problems),
        state_problems=tuple(state_problems),
    )


def verify_migration(connection: Connection, expectation: MigrationExpectation) -> None:
    problems = evaluate_requirements(connection, expectation.checks)
    if problems:
        raise RuntimeError(
            f"migration {expectation.label} did not produce its expected schema:\n  - "
            + "\n  - ".join(problems)
        )


def full_schema_problems(connection: Connection) -> list[str]:
    """Return every problem across all known migrations."""

    problems: list[str] = []
    for version in REQUIRED_MIGRATION_VERSIONS:
        expectation = EXPECTATIONS[version]
        for problem in evaluate_requirements(connection, expectation.checks):
            problems.append(f"{expectation.label}: {problem}")
    return problems


def verify_full_schema(connection: Connection) -> None:
    problems = full_schema_problems(connection)
    if problems:
        raise RuntimeError(
            "database schema does not satisfy the application contract:\n  - "
            + "\n  - ".join(problems)
        )
