"""Staged, job-backed document ingestion with embedding-gated activation.

The HTTP endpoint stays synchronous, but the service is split into explicit
stages:

1. ``prepare_document`` normalizes, validates, hashes, and chunks content.
2. ``_resolve_or_create_job`` durably creates or reuses an ingestion job.
3. ``_prepare_target`` persists or reuses the target version and its chunks as
   **non-current**, links the job, and commits a durable ``running`` state.
4. ``_ensure_embeddings`` embeds missing chunks in batches of at most 32 and
   persists each batch in its own short transaction.
5. ``_verify_embedding_completeness`` checks count, model, dimensions, and
   timestamps for every chunk.
6. ``_activate_version`` atomically verifies the full contract, demotes the
   previous current version, activates the target, and marks the job succeeded.
7. ``ingest_document`` returns the existing response contract.

Transaction boundaries
---------------------
* **TX1 (job resolution):** validate the tenant, insert the job as ``queued`` or
  resolve the existing ``(tenant_id, idempotency_key)`` row, and commit.
* **TX2 (target preparation):** lock the job and document rows, assign the next
  version number while the document lock is held, insert the version with
  ``is_current = FALSE``, insert its chunks, link the job to the target, and
  persist ``running``. Commits before any embedding HTTP call.
* **TX3 (embedding persistence):** one short transaction per batch. Only chunks
  whose embedding is missing, stale for the configured model, dimensionally
  wrong, or missing a timestamp are updated.
* **TX4 (activation):** lock the job, document, and target version; re-check the
  tenant, job associations, chunk count, and every embedding; then demote the
  old current version, activate the target, update the document pointer, bump
  the activation generation, and mark the job succeeded in one commit.
* **TX5 (failure recording):** after a rollback, store a sanitized ``failed``
  state. A job that had already succeeded is never downgraded.

No database transaction is held while waiting for the embedding service.

Authorization and ACL serialization
-----------------------------------
TX1 and TX2 both re-run write authorization. For an existing document the
authorization step takes the per-document PostgreSQL advisory transaction lock
before reading ``document_acl`` and holds it through every document, version,
chunk, and job write in that transaction. Every ACL mutation takes the same
lock, so a grant or revocation cannot commit between the authorization read and
the writes it permits. Lock order is document ACL advisory lock -> ingestion job
row -> document row -> version/chunk writes; see ``app/db/advisory_locks.py``.
A new document has no ACL rows and cannot be targeted by an ACL mutation until it
commits; if a concurrent creator wins the upsert race, authorization is
re-evaluated under the lock before any version or chunk is written. TX4 activation
does not re-authorize because the durable target was already authorized; a
revocation after TX2 does not cancel already-committed ingest work.

Activation and stale executions
-------------------------------
Every execution claims a monotonic token from a PostgreSQL sequence while it
creates or reuses its target version. Activation is allowed only when that token
is greater than ``documents.last_activation_generation``. A deliberately
resubmitted historical snapshot gets a fresh token and may reactivate an older
version number, while an older ingestion that finishes embedding late is
superseded. A superseded execution still marks its job succeeded and returns
``is_current = False``; it never replaces the newer active version.

Idempotency
-----------
The key embeds the normalized content hash and a deterministic SHA-256 over the
tenant, source key, and content hash. The database
``UNIQUE (tenant_id, idempotency_key)`` constraint arbitrates concurrent
creation. Every resolved job is additionally verified against its embedded
content hash and its linked document/version.

If an already ``succeeded`` job is resubmitted for the version that is still
current, the call is a pure read: it refreshes descriptive document metadata and
returns without incrementing attempts or calling the embedding service. If the
same job must be reactivated (historical content), valid embeddings are reused;
a redundant failed execution cannot downgrade the durable success.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import text
from sqlalchemy.engine import Connection
from sqlalchemy.exc import SQLAlchemyError

from app.core.auth import (
    AuthorizationError,
    Principal,
    acl_parameters,
    acl_principal_match_sql,
    has_document_write_role,
)
from app.core.config import get_settings
from app.db.advisory_locks import acquire_document_acl_lock
from app.db.connections import get_engine
from app.db.embedding_schema import (
    EmbeddingSchemaError,
    check_embedding_schema,
    verify_embedding_schema,
)
from app.services.chunking import chunk_text, normalize_document_text
from app.services.embedding_client import (
    MAX_BATCH_SIZE,
    EmbeddingServiceError,
    EmbeddingUnavailableError,
    EmbeddingValidationError,
    embed_texts,
    expected_dimensions,
    expected_model,
    validate_embeddings,
    vector_literal,
)

logger = logging.getLogger(__name__)

IDEMPOTENCY_KEY_VERSION = "v1"
EMBEDDING_STATUS_COMPLETE = "complete"
EMBEDDING_STATUS_PENDING = "pending"
EMBEDDING_STATUS_FAILED = "failed"


class TenantNotFoundError(Exception):
    """Raised when an ingestion request references an unknown tenant."""


class EmptyDocumentError(Exception):
    """Raised when the normalized document contains no text."""


class IdempotencyKeyCollisionError(Exception):
    """Raised when a job key is already bound to different ingestion content.

    This should be practically impossible with the derived SHA-256 key, but the
    explicit check prevents a forced or future key collision from silently
    associating a job with unrelated content.
    """


class EmbeddingCompletenessError(Exception):
    """Raised when the target version does not have complete valid embeddings."""


class ActivationIntegrityError(Exception):
    """Raised when an activation would cross a tenant/document/version boundary."""


class IngestionJobNotFoundError(Exception):
    """Raised when a background task references an unknown ingestion job."""


class IngestionJobStateError(Exception):
    """Raised when a durable job cannot be resumed from its persisted target."""

    def __init__(
        self,
        message: str,
        *,
        job_id: UUID | None = None,
        tenant_id: UUID | None = None,
        source_key: str | None = None,
    ) -> None:
        super().__init__(message)
        self.job_id = job_id
        self.tenant_id = tenant_id
        self.source_key = source_key


class IngestionJobTerminalError(Exception):
    """Raised when a duplicate task targets a terminal, non-retryable job."""


class IngestionJobBusyError(Exception):
    """Raised when a task arrives while another execution holds a live lease.

    This is the expected outcome of a duplicate Celery delivery. The duplicate
    must not interrupt the healthy execution or record a failure.
    """


class IngestionExecutionLeaseLostError(Exception):
    """Raised when a worker no longer owns the execution it is trying to finish.

    The execution token is a database fencing token: after a lease expires and
    another worker claims the job, every fenced write from the expired execution
    is rejected. The late task becomes a no-op instead of overwriting the newer
    outcome or activating a stale version.
    """


class IngestionQueueUnavailableError(Exception):
    """Raised when a durable job exists but its broker publication failed."""


@dataclass(frozen=True)
class PreparedDocument:
    """Output of the deterministic normalization/hash/chunking stage."""

    content: str
    content_hash: str
    chunks: tuple[str, ...]


@dataclass(frozen=True)
class IngestionJob:
    """Minimal durable job state needed by the synchronous execution stages."""

    id: UUID
    status: str
    document_id: UUID | None
    version_id: UUID | None
    attempts: int


@dataclass(frozen=True)
class PreparedTarget:
    """A non-current version that is ready to be embedded and activated."""

    document_id: UUID
    version_id: UUID
    version_number: int
    content_hash: str
    chunk_count: int
    deduplicated: bool
    activation_generation: int
    preserve_succeeded: bool
    already_active: bool = False
    # Set for a synchronous submission that finds another live owner. The
    # caller must wait for that execution to finish and then re-run
    # ``_prepare_target`` instead of stealing ownership.
    wait_for_execution: bool = False
    # Minted by the synchronous path in TX2 so its embedding, activation, and
    # failure writes use the same fencing rules as a background worker.
    execution_token: UUID | None = None


@dataclass(frozen=True)
class ActivationResult:
    """Outcome of the atomic activation stage."""

    is_current: bool


@dataclass(frozen=True)
class SubmittedIngestion:
    """A durable async submission that may or may not need a task publication."""

    job_id: UUID
    status: str
    document_id: UUID | None
    version_id: UUID | None
    created: bool
    needs_publish: bool


@dataclass(frozen=True)
class IngestionExecutionResult:
    """Outcome of one worker execution against a durable job."""

    job_id: UUID
    tenant_id: UUID
    status: str
    attempts: int
    document_id: UUID | None
    version_id: UUID | None
    version_number: int | None
    content_hash: str | None
    chunk_count: int | None
    is_current: bool | None
    outcome: str


@dataclass(frozen=True)
class ResumableJob:
    """Persisted target state needed to finish an already-prepared job."""

    job_id: UUID
    tenant_id: UUID
    status: str
    attempts: int
    document_id: UUID
    version_id: UUID
    version_number: int
    content_hash: str
    chunk_count: int
    activation_generation: int
    source_key: str
    execution_token: UUID


# Error codes that describe a deterministic failure. A duplicate task delivery
# must not re-run these jobs and mint new attempts; only an explicit new
# submission (which resets the durable job to ``running``) may try again.
PERMANENT_JOB_ERROR_CODES = frozenset(
    {
        "activation_rejected",
        "embedding_failed",
        "embedding_incomplete",
        "embedding_invalid_response",
        "embedding_schema_mismatch",
        "ingestion_job_invalid",
    }
)

# Failures that may legitimately succeed on a later execution. Reconciliation
# may recover these only after the worker-recorded dispatch grace has elapsed, so
# it does not race a healthy Celery retry. Permanent failures and dead letters
# are deliberately absent: they are never redispatched.
TRANSIENT_JOB_ERROR_CODES = frozenset(
    {
        "database_error",
        "embedding_service_unavailable",
        "ingestion_failed",
    }
)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def prepare_document(content: str) -> PreparedDocument:
    """Stage 1: normalize, validate, hash, and chunk the submitted content."""

    normalized_content = normalize_document_text(content)
    if not normalized_content:
        raise EmptyDocumentError("Document content must contain non-whitespace text")
    return PreparedDocument(
        content=normalized_content,
        content_hash=sha256_text(normalized_content),
        chunks=tuple(chunk_text(normalized_content)),
    )


def derive_idempotency_key(tenant_id: UUID, source_key: str, content_hash: str) -> str:
    """Return a deterministic, versioned key that commits to the content hash.

    The key embeds the content hash so an existing job can be checked for a
    collision even before it has a document/version link. It never embeds
    readable content, the source key, or the tenant UUID.
    """

    material = json.dumps(
        [str(tenant_id), source_key, content_hash],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return f"{IDEMPOTENCY_KEY_VERSION}:{content_hash}:{digest}"


def _key_content_hash(idempotency_key: str) -> str | None:
    """Return the content hash embedded in a well-formed key, else ``None``."""

    parts = idempotency_key.split(":")
    if len(parts) != 3 or parts[0] != IDEMPOTENCY_KEY_VERSION or len(parts[1]) != 64:
        return None
    return parts[1]


def _job_from_row(row: Any) -> IngestionJob:
    return IngestionJob(
        id=row["id"],
        status=row["status"],
        document_id=row["document_id"],
        version_id=row["version_id"],
        attempts=row["attempts"],
    )


def _verify_job_association(
    connection: Connection,
    job_row: Any,
    *,
    tenant_id: UUID,
    source_key: str,
    content_hash: str,
) -> None:
    """Refuse to reuse a job whose key or durable links point at other content."""

    # The key itself commits to the content hash, so this also protects jobs that
    # are still queued and therefore have no document/version link yet.
    if _key_content_hash(job_row["idempotency_key"]) != content_hash:
        raise IdempotencyKeyCollisionError(
            "Idempotency key is already associated with different ingestion content"
        )

    document_id = job_row["document_id"]
    if document_id is None:
        # A queued job passed the key check but has no links to verify yet;
        # execution re-verifies after taking the row lock.
        return

    association = (
        connection.execute(
            text(
                """
            SELECT d.tenant_id, d.source_key,
                   dv.content_hash AS version_content_hash
            FROM documents AS d
            LEFT JOIN document_versions AS dv ON dv.id = :version_id
            WHERE d.id = :document_id
            """
            ),
            {"document_id": document_id, "version_id": job_row["version_id"]},
        )
        .mappings()
        .first()
    )

    if association is None:
        raise IdempotencyKeyCollisionError(
            "Idempotency key is already associated with different ingestion content"
        )
    if association["tenant_id"] != tenant_id or association["source_key"] != source_key:
        raise IdempotencyKeyCollisionError(
            "Idempotency key is already associated with different ingestion content"
        )
    if job_row["version_id"] is not None and association["version_content_hash"] != content_hash:
        raise IdempotencyKeyCollisionError(
            "Idempotency key is already associated with different ingestion content"
        )


def _find_document_id(
    connection: Connection,
    *,
    tenant_id: UUID,
    source_key: str,
) -> UUID | None:
    """Resolve the logical document for a tenant/source key without locking."""

    return connection.execute(
        text(
            """
            SELECT id
            FROM documents
            WHERE tenant_id = :tenant_id AND source_key = :source_key
            """
        ),
        {"tenant_id": tenant_id, "source_key": source_key},
    ).scalar_one_or_none()


def _authorize_document_write(
    connection: Connection,
    principal: Principal,
    *,
    tenant_id: UUID,
    source_key: str,
    document_id: UUID | None = None,
) -> UUID | None:
    """Raise :class:`AuthorizationError` unless this principal may write.

    This is the single write-authorization policy used by both the synchronous
    and asynchronous submission paths. It runs before any document, version,
    chunk, or job mutation.

    Policy:

    * the principal's authenticated tenant must be the tenant being written;
    * an existing document with one or more ACL rows requires a matching
      ``manage`` grant (a ``read`` grant or a tenant-wide writer role is not
      enough);
    * an existing document with no ACL rows requires the explicit
      ``document_writer`` role, not mere tenant membership;
    * creating a new document requires the same explicit role.

    ACL concurrency: when the document exists, the per-document advisory lock is
    acquired *before* ACL rows are read and is held until the surrounding
    transaction commits or rolls back. Every ACL mutation takes the same lock, so
    a grant or revocation cannot commit between this decision and the writes it
    authorizes. A new document has no ACL rows yet and cannot be targeted by an
    ACL mutation until it commits; the caller re-checks after the upsert if a
    concurrent creator won the race.

    Returns the resolved document ID, or ``None`` when no document exists yet.
    """

    if principal.tenant_id != tenant_id:
        raise AuthorizationError("principal tenant does not match ingestion tenant")

    if document_id is None:
        document_id = _find_document_id(
            connection,
            tenant_id=tenant_id,
            source_key=source_key,
        )

    if document_id is None:
        if not has_document_write_role(principal):
            raise AuthorizationError("document write role is required")
        return None

    acquire_document_acl_lock(connection, document_id)

    row = (
        connection.execute(
            text(
                f"""
                SELECT
                    EXISTS (
                        SELECT 1
                        FROM document_acl AS acl
                        WHERE acl.document_id = :document_id
                          AND acl.permission = 'manage'
                          AND ({acl_principal_match_sql("acl")})
                    ) AS has_manage,
                    EXISTS (
                        SELECT 1
                        FROM document_acl AS acl
                        WHERE acl.document_id = :document_id
                    ) AS has_acl
                """
            ),
            {
                "document_id": document_id,
                **acl_parameters(principal),
            },
        )
        .mappings()
        .one()
    )

    if row["has_acl"]:
        if row["has_manage"]:
            return document_id
        raise AuthorizationError("document manage permission is required")

    if not has_document_write_role(principal):
        raise AuthorizationError("document write role is required")

    return document_id


def _resolve_or_create_job(
    *,
    principal: Principal,
    tenant_id: UUID,
    idempotency_key: str,
    source_key: str,
    content_hash: str,
) -> tuple[IngestionJob, bool]:
    """Stage 2: durably create or resolve the logical job (TX1).

    Returns the job and whether this call created it. A newly created row cannot
    be a collision, so verification only applies to an existing row.
    """

    engine = get_engine()
    with engine.begin() as connection:
        tenant_exists = connection.execute(
            text("SELECT 1 FROM tenants WHERE id = :tenant_id"),
            {"tenant_id": tenant_id},
        ).scalar_one_or_none()
        if tenant_exists is None:
            raise TenantNotFoundError

        # Authorize before the job row (or any later target state) is written.
        # This also runs for duplicate and idempotent submissions because it is
        # evaluated on every request rather than reused from a prior decision.
        _authorize_document_write(
            connection,
            principal,
            tenant_id=tenant_id,
            source_key=source_key,
        )

        inserted_id = connection.execute(
            text(
                """
                INSERT INTO ingestion_jobs (tenant_id, idempotency_key, status)
                VALUES (:tenant_id, :idempotency_key, 'queued')
                ON CONFLICT (tenant_id, idempotency_key) DO NOTHING
                RETURNING id
                """
            ),
            {"tenant_id": tenant_id, "idempotency_key": idempotency_key},
        ).scalar_one_or_none()

        job_row = (
            connection.execute(
                text(
                    """
                SELECT id, idempotency_key, status, document_id, version_id, attempts
                FROM ingestion_jobs
                WHERE tenant_id = :tenant_id AND idempotency_key = :idempotency_key
                """
                ),
                {"tenant_id": tenant_id, "idempotency_key": idempotency_key},
            )
            .mappings()
            .one()
        )

        created = inserted_id is not None
        if not created:
            _verify_job_association(
                connection,
                job_row,
                tenant_id=tenant_id,
                source_key=source_key,
                content_hash=content_hash,
            )
        return _job_from_row(job_row), created


def _embedding_state(
    connection: Connection,
    *,
    version_id: UUID,
    tenant_id: UUID,
    model: str,
    dimensions: int,
) -> dict[str, int]:
    """Return aggregate embedding-integrity counters for one version."""

    row = (
        connection.execute(
            text(
                """
                SELECT
                    count(*) AS total,
                    count(*) FILTER (WHERE tenant_id <> :tenant_id) AS wrong_tenant,
                    count(*) FILTER (WHERE embedding IS NULL) AS missing_embedding,
                    count(*) FILTER (
                        WHERE embedding_model IS DISTINCT FROM :model
                    ) AS wrong_model,
                    count(*) FILTER (
                        WHERE embedding IS NOT NULL
                          AND vector_dims(embedding) IS DISTINCT FROM :dimensions
                    ) AS wrong_dimensions,
                    count(*) FILTER (WHERE embedded_at IS NULL) AS missing_embedded_at
                FROM chunks
                WHERE document_version_id = :version_id
                """
            ),
            {
                "version_id": version_id,
                "tenant_id": tenant_id,
                "model": model,
                "dimensions": dimensions,
            },
        )
        .mappings()
        .one()
    )
    return {key: int(row[key]) for key in row}


def _embedding_state_is_complete(state: dict[str, int], expected_chunk_count: int) -> bool:
    return (
        expected_chunk_count > 0
        and state["total"] == expected_chunk_count
        and state["wrong_tenant"] == 0
        and state["missing_embedding"] == 0
        and state["wrong_model"] == 0
        and state["wrong_dimensions"] == 0
        and state["missing_embedded_at"] == 0
    )


def _require_embedding_completeness(
    state: dict[str, int],
    expected_chunk_count: int,
) -> None:
    if not _embedding_state_is_complete(state, expected_chunk_count):
        raise EmbeddingCompletenessError(
            "Target version does not have complete embeddings for the configured model"
        )


def _find_existing_version(
    connection: Connection,
    *,
    document_id: UUID,
    content_hash: str,
) -> dict[str, Any] | None:
    row = (
        connection.execute(
            text(
                """
                SELECT id, version_number
                FROM document_versions
                WHERE document_id = :document_id AND content_hash = :content_hash
                """
            ),
            {"document_id": document_id, "content_hash": content_hash},
        )
        .mappings()
        .first()
    )
    return dict(row) if row is not None else None


def _count_version_chunks(connection: Connection, version_id: UUID) -> int:
    return int(
        connection.execute(
            text("SELECT count(*) FROM chunks WHERE document_version_id = :version_id"),
            {"version_id": version_id},
        ).scalar_one()
    )


def _prepare_target(
    job_id: UUID,
    *,
    principal: Principal,
    tenant_id: UUID,
    source_key: str,
    title: str,
    source_type: str,
    source_uri: str | None,
    authority_level: int,
    prepared: PreparedDocument,
    started_at: datetime,
    created: bool,
    claim_execution: bool = False,
) -> PreparedTarget:
    """Stage 3: persist/reuse the non-current target and claim a generation (TX2).

    The document row lock serializes version creation for a logical document, so
    version numbers are assigned deterministically. Nothing in this transaction
    changes ``documents.current_version_id`` or any existing ``is_current`` row.

    ``claim_execution`` is true only for the synchronous HTTP path. It mints a
    fencing token and lease in the same transaction as the target, so a
    synchronous execution is indistinguishable from a worker execution to
    reconciliation and every later write can be fenced. The asynchronous path
    leaves the job unclaimed so the worker can mint ownership when it receives
    the task.
    """

    model = expected_model()
    dimensions = expected_dimensions()
    settings = get_settings()
    engine = get_engine()

    with engine.begin() as connection:
        # Fail before touching the target if the live column cannot store the
        # configured dimension. The settings validator catches a compile-time
        # mismatch; this catches schema drift in the database.
        verify_embedding_schema(connection)

        # Lock order: document ACL advisory lock -> ingestion job row -> document
        # row -> version/chunk writes. Authorization must run, and take the ACL
        # lock, before the job row lock so ingestion and ACL mutation never
        # acquire the two locks in opposite order.
        #
        # Re-check within the target transaction before any version/chunk/job
        # write. A document may have been created or restricted between TX1 and
        # TX2, so a decision from the job-resolution transaction is not reused.
        authorized_document_id = _authorize_document_write(
            connection,
            principal,
            tenant_id=tenant_id,
            source_key=source_key,
        )

        # The row lock serializes retries of the same logical job. Re-verifying
        # after acquiring it closes the race between TX1 resolution and TX2.
        job_row = (
            connection.execute(
                text(
                    """
                    SELECT id, idempotency_key, status, document_id, version_id, attempts,
                           execution_token, activation_generation,
                           (
                               execution_lease_expires_at IS NOT NULL
                               AND execution_lease_expires_at > now()
                           ) AS execution_lease_active
                    FROM ingestion_jobs
                    WHERE id = :job_id
                    FOR UPDATE
                    """
                ),
                {"job_id": job_id},
            )
            .mappings()
            .one()
        )

        if not created:
            _verify_job_association(
                connection,
                job_row,
                tenant_id=tenant_id,
                source_key=source_key,
                content_hash=prepared.content_hash,
            )

        previous_status = job_row["status"]
        preserve_succeeded = previous_status == "succeeded"

        # A synchronous duplicate must not steal a live execution from another
        # request or worker. Waiting preserves single ownership and lets the
        # caller turn a completed duplicate into a pure read. The idempotency
        # key already commits to the content hash, and the association check
        # above confirms the job still points at this content.
        if (
            claim_execution
            and previous_status == "running"
            and job_row["execution_token"] is not None
            and job_row["execution_lease_active"]
            and job_row["document_id"] is not None
            and job_row["version_id"] is not None
            and job_row["activation_generation"] is not None
        ):
            live_target = (
                connection.execute(
                    text(
                        """
                        SELECT dv.version_number, dv.content_hash,
                               (
                                   SELECT count(*)
                                   FROM chunks AS c
                                   WHERE c.document_version_id = dv.id
                               ) AS chunk_count
                        FROM document_versions AS dv
                        WHERE dv.id = :version_id
                          AND dv.document_id = :document_id
                        """
                    ),
                    {
                        "version_id": job_row["version_id"],
                        "document_id": job_row["document_id"],
                    },
                )
                .mappings()
                .first()
            )
            if live_target is not None and live_target["content_hash"] == prepared.content_hash:
                return PreparedTarget(
                    document_id=job_row["document_id"],
                    version_id=job_row["version_id"],
                    version_number=int(live_target["version_number"]),
                    content_hash=live_target["content_hash"],
                    chunk_count=int(live_target["chunk_count"]),
                    deduplicated=True,
                    activation_generation=int(job_row["activation_generation"]),
                    preserve_succeeded=False,
                    wait_for_execution=True,
                )

        # A true duplicate of the current, fully embedded version is a read: it
        # must not increment attempts, reset timestamps, or re-embed anything.
        if preserve_succeeded and job_row["document_id"] and job_row["version_id"]:
            existing = (
                connection.execute(
                    text(
                        """
                        SELECT dv.id, dv.version_number, dv.content_hash,
                               d.tenant_id, d.current_version_id,
                               (
                                   SELECT count(*)
                                   FROM chunks AS c
                                   WHERE c.document_version_id = dv.id
                               ) AS chunk_count
                        FROM document_versions AS dv
                        JOIN documents AS d ON d.id = dv.document_id
                        WHERE dv.id = :version_id
                          AND dv.document_id = :document_id
                        """
                    ),
                    {
                        "version_id": job_row["version_id"],
                        "document_id": job_row["document_id"],
                    },
                )
                .mappings()
                .first()
            )

            if (
                existing is not None
                and existing["tenant_id"] == tenant_id
                and existing["content_hash"] == prepared.content_hash
            ):
                state = _embedding_state(
                    connection,
                    version_id=existing["id"],
                    tenant_id=tenant_id,
                    model=model,
                    dimensions=dimensions,
                )
                version_is_current = existing["current_version_id"] == existing["id"]
                version_is_complete = _embedding_state_is_complete(
                    state,
                    int(existing["chunk_count"]),
                )
                if version_is_current and version_is_complete:
                    connection.execute(
                        text(
                            """
                            UPDATE documents
                            SET title = :title,
                                source_type = :source_type,
                                updated_at = now()
                            WHERE id = :document_id AND tenant_id = :tenant_id
                            """
                        ),
                        {
                            "title": title,
                            "source_type": source_type,
                            "document_id": job_row["document_id"],
                            "tenant_id": tenant_id,
                        },
                    )
                    return PreparedTarget(
                        document_id=job_row["document_id"],
                        version_id=existing["id"],
                        version_number=existing["version_number"],
                        content_hash=existing["content_hash"],
                        chunk_count=int(existing["chunk_count"]),
                        deduplicated=True,
                        activation_generation=0,
                        preserve_succeeded=True,
                        already_active=True,
                    )

        execution_token: UUID | None = uuid4() if claim_execution else None
        connection.execute(
            text(
                """
                UPDATE ingestion_jobs
                SET status = 'running',
                    attempts = attempts + 1,
                    started_at = COALESCE(started_at, :started_at),
                    finished_at = NULL,
                    error_code = NULL,
                    error_summary = NULL,
                    -- A fresh submission supersedes any previous execution or
                    -- dispatch backoff. The synchronous path also mints its
                    -- fencing token here; the async path leaves the job
                    -- unclaimed for the worker that receives the task.
                    execution_token = :execution_token,
                    execution_lease_expires_at = CASE
                        WHEN :claim_execution
                        THEN now() + make_interval(secs => :lease_seconds)
                        ELSE NULL
                    END,
                    next_dispatch_at = NULL,
                    dispatch_claimed_until = NULL,
                    dispatch_error_code = NULL
                WHERE id = :job_id
                """
            ),
            {
                "job_id": job_id,
                "started_at": started_at,
                "execution_token": execution_token,
                "claim_execution": claim_execution,
                "lease_seconds": settings.ingestion_execution_lease_seconds,
            },
        )

        connection.execute(
            text(
                """
                INSERT INTO documents (tenant_id, source_key, title, source_type)
                VALUES (:tenant_id, :source_key, :title, :source_type)
                ON CONFLICT (tenant_id, source_key) DO NOTHING
                """
            ),
            {
                "tenant_id": tenant_id,
                "source_key": source_key,
                "title": title,
                "source_type": source_type,
            },
        )

        # Resolve the document ID without a row lock first. A document that did
        # not exist when this transaction started (authorized_document_id is
        # None) may have been created concurrently, possibly with ACL rows that
        # restrict this principal. If so, take the same advisory lock and
        # re-evaluate authorization before the row lock and any version, chunk,
        # or job mutation. Doing it in this order keeps every path on the
        # documented lock order: advisory lock -> document row lock.
        document = (
            connection.execute(
                text(
                    """
                    SELECT id
                    FROM documents
                    WHERE tenant_id = :tenant_id AND source_key = :source_key
                    """
                ),
                {"tenant_id": tenant_id, "source_key": source_key},
            )
            .mappings()
            .one()
        )
        document_id = document["id"]

        if authorized_document_id is None:
            _authorize_document_write(
                connection,
                principal,
                tenant_id=tenant_id,
                source_key=source_key,
                document_id=document_id,
            )

        document = (
            connection.execute(
                text(
                    """
                    SELECT id
                    FROM documents
                    WHERE id = :document_id
                    FOR UPDATE
                    """
                ),
                {"document_id": document_id},
            )
            .mappings()
            .one()
        )
        document_id = document["id"]

        # Keep descriptive metadata current even when content is unchanged.
        connection.execute(
            text(
                """
                UPDATE documents
                SET title = :title, source_type = :source_type, updated_at = now()
                WHERE id = :document_id
                """
            ),
            {"title": title, "source_type": source_type, "document_id": document_id},
        )

        existing_version = _find_existing_version(
            connection,
            document_id=document_id,
            content_hash=prepared.content_hash,
        )

        if existing_version is not None:
            version_id = existing_version["id"]
            version_number = existing_version["version_number"]
            chunk_count = _count_version_chunks(connection, version_id)
            deduplicated = True
        else:
            if not prepared.chunks:
                # Defensive: prepare_document already rejected empty content.
                raise EmbeddingCompletenessError("Document produced no chunks")

            next_version_number = connection.execute(
                text(
                    """
                    SELECT COALESCE(MAX(version_number), 0) + 1
                    FROM document_versions WHERE document_id = :document_id
                    """
                ),
                {"document_id": document_id},
            ).scalar_one()

            version_id = connection.execute(
                text(
                    """
                    INSERT INTO document_versions (
                        document_id, version_number, content_hash, source_uri,
                        authority_level, is_current
                    )
                    VALUES (
                        :document_id, :version_number, :content_hash, :source_uri,
                        :authority_level, FALSE
                    )
                    RETURNING id
                    """
                ),
                {
                    "document_id": document_id,
                    "version_number": next_version_number,
                    "content_hash": prepared.content_hash,
                    "source_uri": source_uri,
                    "authority_level": authority_level,
                },
            ).scalar_one()

            for index, chunk in enumerate(prepared.chunks):
                connection.execute(
                    text(
                        """
                        INSERT INTO chunks (
                            tenant_id, document_id, document_version_id,
                            chunk_index, content, content_hash
                        )
                        VALUES (
                            :tenant_id, :document_id, :version_id,
                            :chunk_index, :content, :chunk_hash
                        )
                        """
                    ),
                    {
                        "tenant_id": tenant_id,
                        "document_id": document_id,
                        "version_id": version_id,
                        "chunk_index": index,
                        "content": chunk,
                        "chunk_hash": sha256_text(chunk),
                    },
                )

            version_number = next_version_number
            chunk_count = len(prepared.chunks)
            deduplicated = False

        activation_generation = int(
            connection.execute(
                text("SELECT nextval('ingestion_activation_generation_seq')")
            ).scalar_one()
        )

        connection.execute(
            text(
                """
                UPDATE ingestion_jobs
                SET document_id = :document_id,
                    version_id = :version_id,
                    activation_generation = :activation_generation
                WHERE id = :job_id
                """
            ),
            {
                "job_id": job_id,
                "document_id": document_id,
                "version_id": version_id,
                "activation_generation": activation_generation,
            },
        )

        return PreparedTarget(
            document_id=document_id,
            version_id=version_id,
            version_number=version_number,
            content_hash=prepared.content_hash,
            chunk_count=chunk_count,
            deduplicated=deduplicated,
            activation_generation=activation_generation,
            preserve_succeeded=preserve_succeeded,
            execution_token=execution_token,
        )


def _load_embedding_gaps(
    *,
    version_id: UUID,
    tenant_id: UUID,
    model: str,
    dimensions: int,
) -> list[dict[str, Any]]:
    """Return chunks whose stored embedding is missing or stale for this model."""

    with get_engine().connect() as connection:
        rows = (
            connection.execute(
                text(
                    """
                    SELECT id, chunk_index, content
                    FROM chunks
                    WHERE document_version_id = :version_id
                      AND tenant_id = :tenant_id
                      AND (
                          embedding IS NULL
                          OR embedding_model IS DISTINCT FROM :model
                          OR embedded_at IS NULL
                          OR vector_dims(embedding) IS DISTINCT FROM :dimensions
                      )
                    ORDER BY chunk_index
                    """
                ),
                {
                    "version_id": version_id,
                    "tenant_id": tenant_id,
                    "model": model,
                    "dimensions": dimensions,
                },
            )
            .mappings()
            .all()
        )
    return [dict(row) for row in rows]


def _persist_embedding_batch(
    *,
    version_id: UUID,
    tenant_id: UUID,
    batch: list[dict[str, Any]],
    vectors: list[list[float]],
    model: str,
    dimensions: int,
) -> None:
    """Persist one validated batch without overwriting already-valid vectors."""

    embedded_at = datetime.now(UTC)
    with get_engine().begin() as connection:
        for chunk, vector in zip(batch, vectors, strict=True):
            connection.execute(
                text(
                    """
                    UPDATE chunks
                    SET embedding = CAST(:embedding AS vector),
                        embedding_model = :model,
                        embedded_at = :embedded_at
                    WHERE id = :chunk_id
                      AND document_version_id = :version_id
                      AND tenant_id = :tenant_id
                      AND (
                          embedding IS NULL
                          OR embedding_model IS DISTINCT FROM :model
                          OR embedded_at IS NULL
                          OR vector_dims(embedding) IS DISTINCT FROM :dimensions
                      )
                    """
                ),
                {
                    "chunk_id": chunk["id"],
                    "version_id": version_id,
                    "tenant_id": tenant_id,
                    "embedding": vector_literal(vector),
                    "model": model,
                    "dimensions": dimensions,
                    "embedded_at": embedded_at,
                },
            )


def verify_embedding_completeness(
    *,
    version_id: UUID,
    tenant_id: UUID,
    expected_chunk_count: int,
) -> None:
    """Stage 5: verify every chunk has a valid embedding for the configured model."""

    with get_engine().connect() as connection:
        state = _embedding_state(
            connection,
            version_id=version_id,
            tenant_id=tenant_id,
            model=expected_model(),
            dimensions=expected_dimensions(),
        )
    _require_embedding_completeness(state, expected_chunk_count)


def renew_ingestion_job_lease(
    job_id: UUID,
    execution_token: UUID,
    *,
    lease_seconds: int | None = None,
) -> bool:
    """Extend a live execution lease, returning ``False`` when ownership is gone.

    The predicate requires the token to still match and the lease to still be
    live. An expired execution cannot renew itself back into ownership; a new
    worker claim is required.
    """

    settings = get_settings()
    lease = lease_seconds or settings.ingestion_execution_lease_seconds
    with get_engine().begin() as connection:
        result = connection.execute(
            text(
                """
                UPDATE ingestion_jobs
                SET execution_lease_expires_at = now() + make_interval(secs => :lease)
                WHERE id = :job_id
                  AND execution_token = :execution_token
                  AND status = 'running'
                  AND execution_lease_expires_at IS NOT NULL
                  AND execution_lease_expires_at > now()
                """
            ),
            {
                "job_id": job_id,
                "execution_token": execution_token,
                "lease": lease,
            },
        )
    return result.rowcount == 1


def _require_execution_lease(
    job_id: UUID,
    execution_token: UUID,
    *,
    lease_seconds: int | None = None,
) -> None:
    """Renew a live lease or raise when this execution no longer owns the job.

    Both the synchronous endpoint and the worker use this helper so there is one
    definition of a valid heartbeat and one failure mode when ownership is lost.
    """

    if not renew_ingestion_job_lease(
        job_id,
        execution_token,
        lease_seconds=lease_seconds,
    ):
        raise IngestionExecutionLeaseLostError(
            "Execution lease was lost while ingestion was still running"
        )


def _wait_for_execution_to_finish(job_id: UUID, *, timeout_seconds: float) -> None:
    """Wait for another execution to release a job before retrying preparation.

    A synchronous duplicate that finds a live owner must not steal ownership.
    It polls short, transaction-free reads (no connection is held across an
    embedding call) until the owner succeeds, fails, or its lease expires, then
    lets the caller re-run ``_prepare_target`` to either return a pure read or
    start a legitimate retry.
    """

    deadline = time.monotonic() + max(0.0, timeout_seconds)
    while True:
        with get_engine().connect() as connection:
            row = (
                connection.execute(
                    text(
                        """
                        SELECT status, execution_token,
                               (
                                   execution_lease_expires_at IS NOT NULL
                                   AND execution_lease_expires_at > now()
                               ) AS execution_lease_active
                        FROM ingestion_jobs
                        WHERE id = :job_id
                        """
                    ),
                    {"job_id": job_id},
                )
                .mappings()
                .first()
            )
        if row is None:
            raise IngestionJobNotFoundError
        if (
            row["status"] != "running"
            or row["execution_token"] is None
            or not row["execution_lease_active"]
        ):
            return
        if time.monotonic() >= deadline:
            raise IngestionJobBusyError(
                "Another execution still owns this ingestion job; retry the request"
            )
        time.sleep(0.05)


def _ensure_embeddings(
    *,
    version_id: UUID,
    tenant_id: UUID,
    expected_chunk_count: int,
    heartbeat: Callable[[], None] | None = None,
) -> None:
    """Stage 4: embed missing chunks in batches and verify the result.

    ``heartbeat`` is supplied only by the background worker. It renews the
    execution lease between batches so a healthy long-running ingestion is not
    mistaken for an abandoned worker. The synchronous endpoint passes nothing.
    """

    model = expected_model()
    dimensions = expected_dimensions()
    gaps = _load_embedding_gaps(
        version_id=version_id,
        tenant_id=tenant_id,
        model=model,
        dimensions=dimensions,
    )

    for start in range(0, len(gaps), MAX_BATCH_SIZE):
        if heartbeat is not None:
            heartbeat()
        batch = gaps[start : start + MAX_BATCH_SIZE]
        vectors = embed_texts([row["content"] for row in batch])
        validated = validate_embeddings(
            vectors,
            expected_count=len(batch),
            expected_dimensions=dimensions,
        )
        _persist_embedding_batch(
            version_id=version_id,
            tenant_id=tenant_id,
            batch=batch,
            vectors=validated,
            model=model,
            dimensions=dimensions,
        )
        if heartbeat is not None:
            heartbeat()

    if heartbeat is not None:
        heartbeat()

    verify_embedding_completeness(
        version_id=version_id,
        tenant_id=tenant_id,
        expected_chunk_count=expected_chunk_count,
    )


def _mark_job_succeeded(
    connection: Connection,
    *,
    job_id: UUID,
    document_id: UUID,
    version_id: UUID,
    execution_token: UUID | None = None,
) -> None:
    if execution_token is None:
        fencing = ""
        parameters: dict[str, Any] = {
            "job_id": job_id,
            "document_id": document_id,
            "version_id": version_id,
        }
    else:
        fencing = """
              AND execution_token = :execution_token
              AND execution_lease_expires_at IS NOT NULL
              AND execution_lease_expires_at > now()
        """
        parameters = {
            "job_id": job_id,
            "document_id": document_id,
            "version_id": version_id,
            "execution_token": execution_token,
        }

    result = connection.execute(
        text(
            f"""
            UPDATE ingestion_jobs
            SET status = 'succeeded',
                document_id = :document_id,
                version_id = :version_id,
                finished_at = now(),
                error_code = NULL,
                error_summary = NULL,
                execution_token = NULL,
                execution_lease_expires_at = NULL,
                next_dispatch_at = NULL,
                dispatch_claimed_until = NULL,
                dispatch_error_code = NULL
            WHERE id = :job_id
            {fencing}
            """
        ),
        parameters,
    )
    if result.rowcount != 1:
        if execution_token is not None:
            raise IngestionExecutionLeaseLostError(
                "Execution lease was lost before the job could be marked succeeded"
            )
        raise ActivationIntegrityError("Ingestion job could not be marked succeeded")


def _activate_version(
    *,
    job_id: UUID,
    tenant_id: UUID,
    document_id: UUID,
    version_id: UUID,
    version_number: int,
    content_hash: str,
    expected_chunk_count: int,
    execution_token: UUID | None = None,
) -> ActivationResult:
    """Stage 6: verify and atomically activate the target version (TX4).

    When ``execution_token`` is supplied, activation additionally requires that
    this worker still owns the job and that its lease has not expired. The check
    runs inside the same transaction and row lock as the activation, so a late
    worker cannot demote a newer current version after another execution claimed
    the job.
    """

    model = expected_model()
    dimensions = expected_dimensions()

    with get_engine().begin() as connection:
        job_row = (
            connection.execute(
                text(
                    """
                    SELECT id, tenant_id, document_id, version_id, status,
                           idempotency_key, activation_generation, execution_token,
                           (
                               execution_lease_expires_at IS NOT NULL
                               AND execution_lease_expires_at > now()
                           ) AS execution_lease_active
                    FROM ingestion_jobs
                    WHERE id = :job_id
                    FOR UPDATE
                    """
                ),
                {"job_id": job_id},
            )
            .mappings()
            .first()
        )

        if job_row is None or job_row["tenant_id"] != tenant_id:
            raise ActivationIntegrityError("Ingestion job does not belong to the tenant")
        if job_row["document_id"] != document_id or job_row["version_id"] != version_id:
            raise ActivationIntegrityError(
                "Ingestion job is not associated with the expected document version"
            )
        if execution_token is not None and (
            job_row["execution_token"] != execution_token or not job_row["execution_lease_active"]
        ):
            raise IngestionExecutionLeaseLostError(
                "Execution lease was lost before the version could be activated"
            )

        document = (
            connection.execute(
                text(
                    """
                    SELECT id, tenant_id, current_version_id, last_activation_generation,
                           status
                    FROM documents
                    WHERE id = :document_id AND tenant_id = :tenant_id
                    FOR UPDATE
                    """
                ),
                {"document_id": document_id, "tenant_id": tenant_id},
            )
            .mappings()
            .first()
        )
        if document is None:
            raise ActivationIntegrityError("Document does not belong to the tenant")

        version = (
            connection.execute(
                text(
                    """
                    SELECT id, document_id, version_number, content_hash, is_current
                    FROM document_versions
                    WHERE id = :version_id
                    FOR UPDATE
                    """
                ),
                {"version_id": version_id},
            )
            .mappings()
            .first()
        )
        if (
            version is None
            or version["document_id"] != document_id
            or version["version_number"] != version_number
            or version["content_hash"] != content_hash
        ):
            raise ActivationIntegrityError(
                "Target version does not belong to the expected document"
            )

        state = _embedding_state(
            connection,
            version_id=version_id,
            tenant_id=tenant_id,
            model=model,
            dimensions=dimensions,
        )
        _require_embedding_completeness(state, expected_chunk_count)

        generation = job_row["activation_generation"]
        if generation is None:
            raise ActivationIntegrityError("Ingestion job has no activation generation")

        if int(generation) <= int(document["last_activation_generation"]):
            # An older execution finished late. Preserve the newer active version
            # and still record this job's successful embedding work.
            is_current = document["current_version_id"] == version_id
            _mark_job_succeeded(
                connection,
                job_id=job_id,
                document_id=document_id,
                version_id=version_id,
                execution_token=execution_token,
            )
            return ActivationResult(is_current=is_current)

        connection.execute(
            text(
                """
                UPDATE document_versions
                SET is_current = FALSE
                WHERE document_id = :document_id
                  AND id <> :version_id
                  AND is_current
                """
            ),
            {"document_id": document_id, "version_id": version_id},
        )

        promote = connection.execute(
            text(
                """
                UPDATE document_versions
                SET is_current = TRUE
                WHERE id = :version_id AND document_id = :document_id
                """
            ),
            {"version_id": version_id, "document_id": document_id},
        )
        if promote.rowcount != 1:
            raise ActivationIntegrityError("Target version could not be activated")

        connection.execute(
            text(
                """
                UPDATE documents
                SET current_version_id = :version_id,
                    status = 'active',
                    last_activation_generation = :activation_generation,
                    updated_at = now()
                WHERE id = :document_id AND tenant_id = :tenant_id
                """
            ),
            {
                "version_id": version_id,
                "document_id": document_id,
                "tenant_id": tenant_id,
                "activation_generation": generation,
            },
        )

        _mark_job_succeeded(
            connection,
            job_id=job_id,
            document_id=document_id,
            version_id=version_id,
            execution_token=execution_token,
        )
        return ActivationResult(is_current=True)


def _failure_details(error: Exception) -> tuple[str, str]:
    """Map an exception to a safe, non-sensitive code and summary."""

    if isinstance(error, EmbeddingUnavailableError):
        return "embedding_service_unavailable", "Embedding service was unavailable"
    if isinstance(error, EmbeddingValidationError):
        return "embedding_invalid_response", "Embedding service returned an invalid response"
    if isinstance(error, EmbeddingCompletenessError):
        return "embedding_incomplete", "Embedding completeness verification failed"
    if isinstance(error, EmbeddingSchemaError):
        return (
            "embedding_schema_mismatch",
            "Embedding configuration does not match the database schema",
        )
    if isinstance(error, EmbeddingServiceError):
        return "embedding_failed", "Embedding generation failed"
    if isinstance(error, ActivationIntegrityError):
        return "activation_rejected", "Version activation integrity check failed"
    if isinstance(error, IngestionJobStateError):
        return "ingestion_job_invalid", "Ingestion job is not in a resumable state"
    if isinstance(error, IngestionQueueUnavailableError):
        return "queue_unavailable", "Background queue publication failed"
    if isinstance(error, SQLAlchemyError):
        return "database_error", "Database error during ingestion"
    return "ingestion_failed", "Ingestion failed during document persistence"


def _record_job_failure(
    job_id: UUID,
    *,
    tenant_id: UUID,
    source_key: str,
    started_at: datetime,
    error: Exception,
    preserve_succeeded: bool,
    execution_token: UUID | None = None,
) -> bool:
    """Stage 7: durably record a sanitized failure (TX5).

    Returns ``False`` only when a fenced update matched no row, which means the
    execution lease was lost and another execution now owns the job. The caller
    must then stop treating its own error as the job's outcome.

    A fenced transient failure sets ``next_dispatch_at`` after a grace period.
    That makes a lost Celery retry recoverable without racing a healthy retry,
    while the job status remains the durable, authoritative ``failed`` state.
    """

    error_code, error_summary = _failure_details(error)
    engine = get_engine()
    settings = get_settings()
    schedule_dispatch = execution_token is not None and error_code in TRANSIENT_JOB_ERROR_CODES

    fencing = ""
    parameters: dict[str, Any] = {
        "job_id": job_id,
        "tenant_id": tenant_id,
        "source_key": source_key,
        "started_at": started_at,
        "error_code": error_code,
        "error_summary": error_summary,
        "schedule_dispatch": schedule_dispatch,
        "dispatch_grace": settings.ingestion_reconcile_transient_grace_seconds,
    }
    if execution_token is not None:
        fencing = """
              AND execution_token = :execution_token
              AND execution_lease_expires_at IS NOT NULL
              AND execution_lease_expires_at > now()
        """
        parameters["execution_token"] = execution_token
    else:
        # A caller that never acquired ownership (for example a synchronous
        # request that failed during target preparation) must not clear a live
        # lease held by another execution. It may still record a failure for a
        # job with no live owner.
        fencing = """
              AND (
                  execution_lease_expires_at IS NULL
                  OR execution_lease_expires_at <= now()
              )
        """

    # A job that succeeded earlier is a durable record of that ingestion. A
    # redundant execution may fail, but it must not erase or downgrade the
    # previous success.
    if preserve_succeeded:
        with engine.begin() as connection:
            result = connection.execute(
                text(
                    f"""
                    UPDATE ingestion_jobs
                    SET status = 'succeeded',
                        finished_at = now(),
                        error_code = NULL,
                        error_summary = NULL,
                        execution_token = NULL,
                        execution_lease_expires_at = NULL,
                        next_dispatch_at = NULL,
                        dispatch_claimed_until = NULL,
                        dispatch_error_code = NULL
                    WHERE id = :job_id AND status = 'running'
                    {fencing}
                    """
                ),
                parameters,
            )
        return result.rowcount == 1

    with engine.begin() as connection:
        result = connection.execute(
            text(
                f"""
                UPDATE ingestion_jobs
                SET status = 'failed',
                    attempts = CASE
                        WHEN status = 'running' THEN attempts
                        ELSE attempts + 1
                    END,
                    started_at = COALESCE(started_at, :started_at),
                    finished_at = now(),
                    error_code = :error_code,
                    error_summary = :error_summary,
                    execution_token = NULL,
                    execution_lease_expires_at = NULL,
                    dispatch_claimed_until = NULL,
                    next_dispatch_at = CASE
                        WHEN :schedule_dispatch
                        THEN now() + make_interval(secs => :dispatch_grace)
                        ELSE next_dispatch_at
                    END,
                    document_id = COALESCE(
                        document_id,
                        (
                            SELECT id FROM documents
                            WHERE tenant_id = :tenant_id AND source_key = :source_key
                        )
                    )
                WHERE id = :job_id AND status <> 'succeeded'
                {fencing}
                """
            ),
            parameters,
        )
    return result.rowcount == 1


def _record_job_failure_safely(
    job_id: UUID,
    *,
    tenant_id: UUID,
    source_key: str,
    started_at: datetime,
    error: Exception,
    preserve_succeeded: bool,
    execution_token: UUID | None = None,
) -> bool | None:
    """Best-effort failure recording that never hides the original exception.

    Returns ``True`` when a row was updated, ``False`` when a fenced token no
    longer owns the job, and ``None`` when the failure could not be recorded
    because PostgreSQL was unavailable. The caller decides whether the original
    exception should still drive a bounded retry.
    """

    try:
        return _record_job_failure(
            job_id,
            tenant_id=tenant_id,
            source_key=source_key,
            started_at=started_at,
            error=error,
            preserve_succeeded=preserve_succeeded,
            execution_token=execution_token,
        )
    except Exception:  # noqa: BLE001 - the original exception still propagates
        # Keep the log free of exception text: driver errors can contain SQL
        # parameters. The caller still receives the original exception.
        logger.error("Could not record failure for ingestion job %s", job_id)
        return None


def _response(target: PreparedTarget, *, is_current: bool) -> dict[str, Any]:
    return {
        "document_id": target.document_id,
        "version_id": target.version_id,
        "version_number": target.version_number,
        "content_hash": target.content_hash,
        "chunk_count": target.chunk_count,
        "deduplicated": target.deduplicated,
        "is_current": is_current,
        "embedding_status": EMBEDDING_STATUS_COMPLETE,
    }


def ingest_document(
    *,
    principal: Principal,
    tenant_id: UUID,
    source_key: str,
    title: str,
    source_type: str,
    content: str,
    source_uri: str | None = None,
    authority_level: int = 50,
) -> dict[str, Any]:
    """Persist a document revision through a durable, idempotent ingestion job.

    The response contract is unchanged. ``embedding_status`` is ``"complete"``
    for every successful result because activation is impossible unless every
    chunk has a finite embedding for the configured model and dimensions.
    ``"pending"`` and ``"failed"`` remain named states for future asynchronous
    execution, but the synchronous endpoint never returns them in a 2xx body.

    ``principal`` is the authenticated request identity. Authorization is
    re-evaluated here, within the job and target transactions, so callers cannot
    bypass the HTTP layer and a duplicate submission never reuses a prior
    permission decision.
    """

    prepared = prepare_document(content)
    idempotency_key = derive_idempotency_key(tenant_id, source_key, prepared.content_hash)
    job, created = _resolve_or_create_job(
        principal=principal,
        tenant_id=tenant_id,
        idempotency_key=idempotency_key,
        source_key=source_key,
        content_hash=prepared.content_hash,
    )

    attempt_started_at = datetime.now(UTC)
    preserve_succeeded = job.status == "succeeded"
    execution_token: UUID | None = None
    settings = get_settings()
    try:
        while True:
            target = _prepare_target(
                job.id,
                principal=principal,
                tenant_id=tenant_id,
                source_key=source_key,
                title=title,
                source_type=source_type,
                source_uri=source_uri,
                authority_level=authority_level,
                prepared=prepared,
                started_at=attempt_started_at,
                created=created,
                claim_execution=True,
            )
            if not target.wait_for_execution:
                break
            # A duplicate request found a live owner. Do not steal its token;
            # wait for it to finish, then resolve the job again.
            _wait_for_execution_to_finish(
                job.id,
                timeout_seconds=settings.ingestion_execution_lease_seconds,
            )

        execution_token = target.execution_token
        preserve_succeeded = target.preserve_succeeded

        if target.already_active:
            return _response(target, is_current=True)

        if execution_token is None:
            raise ActivationIntegrityError(
                "Synchronous ingestion did not acquire execution ownership"
            )

        def heartbeat() -> None:
            _require_execution_lease(
                job.id,
                execution_token,
                lease_seconds=settings.ingestion_execution_lease_seconds,
            )

        heartbeat()
        _ensure_embeddings(
            version_id=target.version_id,
            tenant_id=tenant_id,
            expected_chunk_count=target.chunk_count,
            heartbeat=heartbeat,
        )
        heartbeat()
        activation = _activate_version(
            job_id=job.id,
            tenant_id=tenant_id,
            document_id=target.document_id,
            version_id=target.version_id,
            version_number=target.version_number,
            content_hash=target.content_hash,
            expected_chunk_count=target.chunk_count,
            execution_token=execution_token,
        )
        return _response(target, is_current=activation.is_current)
    except IdempotencyKeyCollisionError:
        # The existing job belongs to other content. Never mark it failed or
        # rewrite its links on behalf of the colliding request.
        raise
    except IngestionExecutionLeaseLostError:
        # Another execution now owns the job. Do not record this execution's
        # error against the newer owner's state.
        raise
    except Exception as error:
        if execution_token is not None or created:
            _record_job_failure_safely(
                job.id,
                tenant_id=tenant_id,
                source_key=source_key,
                started_at=attempt_started_at,
                error=error,
                preserve_succeeded=preserve_succeeded,
                execution_token=execution_token,
            )
        raise


def _isoformat(value: Any) -> str | None:
    return value.isoformat() if value is not None else None


def submit_ingestion_job(
    *,
    principal: Principal,
    tenant_id: UUID,
    source_key: str,
    title: str,
    source_type: str,
    content: str,
    source_uri: str | None = None,
    authority_level: int = 50,
) -> SubmittedIngestion:
    """Prepare a durable target for background embedding and activation.

    The request transaction persists the immutable version and its chunks as
    **non-current**, links the durable job, and commits *before* a task may be
    published. Document content therefore never travels through Redis, and a
    broker outage leaves an inspectable failed job plus a safe non-current
    target rather than an unobservable queued write.

    The submitted principal is used only for this request's authorization. It is
    never written to the job row or passed to Celery; the worker completes the
    already-authorized durable target as a trusted internal operation.
    """

    prepared = prepare_document(content)
    idempotency_key = derive_idempotency_key(tenant_id, source_key, prepared.content_hash)
    job, created = _resolve_or_create_job(
        principal=principal,
        tenant_id=tenant_id,
        idempotency_key=idempotency_key,
        source_key=source_key,
        content_hash=prepared.content_hash,
    )

    # A running job that already has a target is being handled by another
    # request or worker. Returning the same job is the idempotent response;
    # repeated broker publication would only duplicate safe but unnecessary work.
    if not created and job.status == "running" and job.version_id is not None:
        return SubmittedIngestion(
            job_id=job.id,
            status=job.status,
            document_id=job.document_id,
            version_id=job.version_id,
            created=False,
            needs_publish=False,
        )

    attempt_started_at = datetime.now(UTC)
    try:
        target = _prepare_target(
            job.id,
            principal=principal,
            tenant_id=tenant_id,
            source_key=source_key,
            title=title,
            source_type=source_type,
            source_uri=source_uri,
            authority_level=authority_level,
            prepared=prepared,
            started_at=attempt_started_at,
            created=created,
            claim_execution=False,
        )
    except IdempotencyKeyCollisionError:
        raise
    except Exception as error:
        # A validation or database failure during preparation must leave a
        # terminal, inspectable job instead of a queued row that no task owns.
        _record_job_failure_safely(
            job.id,
            tenant_id=tenant_id,
            source_key=source_key,
            started_at=attempt_started_at,
            error=error,
            preserve_succeeded=job.status == "succeeded",
        )
        raise

    if target.already_active:
        return SubmittedIngestion(
            job_id=job.id,
            status="succeeded",
            document_id=target.document_id,
            version_id=target.version_id,
            created=created,
            needs_publish=False,
        )

    return SubmittedIngestion(
        job_id=job.id,
        status="running",
        document_id=target.document_id,
        version_id=target.version_id,
        created=created,
        needs_publish=True,
    )


def get_ingestion_job_status(
    job_id: UUID,
    *,
    principal: Principal,
    tenant_id: UUID,
) -> dict[str, Any]:
    """Return tenant-scoped, sanitized job state for the status endpoint.

    Access policy:

    * a job is always tenant-scoped; another tenant's job is indistinguishable
      from a missing job;
    * a job linked to a document requires the caller to pass the same document
      read-ACL predicate used by retrieval (no ACL rows means tenant-wide read;
      any ACL row requires a matching ``read``/``manage`` grant);
    * a job with no linked document yet is visible to authenticated members of
      its tenant. It contains no source key, title, content, or execution token,
      and the job ID is an unguessable capability returned only to the submitter.

    Unauthorized and missing jobs raise the same exception so the endpoint can
    return the established non-disclosing ``404``.
    """

    if principal.tenant_id != tenant_id:
        raise IngestionJobNotFoundError

    with get_engine().connect() as connection:
        row = (
            connection.execute(
                text(
                    f"""
                    SELECT j.id, j.status, j.attempts, j.document_id,
                           j.version_id, j.error_code, j.error_summary,
                           j.created_at, j.started_at, j.finished_at,
                           CASE
                               WHEN j.document_id IS NULL THEN TRUE
                               WHEN d.id IS NULL THEN FALSE
                               WHEN NOT EXISTS (
                                   SELECT 1
                                   FROM document_acl AS acl
                                   WHERE acl.document_id = d.id
                               ) THEN TRUE
                               ELSE EXISTS (
                                   SELECT 1
                                   FROM document_acl AS acl
                                   WHERE acl.document_id = d.id
                                     AND acl.permission IN ('read', 'manage')
                                     AND ({acl_principal_match_sql("acl")})
                               )
                           END AS can_read
                    FROM ingestion_jobs AS j
                    LEFT JOIN documents AS d
                      ON d.id = j.document_id AND d.tenant_id = j.tenant_id
                    WHERE j.id = :job_id AND j.tenant_id = :tenant_id
                    """
                ),
                {
                    "job_id": job_id,
                    "tenant_id": tenant_id,
                    **acl_parameters(principal),
                },
            )
            .mappings()
            .first()
        )

    if row is None or not row["can_read"]:
        # The tenant and ACL predicates deliberately make another tenant's or
        # another principal's job indistinguishable from a missing one.
        raise IngestionJobNotFoundError

    return {
        "job_id": str(row["id"]),
        "status": row["status"],
        "attempts": int(row["attempts"]),
        "document_id": str(row["document_id"]) if row["document_id"] else None,
        "version_id": str(row["version_id"]) if row["version_id"] else None,
        "error_code": row["error_code"],
        "error_summary": row["error_summary"],
        "created_at": _isoformat(row["created_at"]),
        "started_at": _isoformat(row["started_at"]),
        "finished_at": _isoformat(row["finished_at"]),
    }


def get_ingestion_job_snapshot(job_id: UUID) -> dict[str, Any] | None:
    """Return worker-safe job state without document content or metadata.

    This is intentionally not tenant-scoped: the worker already owns a task for
    the job and only uses the snapshot for structured logging and reconciliation.
    """

    with get_engine().connect() as connection:
        row = (
            connection.execute(
                text(
                    """
                    SELECT id, tenant_id, status, attempts, document_id, version_id,
                           error_code, error_summary, created_at, started_at,
                           finished_at
                    FROM ingestion_jobs
                    WHERE id = :job_id
                    """
                ),
                {"job_id": job_id},
            )
            .mappings()
            .first()
        )
    if row is None:
        return None

    return {
        "job_id": row["id"],
        "tenant_id": row["tenant_id"],
        "status": row["status"],
        "attempts": int(row["attempts"]),
        "document_id": row["document_id"],
        "version_id": row["version_id"],
        "error_code": row["error_code"],
        "error_summary": row["error_summary"],
        "created_at": _isoformat(row["created_at"]),
        "started_at": _isoformat(row["started_at"]),
        "finished_at": _isoformat(row["finished_at"]),
    }


def mark_ingestion_job_enqueue_failed(job_id: UUID) -> None:
    """Record a broker publication failure without exposing broker details.

    The job moves to the resumable ``failed``/``queue_unavailable`` state, not a
    permanent failure. ``next_dispatch_at`` makes it immediately eligible for
    reconciliation; repeated broker outages are then backed off by the
    reconciler instead of being turned into dead letters.
    """

    with get_engine().begin() as connection:
        connection.execute(
            text(
                """
                UPDATE ingestion_jobs
                SET status = 'failed',
                    error_code = 'queue_unavailable',
                    error_summary = 'Background queue publication failed',
                    finished_at = now(),
                    dispatch_error_code = 'queue_unavailable',
                    next_dispatch_at = now(),
                    dispatch_claimed_until = NULL,
                    execution_token = NULL,
                    execution_lease_expires_at = NULL
                WHERE id = :job_id
                  AND status IN ('queued', 'running')
                  -- Do not downgrade a job that a worker already claimed: an
                  -- ambiguous publish may have been delivered, and the worker's
                  -- lease and fencing token must stay authoritative.
                  AND execution_token IS NULL
                """
            ),
            {"job_id": job_id},
        )


def mark_ingestion_job_retries_exhausted(
    job_id: UUID,
    *,
    error_code: str | None = None,
    expected_attempts: int | None = None,
    expected_error_code: str | None = None,
) -> bool:
    """Make an exhausted retry loop terminal and inspectable.

    The original sanitized failure code/summary is preserved when present; the
    status change to ``dead_letter`` is what prevents another duplicate delivery
    from starting new embedding work.

    ``expected_attempts`` and ``expected_error_code`` fence this terminal write
    against a concurrent reclaim. A late task that reaches its retry ceiling
    after another execution took ownership must not dead-letter the newer
    execution. The write also requires ``status = 'failed'`` so an unexpected
    exhaustion can never overwrite a running or succeeded job.
    """

    with get_engine().begin() as connection:
        result = connection.execute(
            text(
                """
                UPDATE ingestion_jobs
                SET status = 'dead_letter',
                    finished_at = now(),
                    error_code = COALESCE(error_code, :error_code),
                    error_summary = COALESCE(
                        error_summary,
                        'Retries exhausted after transient failures'
                    ),
                    execution_token = NULL,
                    execution_lease_expires_at = NULL,
                    next_dispatch_at = NULL,
                    dispatch_claimed_until = NULL
                WHERE id = :job_id
                  AND status = 'failed'
                  AND (:expected_attempts IS NULL OR attempts = :expected_attempts)
                  AND (
                      :expected_error_code IS NULL
                      OR error_code = :expected_error_code
                  )
                """
            ),
            {
                "job_id": job_id,
                "error_code": error_code,
                "expected_attempts": expected_attempts,
                "expected_error_code": expected_error_code,
            },
        )
    return result.rowcount == 1


def _claim_resumable_job(
    job_id: UUID,
    *,
    started_at: datetime,
    lease_seconds: int | None = None,
) -> ResumableJob | IngestionExecutionResult:
    """Lock the durable job and claim one worker execution with a lease.

    A succeeded job returns a pure-read result. A permanent failed/dead-letter
    job returns a terminal signal without incrementing attempts. A queued,
    transiently failed, or never-claimed running job starts an execution. A
    running job with a live execution lease is busy: a duplicate delivery must
    leave the healthy execution alone. A running job with no token (the API
    committed but never got as far as a worker) or an expired token (crashed
    worker) is reclaimed with a fresh fencing token.
    """

    settings = get_settings()
    lease = lease_seconds or settings.ingestion_execution_lease_seconds

    with get_engine().begin() as connection:
        job = (
            connection.execute(
                text(
                    """
                    SELECT id, tenant_id, status, document_id, version_id,
                           activation_generation, error_code, attempts,
                           execution_token,
                           (
                               execution_lease_expires_at IS NOT NULL
                               AND execution_lease_expires_at > now()
                           ) AS execution_lease_active
                    FROM ingestion_jobs
                    WHERE id = :job_id
                    FOR UPDATE
                    """
                ),
                {"job_id": job_id},
            )
            .mappings()
            .first()
        )
        if job is None:
            raise IngestionJobNotFoundError

        tenant_id = job["tenant_id"]
        document_id = job["document_id"]
        version_id = job["version_id"]

        if job["status"] == "succeeded":
            if document_id is None or version_id is None:
                raise IngestionJobStateError(
                    "Succeeded ingestion job has no linked version",
                    job_id=job_id,
                    tenant_id=tenant_id,
                )
            version = (
                connection.execute(
                    text(
                        """
                        SELECT dv.version_number, dv.content_hash,
                               d.current_version_id
                        FROM document_versions AS dv
                        JOIN documents AS d ON d.id = dv.document_id
                        WHERE dv.id = :version_id
                          AND dv.document_id = :document_id
                        """
                    ),
                    {"version_id": version_id, "document_id": document_id},
                )
                .mappings()
                .first()
            )
            if version is None:
                raise IngestionJobStateError(
                    "Succeeded ingestion job points at a missing version",
                    job_id=job_id,
                    tenant_id=tenant_id,
                )
            chunk_count = int(
                connection.execute(
                    text(
                        "SELECT count(*) FROM chunks "
                        "WHERE document_version_id = :version_id "
                        "AND tenant_id = :tenant_id"
                    ),
                    {"version_id": version_id, "tenant_id": tenant_id},
                ).scalar_one()
            )
            return IngestionExecutionResult(
                job_id=job_id,
                tenant_id=tenant_id,
                status="succeeded",
                attempts=int(job["attempts"]),
                document_id=document_id,
                version_id=version_id,
                version_number=int(version["version_number"]),
                content_hash=version["content_hash"],
                chunk_count=chunk_count,
                is_current=version["current_version_id"] == version_id,
                outcome="skipped",
            )

        if job["status"] == "dead_letter" or (
            job["status"] == "failed" and job["error_code"] in PERMANENT_JOB_ERROR_CODES
        ):
            raise IngestionJobTerminalError("Ingestion job is terminal and will not be retried")

        if job["status"] == "running" and job["execution_lease_active"]:
            raise IngestionJobBusyError(
                "Ingestion job already has a live execution lease; duplicate delivery skipped"
            )

        if document_id is None or version_id is None or job["activation_generation"] is None:
            raise IngestionJobStateError(
                "Ingestion job has no prepared target to resume",
                job_id=job_id,
                tenant_id=tenant_id,
            )

        target = (
            connection.execute(
                text(
                    """
                    SELECT dv.version_number, dv.content_hash, d.source_key
                    FROM document_versions AS dv
                    JOIN documents AS d ON d.id = dv.document_id
                    WHERE dv.id = :version_id
                      AND dv.document_id = :document_id
                      AND d.tenant_id = :tenant_id
                    """
                ),
                {
                    "version_id": version_id,
                    "document_id": document_id,
                    "tenant_id": tenant_id,
                },
            )
            .mappings()
            .first()
        )
        if target is None:
            raise IngestionJobStateError(
                "Ingestion job target does not belong to the job tenant",
                job_id=job_id,
                tenant_id=tenant_id,
            )

        chunk_count = int(
            connection.execute(
                text(
                    "SELECT count(*) FROM chunks "
                    "WHERE document_version_id = :version_id "
                    "AND tenant_id = :tenant_id"
                ),
                {"version_id": version_id, "tenant_id": tenant_id},
            ).scalar_one()
        )
        if chunk_count < 1:
            raise IngestionJobStateError(
                "Ingestion job target has no chunks",
                job_id=job_id,
                tenant_id=tenant_id,
                source_key=target["source_key"],
            )

        # ``queued`` and ``failed`` start a new execution. A ``running`` job with
        # no token is the first execution after submission (do not count the
        # submission twice); a running job with an expired token is a genuinely
        # new execution after a crash, so it records another attempt. Attempts
        # are never reset.
        attempts = int(job["attempts"])
        starts_new_execution = job["status"] in {"queued", "failed"} or (
            job["status"] == "running" and job["execution_token"] is not None
        )
        if starts_new_execution:
            attempts += 1

        execution_token = uuid4()
        connection.execute(
            text(
                """
                UPDATE ingestion_jobs
                SET status = 'running',
                    attempts = :attempts,
                    started_at = COALESCE(started_at, :started_at),
                    finished_at = NULL,
                    error_code = NULL,
                    error_summary = NULL,
                    execution_token = :execution_token,
                    execution_lease_expires_at = now() + make_interval(secs => :lease),
                    next_dispatch_at = NULL,
                    dispatch_claimed_until = NULL,
                    dispatch_error_code = NULL
                WHERE id = :job_id
                """
            ),
            {
                "job_id": job_id,
                "attempts": attempts,
                "started_at": started_at,
                "execution_token": execution_token,
                "lease": lease,
            },
        )

        return ResumableJob(
            job_id=job_id,
            tenant_id=tenant_id,
            status="running",
            attempts=attempts,
            document_id=document_id,
            version_id=version_id,
            version_number=int(target["version_number"]),
            content_hash=target["content_hash"],
            chunk_count=chunk_count,
            activation_generation=int(job["activation_generation"]),
            source_key=target["source_key"],
            execution_token=execution_token,
        )


def resume_ingestion_job(job_id: UUID) -> IngestionExecutionResult:
    """Finish an already-prepared durable job using the existing staged service.

    This is the only worker entry point. It never accepts content, never creates
    a new target version, and never claims a new activation generation: the
    target and generation were committed by :func:`submit_ingestion_job` before
    the task was published. Embedding batching, completeness verification, and
    atomic activation are delegated to the same internal stages used by the
    synchronous endpoint.

    The claim mints an execution token and lease. Lease heartbeats are sent
    between embedding batches, and every terminal write is fenced by the token.
    A worker that lost its lease therefore cannot activate, mark a failure, or
    otherwise change the outcome of a newer execution.
    """

    settings = get_settings()
    attempt_started_at = datetime.now(UTC)
    try:
        claim = _claim_resumable_job(job_id, started_at=attempt_started_at)
    except IngestionJobStateError as error:
        if error.tenant_id is not None:
            _record_job_failure_safely(
                job_id,
                tenant_id=error.tenant_id,
                source_key=error.source_key or "",
                started_at=attempt_started_at,
                error=error,
                preserve_succeeded=False,
            )
        raise

    if isinstance(claim, IngestionExecutionResult):
        return claim

    def heartbeat() -> None:
        _require_execution_lease(
            claim.job_id,
            claim.execution_token,
            lease_seconds=settings.ingestion_execution_lease_seconds,
        )

    try:
        # The target was prepared against the live schema by the request, but
        # the worker may run after a deployment or schema drift. Fail before
        # spending provider calls if the vector column no longer matches.
        check_embedding_schema()
        heartbeat()
        _ensure_embeddings(
            version_id=claim.version_id,
            tenant_id=claim.tenant_id,
            expected_chunk_count=claim.chunk_count,
            heartbeat=heartbeat,
        )
        heartbeat()
        activation = _activate_version(
            job_id=claim.job_id,
            tenant_id=claim.tenant_id,
            document_id=claim.document_id,
            version_id=claim.version_id,
            version_number=claim.version_number,
            content_hash=claim.content_hash,
            expected_chunk_count=claim.chunk_count,
            execution_token=claim.execution_token,
        )
    except (IdempotencyKeyCollisionError, IngestionExecutionLeaseLostError):
        # A lease loss means another execution owns the job. Do not record this
        # worker's error against the newer execution's state.
        raise
    except Exception as error:
        recorded = _record_job_failure_safely(
            claim.job_id,
            tenant_id=claim.tenant_id,
            source_key=claim.source_key,
            started_at=attempt_started_at,
            error=error,
            preserve_succeeded=False,
            execution_token=claim.execution_token,
        )
        if recorded is False:
            raise IngestionExecutionLeaseLostError(
                "Execution lease was lost before the failure could be recorded"
            ) from error
        # ``None`` means PostgreSQL could not record the sanitized failure; the
        # original exception still drives the bounded task retry.
        raise

    return IngestionExecutionResult(
        job_id=claim.job_id,
        tenant_id=claim.tenant_id,
        status="succeeded",
        attempts=claim.attempts,
        document_id=claim.document_id,
        version_id=claim.version_id,
        version_number=claim.version_number,
        content_hash=claim.content_hash,
        chunk_count=claim.chunk_count,
        is_current=activation.is_current,
        outcome="succeeded",
    )
