"""Per-document PostgreSQL advisory transaction locks for ACL serialization.

Every ingestion write-authorization read of ``document_acl`` and every writer
that changes ``document_acl`` must take the same transaction-scoped advisory lock
for the document. That makes a permission grant or revocation serialize with an
in-flight ingestion authorization instead of committing between the ACL read and
the document/version/chunk writes. Read-only retrieval and job-status reads are
deliberately not locked; they keep the existing snapshot semantics.

Lock identity
-------------

``document_acl_lock_key`` derives a signed 64-bit key from the document UUID with
SHA-256 over a fixed domain separator plus the UUID bytes. SHA-256 is stable
across processes, hosts, and Python runs; Python's built-in ``hash()`` is
deliberately not used because it is process-randomized. The domain separator
keeps this lock family distinct from any other advisory lock family that may use
a UUID in the future.

Database locks, not Python locks
--------------------------------

``pg_advisory_xact_lock`` is held by PostgreSQL until the surrounding transaction
commits or rolls back. It is therefore valid across API processes, workers, and
scripts. No Redis lock or process-local mutex is involved, and tenant isolation is
unchanged: the key is the document UUID, and authorization still checks the
tenant predicate.

Ordering
--------

Acquire document ACL locks in sorted UUID order (``acquire_document_acl_locks``)
when more than one document is affected. In ingestion, the document ACL lock is
acquired before the ingestion job row lock and the document row lock. ACL
mutations take the document ACL lock before touching ``document_acl`` rows, whose
foreign key check takes a ``KEY SHARE`` lock on the document row.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.engine import Connection

# Versioned domain separator. Changing it changes every derived key, so bump the
# version only with a coordinated rollout of all readers and writers.
DOCUMENT_ACL_LOCK_NAMESPACE = b"opsatlas:document_acl:v1"


def document_acl_lock_key(document_id: UUID) -> int:
    """Return a stable signed 64-bit advisory-lock key for one document."""

    digest = hashlib.sha256(DOCUMENT_ACL_LOCK_NAMESPACE + document_id.bytes).digest()
    return int.from_bytes(digest[:8], "big", signed=True)


def acquire_document_acl_lock(connection: Connection, document_id: UUID) -> None:
    """Block until this transaction owns the document's ACL advisory lock.

    The lock is released automatically at transaction end. Calling this more than
    once for the same document in the same transaction is harmless.
    """

    connection.execute(
        text("SELECT pg_advisory_xact_lock(:lock_key)"),
        {"lock_key": document_acl_lock_key(document_id)},
    )


def acquire_document_acl_locks(
    connection: Connection,
    document_ids: Iterable[UUID],
) -> None:
    """Acquire several document ACL locks in deterministic sorted order.

    Sorting prevents two bulk ACL transactions that touch overlapping documents
    from acquiring the same pair of locks in opposite order.
    """

    for document_id in sorted(set(document_ids), key=str):
        acquire_document_acl_lock(connection, document_id)
