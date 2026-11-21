"""Compatibility checks for the PostgreSQL embedding column.

The application has one authoritative embedding constraint: the configured
dimension must be storable in ``chunks.embedding``. Settings reject a
compile-time mismatch; this module detects drift between those settings and the
live schema, for example a database that was migrated to a different vector
dimension without updating the application contract.
"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Connection

from app.db.connections import get_engine
from app.services.embedding_client import expected_dimensions


class EmbeddingSchemaError(RuntimeError):
    """Configured embedding dimensions do not match the live database column."""


def database_embedding_dimensions(connection: Connection) -> int | None:
    """Return the declared dimension of ``chunks.embedding``, or ``None``.

    pgvector stores the declared dimension directly in ``atttypmod``; an
    unconstrained ``vector`` column and a missing column both return ``None`` or
    a negative typmod and are therefore incompatible with the fixed contract.
    """

    typmod = connection.execute(
        text(
            """
            SELECT a.atttypmod
            FROM pg_attribute AS a
            WHERE a.attrelid = 'chunks'::regclass
              AND a.attname = 'embedding'
              AND NOT a.attisdropped
            """
        )
    ).scalar_one_or_none()

    if typmod is None or int(typmod) < 0:
        return None
    return int(typmod)


def verify_embedding_schema(
    connection: Connection,
    *,
    dimensions: int | None = None,
) -> int:
    """Raise when the configured dimension cannot be stored by the database."""

    configured = expected_dimensions() if dimensions is None else dimensions
    actual = database_embedding_dimensions(connection)
    if actual != configured:
        declared = actual if actual is not None else "unbounded or missing"
        raise EmbeddingSchemaError(
            "Configured embedding dimensions "
            f"({configured}) do not match the chunks.embedding column ({declared})"
        )
    return actual


def check_embedding_schema() -> int:
    """Open a short read-only connection and verify the embedding column."""

    with get_engine().connect() as connection:
        return verify_embedding_schema(connection)
