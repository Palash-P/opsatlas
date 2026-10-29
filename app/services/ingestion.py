"""Transactional document ingestion and immutable version persistence."""

from __future__ import annotations

import hashlib
from typing import Any
from uuid import UUID

from sqlalchemy import text

from app.db.connections import get_engine
from app.services.chunking import chunk_text, normalize_document_text


class TenantNotFoundError(Exception):
    """Raised when an ingestion request references an unknown tenant."""


class EmptyDocumentError(Exception):
    """Raised when the normalized document contains no text."""


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def ingest_document(
    *,
    tenant_id: UUID,
    source_key: str,
    title: str,
    source_type: str,
    content: str,
    source_uri: str | None = None,
    authority_level: int = 50,
) -> dict[str, Any]:
    """Persist a document revision atomically.

    Identical content is deduplicated by SHA-256. New revisions and their chunks are
    committed in one transaction. The current-version pointer changes only after all
    chunks have been inserted successfully. Embeddings are deliberately out of scope
    for this milestone and remain NULL until the embedding pipeline is implemented.
    """
    normalized_content = normalize_document_text(content)
    if not normalized_content:
        raise EmptyDocumentError("Document content must contain non-whitespace text")

    content_hash = sha256_text(normalized_content)
    chunks = chunk_text(normalized_content)
    engine = get_engine()

    with engine.begin() as connection:
        tenant_exists = connection.execute(
            text("SELECT 1 FROM tenants WHERE id = :tenant_id"),
            {"tenant_id": tenant_id},
        ).scalar_one_or_none()
        if tenant_exists is None:
            raise TenantNotFoundError

        # The unique (tenant_id, source_key) constraint makes this safe for concurrent
        # creation. The row lock below serializes version changes for an existing source.
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

        document = connection.execute(
            text(
                """
                SELECT id, current_version_id
                FROM documents
                WHERE tenant_id = :tenant_id AND source_key = :source_key
                FOR UPDATE
                """
            ),
            {"tenant_id": tenant_id, "source_key": source_key},
        ).mappings().one()
        document_id = document["id"]
        current_version_id = document["current_version_id"]

        # Keep descriptive metadata current even when the content itself is unchanged.
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

        existing_version = connection.execute(
            text(
                """
                SELECT id, version_number, is_current
                FROM document_versions
                WHERE document_id = :document_id AND content_hash = :content_hash
                """
            ),
            {"document_id": document_id, "content_hash": content_hash},
        ).mappings().first()

        if existing_version is not None:
            # A content hash is unique per logical document. If submitted content
            # matches an older snapshot, reactivate that immutable snapshot rather
            # than creating a second version with the same hash or returning a
            # successful result that is not current.
            existing_version_id = existing_version["id"]
            is_current = existing_version_id == current_version_id
            if not is_current:
                connection.execute(
                    text(
                        "UPDATE document_versions "
                        "SET is_current = FALSE WHERE document_id = :document_id"
                    ),
                    {"document_id": document_id},
                )
                connection.execute(
                    text(
                        "UPDATE document_versions SET is_current = TRUE "
                        "WHERE id = :version_id"
                    ),
                    {"version_id": existing_version_id},
                )
                connection.execute(
                    text(
                        "UPDATE documents SET current_version_id = :version_id, "
                        "status = 'active', updated_at = now() WHERE id = :document_id"
                    ),
                    {"version_id": existing_version_id, "document_id": document_id},
                )
                is_current = True

            chunk_count = connection.execute(
                text("SELECT count(*) FROM chunks WHERE document_version_id = :version_id"),
                {"version_id": existing_version_id},
            ).scalar_one()
            return {
                "document_id": document_id,
                "version_id": existing_version_id,
                "version_number": existing_version["version_number"],
                "content_hash": content_hash,
                "chunk_count": chunk_count,
                "deduplicated": True,
                "is_current": is_current,
                "embedding_status": "pending",
            }

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
                "content_hash": content_hash,
                "source_uri": source_uri,
                "authority_level": authority_level,
            },
        ).scalar_one()

        for index, chunk in enumerate(chunks):
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

        # These updates and all inserts commit together. If any insert fails, the old
        # current version remains current because the whole transaction rolls back.
        connection.execute(
            text("UPDATE document_versions SET is_current = FALSE WHERE document_id = :document_id"),
            {"document_id": document_id},
        )
        connection.execute(
            text("UPDATE document_versions SET is_current = TRUE WHERE id = :version_id"),
            {"version_id": version_id},
        )
        connection.execute(
            text(
                """
                UPDATE documents
                SET current_version_id = :version_id, status = 'active', updated_at = now()
                WHERE id = :document_id
                """
            ),
            {"version_id": version_id, "document_id": document_id},
        )

        return {
            "document_id": document_id,
            "version_id": version_id,
            "version_number": next_version_number,
            "content_hash": content_hash,
            "chunk_count": len(chunks),
            "deduplicated": False,
            "is_current": True,
            "embedding_status": "pending",
        }
