"""Backfill missing chunk embeddings using the local embedding service."""

from __future__ import annotations

import argparse
import logging

from sqlalchemy import text

from app.db.connections import get_engine
from app.services.embedding_client import (
    embed_texts,
    expected_dimensions,
    expected_model,
    vector_literal,
    vector_sql_type,
)

logger = logging.getLogger("opsatlas.backfill")


def _outdated_predicate() -> str:
    """SQL predicate for chunks whose vector is missing or stale."""

    return """
        embedding IS NULL
        OR embedding_model IS DISTINCT FROM :model
        OR embedded_at IS NULL
        OR vector_dims(embedding) IS DISTINCT FROM :dimensions
    """


def count_missing() -> int:
    with get_engine().connect() as connection:
        return connection.execute(
            text(f"SELECT count(*) FROM chunks WHERE {_outdated_predicate()}"),
            {
                "model": expected_model(),
                "dimensions": expected_dimensions(),
            },
        ).scalar_one()


def backfill(batch_size: int, apply_changes: bool) -> int:
    engine = get_engine()
    processed = 0
    model = expected_model()
    dimensions = expected_dimensions()
    vector_type = vector_sql_type()

    while True:
        # Read a small batch and close the transaction before calling the model.
        with engine.connect() as connection:
            rows = (
                connection.execute(
                    text(
                        f"""
                    SELECT id, content
                    FROM chunks
                    WHERE {_outdated_predicate()}
                    ORDER BY created_at, id
                    LIMIT :batch_size
                    """
                    ),
                    {
                        "batch_size": batch_size,
                        "model": model,
                        "dimensions": dimensions,
                    },
                )
                .mappings()
                .all()
            )

        if not rows:
            break

        logger.info("Found batch of %d chunks needing embeddings", len(rows))

        if not apply_changes:
            processed += len(rows)
            # Dry-run must not loop forever over the same missing rows.
            break

        vectors = embed_texts([row["content"] for row in rows])

        # Only update rows that still lack a valid embedding. If another worker
        # filled a row while inference ran, this update leaves that vector alone.
        with engine.begin() as connection:
            for row, vector in zip(rows, vectors, strict=True):
                result = connection.execute(
                    text(
                        f"""
                        UPDATE chunks
                        SET embedding = CAST(:embedding AS {vector_type}),
                            embedding_model = :model,
                            embedded_at = now()
                        WHERE id = :chunk_id
                          AND ({_outdated_predicate()})
                        """
                    ),
                    {
                        "embedding": vector_literal(vector),
                        "model": model,
                        "dimensions": dimensions,
                        "chunk_id": row["id"],
                    },
                )
                processed += result.rowcount

        logger.info("Updated %d chunks so far", processed)

    return processed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Persist generated embeddings")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    if not 1 <= args.batch_size <= 32:
        parser.error("--batch-size must be between 1 and 32")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    missing = count_missing()
    print(f"Chunks needing embeddings: {missing}")

    if missing == 0:
        print("Nothing to do.")
        return

    if not args.apply:
        print("Dry run only; no database rows changed.")
        print("Run with --apply to generate and store embeddings.")
        return

    processed = backfill(args.batch_size, apply_changes=True)
    print(f"Chunks updated: {processed}")
    print(f"Chunks still needing embeddings: {count_missing()}")


if __name__ == "__main__":
    main()
