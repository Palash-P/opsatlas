"""Backfill missing chunk embeddings using the local embedding service."""

from __future__ import annotations

import argparse
import logging

from sqlalchemy import text

from app.db.connections import get_engine
from app.services.embedding_client import EXPECTED_MODEL, embed_texts

logger = logging.getLogger("opsatlas.backfill")


def count_missing() -> int:
    with get_engine().connect() as connection:
        return connection.execute(
            text("SELECT count(*) FROM chunks WHERE embedding IS NULL")
        ).scalar_one()


def backfill(batch_size: int, apply_changes: bool) -> int:
    engine = get_engine()
    processed = 0

    while True:
        # Read a small batch and close the transaction before calling the model.
        with engine.connect() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT id, content
                    FROM chunks
                    WHERE embedding IS NULL
                    ORDER BY created_at, id
                    LIMIT :batch_size
                    """
                ),
                {"batch_size": batch_size},
            ).mappings().all()

        if not rows:
            break

        logger.info("Found batch of %d chunks missing embeddings", len(rows))

        if not apply_changes:
            processed += len(rows)
            # Dry-run must not loop forever over the same missing rows.
            break

        vectors = embed_texts([row["content"] for row in rows])

        # Only update rows that still lack embeddings. If another worker filled
        # a row while inference ran, this update leaves that existing vector alone.
        with engine.begin() as connection:
            for row, vector in zip(rows, vectors, strict=True):
                vector_text = "[" + ",".join(str(value) for value in vector) + "]"
                result = connection.execute(
                    text(
                        """
                        UPDATE chunks
                        SET embedding = CAST(:embedding AS vector(384)),
                            embedding_model = :model,
                            embedded_at = now()
                        WHERE id = :chunk_id
                          AND embedding IS NULL
                        """
                    ),
                    {
                        "embedding": vector_text,
                        "model": EXPECTED_MODEL,
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
    print(f"Chunks missing embeddings: {missing}")

    if missing == 0:
        print("Nothing to do.")
        return

    if not args.apply:
        print("Dry run only; no database rows changed.")
        print("Run with --apply to generate and store embeddings.")
        return

    processed = backfill(args.batch_size, apply_changes=True)
    print(f"Chunks updated: {processed}")
    print(f"Chunks still missing embeddings: {count_missing()}")


if __name__ == "__main__":
    main()