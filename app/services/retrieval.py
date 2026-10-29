"""Tenant-scoped semantic, keyword, and hybrid retrieval."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import text

from app.db.connections import get_engine
from app.services.embedding_client import EXPECTED_MODEL, embed_texts

MAX_RESULTS = 20
RRF_K = 60
HYBRID_CANDIDATE_MULTIPLIER = 3


def fuse_ranked_results(
    semantic_results: list[dict[str, Any]],
    keyword_results: list[dict[str, Any]],
    *,
    limit: int,
    rrf_k: int = RRF_K,
) -> list[dict[str, Any]]:
    """Fuse two ranked lists with Reciprocal Rank Fusion.

    Each list contributes ``1 / (rrf_k + rank)`` for every chunk, where ranks
    start at 1. A chunk appearing in both lists therefore receives both
    contributions. Ties are broken by best rank, then chunk ID, so output is
    deterministic even when the retrievers assign equal scores.
    """
    if not 1 <= limit <= MAX_RESULTS:
        raise ValueError(f"limit must be between 1 and {MAX_RESULTS}")

    if rrf_k < 1:
        raise ValueError("rrf_k must be at least 1")

    candidates: dict[UUID, dict[str, Any]] = {}

    def add_ranked_results(
        results: list[dict[str, Any]],
        *,
        rank_field: str,
        score_field: str,
    ) -> None:
        seen_in_list: set[UUID] = set()
        rank = 0

        for result in results:
            chunk_id = result["chunk_id"]
            if chunk_id in seen_in_list:
                continue

            seen_in_list.add(chunk_id)
            rank += 1

            entry = candidates.get(chunk_id)
            if entry is None:
                entry = {
                    "item": dict(result),
                    "hybrid_score": 0.0,
                    "semantic_rank": None,
                    "keyword_rank": None,
                    "semantic_similarity": None,
                    "keyword_score": None,
                    "best_rank": None,
                }
                candidates[chunk_id] = entry

            entry[rank_field] = rank
            entry[score_field] = result.get(
                "similarity" if rank_field == "semantic_rank" else "keyword_score"
            )
            entry["hybrid_score"] += 1.0 / (rrf_k + rank)

            ranks = [
                value
                for value in (entry["semantic_rank"], entry["keyword_rank"])
                if value is not None
            ]
            entry["best_rank"] = min(ranks)

    add_ranked_results(
        semantic_results,
        rank_field="semantic_rank",
        score_field="semantic_similarity",
    )
    add_ranked_results(
        keyword_results,
        rank_field="keyword_rank",
        score_field="keyword_score",
    )

    ordered = sorted(
        candidates.items(),
        key=lambda item: (
            -item[1]["hybrid_score"],
            item[1]["best_rank"],
            str(item[0]),
        ),
    )

    fused: list[dict[str, Any]] = []
    for chunk_id, entry in ordered[:limit]:
        item = entry["item"]
        fused.append(
            {
                "chunk_id": chunk_id,
                "document_id": item["document_id"],
                "version_id": item["version_id"],
                "title": item["title"],
                "source_key": item["source_key"],
                "version_number": item["version_number"],
                "authority_level": item["authority_level"],
                "chunk_index": item["chunk_index"],
                "content": item["content"],
                "hybrid_score": entry["hybrid_score"],
                "semantic_rank": entry["semantic_rank"],
                "keyword_rank": entry["keyword_rank"],
                "semantic_similarity": entry["semantic_similarity"],
                "keyword_score": entry["keyword_score"],
            }
        )

    return fused


def semantic_search(
    *,
    tenant_id: UUID,
    query: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Retrieve relevant chunks from current, active documents in one tenant."""
    if not query.strip():
        raise ValueError("query must not be empty")

    if not 1 <= limit <= 20:
        raise ValueError("limit must be between 1 and 20")

    vector = embed_texts([query])[0]
    vector_text = "[" + ",".join(str(value) for value in vector) + "]"

    statement = text(
        """
        SELECT
            c.id AS chunk_id,
            c.document_id,
            c.document_version_id AS version_id,
            d.title,
            d.source_key,
            dv.version_number,
            dv.authority_level,
            c.chunk_index,
            c.content,
            c.embedding <=> CAST(:query_embedding AS vector(384)) AS distance
        FROM chunks AS c
        JOIN documents AS d
          ON d.id = c.document_id
        JOIN document_versions AS dv
          ON dv.id = c.document_version_id
        WHERE c.tenant_id = :tenant_id
          AND d.tenant_id = :tenant_id
          AND d.status = 'active'
          AND dv.is_current = TRUE
          AND c.embedding IS NOT NULL
          AND c.embedding_model = :model
        ORDER BY c.embedding <=> CAST(:query_embedding AS vector(384)), c.id
        LIMIT :result_limit
        """
    )

    with get_engine().connect() as connection:
        rows = connection.execute(
            statement,
            {
                "tenant_id": tenant_id,
                "query_embedding": vector_text,
                "model": EXPECTED_MODEL,
                "result_limit": limit,
            },
        ).mappings().all()

    results: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        distance = float(item.pop("distance"))
        item["cosine_distance"] = distance
        item["similarity"] = 1.0 - distance
        results.append(item)

    return results


def keyword_search(
    *,
    tenant_id: UUID,
    query: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Retrieve keyword-relevant chunks from current, active documents."""
    if not query.strip():
        raise ValueError("query must not be empty")

    if not 1 <= limit <= 20:
        raise ValueError("limit must be between 1 and 20")

    statement = text(
        """
        WITH search_query AS (
            SELECT websearch_to_tsquery('english', :query) AS tsq
        )
        SELECT
            c.id AS chunk_id,
            c.document_id,
            c.document_version_id AS version_id,
            d.title,
            d.source_key,
            dv.version_number,
            dv.authority_level,
            c.chunk_index,
            c.content,
            ts_rank_cd(
                to_tsvector('english', c.content),
                search_query.tsq
            ) AS keyword_score
        FROM chunks AS c
        JOIN documents AS d
          ON d.id = c.document_id
        JOIN document_versions AS dv
          ON dv.id = c.document_version_id
        CROSS JOIN search_query
        WHERE c.tenant_id = :tenant_id
          AND d.tenant_id = :tenant_id
          AND d.status = 'active'
          AND dv.is_current = TRUE
          AND to_tsvector('english', c.content) @@ search_query.tsq
        ORDER BY keyword_score DESC, c.id
        LIMIT :result_limit
        """
    )

    with get_engine().connect() as connection:
        rows = connection.execute(
            statement,
            {
                "tenant_id": tenant_id,
                "query": query,
                "result_limit": limit,
            },
        ).mappings().all()

    return [dict(row) for row in rows]


def hybrid_search(
    *,
    tenant_id: UUID,
    query: str,
    limit: int = 5,
    rrf_k: int = RRF_K,
) -> list[dict[str, Any]]:
    """Fuse semantic and keyword candidates with Reciprocal Rank Fusion.

    The existing retrieval functions cap each candidate list at 20, so the
    configurable multiplier can only widen the pool up to that hard limit.
    """
    if not query.strip():
        raise ValueError("query must not be empty")

    if not 1 <= limit <= MAX_RESULTS:
        raise ValueError(f"limit must be between 1 and {MAX_RESULTS}")

    if rrf_k < 1:
        raise ValueError("rrf_k must be at least 1")

    # Fetch a wider pool than the final result count so agreement outside the
    # top few ranks can still influence fusion. Each retriever returns at most
    # MAX_RESULTS rows, regardless of the multiplier.
    candidate_limit = min(MAX_RESULTS, limit * HYBRID_CANDIDATE_MULTIPLIER)

    semantic_results = semantic_search(
        tenant_id=tenant_id,
        query=query,
        limit=candidate_limit,
    )
    keyword_results = keyword_search(
        tenant_id=tenant_id,
        query=query,
        limit=candidate_limit,
    )

    return fuse_ranked_results(
        semantic_results,
        keyword_results,
        limit=limit,
        rrf_k=rrf_k,
    )