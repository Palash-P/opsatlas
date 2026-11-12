"""Semantic and hybrid search API with authenticated tenant scoping."""

import time
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.auth import Principal, require_identity, resolve_request_tenant
from app.core.limits import request_limits
from app.core.observability import get_telemetry, set_error_category
from app.services.embedding_client import EmbeddingServiceError
from app.services.retrieval import RRF_K, hybrid_search, semantic_search

router = APIRouter(
    prefix="/api/v1",
    tags=["search"],
    dependencies=[Depends(request_limits)],
)


def _update_telemetry(**values) -> None:
    telemetry = get_telemetry()
    if telemetry is None:
        return
    for key, value in values.items():
        setattr(telemetry, key, value)


class SearchRequest(BaseModel):
    # Deprecated compatibility field. When supplied it must match the tenant
    # resolved from the bearer token.
    tenant_id: UUID | None = None
    query: str = Field(min_length=1, max_length=4000)
    limit: int = Field(default=5, ge=1, le=20)


class SearchResult(BaseModel):
    chunk_id: UUID
    document_id: UUID
    version_id: UUID
    title: str
    source_key: str
    version_number: int
    authority_level: int
    chunk_index: int
    content: str
    cosine_distance: float
    similarity: float


class SearchResponse(BaseModel):
    query: str
    result_count: int
    results: list[SearchResult]


class HybridSearchResult(BaseModel):
    chunk_id: UUID
    document_id: UUID
    version_id: UUID
    title: str
    source_key: str
    version_number: int
    authority_level: int
    chunk_index: int
    content: str
    hybrid_score: float
    semantic_rank: int | None
    keyword_rank: int | None
    semantic_similarity: float | None
    keyword_score: float | None


class HybridSearchResponse(BaseModel):
    query: str
    result_count: int
    rrf_k: int
    results: list[HybridSearchResult]


@router.post("/search", response_model=SearchResponse)
def search_endpoint(
    request: SearchRequest,
    identity: Annotated[Principal, Depends(require_identity)],
) -> SearchResponse:
    """Search current document versions using semantic similarity."""
    if not request.query.strip():
        raise HTTPException(status_code=422, detail="Query must not be blank")

    tenant_id = resolve_request_tenant(request.tenant_id, identity)
    _update_telemetry(retrieval_strategy="semantic")
    started = time.perf_counter()
    try:
        results = semantic_search(
            tenant_id=tenant_id,
            query=request.query,
            limit=request.limit,
            principal=identity,
        )
    except EmbeddingServiceError:
        set_error_category("embedding_unavailable")
        raise HTTPException(
            status_code=503,
            detail="Embedding service is unavailable",
        ) from None
    except ValueError as exc:
        set_error_category("invalid_retrieval_option")
        raise HTTPException(status_code=422, detail=str(exc)) from None
    finally:
        _update_telemetry(
            retrieval_latency_ms=int((time.perf_counter() - started) * 1000)
        )

    return SearchResponse(
        query=request.query,
        result_count=len(results),
        results=results,
    )


@router.post("/search/hybrid", response_model=HybridSearchResponse)
def hybrid_search_endpoint(
    request: SearchRequest,
    identity: Annotated[Principal, Depends(require_identity)],
) -> HybridSearchResponse:
    """Search current document versions using fused semantic and keyword ranks."""
    if not request.query.strip():
        raise HTTPException(status_code=422, detail="Query must not be blank")

    tenant_id = resolve_request_tenant(request.tenant_id, identity)
    _update_telemetry(retrieval_strategy="hybrid")
    started = time.perf_counter()
    try:
        results = hybrid_search(
            tenant_id=tenant_id,
            query=request.query,
            limit=request.limit,
            principal=identity,
        )
    except EmbeddingServiceError:
        set_error_category("embedding_unavailable")
        raise HTTPException(
            status_code=503,
            detail="Embedding service is unavailable",
        ) from None
    except ValueError as exc:
        set_error_category("invalid_retrieval_option")
        raise HTTPException(status_code=422, detail=str(exc)) from None
    finally:
        _update_telemetry(
            retrieval_latency_ms=int((time.perf_counter() - started) * 1000)
        )

    return HybridSearchResponse(
        query=request.query,
        result_count=len(results),
        rrf_k=RRF_K,
        results=results,
    )
