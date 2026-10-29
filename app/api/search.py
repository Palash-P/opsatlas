"""Semantic search API."""

from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.embedding_client import EmbeddingServiceError
from app.services.retrieval import RRF_K, hybrid_search, semantic_search

router = APIRouter(prefix="/api/v1", tags=["search"])


class SearchRequest(BaseModel):
    tenant_id: UUID
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
def search_endpoint(request: SearchRequest) -> SearchResponse:
    """Search current document versions using semantic similarity."""
    if not request.query.strip():
        raise HTTPException(status_code=422, detail="Query must not be blank")

    try:
        results = semantic_search(
            tenant_id=request.tenant_id,
            query=request.query,
            limit=request.limit,
        )
    except EmbeddingServiceError:
        raise HTTPException(
            status_code=503,
            detail="Embedding service is unavailable",
        ) from None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None

    return SearchResponse(
        query=request.query,
        result_count=len(results),
        results=results,
    )


@router.post("/search/hybrid", response_model=HybridSearchResponse)
def hybrid_search_endpoint(request: SearchRequest) -> HybridSearchResponse:
    """Search current document versions using fused semantic and keyword ranks."""
    if not request.query.strip():
        raise HTTPException(status_code=422, detail="Query must not be blank")

    try:
        results = hybrid_search(
            tenant_id=request.tenant_id,
            query=request.query,
            limit=request.limit,
        )
    except EmbeddingServiceError:
        raise HTTPException(
            status_code=503,
            detail="Embedding service is unavailable",
        ) from None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None

    return HybridSearchResponse(
        query=request.query,
        result_count=len(results),
        rrf_k=RRF_K,
        results=results,
    )