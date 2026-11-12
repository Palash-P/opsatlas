"""Grounded answer endpoint: retrieve, assemble evidence, generate, cite."""

from __future__ import annotations

import math
import time
from typing import Annotated, Any, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.auth import Principal, require_identity, resolve_request_tenant
from app.core.config import get_settings
from app.core.limits import request_limits
from app.core.observability import (
    current_request_id,
    get_telemetry,
    set_error_category,
)
from app.services.answer_generation import (
    PROMPT_VERSION,
    generate_grounded_answer,
)
from app.services.embedding_client import EmbeddingServiceError
from app.services.evidence import (
    EvidenceAssemblyError,
    EvidenceBundle,
    assemble_evidence,
)
from app.services.llm_providers import (
    CitationValidationError,
    GenerationConfigurationError,
    GenerationInvalidResponseError,
    GenerationProviderError,
    GenerationRateLimitError,
    GenerationTimeoutError,
    InvalidModelOutputError,
    build_llm_provider,
)
from app.services.retrieval import (
    RRF_K,
    VERSION_MODES,
    hybrid_search,
    lexical_search,
    resolve_version_request,
    semantic_search,
)

router = APIRouter(
    prefix="/api/v1",
    tags=["answers"],
    dependencies=[Depends(request_limits)],
)

_SCORE_FIELDS = ("hybrid_score", "lexical_score", "similarity", "keyword_score")


class RetrievalOptions(BaseModel):
    """Optional retrieval configuration. Defaults preserve existing behavior."""

    strategy: Literal["hybrid", "lexical", "semantic"] = "hybrid"
    limit: int = Field(default=5, ge=1, le=20)
    keyword_strategy: Literal["and", "or_idf"] = "and"
    fusion_strategy: Literal[
        "chunk", "source", "weighted", "lexical_primary"
    ] = "chunk"
    semantic_weight: float = Field(default=1.0, gt=0)
    lexical_weight: float = Field(default=1.0, gt=0)
    rrf_k: int = Field(default=RRF_K, ge=1)
    version_mode: Literal[
        "auto", "current", "historical", "explicit_filter"
    ] = "auto"
    version_numbers: list[int] | None = None
    context_token_budget: int | None = Field(default=None, ge=1, le=32_000)
    max_evidence: int | None = Field(default=None, ge=1, le=20)
    citation_excerpt_chars: int | None = Field(default=None, ge=1, le=4_000)


class AnswerRequest(BaseModel):
    # Deprecated compatibility field. If supplied it must equal the tenant
    # resolved from the bearer token; the server never uses it as authorization.
    tenant_id: UUID | None = None
    question: str = Field(min_length=1, max_length=4_000)
    retrieval: RetrievalOptions = Field(default_factory=RetrievalOptions)


class CitationModel(BaseModel):
    evidence_id: str
    document_id: UUID
    version_id: UUID
    source_key: str
    title: str
    version_number: int
    chunk_id: UUID
    chunk_index: int
    excerpt: str


class EvidenceModel(BaseModel):
    evidence_id: str
    retrieval_rank: int
    chunk_id: UUID
    document_id: UUID
    version_id: UUID
    source_key: str
    title: str
    version_number: int
    chunk_index: int
    authority_level: int
    content: str
    excerpt: str
    estimated_tokens: int
    score: float | None
    score_field: str | None
    scores: dict[str, float]
    semantic_rank: int | None
    keyword_rank: int | None
    ranking_origin: str | None


class RetrievalResultModel(BaseModel):
    rank: int
    chunk_id: UUID
    document_id: UUID
    version_id: UUID
    source_key: str
    title: str
    version_number: int
    chunk_index: int
    score: float | None
    score_field: str | None
    scores: dict[str, float]
    semantic_rank: int | None
    keyword_rank: int | None
    lexical_rank: int | None
    ranking_origin: str | None


class RetrievalMetadata(BaseModel):
    strategy: str
    config: dict[str, Any]
    result_count: int
    results: list[RetrievalResultModel]


class GenerationMetadata(BaseModel):
    provider: str
    model: str
    requested_model: str | None = None
    provider_capabilities: dict[str, Any] | None = None
    prompt_version: str
    temperature: float
    max_output_tokens: int
    timeout_seconds: float
    max_retries: int
    latency_ms: int
    attempts: int
    input_tokens_estimate: int
    output_tokens_estimate: int
    provider_usage: dict[str, int] | None


class ConflictModel(BaseModel):
    evidence_ids: list[str]
    description: str


class VersionReferenceModel(BaseModel):
    source_key: str
    version_number: int
    status: str
    evidence_ids: list[str]


class VersionAnalysisModel(BaseModel):
    versions_identified: list[VersionReferenceModel]
    differences: list[str]
    conflicting_instructions: list[str]
    recommendation_applies_to: str | None
    supporting_evidence_ids: list[str]


class AbstentionModel(BaseModel):
    abstained: bool
    reason: str | None
    insufficient_evidence: bool


class AnswerDiagnostics(BaseModel):
    evidence_count: int
    context_tokens_estimate: int
    context_token_budget: int
    dropped_evidence_count: int
    dropped_evidence: list[dict[str, Any]]
    mixed_versions: bool
    version_pairs: list[list[Any]]
    uncited_evidence_ids: list[str]


class AnswerResponse(BaseModel):
    request_id: str
    question: str
    answer: str
    citations: list[CitationModel]
    evidence: list[EvidenceModel]
    retrieval: RetrievalMetadata
    generation: GenerationMetadata
    abstention: AbstentionModel
    conflicts: list[ConflictModel]
    version_analysis: VersionAnalysisModel | None
    uncertainty: str | None
    diagnostics: AnswerDiagnostics


def _numeric_scores(result: dict[str, Any]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for field in _SCORE_FIELDS:
        value = result.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if math.isfinite(float(value)):
            scores[field] = float(value)
    return scores


def _primary_score(result: dict[str, Any]) -> tuple[float | None, str | None]:
    for field in _SCORE_FIELDS:
        value = result.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if math.isfinite(float(value)):
            return float(value), field
    return None, None


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _update_telemetry(**values: Any) -> None:
    telemetry = get_telemetry()
    if telemetry is None:
        return
    for key, value in values.items():
        setattr(telemetry, key, value)


def _record_generation_failure(exc: Exception, started: float) -> None:
    attempts = getattr(exc, "attempts", None)
    _update_telemetry(
        generation_latency_ms=int((time.perf_counter() - started) * 1000),
        retry_count=(
            max(0, attempts - 1)
            if isinstance(attempts, int) and not isinstance(attempts, bool)
            else None
        ),
    )


def _generation_error_response(exc: Exception) -> tuple[str, int, str]:
    """Map a generation failure to an error category, status, and safe detail."""

    if isinstance(exc, GenerationTimeoutError):
        return "provider_timeout", 504, "Answer generation timed out"
    if isinstance(exc, GenerationRateLimitError):
        return "provider_rate_limited", 429, "Answer generation rate limit exceeded"
    if isinstance(exc, CitationValidationError):
        return (
            "citation_validation_error",
            502,
            "Model returned a citation that was not supplied as evidence",
        )
    if isinstance(exc, (InvalidModelOutputError, GenerationInvalidResponseError)):
        return "invalid_model_output", 502, "Model returned an invalid answer"
    if isinstance(exc, GenerationConfigurationError):
        return (
            "provider_configuration_error",
            503,
            "Answer generation is not configured",
        )
    if isinstance(exc, GenerationProviderError):
        return "provider_error", 502, "Answer generation provider failed"
    return "generation_error", 502, "Answer generation failed"


def _token_usage(generated: Any) -> tuple[int, int, str]:
    """Return input/output tokens plus whether they are actual or estimated."""

    usage = generated.provider_usage or {}
    prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
    completion = usage.get("completion_tokens", usage.get("output_tokens"))
    prompt = prompt if isinstance(prompt, int) and not isinstance(prompt, bool) else None
    completion = (
        completion
        if isinstance(completion, int) and not isinstance(completion, bool)
        else None
    )
    if prompt is None and completion is None:
        return (
            generated.input_tokens_estimate,
            generated.output_tokens_estimate,
            "estimated",
        )
    if prompt is not None and completion is not None:
        return prompt, completion, "actual"
    return (
        prompt if prompt is not None else generated.input_tokens_estimate,
        completion if completion is not None else generated.output_tokens_estimate,
        "mixed",
    )


def retrieve_for_answer(
    *,
    tenant_id: UUID,
    question: str,
    options: RetrievalOptions,
    principal: Principal | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Dispatch to an existing retrieval path without changing its defaults.

    Version intent is resolved once and recorded in the retrieval config so the
    effective scope is visible in API responses and evaluation artifacts.
    """

    if options.version_mode not in VERSION_MODES:
        raise ValueError(
            "version_mode must be one of: " + ", ".join(VERSION_MODES)
        )

    version_request = resolve_version_request(
        question,
        mode=options.version_mode,
        version_numbers=(
            tuple(options.version_numbers) if options.version_numbers else None
        ),
    )
    version_config = {
        "mode": options.version_mode,
        "scope": version_request.scope,
        "version_numbers": list(version_request.version_numbers),
        "include_historical": version_request.include_historical,
        "filter_numbers": list(version_request.filter_numbers),
    }
    version_kwargs = {
        "include_historical": version_request.include_historical,
        "filter_version_numbers": version_request.filter_numbers or None,
        "rank_version_numbers": version_request.version_numbers or None,
    }

    if options.strategy == "semantic":
        results = semantic_search(
            tenant_id=tenant_id,
            query=question,
            limit=options.limit,
            principal=principal,
            **version_kwargs,
        )
        config = {
            "strategy": "semantic",
            "limit": options.limit,
            "version": version_config,
        }
        return results, config

    if options.strategy == "lexical":
        results = lexical_search(
            tenant_id=tenant_id,
            query=question,
            limit=options.limit,
            principal=principal,
            **version_kwargs,
        )
        config = {
            "strategy": "lexical",
            "limit": options.limit,
            "version": version_config,
        }
        return results, config

    results = hybrid_search(
        tenant_id=tenant_id,
        query=question,
        limit=options.limit,
        rrf_k=options.rrf_k,
        keyword_strategy=options.keyword_strategy,
        fusion_strategy=options.fusion_strategy,
        semantic_weight=options.semantic_weight,
        lexical_weight=options.lexical_weight,
        version_request=version_request,
        principal=principal,
    )
    config = {
        "strategy": "hybrid",
        "limit": options.limit,
        "keyword_strategy": options.keyword_strategy,
        "fusion_strategy": options.fusion_strategy,
        "semantic_weight": options.semantic_weight,
        "lexical_weight": options.lexical_weight,
        "rrf_k": options.rrf_k,
        "version": version_config,
    }
    return results, config


def _retrieval_metadata(
    results: list[dict[str, Any]],
    config: dict[str, Any],
) -> RetrievalMetadata:
    return RetrievalMetadata(
        strategy=config["strategy"],
        config=config,
        result_count=len(results),
        results=[
            RetrievalResultModel(
                rank=rank,
                chunk_id=result["chunk_id"],
                document_id=result["document_id"],
                version_id=result["version_id"],
                source_key=result["source_key"],
                title=result["title"],
                version_number=result["version_number"],
                chunk_index=result["chunk_index"],
                score=_primary_score(result)[0],
                score_field=_primary_score(result)[1],
                scores=_numeric_scores(result),
                semantic_rank=_optional_int(result.get("semantic_rank")),
                keyword_rank=_optional_int(result.get("keyword_rank")),
                lexical_rank=_optional_int(result.get("lexical_rank")),
                ranking_origin=(
                    result.get("ranking_origin")
                    if isinstance(result.get("ranking_origin"), str)
                    else None
                ),
            )
            for rank, result in enumerate(results, start=1)
        ],
    )


def _evidence_models(bundle: EvidenceBundle) -> list[EvidenceModel]:
    return [
        EvidenceModel(
            evidence_id=item.evidence_id,
            retrieval_rank=item.retrieval_rank,
            chunk_id=item.chunk_id,
            document_id=item.document_id,
            version_id=item.version_id,
            source_key=item.source_key,
            title=item.title,
            version_number=item.version_number,
            chunk_index=item.chunk_index,
            authority_level=item.authority_level,
            content=item.content,
            excerpt=item.excerpt,
            estimated_tokens=item.estimated_tokens,
            score=item.score,
            score_field=item.score_field,
            scores=item.scores,
            semantic_rank=item.semantic_rank,
            keyword_rank=item.keyword_rank,
            ranking_origin=item.ranking_origin,
        )
        for item in bundle.items
    ]


def _version_analysis_model(generated) -> VersionAnalysisModel | None:
    analysis = generated.version_analysis
    if analysis is None:
        return None
    return VersionAnalysisModel(
        versions_identified=[
            VersionReferenceModel(
                source_key=reference.source_key,
                version_number=reference.version_number,
                status=reference.status,
                evidence_ids=list(reference.evidence_ids),
            )
            for reference in analysis.versions_identified
        ],
        differences=list(analysis.differences),
        conflicting_instructions=list(analysis.conflicting_instructions),
        recommendation_applies_to=analysis.recommendation_applies_to,
        supporting_evidence_ids=list(analysis.supporting_evidence_ids),
    )


@router.post("/answer", response_model=AnswerResponse)
def answer_endpoint(
    request: AnswerRequest,
    identity: Annotated[Principal, Depends(require_identity)],
) -> AnswerResponse:
    """Retrieve evidence, generate a grounded answer, and validate citations."""

    # The authenticated tenant is authoritative. A body tenant is only an
    # equality assertion; it never grants access.
    tenant_id = resolve_request_tenant(request.tenant_id, identity)

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be blank")

    _update_telemetry(retrieval_strategy=request.retrieval.strategy)
    retrieval_started = time.perf_counter()
    try:
        results, retrieval_config = retrieve_for_answer(
            tenant_id=tenant_id,
            question=question,
            options=request.retrieval,
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
            retrieval_latency_ms=int((time.perf_counter() - retrieval_started) * 1000)
        )

    settings = get_settings()
    evidence_started = time.perf_counter()
    try:
        bundle = assemble_evidence(
            results,
            context_token_budget=(
                request.retrieval.context_token_budget
                if request.retrieval.context_token_budget is not None
                else settings.answer_context_token_budget
            ),
            max_items=(
                request.retrieval.max_evidence
                if request.retrieval.max_evidence is not None
                else settings.answer_max_evidence
            ),
            citation_excerpt_chars=(
                request.retrieval.citation_excerpt_chars
                if request.retrieval.citation_excerpt_chars is not None
                else settings.answer_citation_excerpt_chars
            ),
        )
    except EvidenceAssemblyError:
        set_error_category("evidence_assembly_error")
        raise HTTPException(
            status_code=500,
            detail="Retrieved evidence could not be assembled safely",
        ) from None
    except ValueError as exc:
        set_error_category("invalid_evidence_configuration")
        raise HTTPException(status_code=422, detail=str(exc)) from None
    finally:
        _update_telemetry(
            evidence_latency_ms=int((time.perf_counter() - evidence_started) * 1000)
        )

    generation_started = time.perf_counter()
    try:
        provider = build_llm_provider(settings)
        _update_telemetry(
            provider=provider.name,
            requested_model=getattr(provider, "model", settings.llm_model),
        )
        generated = generate_grounded_answer(
            question=question,
            bundle=bundle,
            provider=provider,
            settings=settings,
        )
    except Exception as exc:  # noqa: BLE001 - map every generation failure safely
        _record_generation_failure(exc, generation_started)
        category, status_code, detail = _generation_error_response(exc)
        set_error_category(category)
        raise HTTPException(status_code=status_code, detail=detail) from None

    input_tokens, output_tokens, token_source = _token_usage(generated)
    _update_telemetry(
        generation_latency_ms=generated.latency_ms,
        retry_count=max(0, generated.attempts - 1),
        provider=generated.provider,
        observed_model=generated.model,
        requested_model=generated.requested_model or settings.llm_model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        token_source=token_source,
    )

    return AnswerResponse(
        request_id=current_request_id(),
        question=question,
        answer=generated.answer,
        citations=[
            CitationModel(
                evidence_id=citation.evidence_id,
                document_id=citation.document_id,
                version_id=citation.version_id,
                source_key=citation.source_key,
                title=citation.title,
                version_number=citation.version_number,
                chunk_id=citation.chunk_id,
                chunk_index=citation.chunk_index,
                excerpt=citation.excerpt,
            )
            for citation in generated.citations
        ],
        evidence=_evidence_models(bundle),
        retrieval=_retrieval_metadata(results, retrieval_config),
        generation=GenerationMetadata(
            provider=generated.provider,
            model=generated.model,
            requested_model=generated.requested_model,
            provider_capabilities=generated.provider_capabilities,
            prompt_version=generated.prompt_version or PROMPT_VERSION,
            temperature=settings.llm_temperature,
            max_output_tokens=settings.llm_max_output_tokens,
            timeout_seconds=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
            latency_ms=generated.latency_ms,
            attempts=generated.attempts,
            input_tokens_estimate=generated.input_tokens_estimate,
            output_tokens_estimate=generated.output_tokens_estimate,
            provider_usage=generated.provider_usage,
        ),
        abstention=AbstentionModel(
            abstained=generated.abstained,
            reason=generated.abstention_reason,
            insufficient_evidence=generated.insufficient_evidence,
        ),
        conflicts=[
            ConflictModel(
                evidence_ids=list(conflict.evidence_ids),
                description=conflict.description,
            )
            for conflict in generated.conflicts
        ],
        version_analysis=_version_analysis_model(generated),
        uncertainty=generated.uncertainty,
        diagnostics=AnswerDiagnostics(
            evidence_count=len(bundle.items),
            context_tokens_estimate=bundle.context_tokens_estimate,
            context_token_budget=bundle.context_token_budget,
            dropped_evidence_count=len(bundle.dropped),
            dropped_evidence=[
                {
                    "chunk_id": str(dropped.chunk_id),
                    "source_key": dropped.source_key,
                    "version_number": dropped.version_number,
                    "retrieval_rank": dropped.retrieval_rank,
                    "reason": dropped.reason,
                }
                for dropped in bundle.dropped
            ],
            mixed_versions=bundle.mixed_versions,
            version_pairs=[
                [source_key, version_number]
                for source_key, version_number in bundle.version_pairs
            ],
            uncited_evidence_ids=list(generated.uncited_evidence_ids),
        ),
    )
