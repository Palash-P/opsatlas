"""Deterministic evidence assembly for grounded answer generation.

The assembly layer sits between retrieval and generation. It does not change the
retrieval ranking. It removes exact duplicates, preserves every distinct passage
from a document, applies a context-token budget, and labels every passage with
its source version so mixed-version evidence is explicit.
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass
from typing import Any
from uuid import UUID

SCORE_FIELDS = ("hybrid_score", "lexical_score", "similarity", "keyword_score")


class EvidenceAssemblyError(ValueError):
    """Raised when ranked retrieval results cannot be assembled safely."""


@dataclass(frozen=True)
class EvidenceItem:
    """One chunk supplied to the model and eligible for citation."""

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


@dataclass(frozen=True)
class DroppedEvidence:
    """A retrieval result that did not survive assembly."""

    chunk_id: UUID
    source_key: str
    version_number: int
    retrieval_rank: int
    reason: str


@dataclass(frozen=True)
class EvidenceBundle:
    """Structured context returned to generation and the API layer."""

    items: tuple[EvidenceItem, ...]
    dropped: tuple[DroppedEvidence, ...]
    context_text: str
    context_tokens_estimate: int
    context_token_budget: int
    mixed_versions: bool
    version_pairs: tuple[tuple[str, int], ...]

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(item.evidence_id for item in self.items)

    def item_by_id(self, evidence_id: str) -> EvidenceItem | None:
        for item in self.items:
            if item.evidence_id == evidence_id:
                return item
        return None


def estimate_tokens(text: str) -> int:
    """Approximate English token count without adding a tokenizer dependency.

    Four characters per token is a deliberately simple, documented approximation.
    It is used only for context budgeting; it is not billing or model-token
    accounting. Empty text estimates to zero.
    """

    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def _coerce_uuid(value: Any, field: str) -> UUID:
    try:
        return value if isinstance(value, UUID) else UUID(str(value))
    except (TypeError, ValueError, AttributeError) as exc:
        raise EvidenceAssemblyError(f"retrieval result has invalid {field}") from exc


def _coerce_int(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise EvidenceAssemblyError(f"retrieval result has invalid {field}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise EvidenceAssemblyError(f"retrieval result has invalid {field}") from exc
    if parsed < minimum:
        raise EvidenceAssemblyError(f"retrieval result has invalid {field}")
    return parsed


def _coerce_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceAssemblyError(f"retrieval result has invalid {field}")
    return value


def _optional_int(value: Any, field: str) -> int | None:
    if value is None:
        return None
    return _coerce_int(value, field, minimum=1)


def _numeric_scores(result: dict[str, Any]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for field in SCORE_FIELDS:
        value = result.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if math.isfinite(float(value)):
            scores[field] = float(value)
    return scores


def _primary_score(scores: dict[str, float]) -> tuple[float | None, str | None]:
    for field in SCORE_FIELDS:
        if field in scores:
            return scores[field], field
    return None, None


def _make_excerpt(content: str, max_chars: int) -> str:
    """Return a whitespace-normalized citation excerpt of the supplied chunk."""

    normalized = " ".join(content.split())
    if len(normalized) <= max_chars:
        return normalized
    cutoff = normalized[:max_chars].rstrip()
    if " " in cutoff:
        cutoff = cutoff.rsplit(" ", 1)[0]
    return cutoff + "…"


def _render_block(item: EvidenceItem) -> str:
    # Escape content so a document cannot forge evidence block boundaries or
    # inject new pseudo-tags into the prompt. The model still sees the text; it
    # just cannot break out of this block.
    return (
        f'<evidence id="{item.evidence_id}" '
        f'source_key="{html.escape(item.source_key, quote=True)}" '
        f'title="{html.escape(item.title, quote=True)}" '
        f'version="{item.version_number}" '
        f'chunk_id="{item.chunk_id}" '
        f'chunk_index="{item.chunk_index}">\n'
        f"{html.escape(item.content, quote=False)}\n"
        "</evidence>"
    )


def assemble_evidence(
    results: list[dict[str, Any]],
    *,
    context_token_budget: int,
    max_items: int = 8,
    citation_excerpt_chars: int = 600,
) -> EvidenceBundle:
    """Assemble ranked retrieval results into bounded, identifiable evidence.

    Ordering is deterministic: the input ranking is preserved, and the first
    occurrence of a duplicate wins. Exact duplicate chunk IDs and identical
    content from the same document version are removed. Distinct chunks from the
    same document are always retained, so a multi-chunk runbook is not collapsed
    to one passage.

    Every included item carries its document, source key, title, version number,
    version ID, chunk ID, chunk index, retrieval rank, and scores. Every rendered
    evidence block states all of those identifiers, so passages from different
    versions can never be mixed anonymously.
    """

    if context_token_budget < 1:
        raise ValueError("context_token_budget must be at least 1")
    if max_items < 1:
        raise ValueError("max_items must be at least 1")
    if citation_excerpt_chars < 1:
        raise ValueError("citation_excerpt_chars must be at least 1")

    items: list[EvidenceItem] = []
    dropped: list[DroppedEvidence] = []
    seen_chunk_ids: set[UUID] = set()
    seen_content: set[tuple[UUID, int, str]] = set()
    used_tokens = 0

    for retrieval_rank, result in enumerate(results, start=1):
        if not isinstance(result, dict):
            raise EvidenceAssemblyError("retrieval result must be an object")

        chunk_id = _coerce_uuid(result.get("chunk_id"), "chunk_id")
        document_id = _coerce_uuid(result.get("document_id"), "document_id")
        version_id = _coerce_uuid(result.get("version_id"), "version_id")
        source_key = _coerce_text(result.get("source_key"), "source_key")
        title = _coerce_text(result.get("title"), "title")
        version_number = _coerce_int(
            result.get("version_number"),
            "version_number",
            minimum=1,
        )
        chunk_index = _coerce_int(result.get("chunk_index"), "chunk_index")
        authority_level = _coerce_int(result.get("authority_level"), "authority_level")
        content = _coerce_text(result.get("content"), "content")

        if chunk_id in seen_chunk_ids:
            dropped.append(
                DroppedEvidence(
                    chunk_id=chunk_id,
                    source_key=source_key,
                    version_number=version_number,
                    retrieval_rank=retrieval_rank,
                    reason="duplicate_chunk_id",
                )
            )
            continue

        content_key = (document_id, version_number, " ".join(content.split()))
        if content_key in seen_content:
            dropped.append(
                DroppedEvidence(
                    chunk_id=chunk_id,
                    source_key=source_key,
                    version_number=version_number,
                    retrieval_rank=retrieval_rank,
                    reason="duplicate_content",
                )
            )
            continue

        seen_chunk_ids.add(chunk_id)
        seen_content.add(content_key)

        if len(items) >= max_items:
            dropped.append(
                DroppedEvidence(
                    chunk_id=chunk_id,
                    source_key=source_key,
                    version_number=version_number,
                    retrieval_rank=retrieval_rank,
                    reason="max_evidence",
                )
            )
            continue

        scores = _numeric_scores(result)
        score, score_field = _primary_score(scores)
        item = EvidenceItem(
            evidence_id=f"E{len(items) + 1}",
            retrieval_rank=retrieval_rank,
            chunk_id=chunk_id,
            document_id=document_id,
            version_id=version_id,
            source_key=source_key,
            title=title,
            version_number=version_number,
            chunk_index=chunk_index,
            authority_level=authority_level,
            content=content,
            excerpt=_make_excerpt(content, citation_excerpt_chars),
            estimated_tokens=estimate_tokens(content),
            score=score,
            score_field=score_field,
            scores=scores,
            semantic_rank=_optional_int(result.get("semantic_rank"), "semantic_rank"),
            keyword_rank=_optional_int(result.get("keyword_rank"), "keyword_rank"),
            ranking_origin=(
                result.get("ranking_origin")
                if isinstance(result.get("ranking_origin"), str)
                else None
            ),
        )

        block_tokens = estimate_tokens(_render_block(item)) + (1 if items else 0)
        if used_tokens + block_tokens > context_token_budget:
            dropped.append(
                DroppedEvidence(
                    chunk_id=chunk_id,
                    source_key=source_key,
                    version_number=version_number,
                    retrieval_rank=retrieval_rank,
                    reason="context_budget",
                )
            )
            # The item was not supplied, so its ID must not be eligible for citation.
            seen_chunk_ids.remove(chunk_id)
            seen_content.remove(content_key)
            continue

        used_tokens += block_tokens
        items.append(item)

    context_text = "\n".join(_render_block(item) for item in items)
    version_numbers_by_source: dict[str, set[int]] = {}
    for item in items:
        version_numbers_by_source.setdefault(item.source_key, set()).add(
            item.version_number
        )
    mixed_versions = any(
        len(version_numbers) > 1 for version_numbers in version_numbers_by_source.values()
    )
    version_pairs = tuple(
        sorted({(item.source_key, item.version_number) for item in items})
    )

    return EvidenceBundle(
        items=tuple(items),
        dropped=tuple(dropped),
        context_text=context_text,
        context_tokens_estimate=estimate_tokens(context_text),
        context_token_budget=context_token_budget,
        mixed_versions=mixed_versions,
        version_pairs=version_pairs,
    )
