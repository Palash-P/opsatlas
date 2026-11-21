"""Tenant-scoped semantic, keyword, and hybrid retrieval."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from sqlalchemy import text

from app.core.auth import Principal, acl_filter_sql, acl_parameters
from app.db.connections import get_engine
from app.services.embedding_client import (
    embed_texts,
    expected_dimensions,
    expected_model,
    validate_embeddings,
    vector_literal,
    vector_sql_type,
)

MAX_RESULTS = 20
RRF_K = 60
HYBRID_CANDIDATE_MULTIPLIER = 3
KEYWORD_STRATEGIES = ("and", "or_idf")
FUSION_STRATEGIES = ("chunk", "source", "weighted", "lexical_primary")

# Version handling is a ranking and scoping concern, not a new retrieval path.
# The default remains current-version-only retrieval. Explicit or historical
# questions may include superseded rows, and an explicit version number is used
# as a ranking boost by default. Hard filtering is opt-in so a question that
# mentions an API path such as ``/api/v1`` cannot silently remove the document
# that actually answers it.
VERSION_SCOPES = ("current", "explicit", "historical", "ambiguous")
VERSION_MODES = ("auto", "current", "historical", "explicit_filter")

# Exact identifier matches are high-precision signals. A shared identifier with
# document frequency 3 gets an IDF of about 2.0 on the synthetic corpus, while a
# coincidental rare generic term gets about 2.7. Weighting identifiers keeps an
# exact PAY-503 match from being outranked by documents that only contain one
# common query word. Identifiers remain ranking signals; they never filter.
IDENTIFIER_WEIGHT = 2.0

# Named weight profiles for the weighted-RRF experiment. The equal-weight profile
# is not used by the existing chunk RRF baseline, which stays at 1/(k+rank).
FUSION_WEIGHT_PROFILES = {
    "equal": {"semantic": 1.0, "lexical": 1.0},
    "semantic_heavy": {"semantic": 2.0, "lexical": 1.0},
    "lexical_heavy": {"semantic": 1.0, "lexical": 2.0},
}

# Identifiers are matched as literal strings rather than as full-text lexemes.
# A token such as PAY-503 is tokenized by the english configuration as "pay"
# plus "-503", and a bare 503 does not match it. Keeping these patterns separate
# from the generic lexeme terms prevents an identifier match from degenerating
# into a generic number match.
_IDENTIFIER_RE = re.compile(r"(?<![A-Za-z0-9])([A-Za-z]{2,}-\d{2,})(?![A-Za-z0-9])")
_HTTP_STATUS_RE = re.compile(r"(?<![A-Za-z0-9-])([1-5]\d{2})(?![A-Za-z0-9])")
_API_VERSION_RE = re.compile(r"(?i)(?<![A-Za-z0-9])v([1-9]\d*)(?![A-Za-z0-9])")

# Version intent detection. These patterns only decide scoping and ranking; the
# SQL still enforces tenant and active-document boundaries.
_VERSION_PHRASE_RE = re.compile(r"(?i)\b(?:version|revision|rev|release)\s+(\d+)\b")
_VERSION_TOKEN_RE = re.compile(r"(?i)(?<![A-Za-z0-9/])v(\d+)\b")
_HISTORICAL_MARKER_RE = re.compile(
    r"(?i)\b(?:deprecated|historical|older|previous|prior|legacy|stale|"
    r"superseded|old|revision\s+history|version\s+history|"
    r"chang(?:e|ed|es|ing)?\s+between\s+versions?|"
    r"between\s+versions?|across\s+versions?|"
    r"differs?|difference|compare|contrast)\b"
)
_REVISION_COMPARISON_RE = re.compile(r"(?i)\brevision\b")
_CURRENT_MARKER_RE = re.compile(r"(?i)\b(?:current|latest|newest|today|now|as\s+of)\b")


@dataclass(frozen=True)
class VersionRequest:
    """How a question's version language should affect retrieval.

    ``rank_numbers`` boosts supplied rows whose document version matches the
    numbers named in the question. ``filter_numbers`` is only populated in the
    opt-in ``explicit_filter`` mode. ``include_historical`` allows superseded
    versions to be retrieved; the default ``current`` scope does not.
    """

    scope: str
    version_numbers: tuple[int, ...] = ()
    include_historical: bool = False
    filter_numbers: tuple[int, ...] = ()
    source_context: str = ""

    def __post_init__(self) -> None:
        if self.scope not in VERSION_SCOPES:
            raise ValueError("version scope must be one of: " + ", ".join(VERSION_SCOPES))
        for value in (*self.version_numbers, *self.filter_numbers):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("version numbers must be positive integers")


def _extract_version_numbers(query: str) -> tuple[int, ...]:
    """Extract document-version numbers, excluding API path/version phrases."""

    numbers: set[int] = set()
    for match in _VERSION_PHRASE_RE.finditer(query):
        prefix = query[max(0, match.start() - 4) : match.start()].lower()
        phrase = match.group(0).lower()
        # "API version 3" names the API contract, not a document revision.
        if prefix.endswith("api ") or phrase.startswith("api "):
            continue
        numbers.add(int(match.group(1)))

    for match in _VERSION_TOKEN_RE.finditer(query):
        # A version token inside a path such as /api/v3 is an API contract.
        if match.start() > 0 and query[match.start() - 1] == "/":
            continue
        prefix = query[max(0, match.start() - 4) : match.start()].lower()
        if prefix.endswith("api "):
            continue
        numbers.add(int(match.group(1)))

    return tuple(sorted(numbers))


def detect_version_request(query: str) -> VersionRequest:
    """Classify version intent in a question without changing default retrieval."""

    numbers = _extract_version_numbers(query)
    historical = bool(_HISTORICAL_MARKER_RE.search(query))
    if not numbers and _REVISION_COMPARISON_RE.search(query):
        historical = True
    current = bool(_CURRENT_MARKER_RE.search(query))

    if numbers and historical:
        scope = "historical"
    elif numbers:
        scope = "explicit"
    elif historical and current:
        scope = "ambiguous"
    elif historical:
        scope = "historical"
    elif current:
        scope = "current"
    else:
        # Unspecified questions preserve the existing current-version behavior.
        scope = "current"

    return VersionRequest(
        scope=scope,
        version_numbers=numbers,
        include_historical=scope != "current",
    )


def resolve_version_request(
    query: str,
    *,
    mode: str = "auto",
    version_numbers: tuple[int, ...] | None = None,
    filter_versions: bool = False,
) -> VersionRequest:
    """Resolve an explicit retrieval mode into a concrete version request.

    ``auto`` detects intent. ``current`` forces current-only retrieval.
    ``historical`` includes superseded versions. ``explicit_filter`` also hard
    filters to the named version numbers, and is the only mode that filters.
    """

    if mode not in VERSION_MODES:
        raise ValueError("version mode must be one of: " + ", ".join(VERSION_MODES))

    detected = detect_version_request(query)
    numbers = (
        tuple(sorted(set(version_numbers)))
        if version_numbers is not None
        else detected.version_numbers
    )
    for value in numbers:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("version numbers must be positive integers")

    if mode == "current":
        return VersionRequest(scope="current", version_numbers=numbers)
    if mode == "historical":
        return VersionRequest(
            scope="historical",
            version_numbers=numbers,
            include_historical=True,
            source_context="forced_historical",
        )
    if mode == "explicit_filter":
        if not numbers:
            raise ValueError("explicit_filter mode requires at least one version number")
        return VersionRequest(
            scope="explicit",
            version_numbers=numbers,
            include_historical=True,
            filter_numbers=numbers,
            source_context="explicit_filter",
        )

    scope = detected.scope
    if version_numbers is not None:
        scope = "historical" if detected.scope == "historical" else "explicit"
    if filter_versions:
        if not numbers:
            raise ValueError("filter_versions requires at least one version number")
        return VersionRequest(
            scope=scope,
            version_numbers=numbers,
            include_historical=True,
            filter_numbers=numbers,
            source_context="explicit_filter",
        )
    return VersionRequest(
        scope=scope,
        version_numbers=numbers,
        include_historical=scope != "current",
        filter_numbers=(),
        source_context=detected.source_context,
    )


def _version_filter_sql(
    *,
    include_historical: bool,
    filter_numbers: tuple[int, ...],
) -> str:
    """Return a SQL fragment that preserves tenant/active filters by addition."""

    if not include_historical:
        return "AND dv.is_current = TRUE"
    if filter_numbers:
        return "AND dv.version_number = ANY(CAST(:version_numbers AS integer[]))"
    return ""


def _version_order_sql(
    *,
    rank_numbers: tuple[int, ...],
    include_historical: bool,
    version_number_expr: str = "dv.version_number",
    is_current_expr: str = "dv.is_current",
) -> str:
    """Build the ORDER BY prefix for version-aware ranking.

    The expressions are configurable because :func:`lexical_search` ranks over
    a CTE rather than the joined tables directly. Referencing ``dv`` there would
    put an out-of-scope alias in ``ORDER BY`` and fail at execution time.
    """

    clauses: list[str] = []
    if rank_numbers:
        clauses.append(
            f"CASE WHEN {version_number_expr} = ANY("
            "CAST(:rank_version_numbers AS integer[])) THEN 0 ELSE 1 END"
        )
    # Prefer current versions only when the caller did not name a version. An
    # explicit "version 1" request must not push the requested historical row
    # below unrelated current rows that merely share the version-1 number.
    if include_historical and not rank_numbers:
        clauses.append(f"CASE WHEN {is_current_expr} THEN 0 ELSE 1 END")
    if not clauses:
        return ""
    return ", ".join(clauses) + ", "


def _version_params(
    *,
    rank_numbers: tuple[int, ...],
    filter_numbers: tuple[int, ...],
) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if rank_numbers:
        params["rank_version_numbers"] = list(rank_numbers)
    if filter_numbers:
        params["version_numbers"] = list(filter_numbers)
    return params


class TenantIsolationError(RuntimeError):
    """Raised when a retrieval row escapes the requested tenant boundary."""


def _acl_scope(principal: Principal | None) -> tuple[str, dict[str, Any]]:
    """Return the optional ACL predicate and its parameters."""

    if principal is None:
        return "", {}
    return acl_filter_sql("d"), acl_parameters(principal)


def _verify_tenant_scope(
    rows: list[dict[str, Any]],
    tenant_id: UUID,
) -> list[dict[str, Any]]:
    """Defense-in-depth check on returned rows.

    The SQL predicate is the primary tenant boundary. This assertion catches a
    row that a future SQL change or a misbehaving database proxy returns across
    tenants. Unit-test fixtures that predate the ``tenant_id`` projection do not
    include the column; production queries always do.
    """

    for row in rows:
        value = row.get("tenant_id")
        if value is None:
            continue
        try:
            row_tenant = value if isinstance(value, UUID) else UUID(str(value))
        except (TypeError, ValueError) as exc:
            raise TenantIsolationError(
                "retrieval returned a row with an invalid tenant identifier"
            ) from exc
        if row_tenant != tenant_id:
            raise TenantIsolationError("retrieval returned a row outside the requested tenant")
    return rows


@dataclass(frozen=True)
class IdentifierTerm:
    """A technical identifier that must be matched literally, not lexically."""

    kind: str
    value: str
    sql_pattern: str


@dataclass(frozen=True)
class LexicalQuery:
    """Query text and identifiers split apart for the OR lexical strategy."""

    terms_query: str
    identifiers: tuple[IdentifierTerm, ...]


def _identifier_sql_pattern(value: str, *, allow_hyphen_before: bool = True) -> str:
    """Build a PostgreSQL ARE pattern with safe token boundaries.

    PostgreSQL's ARE dialect does not support lookbehind, so the boundary is
    expressed as an alternation. ``allow_hyphen_before`` is False for standalone
    numeric status codes so that ``503`` does not match inside ``PAY-503``.
    """

    preceding = "[^A-Za-z0-9-]" if not allow_hyphen_before else "[^A-Za-z0-9]"
    return f"(^|{preceding}){re.escape(value)}($|[^A-Za-z0-9])"


def prepare_lexical_query(query: str) -> LexicalQuery:
    """Split a query into generic terms and literal technical identifiers.

    Recognized identifiers are removed from the text that is later tokenized by
    PostgreSQL's ``english`` configuration. This prevents ``PAY-503`` from being
    reduced to the generic lexemes ``pay`` and ``-503`` and prevents ``v1`` from
    being treated as an ordinary word.
    """

    spans: list[tuple[int, int]] = []
    identifiers: list[IdentifierTerm] = []
    seen: set[tuple[str, str]] = set()

    def overlaps(start: int, end: int) -> bool:
        return any(start < span_end and span_start < end for span_start, span_end in spans)

    def collect(
        pattern: re.Pattern[str],
        *,
        kind: str,
        allow_hyphen_before: bool = True,
    ) -> None:
        for match in pattern.finditer(query):
            start, end = match.span()
            if overlaps(start, end):
                continue
            value = match.group(0)
            key = (kind, value.lower())
            if key in seen:
                continue
            seen.add(key)
            identifiers.append(
                IdentifierTerm(
                    kind=kind,
                    value=value,
                    sql_pattern=_identifier_sql_pattern(
                        value,
                        allow_hyphen_before=allow_hyphen_before,
                    ),
                )
            )
            spans.append((start, end))

    # Order matters: remove PAY-503 before looking for a standalone 503.
    collect(_IDENTIFIER_RE, kind="identifier")
    collect(_HTTP_STATUS_RE, kind="http_status", allow_hyphen_before=False)
    collect(_API_VERSION_RE, kind="api_version")

    if not spans:
        return LexicalQuery(terms_query=query, identifiers=tuple(identifiers))

    characters = list(query)
    for start, end in spans:
        characters[start:end] = " " * (end - start)

    return LexicalQuery(
        terms_query="".join(characters).strip(),
        identifiers=tuple(identifiers),
    )


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
            score = result.get("similarity" if rank_field == "semantic_rank" else "keyword_score")
            if score is None and rank_field == "keyword_rank":
                # The OR lexical strategy reports lexical_score instead of
                # keyword_score. Preserve it in the same metadata slot so fused
                # output does not silently drop the lexical score.
                score = result.get("lexical_score")
            entry[score_field] = score
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


def fuse_weighted_ranked_results(
    semantic_results: list[dict[str, Any]],
    keyword_results: list[dict[str, Any]],
    *,
    limit: int,
    semantic_weight: float,
    lexical_weight: float,
    rrf_k: int = RRF_K,
) -> list[dict[str, Any]]:
    """Fuse two ranked lists with explicit semantic and lexical RRF weights.

    This is an experimental companion to :func:`fuse_ranked_results`; that
    function and its equal-weight formula are deliberately unchanged. Each list
    contributes ``weight / (rrf_k + rank)``. Ties use the same deterministic
    order as the baseline: best rank, then chunk ID.
    """
    if not 1 <= limit <= MAX_RESULTS:
        raise ValueError(f"limit must be between 1 and {MAX_RESULTS}")

    if rrf_k < 1:
        raise ValueError("rrf_k must be at least 1")

    if semantic_weight <= 0 or lexical_weight <= 0:
        raise ValueError("semantic_weight and lexical_weight must be greater than 0")

    if not math.isfinite(semantic_weight) or not math.isfinite(lexical_weight):
        raise ValueError("semantic_weight and lexical_weight must be finite")

    candidates: dict[UUID, dict[str, Any]] = {}

    def add_ranked_results(
        results: list[dict[str, Any]],
        *,
        rank_field: str,
        score_field: str,
        weight: float,
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
            score = result.get("similarity" if rank_field == "semantic_rank" else "keyword_score")
            if score is None and rank_field == "keyword_rank":
                score = result.get("lexical_score")
            entry[score_field] = score
            entry["hybrid_score"] += weight / (rrf_k + rank)

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
        weight=semantic_weight,
    )
    add_ranked_results(
        keyword_results,
        rank_field="keyword_rank",
        score_field="keyword_score",
        weight=lexical_weight,
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
                "semantic_weight": semantic_weight,
                "lexical_weight": lexical_weight,
            }
        )

    return fused


def fuse_lexical_primary_results(
    semantic_results: list[dict[str, Any]],
    keyword_results: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Return lexical results first, then fill unused slots from semantic.

    The lexical list is the primary ranking. Semantic results are a fallback,
    not a re-ranking signal: they are appended only when the lexical list does
    not already fill ``limit`` unique chunks. Duplicate chunk IDs are removed.
    """
    if not 1 <= limit <= MAX_RESULTS:
        raise ValueError(f"limit must be between 1 and {MAX_RESULTS}")

    fused: list[dict[str, Any]] = []
    seen: set[UUID] = set()

    for lexical_rank, result in enumerate(keyword_results, start=1):
        chunk_id = result["chunk_id"]
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        item = dict(result)
        item["hybrid_score"] = result.get(
            "lexical_score",
            result.get("keyword_score", 0.0),
        )
        item["lexical_rank"] = lexical_rank
        item["semantic_rank"] = None
        item["ranking_origin"] = "lexical"
        fused.append(item)
        if len(fused) == limit:
            return fused

    for semantic_rank, result in enumerate(semantic_results, start=1):
        chunk_id = result["chunk_id"]
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        item = dict(result)
        item["hybrid_score"] = result.get("similarity", 0.0)
        item["semantic_rank"] = semantic_rank
        item["lexical_rank"] = None
        item["ranking_origin"] = "semantic_fallback"
        fused.append(item)
        if len(fused) == limit:
            break

    return fused


def _collapse_to_source_units(
    results: list[dict[str, Any]],
) -> list[tuple[tuple[str, int], dict[str, Any]]]:
    """Collapse chunk results to unique sources, keeping the best chunk rank."""

    units: dict[tuple[str, int], dict[str, Any]] = {}
    order: list[tuple[str, int]] = []

    for chunk_rank, result in enumerate(results, start=1):
        key = (result["source_key"], result["version_number"])
        entry = units.get(key)
        if entry is None:
            entry = {
                "item": result,
                "best_chunk_rank": chunk_rank,
                "chunk_count": 0,
            }
            units[key] = entry
            order.append(key)
        entry["chunk_count"] += 1

    return [(key, units[key]) for key in order]


def fuse_source_ranked_results(
    semantic_results: list[dict[str, Any]],
    keyword_results: list[dict[str, Any]],
    *,
    limit: int,
    rrf_k: int = RRF_K,
) -> list[dict[str, Any]]:
    """Fuse two chunk lists after collapsing each to its best source rank.

    This is an additional fusion mode, not a replacement for chunk-level RRF.
    Each retriever contributes ``1 / (rrf_k + source_rank)`` per source unit,
    where ``source_rank`` comes from the best (lowest) chunk rank for that
    ``(source_key, version_number)`` source. Duplicate chunks from one source
    therefore cannot crowd the fused ranking.
    """

    if not 1 <= limit <= MAX_RESULTS:
        raise ValueError(f"limit must be between 1 and {MAX_RESULTS}")

    if rrf_k < 1:
        raise ValueError("rrf_k must be at least 1")

    candidates: dict[tuple[str, int], dict[str, Any]] = {}

    def add_ranked_results(
        results: list[dict[str, Any]],
        *,
        prefix: str,
        score_field: str,
    ) -> None:
        for source_rank, (key, collapsed) in enumerate(
            _collapse_to_source_units(results),
            start=1,
        ):
            entry = candidates.get(key)
            if entry is None:
                entry = {
                    "item": dict(collapsed["item"]),
                    "hybrid_score": 0.0,
                    "best_source_rank": None,
                    "best_chunk_rank": None,
                    "semantic_source_rank": None,
                    "keyword_source_rank": None,
                    "semantic_chunk_rank": None,
                    "keyword_chunk_rank": None,
                    "semantic_source_chunk_count": None,
                    "keyword_source_chunk_count": None,
                    "semantic_similarity": None,
                    "keyword_score": None,
                }
                candidates[key] = entry

            entry[f"{prefix}_source_rank"] = source_rank
            entry[f"{prefix}_chunk_rank"] = collapsed["best_chunk_rank"]
            entry[f"{prefix}_source_chunk_count"] = collapsed["chunk_count"]
            score = collapsed["item"].get(score_field)
            if score is None and score_field == "keyword_score":
                score = collapsed["item"].get("lexical_score")
            entry[f"{prefix}_score"] = score
            entry["hybrid_score"] += 1.0 / (rrf_k + source_rank)

            source_ranks = [
                value
                for value in (
                    entry["semantic_source_rank"],
                    entry["keyword_source_rank"],
                )
                if value is not None
            ]
            chunk_ranks = [
                value
                for value in (
                    entry["semantic_chunk_rank"],
                    entry["keyword_chunk_rank"],
                )
                if value is not None
            ]
            entry["best_source_rank"] = min(source_ranks)
            entry["best_chunk_rank"] = min(chunk_ranks)

    add_ranked_results(
        semantic_results,
        prefix="semantic",
        score_field="similarity",
    )
    add_ranked_results(
        keyword_results,
        prefix="keyword",
        score_field="keyword_score",
    )

    ordered = sorted(
        candidates.items(),
        key=lambda item: (
            -item[1]["hybrid_score"],
            item[1]["best_source_rank"],
            item[1]["best_chunk_rank"],
            item[0][0],
            item[0][1],
        ),
    )

    fused: list[dict[str, Any]] = []
    for (source_key, version_number), entry in ordered[:limit]:
        item = entry["item"]
        fused.append(
            {
                "chunk_id": item["chunk_id"],
                "document_id": item["document_id"],
                "version_id": item["version_id"],
                "title": item["title"],
                "source_key": source_key,
                "version_number": version_number,
                "authority_level": item["authority_level"],
                "chunk_index": item["chunk_index"],
                "content": item["content"],
                "hybrid_score": entry["hybrid_score"],
                "semantic_rank": entry["semantic_chunk_rank"],
                "keyword_rank": entry["keyword_chunk_rank"],
                "semantic_source_rank": entry["semantic_source_rank"],
                "keyword_source_rank": entry["keyword_source_rank"],
                "semantic_source_chunk_count": entry["semantic_source_chunk_count"],
                "keyword_source_chunk_count": entry["keyword_source_chunk_count"],
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
    include_historical: bool = False,
    filter_version_numbers: tuple[int, ...] | None = None,
    rank_version_numbers: tuple[int, ...] | None = None,
    principal: Principal | None = None,
) -> list[dict[str, Any]]:
    """Retrieve relevant chunks from active documents in one tenant.

    By default only current versions are eligible, preserving the previous
    behavior. Historical versions are included only when the caller opts in.
    Version numbers affect ordering and, when explicitly requested, filtering;
    they never bypass tenant or active-document checks.

    When ``principal`` is supplied, document ACL grants are applied in the same
    query. The authenticated caller must pass its principal; an absent principal
    means "tenant-scoped, no ACL filter" for internal evaluation scripts.
    """
    if not query.strip():
        raise ValueError("query must not be empty")

    if not 1 <= limit <= 20:
        raise ValueError("limit must be between 1 and 20")

    filter_numbers = tuple(filter_version_numbers or ())
    rank_numbers = tuple(rank_version_numbers or ())
    for value in (*filter_numbers, *rank_numbers):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("version numbers must be positive integers")

    version_filter = _version_filter_sql(
        include_historical=include_historical,
        filter_numbers=filter_numbers,
    )
    version_order = _version_order_sql(
        rank_numbers=rank_numbers,
        include_historical=include_historical,
    )
    acl_sql, acl_params = _acl_scope(principal)

    # Validate again here even though the HTTP client already does: tests and
    # future callers may substitute the embedding function. A query vector that
    # does not match the stored column dimension must fail before PostgreSQL
    # sees it, rather than surfacing as an operator error mid-retrieval.
    vector = validate_embeddings(
        embed_texts([query]),
        expected_count=1,
        expected_dimensions=expected_dimensions(),
    )[0]
    vector_text = vector_literal(vector)
    vector_type = vector_sql_type()

    statement = text(
        f"""
        SELECT
            c.id AS chunk_id,
            c.document_id,
            c.tenant_id AS tenant_id,
            c.document_version_id AS version_id,
            d.title,
            d.source_key,
            dv.version_number,
            dv.authority_level,
            c.chunk_index,
            c.content,
            c.embedding <=> CAST(:query_embedding AS {vector_type}) AS distance
        FROM chunks AS c
        JOIN documents AS d
          ON d.id = c.document_id
        JOIN document_versions AS dv
          ON dv.id = c.document_version_id
        WHERE c.tenant_id = :tenant_id
          AND d.tenant_id = :tenant_id
          AND d.status = 'active'
          {version_filter}
          {acl_sql}
          AND c.embedding IS NOT NULL
          AND c.embedding_model = :model
        ORDER BY {version_order}c.embedding <=> CAST(:query_embedding AS {vector_type}), c.id
        LIMIT :result_limit
        """
    )

    params: dict[str, Any] = {
        "tenant_id": tenant_id,
        "query_embedding": vector_text,
        "model": expected_model(),
        "result_limit": limit,
        **_version_params(
            rank_numbers=rank_numbers,
            filter_numbers=filter_numbers,
        ),
        **acl_params,
    }

    with get_engine().connect() as connection:
        rows = connection.execute(statement, params).mappings().all()

    results: list[dict[str, Any]] = []
    for row in _verify_tenant_scope([dict(row) for row in rows], tenant_id):
        distance = float(row.pop("distance"))
        row["cosine_distance"] = distance
        row["similarity"] = 1.0 - distance
        results.append(row)

    return results


def keyword_search(
    *,
    tenant_id: UUID,
    query: str,
    limit: int = 5,
    include_historical: bool = False,
    filter_version_numbers: tuple[int, ...] | None = None,
    rank_version_numbers: tuple[int, ...] | None = None,
    principal: Principal | None = None,
) -> list[dict[str, Any]]:
    """Retrieve keyword-relevant chunks from active documents.

    The default remains current-version-only. Version parameters behave the same
    way as in :func:`semantic_search`.
    """
    if not query.strip():
        raise ValueError("query must not be empty")

    if not 1 <= limit <= 20:
        raise ValueError("limit must be between 1 and 20")

    filter_numbers = tuple(filter_version_numbers or ())
    rank_numbers = tuple(rank_version_numbers or ())
    for value in (*filter_numbers, *rank_numbers):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("version numbers must be positive integers")

    version_filter = _version_filter_sql(
        include_historical=include_historical,
        filter_numbers=filter_numbers,
    )
    version_order = _version_order_sql(
        rank_numbers=rank_numbers,
        include_historical=include_historical,
    )
    acl_sql, acl_params = _acl_scope(principal)

    statement = text(
        f"""
        WITH search_query AS (
            SELECT websearch_to_tsquery('english', :query) AS tsq
        )
        SELECT
            c.id AS chunk_id,
            c.document_id,
            c.tenant_id AS tenant_id,
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
          {version_filter}
          {acl_sql}
          AND to_tsvector('english', c.content) @@ search_query.tsq
        ORDER BY {version_order}keyword_score DESC, c.id
        LIMIT :result_limit
        """
    )

    params: dict[str, Any] = {
        "tenant_id": tenant_id,
        "query": query,
        "result_limit": limit,
        **_version_params(
            rank_numbers=rank_numbers,
            filter_numbers=filter_numbers,
        ),
        **acl_params,
    }

    with get_engine().connect() as connection:
        rows = connection.execute(statement, params).mappings().all()

    return _verify_tenant_scope([dict(row) for row in rows], tenant_id)


def lexical_search(
    *,
    tenant_id: UUID,
    query: str,
    limit: int = 5,
    identifier_weight: float = IDENTIFIER_WEIGHT,
    include_historical: bool = False,
    filter_version_numbers: tuple[int, ...] | None = None,
    rank_version_numbers: tuple[int, ...] | None = None,
    principal: Principal | None = None,
) -> list[dict[str, Any]]:
    """Retrieve chunks with OR lexeme matching and IDF-weighted coverage.

    Unlike :func:`keyword_search`, which keeps ``websearch_to_tsquery``'s AND
    semantics, this strategy matches any query lexeme and ranks by how much of
    the query's distinct-term IDF mass a chunk covers. Terms are weighted by
    distinct *source-document* frequency, not chunk frequency. Recognized
    identifiers such as ``PAY-503``, standalone HTTP status codes, and API
    versions (``v1``, ``v2``) are matched literally and separately from generic
    number terms. Their IDF sum is multiplied by ``identifier_weight`` so an
    exact identifier match is not outranked by a coincidental rare generic term.
    ``ts_rank_cd`` is a secondary signal after coverage and matched-term count,
    and chunk ID is the final deterministic tie-breaker.

    When a version is named in the question, ``rank_version_numbers`` moves the
    matching document version ahead of other candidates without removing them.
    ``filter_version_numbers`` performs an explicit hard filter.
    """

    if not query.strip():
        raise ValueError("query must not be empty")

    if not 1 <= limit <= 20:
        raise ValueError("limit must be between 1 and 20")

    if not identifier_weight > 0:
        raise ValueError("identifier_weight must be greater than 0")

    if not math.isfinite(identifier_weight):
        raise ValueError("identifier_weight must be finite")

    filter_numbers = tuple(filter_version_numbers or ())
    rank_numbers = tuple(rank_version_numbers or ())
    for value in (*filter_numbers, *rank_numbers):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("version numbers must be positive integers")

    prepared = prepare_lexical_query(query)
    if not prepared.identifiers and not re.search(r"\w", prepared.terms_query):
        return []

    identifier_values = [identifier.value for identifier in prepared.identifiers]
    identifier_patterns = [identifier.sql_pattern for identifier in prepared.identifiers]

    version_filter = _version_filter_sql(
        include_historical=include_historical,
        filter_numbers=filter_numbers,
    )
    version_order = _version_order_sql(
        rank_numbers=rank_numbers,
        include_historical=include_historical,
        # lexical_search orders over the `scored` CTE, so `dv` is not in scope.
        version_number_expr="scored.version_number",
        is_current_expr="scored.is_current",
    )
    acl_sql, acl_params = _acl_scope(principal)

    statement = text(
        f"""
        WITH filtered AS (
            SELECT
                c.id AS chunk_id,
                c.document_id,
                c.tenant_id AS tenant_id,
                c.document_version_id AS version_id,
                d.title,
                d.source_key,
                dv.version_number,
                dv.is_current,
                dv.authority_level,
                c.chunk_index,
                c.content,
                tsvector_to_array(
                    to_tsvector('english', c.content)
                ) AS content_lexemes
            FROM chunks AS c
            JOIN documents AS d
              ON d.id = c.document_id
            JOIN document_versions AS dv
              ON dv.id = c.document_version_id
            WHERE c.tenant_id = :tenant_id
              AND d.tenant_id = :tenant_id
              AND d.status = 'active'
              {version_filter}
              {acl_sql}
        ),
        query_terms AS (
            SELECT DISTINCT term
            FROM ts_debug('english', :lexical_query) AS debug
            CROSS JOIN LATERAL unnest(
                COALESCE(debug.lexemes, '{{}}'::text[])
            ) AS term
            WHERE term <> ''
        ),
        corpus_size AS (
            SELECT count(DISTINCT document_id)::double precision AS document_count
            FROM filtered
        ),
        term_doc_freq AS (
            SELECT term, count(DISTINCT document_id) AS document_frequency
            FROM filtered
            CROSS JOIN LATERAL unnest(content_lexemes) AS term
            GROUP BY term
        ),
        identifier_hits AS (
            SELECT
                filtered.chunk_id,
                identifier.value,
                filtered.document_id
            FROM filtered
            CROSS JOIN LATERAL unnest(
                CAST(:identifier_values AS text[]),
                CAST(:identifier_patterns AS text[])
            ) AS identifier(value, pattern)
            WHERE filtered.content ~* identifier.pattern
        ),
        identifier_doc_freq AS (
            SELECT value, count(DISTINCT document_id) AS document_frequency
            FROM identifier_hits
            GROUP BY value
        ),
        scored AS (
            SELECT
                filtered.*,
                (
                    SELECT COALESCE(
                        sum(
                            ln(
                                (corpus.document_count + 1.0)
                                / (term_doc_freq.document_frequency + 1.0)
                            ) + 1.0
                        ),
                        0.0
                    )
                    FROM (
                        SELECT unnest(filtered.content_lexemes)
                        INTERSECT
                        SELECT term FROM query_terms
                    ) AS matched(term)
                    JOIN term_doc_freq ON term_doc_freq.term = matched.term
                    CROSS JOIN corpus_size AS corpus
                ) AS term_coverage_score,
                (
                    SELECT COALESCE(
                        sum(
                            ln(
                                (corpus.document_count + 1.0)
                                / (identifier_doc_freq.document_frequency + 1.0)
                            ) + 1.0
                        ),
                        0.0
                    )
                    FROM (
                        SELECT DISTINCT value
                        FROM identifier_hits
                        WHERE identifier_hits.chunk_id = filtered.chunk_id
                    ) AS hit
                    JOIN identifier_doc_freq
                      ON identifier_doc_freq.value = hit.value
                    CROSS JOIN corpus_size AS corpus
                ) * CAST(:identifier_weight AS double precision) AS identifier_score,
                (
                    SELECT count(*)
                    FROM (
                        SELECT unnest(filtered.content_lexemes)
                        INTERSECT
                        SELECT term FROM query_terms
                    ) AS matched(term)
                ) AS lexical_term_matches,
                (
                    SELECT count(DISTINCT value)
                    FROM identifier_hits
                    WHERE identifier_hits.chunk_id = filtered.chunk_id
                ) AS identifier_matches
            FROM filtered
            WHERE filtered.content_lexemes && ARRAY(
                    SELECT term FROM query_terms
                )
               OR EXISTS (
                    SELECT 1
                    FROM identifier_hits
                    WHERE identifier_hits.chunk_id = filtered.chunk_id
                )
        ),
        query_tsquery AS (
            SELECT to_tsquery(
                'simple',
                string_agg(quote_literal(term), ' | ')
            ) AS tsq
            FROM query_terms
        )
        SELECT
            chunk_id,
            document_id,
            tenant_id,
            version_id,
            title,
            source_key,
            version_number,
            authority_level,
            chunk_index,
            content,
            term_coverage_score,
            identifier_score,
            (term_coverage_score + identifier_score) AS lexical_score,
            (lexical_term_matches + identifier_matches) AS matched_term_count,
            COALESCE(
                ts_rank_cd(
                    to_tsvector('english', content),
                    query_tsquery.tsq
                ),
                0.0
            ) AS rank_cd
        FROM scored
        CROSS JOIN query_tsquery
        ORDER BY
            {version_order}lexical_score DESC,
            matched_term_count DESC,
            rank_cd DESC,
            chunk_id
        LIMIT :result_limit
        """
    )

    params: dict[str, Any] = {
        "tenant_id": tenant_id,
        "lexical_query": prepared.terms_query,
        "identifier_values": identifier_values,
        "identifier_patterns": identifier_patterns,
        "identifier_weight": identifier_weight,
        "result_limit": limit,
        **_version_params(
            rank_numbers=rank_numbers,
            filter_numbers=filter_numbers,
        ),
        **acl_params,
    }

    with get_engine().connect() as connection:
        rows = connection.execute(statement, params).mappings().all()

    return _verify_tenant_scope([dict(row) for row in rows], tenant_id)


def hybrid_search(
    *,
    tenant_id: UUID,
    query: str,
    limit: int = 5,
    rrf_k: int = RRF_K,
    keyword_strategy: str = "and",
    fusion_strategy: str = "chunk",
    semantic_weight: float = 1.0,
    lexical_weight: float = 1.0,
    version_request: VersionRequest | None = None,
    principal: Principal | None = None,
) -> list[dict[str, Any]]:
    """Fuse semantic and keyword candidates with Reciprocal Rank Fusion.

    The defaults preserve the existing behavior: AND-based keyword retrieval,
    chunk-level equal-weight RRF, and current-version-only retrieval. Version
    intent is detected from the question unless the caller supplies an explicit
    :class:`VersionRequest`. Explicit version numbers boost matching versions;
    they do not remove other candidates unless ``filter_numbers`` is set.

    Set ``keyword_strategy='or_idf'`` to use :func:`lexical_search`, or
    ``fusion_strategy='source'`` to collapse duplicate chunks to their best
    source rank before fusion.

    Experimental strategies are opt-in:

    - ``fusion_strategy='weighted'`` uses ``semantic_weight`` and
      ``lexical_weight`` in place of equal weights.
    - ``fusion_strategy='lexical_primary'`` returns lexical results first and
      fills any remaining slots from semantic results.

    The existing retrieval functions cap each candidate list at 20, so the
    configurable multiplier can only widen the pool up to that hard limit.
    """
    if not query.strip():
        raise ValueError("query must not be empty")

    if not 1 <= limit <= MAX_RESULTS:
        raise ValueError(f"limit must be between 1 and {MAX_RESULTS}")

    if rrf_k < 1:
        raise ValueError("rrf_k must be at least 1")

    if keyword_strategy not in KEYWORD_STRATEGIES:
        raise ValueError("keyword_strategy must be one of: " + ", ".join(KEYWORD_STRATEGIES))

    if fusion_strategy not in FUSION_STRATEGIES:
        raise ValueError("fusion_strategy must be one of: " + ", ".join(FUSION_STRATEGIES))

    if not (semantic_weight > 0) or not (lexical_weight > 0):
        raise ValueError("semantic_weight and lexical_weight must be greater than 0")

    if not math.isfinite(semantic_weight) or not math.isfinite(lexical_weight):
        raise ValueError("fusion weights must be finite")

    if version_request is None:
        version_request = detect_version_request(query)
    elif not isinstance(version_request, VersionRequest):
        raise ValueError("version_request must be a VersionRequest")

    # Fetch a wider pool than the final result count so agreement outside the
    # top few ranks can still influence fusion. Each retriever returns at most
    # MAX_RESULTS rows, regardless of the multiplier.
    candidate_limit = min(MAX_RESULTS, limit * HYBRID_CANDIDATE_MULTIPLIER)
    version_kwargs = {
        "include_historical": version_request.include_historical,
        "filter_version_numbers": version_request.filter_numbers or None,
        "rank_version_numbers": version_request.version_numbers or None,
    }

    semantic_results = semantic_search(
        tenant_id=tenant_id,
        query=query,
        limit=candidate_limit,
        principal=principal,
        **version_kwargs,
    )
    keyword_retriever = keyword_search if keyword_strategy == "and" else lexical_search
    keyword_results = keyword_retriever(
        tenant_id=tenant_id,
        query=query,
        limit=candidate_limit,
        principal=principal,
        **version_kwargs,
    )

    if fusion_strategy == "chunk":
        return fuse_ranked_results(
            semantic_results,
            keyword_results,
            limit=limit,
            rrf_k=rrf_k,
        )
    if fusion_strategy == "source":
        return fuse_source_ranked_results(
            semantic_results,
            keyword_results,
            limit=limit,
            rrf_k=rrf_k,
        )
    if fusion_strategy == "weighted":
        return fuse_weighted_ranked_results(
            semantic_results,
            keyword_results,
            limit=limit,
            semantic_weight=semantic_weight,
            lexical_weight=lexical_weight,
            rrf_k=rrf_k,
        )

    return fuse_lexical_primary_results(
        semantic_results,
        keyword_results,
        limit=limit,
    )
