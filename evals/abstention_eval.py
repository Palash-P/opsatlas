"""Read-only abstention-readiness diagnostics for unanswerable golden records.

This module deliberately does not change retrieval metrics. It runs the same
retrievers on unanswerable records, reports what they return, and measures
whether top-score thresholds separate answerable from unanswerable queries.

A retrieval-only threshold is not a generation abstention policy. These results
say nothing about whether a future answer generator would refuse, cite, or
answer correctly.
"""

from __future__ import annotations

from collections.abc import Callable
from statistics import median
from typing import Any

from evals.retrieval_eval import RETRIEVER_SCORE_FIELDS, K

# Author-assigned from a corpus review, not inferred from retrieval results.
CORPUS_EVIDENCE_BY_CATEGORY = {
    "insufficient_evidence": "related_context_only",
    "no_relevant_source": "none",
    "sensitive_unanswerable": "none",
}


def _score_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "max": None}
    return {
        "min": min(values),
        "median": median(values),
        "max": max(values),
    }


def _result_top_score(
    results: list[dict[str, Any]],
    retriever_name: str,
) -> float | None:
    if not results:
        return None
    score_field = RETRIEVER_SCORE_FIELDS[retriever_name]
    score = results[0].get(score_field)
    if score is None:
        # Report rows serialize the retriever-specific score as "score".
        # Live retriever rows keep the named field.
        score = results[0].get("score")
    if score is None:
        return None
    return float(score)


def _unique_source_keys(results: list[dict[str, Any]], *, k: int) -> list[str]:
    keys: list[str] = []
    for result in results[:k]:
        source_key = result["source_key"]
        if source_key not in keys:
            keys.append(source_key)
    return keys


def _best_threshold(
    answerable_scores: list[float],
    unanswerable_scores: list[float],
) -> dict[str, Any]:
    """Find the threshold with the best Youden J on observed top scores.

    The threshold is a diagnostic upper bound on one score, not a calibrated
    confidence value. Ties prefer the higher threshold so the reported boundary
    is the least permissive one with the same separation.
    """
    if not answerable_scores or not unanswerable_scores:
        return {
            "threshold": None,
            "answerable_retention": None,
            "unanswerable_retrieval_rate": None,
            "youden_j": None,
        }

    candidates = sorted(set(answerable_scores) | set(unanswerable_scores))
    best: dict[str, Any] | None = None
    for threshold in candidates:
        retention = sum(
            1 for score in answerable_scores if score >= threshold
        ) / len(answerable_scores)
        false_positive = sum(
            1 for score in unanswerable_scores if score >= threshold
        ) / len(unanswerable_scores)
        youden = retention - false_positive
        candidate = {
            "threshold": threshold,
            "answerable_retention": retention,
            "unanswerable_retrieval_rate": false_positive,
            "youden_j": youden,
        }
        if best is None or (
            youden,
            threshold,
        ) > (
            best["youden_j"],
            best["threshold"],
        ):
            best = candidate

    return best


def build_abstention_readiness(
    *,
    report: dict[str, Any],
    retrievers: dict[str, Callable[..., list[dict[str, Any]]]],
    tenant_id: Any,
    fetch_limit: int = K,
    k: int = K,
) -> dict[str, Any]:
    """Run unanswerable records separately and describe threshold behaviour."""
    retriever_names = [
        name
        for name in report["aggregate"]["retrievers"]
        if name in retrievers
    ]

    answerable_scores: dict[str, list[float]] = {
        name: [] for name in retriever_names
    }
    for query in report["queries"]:
        for name in retriever_names:
            results = query["retrievers"][name]["results"]
            score = _result_top_score(results, name)
            if score is not None:
                answerable_scores[name].append(score)

    per_query: list[dict[str, Any]] = []
    unanswerable_scores: dict[str, list[float]] = {
        name: [] for name in retriever_names
    }
    for record in report.get("unanswerable", []):
        record_results: dict[str, Any] = {}
        for name in retriever_names:
            rows = retrievers[name](
                tenant_id=tenant_id,
                query=record["question"],
                limit=fetch_limit,
            )
            top_score = _result_top_score(rows, name)
            if top_score is not None:
                unanswerable_scores[name].append(top_score)
            record_results[name] = {
                "result_count": len(rows),
                "unique_source_count": len(
                    {row["source_key"] for row in rows[:k]}
                ),
                "top_source_keys": _unique_source_keys(rows, k=k),
                "top_score": top_score,
                "returned_evidence": bool(rows),
            }
        per_query.append(
            {
                "id": record["id"],
                "category": record["category"],
                "corpus_evidence": CORPUS_EVIDENCE_BY_CATEGORY.get(
                    record["category"],
                    "not_classified",
                ),
                "retrievers": record_results,
            }
        )

    by_retriever: dict[str, Any] = {}
    for name in retriever_names:
        answerable = answerable_scores[name]
        unanswerable = unanswerable_scores[name]
        any_result = sum(
            1
            for record in per_query
            if record["retrievers"][name]["returned_evidence"]
        )
        total_results = sum(
            record["retrievers"][name]["result_count"] for record in per_query
        )
        total_unique_sources = sum(
            record["retrievers"][name]["unique_source_count"]
            for record in per_query
        )
        by_retriever[name] = {
            "answerable_query_count": len(answerable),
            "answerable_top_score": _score_summary(answerable),
            "unanswerable_query_count": len(unanswerable),
            "unanswerable_queries_with_any_result": any_result,
            "unanswerable_queries_with_any_result_rate": (
                any_result / len(per_query) if per_query else None
            ),
            "mean_unanswerable_result_count": (
                total_results / len(per_query) if per_query else None
            ),
            "mean_unanswerable_unique_source_count": (
                total_unique_sources / len(per_query) if per_query else None
            ),
            "unanswerable_top_score": _score_summary(unanswerable),
            "score_separation": {
                "answerable_min": min(answerable) if answerable else None,
                "unanswerable_max": max(unanswerable) if unanswerable else None,
                "perfect_separation": (
                    bool(answerable)
                    and bool(unanswerable)
                    and min(answerable) > max(unanswerable)
                ),
            },
            "best_threshold": _best_threshold(answerable, unanswerable),
        }

    return {
        "scored_for_abstention": len(per_query),
        "mixed_into_retrieval_metrics": False,
        "corpus_evidence_by_category": dict(CORPUS_EVIDENCE_BY_CATEGORY),
        "by_retriever": by_retriever,
        "per_query": per_query,
        "limitations": [
            (
                "Unanswerable records are run separately and are never included "
                "in Recall@5, MRR, nDCG, or AllExpected@5."
            ),
            (
                "corpus_evidence is an author-assigned corpus review category. "
                "insufficient_evidence means related context exists but does not "
                "answer the question; none means no relevant source exists."
            ),
            (
                "A score threshold is not a calibrated abstention policy and "
                "cannot establish generation refusal, citation correctness, or "
                "answer faithfulness."
            ),
            (
                "The threshold analysis is in-sample over the same records used "
                "to observe the scores. It is descriptive, not validated."
            ),
        ],
    }
