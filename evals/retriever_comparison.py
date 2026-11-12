"""Comparison helpers for the retrieval strategies added in lexical v1.

The evaluation report already scores every retriever that the CLI passes to
``run_evaluation``. This module turns those per-query evaluations into the
cross-retriever view requested by the lexical v1 review: aggregate metrics,
per-category metrics, false-positive source rankings, retriever-only relevant
sources, ranking differences, duplicate-source effects, and query-level failures.

It is deliberately presentation-only. It never executes retrieval, changes a
ranking, or writes a baseline report.
"""

from __future__ import annotations

from typing import Any

METRIC_NAMES = ("recall_at_5", "mrr", "any_hit_at_5", "all_expected_at_5")


def _mean(values: list[float | int]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _metric_block(
    queries: list[dict[str, Any]],
    retriever_name: str,
    rank_unit: str,
) -> dict[str, Any]:
    key = "metrics" if rank_unit == "chunk" else "source_metrics"
    metrics = [
        query["retrievers"][retriever_name][key]
        for query in queries
    ]
    ndcg_values = [
        metric["ndcg_at_5"]
        for metric in metrics
        if metric.get("ndcg_at_5") is not None
    ]
    return {
        "query_count": len(metrics),
        "rank_unit": rank_unit,
        "recall_at_5": _mean([metric["recall_at_5"] for metric in metrics]),
        "mrr": _mean([metric["mrr"] for metric in metrics]),
        "any_hit_at_5": _mean([metric["any_hit_at_5"] for metric in metrics]),
        "all_expected_at_5": _mean(
            [metric["all_expected_at_5"] for metric in metrics]
        ),
        "ndcg_at_5": _mean(ndcg_values),
        "ndcg_status": (
            "computed_graded_relevance"
            if ndcg_values
            else "not_applicable_binary_relevance_only"
        ),
    }


def _delta(
    current: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, float | None]:
    return {
        f"{name}_delta": (
            None
            if current[name] is None or baseline[name] is None
            else current[name] - baseline[name]
        )
        for name in METRIC_NAMES
    }


def _chunk_order(query: dict[str, Any], retriever_name: str) -> list[str]:
    return [
        result["chunk_id"]
        for result in query["retrievers"][retriever_name]["results"]
    ]


def _source_order(query: dict[str, Any], retriever_name: str) -> list[tuple[str, int]]:
    return [
        (unit["source_key"], unit["version_number"])
        for unit in query["retrievers"][retriever_name]["source_units"]
    ]


def _false_positive_sources(query: dict[str, Any], retriever_name: str) -> list[dict[str, Any]]:
    return [
        {
            "rank": unit["rank"],
            "source_key": unit["source_key"],
            "version_number": unit["version_number"],
            "mapping_status": unit["mapping_status"],
        }
        for unit in query["retrievers"][retriever_name]["source_units"]
        if unit["mapping_status"] != "matched"
    ]


def _duplicate_source_effect(
    query: dict[str, Any],
    retriever_name: str,
) -> dict[str, Any]:
    source_counts: dict[tuple[str, int], int] = {}
    for result in query["retrievers"][retriever_name]["results"]:
        key = (result["source_key"], result["version_number"])
        source_counts[key] = source_counts.get(key, 0) + 1

    chunks = sum(source_counts.values())
    unique_sources = len(source_counts)
    return {
        "top_k_chunk_count": chunks,
        "unique_source_count": unique_sources,
        "duplicate_chunk_count": chunks - unique_sources,
        "max_chunks_for_one_source": max(source_counts.values(), default=0),
    }


def _relevant_sources_by_retriever(query: dict[str, Any], retriever_name: str) -> list[str]:
    evaluation = query["retrievers"][retriever_name]
    return list(
        evaluation.get("source_matched_source_ids", evaluation["matched_source_ids"])
    )


def _missing_sources_by_retriever(query: dict[str, Any], retriever_name: str) -> list[str]:
    evaluation = query["retrievers"][retriever_name]
    return list(
        evaluation.get(
            "source_missing_source_ids",
            evaluation["missing_source_ids"],
        )
    )


def _ranking_differences(
    queries: list[dict[str, Any]],
    retriever_names: list[str],
) -> dict[str, Any]:
    per_retriever: dict[str, Any] = {}
    for retriever_name in retriever_names:
        differing_chunk: list[str] = []
        differing_source: list[str] = []
        for query in queries:
            if _chunk_order(query, retriever_name) != _chunk_order(query, "semantic"):
                differing_chunk.append(query["id"])
            if _source_order(query, retriever_name) != _source_order(
                query, "semantic"
            ):
                differing_source.append(query["id"])
        per_retriever[retriever_name] = {
            "reference": "semantic",
            "queries_with_different_chunk_ranking": len(differing_chunk),
            "queries_with_different_source_ranking": len(differing_source),
            "chunk_ranking_query_ids": differing_chunk,
            "source_ranking_query_ids": differing_source,
        }
    return per_retriever


def _duplicate_source_effects(
    queries: list[dict[str, Any]],
    retriever_names: list[str],
) -> dict[str, Any]:
    by_retriever: dict[str, Any] = {}
    for retriever_name in retriever_names:
        effects = [
            _duplicate_source_effect(query, retriever_name)
            for query in queries
        ]
        queries_with_duplicates = [
            query["id"]
            for query, effect in zip(queries, effects)
            if effect["duplicate_chunk_count"] > 0
        ]
        by_retriever[retriever_name] = {
            "queries_with_duplicate_sources_in_top_5": len(
                queries_with_duplicates
            ),
            "query_ids_with_duplicate_sources_in_top_5": queries_with_duplicates,
            "total_duplicate_chunks_in_top_5": sum(
                effect["duplicate_chunk_count"] for effect in effects
            ),
            "max_chunks_for_one_source_in_top_5": max(
                (effect["max_chunks_for_one_source"] for effect in effects),
                default=0,
            ),
        }

    chunk_vs_source: dict[str, Any] = {}
    if "hybrid" in retriever_names and "hybrid_source" in retriever_names:
        differing = [
            query["id"]
            for query in queries
            if _source_order(query, "hybrid")
            != _source_order(query, "hybrid_source")
        ]
        chunk_vs_source = {
            "compared_retrievers": ["hybrid", "hybrid_source"],
            "queries_with_different_source_ranking": len(differing),
            "query_ids_with_different_source_ranking": differing,
        }

    return {
        "by_retriever": by_retriever,
        "chunk_level_vs_source_aware_fusion": chunk_vs_source,
    }


def _query_failures(
    queries: list[dict[str, Any]],
    retriever_names: list[str],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for query in queries:
        for retriever_name in retriever_names:
            evaluation = query["retrievers"][retriever_name]
            missing_sources = _missing_sources_by_retriever(query, retriever_name)
            if not missing_sources:
                continue
            failures.append(
                {
                    "id": query["id"],
                    "category": query["category"],
                    "retriever": retriever_name,
                    "expected_source_ids": query["expected_source_ids"],
                    "missing_source_ids": missing_sources,
                    "first_expected_rank_source": evaluation[
                        "source_metrics"
                    ]["first_expected_rank"],
                    "source_ranking": [
                        {
                            "rank": unit["rank"],
                            "source_key": unit["source_key"],
                            "version_number": unit["version_number"],
                            "mapped_source_id": unit["mapped_source_id"],
                            "mapping_status": unit["mapping_status"],
                        }
                        for unit in evaluation["source_units"]
                    ],
                }
            )
    return failures


def _non_expected_sources(
    query: dict[str, Any],
    retriever_name: str,
) -> list[dict[str, Any]]:
    """Matched source units that are not in the query's expected set.

    The source map may cover the whole corpus, so a non-expected document is
    still 'matched'. Those units occupy top-k slots and are false positives for
    that query even though they are not mapping errors.
    """
    expected = set(query["expected_source_ids"])
    return [
        {
            "rank": unit["rank"],
            "source_key": unit["source_key"],
            "version_number": unit["version_number"],
            "mapped_source_id": unit["mapped_source_id"],
        }
        for unit in query["retrievers"][retriever_name]["source_units"]
        if unit["mapping_status"] == "matched"
        and unit["mapped_source_id"] not in expected
    ]


def _false_positive_summary(
    queries: list[dict[str, Any]],
    retriever_names: list[str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name in retriever_names:
        per_query = [
            len(_non_expected_sources(query, name)) for query in queries
        ]
        summary[name] = {
            "definition": (
                "Matched source units in the top-5 that are not in the query's "
                "expected set. Mapping errors are counted separately."
            ),
            "total_non_expected_source_units_in_top_5": sum(per_query),
            "queries_with_non_expected_sources_in_top_5": sum(
                1 for count in per_query if count > 0
            ),
            "max_non_expected_source_units_for_one_query": max(
                per_query, default=0
            ),
        }
    return summary


def _candidate_counts(queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Report how many candidates each lexical path supplied per query.

    A hybrid run that receives an empty keyword list cannot demonstrate fusion
    behaviour. These counts make that failure visible instead of inferring it
    from an unchanged ranking.
    """
    counts: list[dict[str, Any]] = []
    for query in queries:
        diagnostics = query.get("diagnostics") or {}
        keyword = diagnostics.get("keyword") or {}
        lexical = diagnostics.get("lexical") or {}
        counts.append(
            {
                "id": query["id"],
                "category": query["category"],
                "keyword_candidate_count": keyword.get("result_count", 0)
                if keyword.get("provided")
                else None,
                "lexical_candidate_count": lexical.get("result_count", 0)
                if lexical.get("provided")
                else None,
                "lexical_chunks_in_hybrid_top_5": lexical.get(
                    "chunks_contributing_to_hybrid_top_k", 0
                )
                if lexical.get("provided")
                else None,
                "fusion_contributions": diagnostics.get(
                    "fusion_contributions", {}
                ),
            }
        )
    return counts


def _harm_vs_semantic(
    queries: list[dict[str, Any]],
    retriever_names: list[str],
) -> dict[str, Any]:
    """Find queries where a retriever raises or lowers quality vs semantic."""
    analysis: dict[str, Any] = {}
    for name in retriever_names:
        if name == "semantic":
            continue
        harmed_mrr: list[dict[str, Any]] = []
        improved_mrr: list[dict[str, Any]] = []
        harmed_recall: list[dict[str, Any]] = []
        improved_recall: list[dict[str, Any]] = []
        equal_mrr = 0
        for query in queries:
            semantic = query["retrievers"]["semantic"]["source_metrics"]
            current = query["retrievers"][name]["source_metrics"]
            mrr_delta = current["mrr"] - semantic["mrr"]
            recall_delta = (
                current["recall_at_5"] - semantic["recall_at_5"]
            )
            if mrr_delta == 0:
                equal_mrr += 1
            detail = {
                "id": query["id"],
                "category": query["category"],
                "semantic_first_expected_rank": semantic["first_expected_rank"],
                "retriever_first_expected_rank": current["first_expected_rank"],
                "semantic_mrr": semantic["mrr"],
                "retriever_mrr": current["mrr"],
                "mrr_delta": mrr_delta,
                "semantic_recall_at_5": semantic["recall_at_5"],
                "retriever_recall_at_5": current["recall_at_5"],
                "recall_at_5_delta": recall_delta,
            }
            if mrr_delta < 0:
                harmed_mrr.append(detail)
            elif mrr_delta > 0:
                improved_mrr.append(detail)
            if recall_delta < 0:
                harmed_recall.append(detail)
            elif recall_delta > 0:
                improved_recall.append(detail)

        analysis[name] = {
            "reference": "semantic",
            "source_mrr_improved_query_count": len(improved_mrr),
            "source_mrr_equal_query_count": equal_mrr,
            "source_mrr_harmed_query_count": len(harmed_mrr),
            "source_mrr_improved_query_ids": [item["id"] for item in improved_mrr],
            "source_mrr_harmed_query_ids": [item["id"] for item in harmed_mrr],
            "source_recall_improved_query_count": len(improved_recall),
            "source_recall_harmed_query_count": len(harmed_recall),
            "source_recall_improved_query_ids": [
                item["id"] for item in improved_recall
            ],
            "source_recall_harmed_query_ids": [
                item["id"] for item in harmed_recall
            ],
            "improved_queries": improved_mrr,
            "harmed_queries": harmed_mrr,
        }
    return analysis


def _classify_deltas(deltas: list[float]) -> dict[str, Any]:
    if not deltas:
        return {"improved": 0, "equal": 0, "harmed": 0, "mean_delta": None}
    return {
        "improved": sum(1 for delta in deltas if delta > 0),
        "equal": sum(1 for delta in deltas if delta == 0),
        "harmed": sum(1 for delta in deltas if delta < 0),
        "mean_delta": _mean(deltas),
    }


def _strategy_deltas(
    queries: list[dict[str, Any]],
    retriever_name: str,
    reference_name: str,
) -> dict[str, Any]:
    """Compare one retriever's source metrics against a reference retriever."""
    recall_deltas: list[float] = []
    mrr_deltas: list[float] = []
    all_expected_deltas: list[float] = []
    improved_queries: list[dict[str, Any]] = []
    harmed_queries: list[dict[str, Any]] = []

    for query in queries:
        reference = query["retrievers"][reference_name]["source_metrics"]
        current = query["retrievers"][retriever_name]["source_metrics"]
        recall_delta = current["recall_at_5"] - reference["recall_at_5"]
        mrr_delta = current["mrr"] - reference["mrr"]
        all_expected_delta = (
            current["all_expected_at_5"] - reference["all_expected_at_5"]
        )
        recall_deltas.append(recall_delta)
        mrr_deltas.append(mrr_delta)
        all_expected_deltas.append(all_expected_delta)

        detail = {
            "id": query["id"],
            "category": query["category"],
            "reference_first_expected_rank": reference["first_expected_rank"],
            "retriever_first_expected_rank": current["first_expected_rank"],
            "mrr_delta": mrr_delta,
            "recall_at_5_delta": recall_delta,
            "all_expected_at_5_delta": all_expected_delta,
        }
        if mrr_delta > 0 or recall_delta > 0:
            improved_queries.append(detail)
        elif mrr_delta < 0 or recall_delta < 0:
            harmed_queries.append(detail)

    return {
        "reference": reference_name,
        "recall_at_5": _classify_deltas(recall_deltas),
        "mrr": _classify_deltas(mrr_deltas),
        "all_expected_at_5": _classify_deltas(all_expected_deltas),
        "improved_query_ids": [item["id"] for item in improved_queries],
        "harmed_query_ids": [item["id"] for item in harmed_queries],
        "improved_queries": improved_queries,
        "harmed_queries": harmed_queries,
    }


def _fusion_vs_semantic(
    queries: list[dict[str, Any]],
    retriever_names: list[str],
) -> dict[str, Any]:
    """Summarize whether each fusion strategy improves on semantic-only."""
    return {
        name: _strategy_deltas(queries, name, "semantic")
        for name in retriever_names
        if name.startswith("hybrid")
    }


def _fusion_vs_equal_weight(
    queries: list[dict[str, Any]],
    retriever_names: list[str],
) -> dict[str, Any]:
    """Compare experimental fusion strategies with the equal-weight baseline.

    ``hybrid`` is the unchanged equal-weight chunk-RRF baseline. Every other
    hybrid strategy is compared against it so a weight or ordering experiment is
    judged on the change it makes, not only on its distance from semantic-only.
    """
    if "hybrid" not in retriever_names:
        return {}
    return {
        name: _strategy_deltas(queries, name, "hybrid")
        for name in retriever_names
        if name.startswith("hybrid") and name != "hybrid"
    }


def build_retriever_comparison(report: dict[str, Any]) -> dict[str, Any]:
    """Build the cross-retriever comparison section for an evaluation report."""

    queries = report["queries"]
    retriever_names = list(report["aggregate"]["retrievers"])

    source_metrics = {
        name: _metric_block(queries, name, "source") for name in retriever_names
    }
    chunk_metrics = {
        name: _metric_block(queries, name, "chunk") for name in retriever_names
    }

    deltas_vs: dict[str, Any] = {}
    for reference in ("semantic", "hybrid"):
        if reference not in retriever_names:
            continue
        deltas_vs[reference] = {
            name: {
                "chunk": _delta(chunk_metrics[name], chunk_metrics[reference]),
                "source": _delta(source_metrics[name], source_metrics[reference]),
            }
            for name in retriever_names
            if name != reference
        }

    relevant_sources: dict[str, Any] = {}
    semantic_matched = {
        query["id"]: set(_relevant_sources_by_retriever(query, "semantic"))
        for query in queries
    }
    for name in retriever_names:
        only_in_retriever = []
        for query in queries:
            matched = set(_relevant_sources_by_retriever(query, name))
            semantic = semantic_matched[query["id"]]
            only = sorted(matched - semantic)
            if only:
                only_in_retriever.append({"id": query["id"], "source_ids": only})
        relevant_sources[name] = {
            "queries_with_relevant_sources_not_in_semantic_top_5": len(
                only_in_retriever
            ),
            "queries": only_in_retriever,
        }

    per_query = []
    for query in queries:
        per_query.append(
            {
                "id": query["id"],
                "category": query["category"],
                "expected_source_ids": query["expected_source_ids"],
                "retrievers": {
                    name: {
                        "source_ranking": [
                            {
                                "rank": unit["rank"],
                                "source_key": unit["source_key"],
                                "version_number": unit["version_number"],
                                "mapped_source_id": unit["mapped_source_id"],
                                "mapping_status": unit["mapping_status"],
                                "relevant": unit["relevant"],
                            }
                            for unit in query["retrievers"][name]["source_units"]
                        ],
                        "false_positive_sources": _false_positive_sources(
                            query, name
                        ),
                        "non_expected_sources": _non_expected_sources(
                            query, name
                        ),
                        "duplicate_source_effect": _duplicate_source_effect(
                            query, name
                        ),
                        "relevant_source_ids": _relevant_sources_by_retriever(
                            query, name
                        ),
                        "missing_source_ids": _missing_sources_by_retriever(
                            query, name
                        ),
                    }
                    for name in retriever_names
                },
            }
        )

    return {
        "retrievers": retriever_names,
        "source_metrics": source_metrics,
        "chunk_metrics": chunk_metrics,
        "deltas_vs": deltas_vs,
        "relevant_sources_not_in_semantic": relevant_sources,
        "ranking_differences": _ranking_differences(queries, retriever_names),
        "duplicate_source_effects": _duplicate_source_effects(
            queries, retriever_names
        ),
        "query_level_failures": _query_failures(queries, retriever_names),
        "false_positive_summary": _false_positive_summary(
            queries, retriever_names
        ),
        "candidate_counts": _candidate_counts(queries),
        "harm_vs_semantic": _harm_vs_semantic(queries, retriever_names),
        "fusion_vs_semantic": _fusion_vs_semantic(queries, retriever_names),
        "fusion_vs_equal_weight": _fusion_vs_equal_weight(
            queries, retriever_names
        ),
        "per_query": per_query,
        "limitations": [
            (
                f"This comparison covers one {len(queries)}-query synthetic "
                "dataset. It does not establish generalization to other queries, "
                "corpora, tenants, or languages."
            ),
            (
                "Relevant sources are binary for Recall@5, MRR, and "
                "AllExpected@5. nDCG@5 is reported only when the dataset "
                "provides explicit integer grades."
            ),
            (
                "Only top-5 source and chunk rankings are compared. Sources "
                "below the cutoff cannot be ranked from this report."
            ),
            (
                "The AND keyword retriever frequently returns zero results on "
                "this corpus. A lower keyword MRR is a baseline property, not a "
                "bug in the comparison."
            ),
            (
                "Non-expected sources that contain supporting text are still "
                "counted as false positives, because relevance is assigned only "
                "through the explicit source map."
            ),
        ],
    }
