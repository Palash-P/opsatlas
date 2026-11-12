from evals.retriever_comparison import build_retriever_comparison


def make_unit(
    rank: int,
    source_key: str,
    *,
    version_number: int = 1,
    mapped_source_id: str | None = None,
    status: str = "matched",
    relevant: bool = True,
) -> dict:
    return {
        "rank": rank,
        "source_key": source_key,
        "version_number": version_number,
        "mapped_source_id": mapped_source_id,
        "mapping_status": status,
        "relevant": relevant,
    }


def make_result(rank: int, source_key: str, *, version_number: int = 1) -> dict:
    return {
        "rank": rank,
        "chunk_id": f"chunk-{rank}",
        "document_id": f"doc-{rank}",
        "version_id": f"version-{rank}",
        "source_key": source_key,
        "version_number": version_number,
        "mapped_source_id": None,
        "mapping_status": "matched",
        "score": 1.0 / rank,
        "score_field": "similarity",
    }


def make_metrics(recall: float, mrr: float) -> dict:
    return {
        "rank_unit": "source",
        "recall_at_5": recall,
        "mrr": mrr,
        "any_hit_at_5": int(recall > 0),
        "all_expected_at_5": int(recall == 1.0),
        "first_expected_rank": None if mrr == 0 else round(1 / mrr),
        "ndcg_at_5": None,
        "ndcg_status": "not_applicable_binary_relevance_only",
    }


def make_evaluation(
    *,
    results: list[dict],
    source_units: list[dict],
    matched: list[str],
    missing: list[str],
    recall: float,
    mrr: float,
) -> dict:
    return {
        "results": results,
        "source_units": source_units,
        "matched_source_ids": matched,
        "missing_source_ids": missing,
        "source_metrics": make_metrics(recall, mrr),
        "metrics": make_metrics(recall, mrr),
    }


def build_report() -> dict:
    # Query 1: semantic misses source-b; lexical finds it at rank 1.
    semantic_query_1 = make_evaluation(
        results=[make_result(1, "runbooks/a.md")],
        source_units=[
            make_unit(1, "runbooks/a.md", mapped_source_id="source-a"),
        ],
        matched=["source-a"],
        missing=["source-b"],
        recall=0.5,
        mrr=1.0,
    )
    lexical_query_1 = make_evaluation(
        results=[
            make_result(1, "runbooks/b.md"),
            make_result(2, "runbooks/a.md"),
        ],
        source_units=[
            make_unit(1, "runbooks/b.md", mapped_source_id="source-b"),
            make_unit(2, "runbooks/a.md", mapped_source_id="source-a"),
        ],
        matched=["source-a", "source-b"],
        missing=[],
        recall=1.0,
        mrr=1.0,
    )
    # Query 2: duplicate chunks from one source in the semantic list.
    semantic_query_2 = make_evaluation(
        results=[
            make_result(1, "runbooks/a.md"),
            make_result(2, "runbooks/a.md"),
            make_result(3, "runbooks/a.md"),
        ],
        source_units=[
            make_unit(1, "runbooks/a.md", mapped_source_id="source-a"),
            make_unit(2, "runbooks/unknown.md", status="unmapped", relevant=False),
        ],
        matched=["source-a"],
        missing=[],
        recall=1.0,
        mrr=1.0,
    )
    lexical_query_2 = make_evaluation(
        results=[make_result(1, "runbooks/a.md")],
        source_units=[
            make_unit(1, "runbooks/a.md", mapped_source_id="source-a"),
        ],
        matched=["source-a"],
        missing=[],
        recall=1.0,
        mrr=1.0,
    )

    queries = [
        {
            "id": "q1",
            "category": "multi_document",
            "expected_source_ids": ["source-a", "source-b"],
            "retrievers": {
                "semantic": semantic_query_1,
                "lexical": lexical_query_1,
            },
        },
        {
            "id": "q2",
            "category": "exact_identifier",
            "expected_source_ids": ["source-a"],
            "retrievers": {
                "semantic": semantic_query_2,
                "lexical": lexical_query_2,
            },
        },
    ]
    return {
        "aggregate": {"retrievers": ["semantic", "lexical"]},
        "queries": queries,
    }


def test_comparison_reports_source_and_chunk_metrics_and_deltas() -> None:
    comparison = build_retriever_comparison(build_report())

    assert comparison["retrievers"] == ["semantic", "lexical"]
    assert comparison["source_metrics"]["semantic"]["recall_at_5"] == 0.75
    assert comparison["source_metrics"]["lexical"]["recall_at_5"] == 1.0
    assert comparison["deltas_vs"]["semantic"]["lexical"]["source"][
        "recall_at_5_delta"
    ] == 0.25


def test_comparison_reports_false_positive_sources_and_duplicate_effects() -> None:
    comparison = build_retriever_comparison(build_report())

    query_2 = comparison["per_query"][1]
    assert query_2["retrievers"]["semantic"]["false_positive_sources"] == [
        {
            "rank": 2,
            "source_key": "runbooks/unknown.md",
            "version_number": 1,
            "mapping_status": "unmapped",
        }
    ]
    duplicate = query_2["retrievers"]["semantic"]["duplicate_source_effect"]
    assert duplicate == {
        "top_k_chunk_count": 3,
        "unique_source_count": 1,
        "duplicate_chunk_count": 2,
        "max_chunks_for_one_source": 3,
    }
    aggregate = comparison["duplicate_source_effects"]["by_retriever"]["semantic"]
    assert aggregate["queries_with_duplicate_sources_in_top_5"] == 1
    assert aggregate["total_duplicate_chunks_in_top_5"] == 2
    assert aggregate["max_chunks_for_one_source_in_top_5"] == 3


def test_comparison_reports_retriever_only_relevant_sources() -> None:
    comparison = build_retriever_comparison(build_report())

    lexical_only = comparison["relevant_sources_not_in_semantic"]["lexical"]
    assert lexical_only["queries"] == [
        {"id": "q1", "source_ids": ["source-b"]}
    ]


def test_comparison_reports_ranking_differences_and_failures() -> None:
    comparison = build_retriever_comparison(build_report())

    differences = comparison["ranking_differences"]["lexical"]
    assert differences["queries_with_different_source_ranking"] == 2
    assert differences["source_ranking_query_ids"] == ["q1", "q2"]

    failures = [
        failure
        for failure in comparison["query_level_failures"]
        if failure["retriever"] == "semantic"
    ]
    assert [failure["id"] for failure in failures] == ["q1"]
    assert failures[0]["missing_source_ids"] == ["source-b"]

    assert "2-query" in comparison["limitations"][0]


def test_comparison_reports_chunk_vs_source_fusion_when_both_present() -> None:
    report = build_report()
    for query in report["queries"]:
        query["retrievers"]["hybrid"] = query["retrievers"]["semantic"]
        query["retrievers"]["hybrid_source"] = query["retrievers"]["lexical"]
    report["aggregate"]["retrievers"] = [
        "semantic",
        "lexical",
        "hybrid",
        "hybrid_source",
    ]

    comparison = build_retriever_comparison(report)

    fusion = comparison["duplicate_source_effects"][
        "chunk_level_vs_source_aware_fusion"
    ]
    assert fusion["compared_retrievers"] == ["hybrid", "hybrid_source"]
    assert fusion["queries_with_different_source_ranking"] == 2
    assert fusion["query_ids_with_different_source_ranking"] == ["q1", "q2"]


def test_comparison_reports_queries_where_lexical_harms_semantic() -> None:
    semantic = make_evaluation(
        results=[make_result(1, "runbooks/a.md")],
        source_units=[make_unit(1, "runbooks/a.md", mapped_source_id="source-a")],
        matched=["source-a"],
        missing=[],
        recall=1.0,
        mrr=1.0,
    )
    lexical = make_evaluation(
        results=[make_result(1, "runbooks/other.md")],
        source_units=[
            make_unit(1, "runbooks/other.md", status="unmapped", relevant=False)
        ],
        matched=[],
        missing=["source-a"],
        recall=0.0,
        mrr=0.0,
    )
    report = {
        "aggregate": {"retrievers": ["semantic", "lexical"]},
        "queries": [
            {
                "id": "q1",
                "category": "paraphrase_no_rare_term",
                "expected_source_ids": ["source-a"],
                "retrievers": {"semantic": semantic, "lexical": lexical},
            }
        ],
    }

    comparison = build_retriever_comparison(report)
    harm = comparison["harm_vs_semantic"]["lexical"]

    assert harm["source_mrr_harmed_query_count"] == 1
    assert harm["source_mrr_harmed_query_ids"] == ["q1"]
    assert harm["source_recall_harmed_query_ids"] == ["q1"]
    assert harm["harmed_queries"][0]["mrr_delta"] == -1.0


def test_comparison_reports_lexical_candidate_counts_and_fusion_delta() -> None:
    semantic = make_evaluation(
        results=[make_result(1, "runbooks/a.md")],
        source_units=[make_unit(1, "runbooks/a.md", mapped_source_id="source-a")],
        matched=["source-a"],
        missing=[],
        recall=1.0,
        mrr=1.0,
    )
    lexical = make_evaluation(
        results=[make_result(1, "runbooks/a.md")],
        source_units=[make_unit(1, "runbooks/a.md", mapped_source_id="source-a")],
        matched=["source-a"],
        missing=[],
        recall=1.0,
        mrr=1.0,
    )
    report = {
        "aggregate": {"retrievers": ["semantic", "lexical", "hybrid"]},
        "queries": [
            {
                "id": "q1",
                "category": "multi_source",
                "expected_source_ids": ["source-a"],
                "diagnostics": {
                    "keyword": {"provided": True, "result_count": 0},
                    "lexical": {
                        "provided": True,
                        "result_count": 4,
                        "chunks_contributing_to_hybrid_top_k": 1,
                    },
                },
                "retrievers": {
                    "semantic": semantic,
                    "lexical": lexical,
                    "hybrid": semantic,
                },
            }
        ],
    }

    comparison = build_retriever_comparison(report)
    counts = comparison["candidate_counts"][0]
    fusion = comparison["fusion_vs_semantic"]["hybrid"]

    assert counts["keyword_candidate_count"] == 0
    assert counts["lexical_candidate_count"] == 4
    assert counts["lexical_chunks_in_hybrid_top_5"] == 1
    assert fusion["mrr"]["equal"] == 1
    assert fusion["mrr"]["improved"] == 0
    assert fusion["mrr"]["harmed"] == 0


def test_comparison_reports_matched_but_non_expected_sources() -> None:
    evaluation = make_evaluation(
        results=[make_result(1, "runbooks/a.md"), make_result(2, "runbooks/b.md")],
        source_units=[
            make_unit(1, "runbooks/a.md", mapped_source_id="source-a"),
            make_unit(2, "runbooks/b.md", mapped_source_id="source-b"),
        ],
        matched=["source-a"],
        missing=[],
        recall=1.0,
        mrr=1.0,
    )
    report = {
        "aggregate": {"retrievers": ["semantic"]},
        "queries": [
            {
                "id": "q1",
                "category": "multi_source",
                "expected_source_ids": ["source-a"],
                "retrievers": {"semantic": evaluation},
            }
        ],
    }

    comparison = build_retriever_comparison(report)
    per_query = comparison["per_query"][0]["retrievers"]["semantic"]

    assert per_query["false_positive_sources"] == []
    assert [unit["mapped_source_id"] for unit in per_query["non_expected_sources"]] == [
        "source-b"
    ]
    assert comparison["false_positive_summary"]["semantic"][
        "total_non_expected_source_units_in_top_5"
    ] == 1


def test_comparison_reports_all_fusion_strategies_against_semantic_and_equal_weight() -> None:
    def evaluation(source_ids: list[str], *, recall: float, mrr: float) -> dict:
        units = [
            make_unit(
                index,
                source_key,
                mapped_source_id=f"source-{source_key}",
            )
            for index, source_key in enumerate(source_ids, start=1)
        ]
        return make_evaluation(
            results=[
                make_result(index, source_key)
                for index, source_key in enumerate(source_ids, start=1)
            ],
            source_units=units,
            matched=[f"source-{source_key}" for source_key in source_ids],
            missing=[],
            recall=recall,
            mrr=mrr,
        )

    semantic = evaluation(["a.md"], recall=0.5, mrr=1.0)
    hybrid = evaluation(["a.md"], recall=0.5, mrr=0.5)
    weighted = evaluation(["a.md", "b.md"], recall=1.0, mrr=1.0)
    report = {
        "aggregate": {
            "retrievers": [
                "semantic",
                "hybrid",
                "hybrid_weighted_semantic",
                "hybrid_weighted_lexical",
                "hybrid_lexical_primary",
            ]
        },
        "queries": [
            {
                "id": "q1",
                "category": "multi_source",
                "expected_source_ids": ["source-a.md", "source-b.md"],
                "retrievers": {
                    "semantic": semantic,
                    "hybrid": hybrid,
                    "hybrid_weighted_semantic": weighted,
                    "hybrid_weighted_lexical": weighted,
                    "hybrid_lexical_primary": weighted,
                },
            }
        ],
    }

    comparison = build_retriever_comparison(report)

    assert set(comparison["fusion_vs_semantic"]) == {
        "hybrid",
        "hybrid_weighted_semantic",
        "hybrid_weighted_lexical",
        "hybrid_lexical_primary",
    }
    assert set(comparison["fusion_vs_equal_weight"]) == {
        "hybrid_weighted_semantic",
        "hybrid_weighted_lexical",
        "hybrid_lexical_primary",
    }
    weighted_vs_equal = comparison["fusion_vs_equal_weight"][
        "hybrid_weighted_semantic"
    ]
    assert weighted_vs_equal["mrr"]["improved"] == 1
    assert weighted_vs_equal["recall_at_5"]["improved"] == 1
    assert weighted_vs_equal["improved_query_ids"] == ["q1"]
    assert weighted_vs_equal["harmed_query_ids"] == []


def test_comparison_reports_fusion_contributions_per_query() -> None:
    evaluation = make_evaluation(
        results=[make_result(1, "runbooks/a.md")],
        source_units=[make_unit(1, "runbooks/a.md", mapped_source_id="source-a")],
        matched=["source-a"],
        missing=[],
        recall=1.0,
        mrr=1.0,
    )
    contributions = {
        "hybrid": {
            "top_k_count": 1,
            "lexical_candidate_matches": 1,
            "semantic_candidate_matches": 1,
            "lexical_only_matches": 0,
            "semantic_only_matches": 0,
            "shared_matches": 1,
            "origin_counts": {"fused": 1},
        }
    }
    report = {
        "aggregate": {"retrievers": ["semantic", "hybrid"]},
        "queries": [
            {
                "id": "q1",
                "category": "multi_source",
                "expected_source_ids": ["source-a"],
                "diagnostics": {"fusion_contributions": contributions},
                "retrievers": {"semantic": evaluation, "hybrid": evaluation},
            }
        ],
    }

    comparison = build_retriever_comparison(report)

    assert comparison["candidate_counts"][0]["fusion_contributions"] == contributions
