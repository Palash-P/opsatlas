"""Run deterministic retrieval evaluation against semantic and hybrid search.

The runner is read-only with respect to the corpus. It validates the dataset,
source map, optional quality profile, and corpus preflight before any retrieval
call, then writes a report that contains no document content.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import UUID

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.abstention_eval import build_abstention_readiness
from evals.retrieval_eval import (
    DEFAULT_FETCH_LIMIT,
    RANK_UNITS,
    EvaluationRunError,
    EvalValidationError,
    K,
    compute_corpus_fingerprint,
    load_dataset_profile,
    load_golden_dataset,
    load_source_map,
    preflight_source_map,
    run_evaluation,
    validate_dataset_profile,
    write_report,
)
from evals.retriever_comparison import build_retriever_comparison

DEFAULT_DATASET = REPO_ROOT / "evals" / "datasets" / "golden_v0.jsonl"
DEFAULT_SOURCE_MAP = REPO_ROOT / "evals" / "datasets" / "golden_v0.source_map.json"
DEFAULT_REPORT_DIR = REPO_ROOT / "evals" / "reports"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to the golden JSONL dataset.",
    )
    parser.add_argument(
        "--source-map",
        type=Path,
        default=DEFAULT_SOURCE_MAP,
        help="Path to the explicit expected-source to database-source mapping.",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        help="Optional dataset quality profile validated before retrieval.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Optional previous report used for a fingerprint-guarded comparison.",
    )
    parser.add_argument(
        "--tenant-id",
        help="Tenant UUID used for the live retrieval run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Path for the machine-readable JSON report. Defaults to "
            "evals/reports/<dataset-id>_semantic_vs_hybrid.json."
        ),
    )
    parser.add_argument(
        "--fetch-limit",
        type=int,
        default=DEFAULT_FETCH_LIMIT,
        help=(
            "Maximum chunks requested from each retriever. Chunk metrics still "
            f"use the first {K} chunks; source metrics collapse duplicates first. "
            f"Allowed range: 1-{DEFAULT_FETCH_LIMIT}."
        ),
    )
    parser.add_argument(
        "--include-lexical-comparison",
        action="store_true",
        help=(
            "Also score the AND keyword baseline, the OR+IDF lexical strategy, "
            "and source-aware hybrid fusion, then attach a cross-retriever "
            "comparison section. The default report remains semantic vs hybrid."
        ),
    )
    parser.add_argument(
        "--include-fusion-experiments",
        action="store_true",
        help=(
            "Score the six fusion strategies: semantic, OR+IDF lexical, "
            "equal-weight chunk RRF, semantic-heavy weighted RRF, lexical-heavy "
            "weighted RRF, and lexical-primary with semantic fallback. Uses "
            "OR+IDF lexical candidates for every hybrid strategy and writes "
            "evals/reports/<dataset-id>_fusion_comparison.json by default."
        ),
    )
    parser.add_argument(
        "--include-abstention-analysis",
        action="store_true",
        help=(
            "Run unanswerable records separately and attach a read-only "
            "abstention-readiness section. They remain excluded from retrieval "
            "quality metrics."
        ),
    )
    parser.add_argument(
        "--hybrid-keyword-strategy",
        choices=("and", "or_idf"),
        default="and",
        help=(
            "Keyword candidate strategy used inside hybrid retrieval. The "
            "default 'and' preserves existing behaviour; 'or_idf' evaluates "
            "chunk/source RRF with the OR+IDF lexical strategy."
        ),
    )
    parser.add_argument(
        "--rank-unit",
        choices=RANK_UNITS,
        default="chunk",
        help="Rank unit used for the console summary. The report always contains both.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate dataset, profile, and source map without calling services.",
    )
    return parser


def _format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _load_baseline(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise EvalValidationError(
            f"Cannot read baseline report '{path}': "
            f"{exc.strerror or type(exc).__name__}"
        ) from None
    except json.JSONDecodeError as exc:
        raise EvalValidationError(
            f"Baseline report '{path}' is not valid JSON: {exc.msg}"
        ) from None
    if not isinstance(payload, dict):
        raise EvalValidationError(f"Baseline report '{path}' must be a JSON object")
    return payload


def read_corpus_state(tenant_id: UUID) -> tuple[list[dict], dict]:
    """Read the current corpus and its inventory without modifying anything."""
    from sqlalchemy import text

    from app.db.connections import get_engine

    with get_engine().connect() as connection:
        inventory = [
            dict(row)
            for row in connection.execute(
                text(
                    """
                    SELECT
                        d.source_key,
                        dv.version_number,
                        dv.is_current,
                        d.status,
                        (
                            SELECT count(*) FROM chunks c
                            WHERE c.document_version_id = dv.id
                        ) AS chunk_count
                    FROM documents d
                    JOIN document_versions dv ON dv.document_id = d.id
                    WHERE d.tenant_id = :tenant_id
                    ORDER BY d.source_key, dv.version_number
                    """
                ),
                {"tenant_id": tenant_id},
            ).mappings()
        ]

        fingerprint_rows = [
            dict(row)
            for row in connection.execute(
                text(
                    """
                    SELECT
                        d.source_key,
                        dv.version_number,
                        c.chunk_index,
                        c.content_hash
                    FROM chunks c
                    JOIN documents d ON d.id = c.document_id
                    JOIN document_versions dv ON dv.id = c.document_version_id
                    WHERE c.tenant_id = :tenant_id
                      AND d.status = 'active'
                      AND dv.is_current = TRUE
                    """
                ),
                {"tenant_id": tenant_id},
            ).mappings()
        ]

    return inventory, compute_corpus_fingerprint(fingerprint_rows)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not 1 <= args.fetch_limit <= DEFAULT_FETCH_LIMIT:
        print(
            f"--fetch-limit must be between 1 and {DEFAULT_FETCH_LIMIT}.",
            file=sys.stderr,
        )
        return 2

    try:
        dataset = load_golden_dataset(args.dataset)
        source_map = load_source_map(args.source_map, dataset)
        if args.profile is not None:
            validate_dataset_profile(
                dataset,
                load_dataset_profile(args.profile),
            )
        baseline = _load_baseline(args.baseline) if args.baseline else None
    except EvalValidationError as exc:
        print(f"Evaluation validation failed: {exc}", file=sys.stderr)
        return 2

    if args.validate_only:
        profile_note = (
            f" and profile '{_display_path(args.profile)}'"
            if args.profile is not None
            else ""
        )
        print(
            f"Validated dataset '{dataset.dataset_id}' with {len(dataset.records)} "
            f"records and {len(source_map.sources)} explicit source mappings"
            f"{profile_note}. No retrieval was executed."
        )
        return 0

    if not args.tenant_id:
        print(
            "Live evaluation requires --tenant-id UUID. Use --validate-only to "
            "validate files without calling PostgreSQL or the embedding service.",
            file=sys.stderr,
        )
        return 2

    try:
        tenant_id = UUID(args.tenant_id)
    except ValueError:
        print("--tenant-id must be a valid UUID.", file=sys.stderr)
        return 2

    from app.services.embedding_client import EXPECTED_DIMENSIONS, EXPECTED_MODEL
    from app.services.retrieval import (
        FUSION_STRATEGIES,
        FUSION_WEIGHT_PROFILES,
        HYBRID_CANDIDATE_MULTIPLIER,
        IDENTIFIER_WEIGHT,
        KEYWORD_STRATEGIES,
        MAX_RESULTS,
        RRF_K,
        hybrid_search,
        keyword_search,
        lexical_search,
        semantic_search,
    )

    try:
        inventory, corpus_metadata = read_corpus_state(tenant_id)
    except Exception as exc:  # noqa: BLE001 - read-only preflight failure
        print(
            "Could not read the corpus for preflight "
            f"({type(exc).__name__}). Check the database connection and tenant.",
            file=sys.stderr,
        )
        return 2

    preflight_issues = preflight_source_map(source_map, inventory)
    if preflight_issues:
        print(
            "Evaluation preflight failed. No retrieval was executed and no report "
            "was written.",
            file=sys.stderr,
        )
        for issue in preflight_issues:
            print(f"  - {issue}", file=sys.stderr)
        print(
            "Seed or repair the corpus, then rerun. Pinned versions must match the "
            "current version before a current-version query can be evaluated.",
            file=sys.stderr,
        )
        return 2

    if args.include_fusion_experiments:
        comparison_retriever_names = [
            "semantic",
            "lexical",
            "hybrid",
            "hybrid_weighted_semantic",
            "hybrid_weighted_lexical",
            "hybrid_lexical_primary",
        ]
    elif args.include_lexical_comparison:
        comparison_retriever_names = [
            "semantic",
            "keyword",
            "lexical",
            "hybrid",
            "hybrid_source",
        ]
    else:
        comparison_retriever_names = None

    retrieval_config = {
        "k": K,
        "fetch_limit": args.fetch_limit,
        "rank_units": list(RANK_UNITS),
        "semantic": {
            "function": "app.services.retrieval.semantic_search",
            "score_field": "similarity",
            "embedding_model": EXPECTED_MODEL,
            "embedding_dimensions": EXPECTED_DIMENSIONS,
            "filters": [
                "tenant_id",
                "active_document",
                "current_version",
                "embedding_not_null",
                "embedding_model",
            ],
        },
        "keyword": {
            "function": "app.services.retrieval.keyword_search",
            "score_field": "keyword_score",
            "keyword_language": "english",
            "filters": [
                "tenant_id",
                "active_document",
                "current_version",
            ],
            "diagnostic_only": True,
        },
        "lexical": {
            "function": "app.services.retrieval.lexical_search",
            "score_field": "lexical_score",
            "strategy": "or_idf",
            "keyword_language": "english",
            "ranking": [
                "idf_weighted_distinct_term_coverage",
                "matched_term_count",
                "ts_rank_cd",
                "chunk_id",
            ],
            "idf": "distinct source-document frequency",
            "identifier_weight": IDENTIFIER_WEIGHT,
            "identifier_kinds": [
                "identifier",
                "http_status",
                "api_version",
            ],
            "filters": [
                "tenant_id",
                "active_document",
                "current_version",
            ],
        },
        "hybrid": {
            "function": "app.services.retrieval.hybrid_search",
            "score_field": "hybrid_score",
            "rrf_k": RRF_K,
            "candidate_multiplier": HYBRID_CANDIDATE_MULTIPLIER,
            "candidate_limit": min(
                MAX_RESULTS,
                args.fetch_limit * HYBRID_CANDIDATE_MULTIPLIER,
            ),
            "candidate_cap": MAX_RESULTS,
            "keyword_language": "english",
            "keyword_strategy": args.hybrid_keyword_strategy,
            "fusion_strategy": "chunk",
            "filters": [
                "tenant_id",
                "active_document",
                "current_version",
                "semantic_embedding_model_when_applicable",
            ],
        },
        "hybrid_source": {
            "function": "app.services.retrieval.hybrid_search",
            "score_field": "hybrid_score",
            "rrf_k": RRF_K,
            "candidate_multiplier": HYBRID_CANDIDATE_MULTIPLIER,
            "candidate_limit": min(
                MAX_RESULTS,
                args.fetch_limit * HYBRID_CANDIDATE_MULTIPLIER,
            ),
            "candidate_cap": MAX_RESULTS,
            "keyword_language": "english",
            "keyword_strategy": args.hybrid_keyword_strategy,
            "fusion_strategy": "source",
            "filters": [
                "tenant_id",
                "active_document",
                "current_version",
                "semantic_embedding_model_when_applicable",
            ],
        },
        "hybrid_weighted_semantic": {
            "function": "app.services.retrieval.hybrid_search",
            "score_field": "hybrid_score",
            "rrf_k": RRF_K,
            "candidate_multiplier": HYBRID_CANDIDATE_MULTIPLIER,
            "candidate_limit": min(
                MAX_RESULTS,
                args.fetch_limit * HYBRID_CANDIDATE_MULTIPLIER,
            ),
            "candidate_cap": MAX_RESULTS,
            "keyword_language": "english",
            "keyword_strategy": "or_idf",
            "fusion_strategy": "weighted",
            "fusion_weight_profile": "semantic_heavy",
            "semantic_weight": FUSION_WEIGHT_PROFILES["semantic_heavy"][
                "semantic"
            ],
            "lexical_weight": FUSION_WEIGHT_PROFILES["semantic_heavy"][
                "lexical"
            ],
            "filters": [
                "tenant_id",
                "active_document",
                "current_version",
                "semantic_embedding_model_when_applicable",
            ],
        },
        "hybrid_weighted_lexical": {
            "function": "app.services.retrieval.hybrid_search",
            "score_field": "hybrid_score",
            "rrf_k": RRF_K,
            "candidate_multiplier": HYBRID_CANDIDATE_MULTIPLIER,
            "candidate_limit": min(
                MAX_RESULTS,
                args.fetch_limit * HYBRID_CANDIDATE_MULTIPLIER,
            ),
            "candidate_cap": MAX_RESULTS,
            "keyword_language": "english",
            "keyword_strategy": "or_idf",
            "fusion_strategy": "weighted",
            "fusion_weight_profile": "lexical_heavy",
            "semantic_weight": FUSION_WEIGHT_PROFILES["lexical_heavy"][
                "semantic"
            ],
            "lexical_weight": FUSION_WEIGHT_PROFILES["lexical_heavy"][
                "lexical"
            ],
            "filters": [
                "tenant_id",
                "active_document",
                "current_version",
                "semantic_embedding_model_when_applicable",
            ],
        },
        "hybrid_lexical_primary": {
            "function": "app.services.retrieval.hybrid_search",
            "score_field": "hybrid_score",
            "rrf_k": RRF_K,
            "candidate_multiplier": HYBRID_CANDIDATE_MULTIPLIER,
            "candidate_limit": min(
                MAX_RESULTS,
                args.fetch_limit * HYBRID_CANDIDATE_MULTIPLIER,
            ),
            "candidate_cap": MAX_RESULTS,
            "keyword_language": "english",
            "keyword_strategy": "or_idf",
            "fusion_strategy": "lexical_primary",
            "fallback": "unique semantic results fill remaining top-k slots",
            "filters": [
                "tenant_id",
                "active_document",
                "current_version",
                "semantic_embedding_model_when_applicable",
            ],
        },
        "available_keyword_strategies": list(KEYWORD_STRATEGIES),
        "available_fusion_strategies": list(FUSION_STRATEGIES),
        "fusion_weight_profiles": {
            name: dict(weights)
            for name, weights in FUSION_WEIGHT_PROFILES.items()
        },
        "comparison_retrievers": comparison_retriever_names,
    }

    include_comparison = (
        args.include_lexical_comparison or args.include_fusion_experiments
    )
    if args.include_fusion_experiments:
        default_report_suffix = "_fusion_comparison.json"
    elif args.include_lexical_comparison:
        default_report_suffix = "_lexical_comparison.json"
    else:
        default_report_suffix = "_semantic_vs_hybrid.json"
    output_path = args.output or (
        DEFAULT_REPORT_DIR / f"{dataset.dataset_id}{default_report_suffix}"
    )

    if args.include_fusion_experiments:
        semantic_heavy = FUSION_WEIGHT_PROFILES["semantic_heavy"]
        lexical_heavy = FUSION_WEIGHT_PROFILES["lexical_heavy"]
        retriever_functions = {
            "semantic": semantic_search,
            "lexical": lexical_search,
            "hybrid": lambda **kwargs: hybrid_search(
                **kwargs,
                keyword_strategy="or_idf",
                fusion_strategy="chunk",
            ),
            "hybrid_weighted_semantic": lambda **kwargs: hybrid_search(
                **kwargs,
                keyword_strategy="or_idf",
                fusion_strategy="weighted",
                semantic_weight=semantic_heavy["semantic"],
                lexical_weight=semantic_heavy["lexical"],
            ),
            "hybrid_weighted_lexical": lambda **kwargs: hybrid_search(
                **kwargs,
                keyword_strategy="or_idf",
                fusion_strategy="weighted",
                semantic_weight=lexical_heavy["semantic"],
                lexical_weight=lexical_heavy["lexical"],
            ),
            "hybrid_lexical_primary": lambda **kwargs: hybrid_search(
                **kwargs,
                keyword_strategy="or_idf",
                fusion_strategy="lexical_primary",
            ),
        }
    elif args.include_lexical_comparison:
        retriever_functions = {
            "semantic": semantic_search,
            "keyword": keyword_search,
            "lexical": lexical_search,
            "hybrid": lambda **kwargs: hybrid_search(
                **kwargs,
                keyword_strategy=args.hybrid_keyword_strategy,
            ),
            "hybrid_source": lambda **kwargs: hybrid_search(
                **kwargs,
                keyword_strategy=args.hybrid_keyword_strategy,
                fusion_strategy="source",
            ),
        }
    else:
        retriever_functions = {
            "semantic": semantic_search,
            "hybrid": lambda **kwargs: hybrid_search(
                **kwargs,
                keyword_strategy=args.hybrid_keyword_strategy,
            ),
            "keyword": keyword_search,
        }

    try:
        report = run_evaluation(
            dataset=dataset,
            source_map=source_map,
            tenant_id=tenant_id,
            retrievers=retriever_functions,
            retrieval_config=retrieval_config,
            fetch_limit=args.fetch_limit,
            corpus_metadata=corpus_metadata,
            baseline_report=baseline,
            repo_root=REPO_ROOT,
        )
        if include_comparison:
            report["retriever_comparison"] = build_retriever_comparison(report)
        if args.include_abstention_analysis:
            report["abstention_readiness"] = build_abstention_readiness(
                report=report,
                retrievers=retriever_functions,
                tenant_id=tenant_id,
                fetch_limit=args.fetch_limit,
            )
        write_report(report, output_path)
    except EvaluationRunError as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(
            f"Could not write evaluation report to '{_display_path(output_path)}': "
            f"{exc.strerror or type(exc).__name__}",
            file=sys.stderr,
        )
        return 2

    aggregate = report["aggregate"]
    rank_unit = args.rank_unit
    semantic = aggregate["by_rank_unit"][rank_unit]["semantic"]
    hybrid = aggregate["by_rank_unit"][rank_unit]["hybrid"]

    print(f"Report written: {_display_path(output_path)}")
    print(f"Corpus fingerprint: {corpus_metadata['sha256']}")
    print(f"Scored answerable queries: {aggregate['scored_query_count']}")
    print(f"Rank unit: {rank_unit}")
    if args.include_fusion_experiments:
        print("Hybrid keyword strategy: or_idf (fusion experiments)")
    else:
        print(f"Hybrid keyword strategy: {args.hybrid_keyword_strategy}")
    print(
        "Semantic  Recall@5="
        f"{_format_metric(semantic['recall_at_5'])} "
        f"MRR={_format_metric(semantic['mrr'])} "
        f"AllExpected@5={_format_metric(semantic['all_expected_at_5'])}"
    )
    print(
        "Hybrid    Recall@5="
        f"{_format_metric(hybrid['recall_at_5'])} "
        f"MRR={_format_metric(hybrid['mrr'])} "
        f"AllExpected@5={_format_metric(hybrid['all_expected_at_5'])}"
    )

    if include_comparison:
        print("Retriever comparison (source-level metrics):")
        for retriever_name in report["aggregate"]["retrievers"]:
            metrics = report["aggregate"]["by_retriever"][retriever_name][
                "source"
            ]
            print(
                f"  {retriever_name:<26} Recall@5="
                f"{_format_metric(metrics['recall_at_5'])} "
                f"MRR={_format_metric(metrics['mrr'])} "
                f"nDCG@5={_format_metric(metrics.get('ndcg_at_5'))} "
                f"AllExpected@5={_format_metric(metrics['all_expected_at_5'])}"
            )
        comparison = report["retriever_comparison"]
        print(
            "Duplicate sources in top-5 chunks: "
            + ", ".join(
                f"{name}={effect['queries_with_duplicate_sources_in_top_5']}"
                f"/{report['aggregate']['scored_query_count']}"
                for name, effect in comparison["duplicate_source_effects"][
                    "by_retriever"
                ].items()
            )
        )
        chunk_vs_source = comparison["duplicate_source_effects"][
            "chunk_level_vs_source_aware_fusion"
        ]
        if chunk_vs_source:
            print(
                "Chunk-level vs source-aware fusion ranking differences: "
                f"{chunk_vs_source['queries_with_different_source_ranking']}"
                f"/{report['aggregate']['scored_query_count']}"
            )
        print(
            f"Query-level failures (source Recall@5 < 1): "
            f"{len(comparison['query_level_failures'])} retriever-query pairs"
        )
        fusion_vs_equal = comparison.get("fusion_vs_equal_weight", {})
        for name, summary in fusion_vs_equal.items():
            print(
                f"  {name} vs equal-weight hybrid: "
                f"MRR improved={summary['mrr']['improved']} "
                f"equal={summary['mrr']['equal']} "
                f"harmed={summary['mrr']['harmed']} "
                f"mean_delta={_format_metric(summary['mrr']['mean_delta'])}; "
                f"Recall@5 improved={summary['recall_at_5']['improved']} "
                f"harmed={summary['recall_at_5']['harmed']}"
            )
        candidate_counts = comparison["candidate_counts"]
        lexical_supplied = sum(
            1
            for item in candidate_counts
            if (item["lexical_candidate_count"] or 0) > 0
        )
        print(
            "Lexical candidates supplied: "
            f"{lexical_supplied}/{len(candidate_counts)} queries"
        )
    diagnostics = aggregate["diagnostics"]
    print(
        "Ranking diagnostics: "
        f"identical_chunk={diagnostics['queries_with_identical_chunk_ranking']}"
        f"/{diagnostics['scored_query_count']} "
        f"identical_source={diagnostics['queries_with_identical_source_ranking']}"
        f"/{diagnostics['scored_query_count']} "
        f"rank_difference={diagnostics['queries_with_rank_difference']} "
        f"keyword_contribution={diagnostics['queries_with_keyword_contribution']} "
        f"lexical_contribution={diagnostics['queries_with_lexical_contribution']}"
    )
    print(
        f"Unanswerable records excluded: {len(report['unanswerable'])} "
        "(not an abstention evaluation)"
    )
    if args.include_abstention_analysis and "abstention_readiness" in report:
        abstention = report["abstention_readiness"]
        total = abstention["scored_for_abstention"]
        print(
            "Abstention readiness (separate, not part of recall): "
            f"{total} unanswerable record(s)"
        )
        for retriever_name, summary in abstention["by_retriever"].items():
            threshold = summary["best_threshold"]
            separation = summary["score_separation"]
            print(
                f"  {retriever_name:<26} "
                f"any_result="
                f"{summary['unanswerable_queries_with_any_result']}/{total} "
                f"perfect_separation={separation['perfect_separation']} "
                f"threshold={_format_metric(threshold['threshold'])} "
                f"answerable_retention="
                f"{_format_metric(threshold['answerable_retention'])} "
                f"unanswerable_retrieval_rate="
                f"{_format_metric(threshold['unanswerable_retrieval_rate'])} "
                f"youden_j={_format_metric(threshold['youden_j'])}"
            )
    hybrid_ndcg = aggregate["by_retriever"]["hybrid"]["source"].get("ndcg_at_5")
    if hybrid_ndcg is None:
        print("nDCG@5 (graded, source): not applicable (no graded relevance labels)")
    else:
        print(f"nDCG@5 (graded, source): {hybrid_ndcg:.4f}")
    if aggregate["saturation_warning"]:
        print(f"WARNING: {aggregate['saturation_warning']}")
    if "baseline_comparison" in report:
        comparison = report["baseline_comparison"]
        print(
            "Baseline comparison: "
            f"comparable={comparison['comparable']} "
            f"informational_only={comparison['informational_only']}"
        )
        for reason in comparison["reasons"]:
            print(f"  - {reason}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
