"""Tests for the read-only abstention-readiness diagnostics."""

from __future__ import annotations

from uuid import uuid4

from evals.abstention_eval import CORPUS_EVIDENCE_BY_CATEGORY, build_abstention_readiness


def _report(answerable_scores: list[float]) -> dict:
    return {
        "aggregate": {"retrievers": ["semantic"]},
        "queries": [
            {
                "id": f"q{index}",
                "retrievers": {
                    "semantic": {
                        "results": [
                            {"source_key": "runbooks/a.md", "similarity": score}
                        ]
                    }
                },
            }
            for index, score in enumerate(answerable_scores)
        ],
        "unanswerable": [
            {
                "id": "u1",
                "question": "What revenue was lost?",
                "category": "insufficient_evidence",
            },
            {
                "id": "u2",
                "question": "Provide the API key.",
                "category": "sensitive_unanswerable",
            },
        ],
    }


def test_abstention_diagnostic_runs_unanswerable_records_separately() -> None:
    calls: list[str] = []

    def retriever(*, tenant_id, query, limit):
        calls.append(query)
        return [
            {"source_key": "postmortems/db-pool.md", "similarity": 0.1},
            {"source_key": "postmortems/db-pool.md", "similarity": 0.05},
        ]

    readiness = build_abstention_readiness(
        report=_report([0.8, 0.9]),
        retrievers={"semantic": retriever},
        tenant_id=uuid4(),
    )

    assert readiness["mixed_into_retrieval_metrics"] is False
    assert readiness["scored_for_abstention"] == 2
    assert calls == [
        "What revenue was lost?",
        "Provide the API key.",
    ]
    assert readiness["per_query"][0]["corpus_evidence"] == "related_context_only"
    assert readiness["per_query"][1]["corpus_evidence"] == "none"
    assert readiness["per_query"][0]["retrievers"]["semantic"][
        "unique_source_count"
    ] == 1
    assert readiness["per_query"][0]["retrievers"]["semantic"][
        "returned_evidence"
    ] is True
    assert set(CORPUS_EVIDENCE_BY_CATEGORY) == {
        "insufficient_evidence",
        "no_relevant_source",
        "sensitive_unanswerable",
    }


def test_abstention_reports_separated_scores_and_threshold() -> None:
    returned = {"value": 0.2}

    def retriever(*, tenant_id, query, limit):
        return [{"source_key": "runbooks/a.md", "similarity": returned["value"]}]

    readiness = build_abstention_readiness(
        report=_report([0.8, 0.9]),
        retrievers={"semantic": retriever},
        tenant_id=uuid4(),
    )

    summary = readiness["by_retriever"]["semantic"]
    assert summary["answerable_query_count"] == 2
    assert summary["unanswerable_query_count"] == 2
    assert summary["unanswerable_queries_with_any_result"] == 2
    assert summary["score_separation"]["perfect_separation"] is True
    assert summary["best_threshold"] == {
        "threshold": 0.8,
        "answerable_retention": 1.0,
        "unanswerable_retrieval_rate": 0.0,
        "youden_j": 1.0,
    }

    returned["value"] = 0.85
    readiness = build_abstention_readiness(
        report=_report([0.8, 0.9]),
        retrievers={"semantic": retriever},
        tenant_id=uuid4(),
    )
    summary = readiness["by_retriever"]["semantic"]
    assert summary["score_separation"]["perfect_separation"] is False
    assert summary["best_threshold"] == {
        "threshold": 0.9,
        "answerable_retention": 0.5,
        "unanswerable_retrieval_rate": 0.0,
        "youden_j": 0.5,
    }


def test_abstention_reads_serialized_report_scores() -> None:
    report = {
        "aggregate": {"retrievers": ["semantic"]},
        "queries": [
            {
                "id": "q1",
                "retrievers": {
                    "semantic": {
                        "results": [
                            {"source_key": "runbooks/a.md", "score": 0.75}
                        ]
                    }
                },
            }
        ],
        "unanswerable": [],
    }

    def retriever(*, tenant_id, query, limit):
        return []

    readiness = build_abstention_readiness(
        report=report,
        retrievers={"semantic": retriever},
        tenant_id=uuid4(),
    )

    summary = readiness["by_retriever"]["semantic"]
    assert summary["answerable_query_count"] == 1
    assert summary["answerable_top_score"] == {
        "min": 0.75,
        "median": 0.75,
        "max": 0.75,
    }


def test_abstention_with_no_unanswerable_records_is_empty() -> None:
    def retriever(*, tenant_id, query, limit):
        return []

    report = _report([0.8])
    report["unanswerable"] = []

    readiness = build_abstention_readiness(
        report=report,
        retrievers={"semantic": retriever},
        tenant_id=uuid4(),
    )

    summary = readiness["by_retriever"]["semantic"]
    assert readiness["scored_for_abstention"] == 0
    assert summary["unanswerable_query_count"] == 0
    assert summary["unanswerable_queries_with_any_result"] == 0
    assert summary["best_threshold"]["threshold"] is None
    assert readiness["limitations"]
