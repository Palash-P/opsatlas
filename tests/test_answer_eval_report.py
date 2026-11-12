"""Regression tests for evaluation report attribution and agreement helpers."""

from __future__ import annotations

from scripts.answer_eval_report import (
    FAILURE_LABELS,
    expected_evidence_matches,
    failure_attribution,
    scorer_agreement,
)

METRICS = (
    "answer_relevance",
    "faithfulness",
    "citation_correctness",
    "citation_completeness",
    "abstention_correctness",
    "conflict_handling",
)


def make_prediction(**overrides) -> dict:
    prediction = {
        "record_id": "record-1",
        "status": "ok",
        "error": None,
        "expected_citations": [],
        "evidence": [],
        "dropped_evidence": [],
        "diagnostics": {"evidence_count": 1},
        "retrieval": {"result_count": 1},
    }
    prediction.update(overrides)
    return prediction


def test_expected_evidence_matches_ignores_optional_citations() -> None:
    prediction = make_prediction(
        expected_citations=[
            {
                "source_key": "runbooks/payment-api.md",
                "version_number": 2,
                "must_include": True,
            },
            {
                "source_key": "runbooks/payment-api.md",
                "version_number": 1,
                "must_include": False,
            },
        ],
        evidence=[
            {
                "source_key": "runbooks/payment-api.md",
                "version_number": 2,
            }
        ],
    )

    missing, budget_dropped = expected_evidence_matches(prediction)

    assert missing == []
    assert budget_dropped == []


def test_expected_evidence_matches_reports_required_missing() -> None:
    prediction = make_prediction(
        expected_citations=[
            {
                "source_key": "runbooks/payment-api.md",
                "version_number": 2,
                "must_include": True,
            }
        ],
        evidence=[],
        dropped_evidence=[
            {
                "source_key": "runbooks/payment-api.md",
                "version_number": 2,
                "reason": "context_budget",
            }
        ],
    )

    missing, budget_dropped = expected_evidence_matches(prediction)

    assert missing == ["runbooks/payment-api.md@v2"]
    assert budget_dropped == ["runbooks/payment-api.md@v2"]


def test_scorer_agreement_ignores_null_and_counts_exact_matches() -> None:
    predictions = [{"record_id": "r1"}, {"record_id": "r2"}]
    scores_a = {
        "r1": {
            "scores": {
                **dict.fromkeys(METRICS, 1),
                "conflict_handling": None,
            }
        },
        "r2": {
            "scores": {
                **dict.fromkeys(METRICS, 1),
                "conflict_handling": None,
            }
        },
    }
    scores_b = {
        "r1": {
            "scores": {
                **dict.fromkeys(METRICS, 1),
                "conflict_handling": None,
            }
        },
        "r2": {
            "scores": {
                **dict.fromkeys(METRICS, 0),
                "conflict_handling": None,
            }
        },
    }

    agreement = scorer_agreement(predictions, scores_a, scores_b)

    assert agreement["records"] == 12
    assert agreement["exact"] == 7
    assert agreement["exact_rate"] == 0.5833
    assert agreement["per_metric"]["answer_relevance"]["both_scored"] == 2
    assert agreement["per_metric"]["answer_relevance"]["mean_absolute_difference"] == 0.5
    assert agreement["per_metric"]["conflict_handling"]["both_scored"] == 0
    assert agreement["per_metric"]["conflict_handling"]["null_mismatch"] == 0


def test_provider_error_is_not_attributed_to_hallucination() -> None:
    prediction = make_prediction(
        status="error",
        error={"type": "GenerationProviderError", "message": "failed"},
        expected_citations=[
            {
                "source_key": "runbooks/kafka-consumer-lag.md",
                "version_number": 1,
                "must_include": True,
            }
        ],
        evidence=[
            {"source_key": "runbooks/kafka-consumer-lag.md", "version_number": 1}
        ],
    )
    final_scores = dict.fromkeys(METRICS, 0)

    attribution = failure_attribution(prediction, final_scores)

    assert attribution["primary_failure"] == "provider_failure"
    assert attribution["all_failures"] == ["provider_failure"]


def test_retrieval_failure_is_primary_when_required_evidence_is_missing() -> None:
    prediction = make_prediction(
        expected_citations=[
            {
                "source_key": "runbooks/payment-api.md",
                "version_number": 2,
                "must_include": True,
            }
        ],
        evidence=[],
    )
    final_scores = {
        **dict.fromkeys(METRICS, 1),
        "abstention_correctness": 0,
        "citation_correctness": 0,
    }

    attribution = failure_attribution(prediction, final_scores)

    assert attribution["primary_failure"] == "retrieval_failure"
    assert "abstention_failure" in attribution["all_failures"]


def test_faithful_omission_is_incomplete_generation_not_hallucination() -> None:
    final_scores = {
        **dict.fromkeys(METRICS, 1),
        "answer_relevance": 0.5,
    }

    attribution = failure_attribution(make_prediction(), final_scores)

    assert attribution["primary_failure"] == "incomplete_generation"
    assert attribution["all_failures"] == ["incomplete_generation"]
    assert "omitted required content" in attribution["notes"]


def test_unsupported_claim_remains_hallucination_primary() -> None:
    final_scores = {
        **dict.fromkeys(METRICS, 1),
        "answer_relevance": 0.5,
        "faithfulness": 0,
    }

    attribution = failure_attribution(make_prediction(), final_scores)

    assert attribution["primary_failure"] == "generation_hallucination"
    assert "incomplete_generation" not in attribution["all_failures"]


def test_invalid_model_output_is_incomplete_generation_not_hallucination() -> None:
    prediction = make_prediction(
        status="error",
        error={"type": "InvalidModelOutputError", "message": "bad JSON"},
    )

    attribution = failure_attribution(prediction, dict.fromkeys(METRICS, None))

    assert attribution["primary_failure"] == "incomplete_generation"
    assert attribution["all_failures"] == ["incomplete_generation"]


def test_fabricated_citation_error_is_citation_mismatch() -> None:
    prediction = make_prediction(
        status="error",
        error={"type": "CitationValidationError", "message": "unsupplied ID"},
    )

    attribution = failure_attribution(prediction, dict.fromkeys(METRICS, None))

    assert attribution["primary_failure"] == "citation_mismatch"
    assert attribution["all_failures"] == ["citation_mismatch"]


def test_failure_taxonomy_separates_incomplete_generation() -> None:
    assert "incomplete_generation" in FAILURE_LABELS
    assert "generation_hallucination" in FAILURE_LABELS
    assert FAILURE_LABELS.index("incomplete_generation") < FAILURE_LABELS.index(
        "generation_hallucination"
    )

