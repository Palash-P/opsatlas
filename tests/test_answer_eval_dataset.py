"""Validation tests for the design-only answer evaluation dataset."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from evals.answer_eval import (
    DATASET_PATH,
    GOLDEN_SOURCE_MAP_PATH,
    METRIC_DEFINITIONS,
    REPORT_SCHEMA_PATH,
    AnswerEvalValidationError,
    build_report_template,
    load_answer_eval_records,
    load_profile,
    validate_answer_eval_records,
    validate_report_template,
)

GOLDEN_V3_SHA256 = "1fe9b87582720e272abe36f45063e8d34ad4266c9e6694870f51b57100d80497"


@pytest.fixture(scope="module")
def records() -> list[dict]:
    return load_answer_eval_records()


def test_answer_eval_dataset_is_internally_consistent(records) -> None:
    profile = load_profile()

    validate_answer_eval_records(records, profile=profile)

    assert len(records) == profile["record_count"] == 26
    assert sum(record["answerable"] for record in records) == 20
    assert sum(not record["answerable"] for record in records) == 6


def test_answer_eval_covers_required_categories(records) -> None:
    counts: dict[str, int] = {}
    for record in records:
        counts[record["category"]] = counts.get(record["category"], 0) + 1

    assert counts == {
        "single_source_factual": 9,
        "multi_source": 3,
        "multi_chunk": 2,
        "version_conflict": 4,
        "insufficient_evidence": 4,
        "sensitive_unanswerable": 2,
        "distractor": 2,
    }


def test_answer_eval_includes_version_and_conflict_coverage(records) -> None:
    conflicted = [record for record in records if record["conflict_expected"]]
    versioned = [
        citation
        for record in records
        for citation in record["expected_citations"]
        if citation.get("version_number") is not None
    ]

    assert len(conflicted) == 2
    assert versioned
    assert any(record["category"] == "version_conflict" for record in records)


def test_answer_eval_dataset_does_not_touch_golden_v3(records) -> None:
    digest = hashlib.sha256(Path(DATASET_PATH.parent, "golden_v3.jsonl").read_bytes())
    assert digest.hexdigest() == GOLDEN_V3_SHA256

    source_map = json.loads(GOLDEN_SOURCE_MAP_PATH.read_text(encoding="utf-8"))
    assert source_map["dataset_id"] == "golden_v3"


def test_report_template_is_unfilled_and_structurally_valid() -> None:
    report = build_report_template()

    validate_report_template(report)

    assert report["status"] == "not_run"
    assert set(report["metric_definitions"]) == set(METRIC_DEFINITIONS)
    assert report["records"] == []
    for metric in METRIC_DEFINITIONS:
        assert report["aggregate"]["overall"][metric] is None


def test_report_schema_declares_all_required_metrics() -> None:
    schema = json.loads(REPORT_SCHEMA_PATH.read_text(encoding="utf-8"))

    assert schema["$schema"].endswith("2020-12/schema")
    required = set(schema["properties"]["metric_definitions"]["required"])
    assert required == set(METRIC_DEFINITIONS)
    score_properties = schema["properties"]["records"]["items"]["properties"]["scores"]
    assert set(score_properties["required"]) == set(METRIC_DEFINITIONS)


def test_report_schema_declares_version_and_failure_extensions() -> None:
    schema = json.loads(REPORT_SCHEMA_PATH.read_text(encoding="utf-8"))

    assert "failure_taxonomy" in schema["properties"]
    prediction = schema["properties"]["records"]["items"]["properties"]["prediction"]
    assert "version_analysis" in prediction["properties"]
    assert "version_analysis" in schema["$defs"]
    assert schema["$defs"]["version_analysis"]["required"]
    assert "provider_failure_rate" in schema["properties"]["aggregate"]["properties"]


def test_dataset_rejects_duplicate_ids(records) -> None:
    mutated = copy.deepcopy(records)
    mutated[1]["id"] = mutated[0]["id"]

    with pytest.raises(AnswerEvalValidationError, match="duplicate record id"):
        validate_answer_eval_records(mutated)


def test_dataset_rejects_unknown_source_key(records) -> None:
    mutated = copy.deepcopy(records)
    mutated[0]["expected_citations"][0]["source_key"] = "runbooks/does-not-exist.md"

    with pytest.raises(AnswerEvalValidationError, match="unknown source_key"):
        validate_answer_eval_records(mutated)


def test_dataset_rejects_nonexistent_version(records) -> None:
    mutated = copy.deepcopy(records)
    mutated[0]["expected_citations"][0]["version_number"] = 99

    with pytest.raises(AnswerEvalValidationError, match="does not exist"):
        validate_answer_eval_records(mutated)


def test_dataset_rejects_answerable_record_without_citations(records) -> None:
    mutated = copy.deepcopy(records)
    mutated[0]["expected_citations"] = []
    mutated[0]["min_required_citations"] = 1

    with pytest.raises(AnswerEvalValidationError, match="needs expected citations"):
        validate_answer_eval_records(mutated)


def test_dataset_rejects_unanswerable_record_with_citations(records) -> None:
    mutated = copy.deepcopy(records)
    unanswerable = next(record for record in mutated if not record["answerable"])
    unanswerable["expected_citations"] = [
        {
            "source_key": "runbooks/payment-errors.md",
            "version_number": 1,
            "supporting_text": "Check the payment service health endpoint.",
        }
    ]

    with pytest.raises(AnswerEvalValidationError, match="must not expect facts"):
        validate_answer_eval_records(mutated)


def test_dataset_rejects_answerable_record_with_abstention_reason(records) -> None:
    mutated = copy.deepcopy(records)
    mutated[0]["abstention_reason"] = "insufficient_evidence"

    with pytest.raises(AnswerEvalValidationError, match="abstention_reason"):
        validate_answer_eval_records(mutated)


def test_report_template_rejects_premature_metric_values() -> None:
    report = build_report_template()
    report["aggregate"]["overall"]["faithfulness"] = 0.9

    with pytest.raises(AnswerEvalValidationError, match="must not contain metric values"):
        validate_report_template(report)
