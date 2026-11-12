"""Validation and report scaffolding for the design-only answer evaluation set.

This module does not call a model and does not compute answer-quality metrics. It
validates that ``answer_eval_v1`` is internally consistent with the synthetic
corpus manifest and produces an explicitly unfilled report template. Answer
metrics must not be reported as validated until an actual run has been scored.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

EVALS_DIR = Path(__file__).resolve().parent
DATASET_PATH = EVALS_DIR / "datasets" / "answer_eval_v1.jsonl"
PROFILE_PATH = EVALS_DIR / "datasets" / "answer_eval_v1.profile.json"
REPORT_SCHEMA_PATH = EVALS_DIR / "answer_eval_v1.report_schema.json"
GOLDEN_SOURCE_MAP_PATH = EVALS_DIR / "datasets" / "golden_v3.source_map.json"

# Provider transport/envelope failures. Invalid model output is deliberately not
# in this set: it is a generation-contract failure, tracked separately so the
# provider failure rate is not inflated by schema violations.
PROVIDER_FAILURE_ERROR_TYPES = (
    "GenerationTimeoutError",
    "GenerationRateLimitError",
    "GenerationProviderError",
    "GenerationInvalidResponseError",
)

REQUIRED_FIELDS = (
    "id",
    "question",
    "category",
    "answerable",
    "expected_behavior",
    "expected_facts",
    "expected_citations",
    "min_required_citations",
    "conflict_expected",
    "abstention_reason",
    "notes",
)

EXPECTED_BEHAVIORS = ("answer", "abstain", "identify_conflict")

METRIC_DEFINITIONS = {
    "answer_relevance": (
        "0-1 rubric: does the answer address the question and include the expected "
        "facts without unrelated material? Unanswerable questions score 1 when the "
        "system abstains appropriately."
    ),
    "faithfulness": (
        "0-1 rubric: is every factual claim in the answer supported by a supplied "
        "evidence chunk, with no outside knowledge or invented procedure?"
    ),
    "citation_correctness": (
        "0-1 rubric: do all returned citations reference chunks actually supplied to "
        "the model, with the correct source key, version number, and chunk ID, and "
        "does each excerpt support its claim? Fabricated citations score 0."
    ),
    "citation_completeness": (
        "0-1 rubric: does the answer cite every expected required source and version "
        "needed to support its claims? Missing a required source lowers the score."
    ),
    "abstention_correctness": (
        "Binary per record: does the system abstain exactly when the question is "
        "unanswerable, and answer when sufficient evidence exists? Applies to all "
        "records, not only unanswerable ones."
    ),
    "conflict_handling": (
        "0-1 rubric for records with conflict_expected or multiple supplied versions: "
        "does the answer identify the conflict, name the versions involved, and avoid "
        "silently presenting a stale passage as current?"
    ),
}


class AnswerEvalValidationError(ValueError):
    """Raised when the answer evaluation dataset or report is malformed."""


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise AnswerEvalValidationError(f"{path.name} must contain a JSON object")
    return payload


def load_answer_eval_records(path: Path = DATASET_PATH) -> list[dict[str, Any]]:
    """Load JSONL records, rejecting blank lines and non-object entries."""

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise AnswerEvalValidationError(
                    f"line {line_number} is not valid JSON"
                ) from exc
            if not isinstance(record, dict):
                raise AnswerEvalValidationError(
                    f"line {line_number} must be a JSON object"
                )
            records.append(record)
    return records


def load_profile(path: Path = PROFILE_PATH) -> dict[str, Any]:
    return _load_json(path)


def _manifest_versions(corpus_id: str) -> dict[str, set[int]]:
    manifest_path = EVALS_DIR / "corpus" / corpus_id / "manifest.json"
    manifest = _load_json(manifest_path)
    versions: dict[str, set[int]] = {}
    for document in manifest.get("documents", []):
        source_key = document.get("source_key")
        if not isinstance(source_key, str):
            continue
        versions[source_key] = {
            version["version_number"]
            for version in document.get("versions", [])
            if isinstance(version.get("version_number"), int)
        }
    return versions


def _check_citation(
    citation: Any,
    *,
    manifest_versions: dict[str, set[int]],
    record_id: str,
    field: str,
) -> None:
    if not isinstance(citation, dict):
        raise AnswerEvalValidationError(f"{record_id}: {field} must be an object")
    source_key = citation.get("source_key")
    if source_key not in manifest_versions:
        raise AnswerEvalValidationError(
            f"{record_id}: {field} references unknown source_key {source_key!r}"
        )
    version_number = citation.get("version_number")
    if version_number is not None:
        if (
            isinstance(version_number, bool)
            or not isinstance(version_number, int)
            or version_number < 1
        ):
            raise AnswerEvalValidationError(
                f"{record_id}: {field} version_number must be a positive integer"
            )
        if version_number not in manifest_versions[source_key]:
            raise AnswerEvalValidationError(
                f"{record_id}: {field} references version {version_number} which "
                f"does not exist for {source_key}"
            )


def validate_answer_eval_records(
    records: list[dict[str, Any]],
    *,
    profile: dict[str, Any] | None = None,
) -> None:
    """Validate fields, counts, and source/version references."""

    profile = profile if profile is not None else load_profile()
    manifest_versions = _manifest_versions(profile["corpus_id"])

    ids: set[str] = set()
    questions: set[str] = set()
    category_counts: dict[str, int] = {}
    answerable = 0
    unanswerable = 0

    for record in records:
        missing = [field for field in REQUIRED_FIELDS if field not in record]
        if missing:
            raise AnswerEvalValidationError(
                f"record is missing fields: {', '.join(missing)}"
            )

        record_id = record["id"]
        if not isinstance(record_id, str) or not record_id.strip():
            raise AnswerEvalValidationError("record id must be a nonempty string")
        if record_id in ids:
            raise AnswerEvalValidationError(f"duplicate record id {record_id!r}")
        ids.add(record_id)

        question = record["question"]
        if not isinstance(question, str) or not question.strip():
            raise AnswerEvalValidationError(
                f"{record_id}: question must be a nonempty string"
            )
        if question in questions:
            raise AnswerEvalValidationError(
                f"{record_id}: duplicate question text"
            )
        questions.add(question)

        category = record["category"]
        if not isinstance(category, str) or not category:
            raise AnswerEvalValidationError(
                f"{record_id}: category must be a nonempty string"
            )
        category_counts[category] = category_counts.get(category, 0) + 1

        if not isinstance(record["answerable"], bool):
            raise AnswerEvalValidationError(
                f"{record_id}: answerable must be a boolean"
            )
        if record["expected_behavior"] not in EXPECTED_BEHAVIORS:
            raise AnswerEvalValidationError(
                f"{record_id}: expected_behavior must be one of "
                f"{', '.join(EXPECTED_BEHAVIORS)}"
            )
        if not isinstance(record["conflict_expected"], bool):
            raise AnswerEvalValidationError(
                f"{record_id}: conflict_expected must be a boolean"
            )

        facts = record["expected_facts"]
        citations = record["expected_citations"]
        if not isinstance(facts, list):
            raise AnswerEvalValidationError(
                f"{record_id}: expected_facts must be a list"
            )
        if not isinstance(citations, list):
            raise AnswerEvalValidationError(
                f"{record_id}: expected_citations must be a list"
            )

        for index, fact in enumerate(facts):
            if not isinstance(fact, dict):
                raise AnswerEvalValidationError(
                    f"{record_id}: expected_facts[{index}] must be an object"
                )
            if not isinstance(fact.get("fact"), str) or not fact["fact"].strip():
                raise AnswerEvalValidationError(
                    f"{record_id}: expected_facts[{index}] needs a fact string"
                )
            _check_citation(
                fact,
                manifest_versions=manifest_versions,
                record_id=record_id,
                field=f"expected_facts[{index}]",
            )

        for index, citation in enumerate(citations):
            _check_citation(
                citation,
                manifest_versions=manifest_versions,
                record_id=record_id,
                field=f"expected_citations[{index}]",
            )
            supporting_text = citation.get("supporting_text")
            if not isinstance(supporting_text, str) or not supporting_text.strip():
                raise AnswerEvalValidationError(
                    f"{record_id}: expected_citations[{index}] needs supporting_text"
                )

        min_citations = record["min_required_citations"]
        if (
            isinstance(min_citations, bool)
            or not isinstance(min_citations, int)
            or min_citations < 0
        ):
            raise AnswerEvalValidationError(
                f"{record_id}: min_required_citations must be a nonnegative integer"
            )

        if record["answerable"]:
            answerable += 1
            if not facts:
                raise AnswerEvalValidationError(
                    f"{record_id}: answerable record needs at least one expected fact"
                )
            if not citations:
                raise AnswerEvalValidationError(
                    f"{record_id}: answerable record needs expected citations"
                )
            if min_citations < 1 or min_citations > len(citations):
                raise AnswerEvalValidationError(
                    f"{record_id}: min_required_citations must be between 1 and the "
                    "number of expected citations"
                )
            if record["abstention_reason"] is not None:
                raise AnswerEvalValidationError(
                    f"{record_id}: answerable record must not set abstention_reason"
                )
        else:
            unanswerable += 1
            if facts or citations:
                raise AnswerEvalValidationError(
                    f"{record_id}: unanswerable record must not expect facts or citations"
                )
            if min_citations != 0:
                raise AnswerEvalValidationError(
                    f"{record_id}: unanswerable record must require zero citations"
                )
            if record["expected_behavior"] != "abstain":
                raise AnswerEvalValidationError(
                    f"{record_id}: unanswerable record must expect abstention"
                )
            if not isinstance(record["abstention_reason"], str):
                raise AnswerEvalValidationError(
                    f"{record_id}: unanswerable record needs an abstention_reason"
                )

        if record["conflict_expected"] and record["expected_behavior"] != "identify_conflict":
            raise AnswerEvalValidationError(
                f"{record_id}: conflict_expected records must use "
                "expected_behavior='identify_conflict'"
            )

    min_counts = profile.get("min_category_counts", {})
    for category, minimum in min_counts.items():
        actual = category_counts.get(category, 0)
        if actual < minimum:
            raise AnswerEvalValidationError(
                f"category {category!r} has {actual} records; profile requires {minimum}"
            )

    if answerable < profile.get("require_min_answerable", 0):
        raise AnswerEvalValidationError("not enough answerable records")
    if unanswerable < profile.get("require_min_unanswerable", 0):
        raise AnswerEvalValidationError("not enough unanswerable records")


def _empty_metric_block() -> dict[str, Any]:
    block = dict.fromkeys(METRIC_DEFINITIONS, None)
    block["n"] = 0
    return block


def build_report_template(
    *,
    system_name: str = "OpsAtlas",
    system_version: str = "0.4.0",
    retrieval_config: dict[str, Any] | None = None,
    generation_config: dict[str, Any] | None = None,
    prompt_version: str | None = None,
) -> dict[str, Any]:
    """Return an unfilled report matching ``answer_eval_v1.report_schema.json``.

    The template deliberately sets all metric values to null and status to
    ``not_run``. Filling it requires an actual evaluation run.
    """

    return {
        "report_version": 1,
        "dataset_id": "answer_eval_v1",
        "status": "not_run",
        "run": {
            "system_name": system_name,
            "system_version": system_version,
            "retrieval_config": retrieval_config or {},
            "generation_config": generation_config
            or {
                "provider": "unset",
                "model": "unset",
                "temperature": 0.0,
                "max_output_tokens": 0,
            },
            "prompt_version": prompt_version,
            "started_at": None,
            "completed_at": None,
        },
        "metric_definitions": dict(METRIC_DEFINITIONS),
        "records": [],
        "aggregate": {
            "status": "not_run",
            "sample_counts": {
                "records": 0,
                "scored": 0,
                "abstained": 0,
                "answerable": 0,
                "unanswerable": 0,
            },
            "overall": _empty_metric_block(),
            "by_category": {},
            "caveats": [
                "No answer evaluation has been run. All metric values are null.",
                "The corpus is synthetic; results would not establish production quality.",
            ],
        },
        "limitations": [
            "Design-only report template; metric values are null.",
            "Answer metrics require human or rubric-based scoring that has not been performed.",
            "The synthetic corpus does not represent production documents or traffic.",
        ],
    }


def validate_report_template(report: dict[str, Any]) -> None:
    """Structural check for the design template without a JSON Schema dependency."""

    for field in (
        "report_version",
        "dataset_id",
        "status",
        "run",
        "metric_definitions",
        "records",
        "aggregate",
        "limitations",
    ):
        if field not in report:
            raise AnswerEvalValidationError(f"report is missing {field}")

    if report["report_version"] != 1:
        raise AnswerEvalValidationError("report_version must be 1")
    if report["dataset_id"] != "answer_eval_v1":
        raise AnswerEvalValidationError("dataset_id must be answer_eval_v1")

    definitions = report["metric_definitions"]
    for metric in METRIC_DEFINITIONS:
        if metric not in definitions:
            raise AnswerEvalValidationError(
                f"metric_definitions is missing {metric}"
            )

    aggregate = report["aggregate"]
    for field in ("status", "sample_counts", "overall", "by_category", "caveats"):
        if field not in aggregate:
            raise AnswerEvalValidationError(f"aggregate is missing {field}")
    for metric in METRIC_DEFINITIONS:
        if metric not in aggregate["overall"]:
            raise AnswerEvalValidationError(
                f"aggregate.overall is missing {metric}"
            )
    if aggregate["status"] == "not_run":
        if any(
            value is not None
            for metric, value in aggregate["overall"].items()
            if metric != "n"
        ):
            raise AnswerEvalValidationError(
                "a not_run report must not contain metric values"
            )
        if report["records"]:
            raise AnswerEvalValidationError(
                "a not_run report must not contain scored records"
            )
