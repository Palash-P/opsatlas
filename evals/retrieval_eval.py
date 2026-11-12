"""Deterministic validation, source mapping, and metrics for retrieval evaluation.

The evaluation harness is deliberately independent of PostgreSQL and the embedding
service. Live retrieval is injected by the CLI runner, while tests use deterministic
fake retrievers and synthetic fixtures.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

K = 5
DEFAULT_FETCH_LIMIT = 20
REPORT_VERSION = 2
HARNESS_VERSION = 2
RETRIEVER_ORDER = (
    "semantic",
    "keyword",
    "lexical",
    "hybrid",
    "hybrid_source",
    "hybrid_weighted_semantic",
    "hybrid_weighted_lexical",
    "hybrid_lexical_primary",
)
RETRIEVER_SCORE_FIELDS = {
    "semantic": "similarity",
    "keyword": "keyword_score",
    "lexical": "lexical_score",
    "hybrid": "hybrid_score",
    "hybrid_source": "hybrid_score",
    "hybrid_weighted_semantic": "hybrid_score",
    "hybrid_weighted_lexical": "hybrid_score",
    "hybrid_lexical_primary": "hybrid_score",
}
RANK_UNITS = ("chunk", "source")
IRRELEVANT_SLOT_PREFIX = "\x00irrelevant:"
REQUIRED_RECORD_FIELDS = (
    "id",
    "question",
    "expected_source_ids",
    "answerable",
    "category",
    "notes",
)


class EvaluationError(RuntimeError):
    """Base class for evaluation failures."""


class EvalValidationError(EvaluationError):
    """Raised when a dataset or source map is invalid."""


class EvaluationRunError(EvaluationError):
    """Raised when a retriever fails during a live evaluation run."""


@dataclass(frozen=True)
class GoldenRecord:
    id: str
    question: str
    expected_source_ids: tuple[str, ...]
    answerable: bool
    category: str
    notes: str
    # Optional explicit grades keyed by expected source ID. Datasets without
    # graded labels keep this empty and continue to report ndcg_at_5 as null.
    relevance_grades: dict[str, int]


@dataclass(frozen=True)
class GoldenDataset:
    dataset_id: str
    path: Path
    sha256: str
    records: tuple[GoldenRecord, ...]

    @property
    def answerable_records(self) -> tuple[GoldenRecord, ...]:
        return tuple(record for record in self.records if record.answerable)

    @property
    def unanswerable_records(self) -> tuple[GoldenRecord, ...]:
        return tuple(record for record in self.records if not record.answerable)


@dataclass(frozen=True)
class SourceTarget:
    source_key: str
    version_number: int | None = None


@dataclass(frozen=True)
class SourceMap:
    mapping_version: int
    dataset_id: str
    path: Path
    sha256: str
    sources: dict[str, SourceTarget]

    @property
    def by_source_key(self) -> dict[str, tuple[str, SourceTarget]]:
        return {
            target.source_key: (expected_source_id, target)
            for expected_source_id, target in self.sources.items()
        }


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise EvalValidationError(
            f"Cannot read '{path}': {exc.strerror or type(exc).__name__}"
        ) from None


def _read_json(path: Path) -> tuple[Any, bytes]:
    raw = _read_bytes(path)
    try:
        return json.loads(raw.decode("utf-8")), raw
    except UnicodeDecodeError:
        raise EvalValidationError(f"'{path}' is not valid UTF-8") from None
    except json.JSONDecodeError as exc:
        raise EvalValidationError(
            f"'{path}' is not valid JSON: {exc.msg} at line {exc.lineno}"
        ) from None


def _require_nonempty_string(value: Any, *, field: str, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvalValidationError(f"{location}: '{field}' must be a nonempty string")
    return value


def _require_string(value: Any, *, field: str, location: str) -> str:
    if not isinstance(value, str):
        raise EvalValidationError(f"{location}: '{field}' must be a string")
    return value


def _parse_relevance_grades(
    payload: dict[str, Any],
    expected_source_ids: list[str],
    *,
    location: str,
) -> dict[str, int]:
    """Parse optional graded relevance labels.

    The field is opt-in. When present it must grade every expected source and
    only expected sources, so a partially labelled query can never silently
    change the denominator of nDCG. Grades are integers 1-3.
    """
    raw_grades = payload.get("relevance_grades")
    if raw_grades is None:
        return {}

    if not isinstance(raw_grades, dict):
        raise EvalValidationError(
            f"{location}: 'relevance_grades' must be an object"
        )

    grades: dict[str, int] = {}
    for source_id, grade in raw_grades.items():
        if not isinstance(source_id, str) or not source_id.strip():
            raise EvalValidationError(
                f"{location}: 'relevance_grades' keys must be nonempty strings"
            )
        if source_id not in expected_source_ids:
            raise EvalValidationError(
                f"{location}: 'relevance_grades' contains unknown source ID "
                f"'{source_id}'"
            )
        if type(grade) is not int or not 1 <= grade <= 3:
            raise EvalValidationError(
                f"{location}: relevance grade for '{source_id}' must be an "
                "integer between 1 and 3"
            )
        grades[source_id] = grade

    missing = sorted(set(expected_source_ids) - set(grades))
    if missing:
        raise EvalValidationError(
            f"{location}: 'relevance_grades' must grade every expected source; "
            f"missing: {', '.join(missing)}"
        )

    return grades


def _parse_record(payload: dict[str, Any], *, line_number: int, path: Path) -> GoldenRecord:
    location = f"{path}:{line_number}"

    missing = [field for field in REQUIRED_RECORD_FIELDS if field not in payload]
    if missing:
        raise EvalValidationError(
            f"{location}: missing required field(s): {', '.join(missing)}"
        )

    record_id = _require_nonempty_string(payload["id"], field="id", location=location)
    question = _require_nonempty_string(
        payload["question"], field="question", location=location
    )
    category = _require_nonempty_string(
        payload["category"], field="category", location=location
    )
    notes = _require_string(payload["notes"], field="notes", location=location)

    answerable = payload["answerable"]
    if type(answerable) is not bool:
        raise EvalValidationError(
            f"{location}: 'answerable' must be a JSON boolean"
        )

    expected_source_ids = payload["expected_source_ids"]
    if not isinstance(expected_source_ids, list):
        raise EvalValidationError(
            f"{location}: 'expected_source_ids' must be a list"
        )

    for value in expected_source_ids:
        if not isinstance(value, str) or not value.strip():
            raise EvalValidationError(
                f"{location}: every 'expected_source_ids' entry must be a "
                "nonempty string"
            )

    if len(set(expected_source_ids)) != len(expected_source_ids):
        raise EvalValidationError(
            f"{location}: 'expected_source_ids' contains duplicate IDs"
        )

    if answerable and not expected_source_ids:
        raise EvalValidationError(
            f"{location}: answerable records must have at least one expected source ID"
        )

    if not answerable and expected_source_ids:
        raise EvalValidationError(
            f"{location}: unanswerable records must have an empty "
            "'expected_source_ids' list"
        )

    relevance_grades = _parse_relevance_grades(
        payload,
        expected_source_ids,
        location=location,
    )

    return GoldenRecord(
        id=record_id,
        question=question,
        expected_source_ids=tuple(expected_source_ids),
        answerable=answerable,
        category=category,
        notes=notes,
        relevance_grades=relevance_grades,
    )


def load_golden_dataset(
    path: Path | str,
    *,
    dataset_id: str | None = None,
) -> GoldenDataset:
    """Load and validate a golden JSONL dataset."""
    dataset_path = Path(path)
    raw = _read_bytes(dataset_path)

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise EvalValidationError(f"'{dataset_path}' is not valid UTF-8") from None

    records: list[GoldenRecord] = []
    line_by_id: dict[str, int] = {}

    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue

        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvalValidationError(
                f"{dataset_path}:{line_number}: invalid JSON: {exc.msg}"
            ) from None

        if not isinstance(payload, dict):
            raise EvalValidationError(
                f"{dataset_path}:{line_number}: each JSONL line must be an object"
            )

        record = _parse_record(payload, line_number=line_number, path=dataset_path)
        if record.id in line_by_id:
            raise EvalValidationError(
                f"{dataset_path}:{line_number}: duplicate record id '{record.id}' "
                f"(first seen at line {line_by_id[record.id]})"
            )

        line_by_id[record.id] = line_number
        records.append(record)

    if not records:
        raise EvalValidationError(f"'{dataset_path}' contains no evaluation records")

    return GoldenDataset(
        dataset_id=dataset_id or dataset_path.stem,
        path=dataset_path,
        sha256=_sha256_bytes(raw),
        records=tuple(records),
    )


def load_dataset_profile(path: Path | str) -> dict[str, Any]:
    """Load a dataset quality profile used by the CLI before a live run."""
    profile_path = Path(path)
    payload, _ = _read_json(profile_path)
    if not isinstance(payload, dict):
        raise EvalValidationError(f"'{profile_path}' must contain a JSON object")

    allowed = {
        "require_unique_questions",
        "min_category_counts",
        "require_min_answerable",
        "require_min_unanswerable",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise EvalValidationError(
            f"'{profile_path}': unknown field(s): {', '.join(unknown)}"
        )

    unique_questions = payload.get("require_unique_questions", False)
    if type(unique_questions) is not bool:
        raise EvalValidationError(
            f"'{profile_path}': 'require_unique_questions' must be a JSON boolean"
        )

    category_counts = payload.get("min_category_counts", {})
    if not isinstance(category_counts, dict):
        raise EvalValidationError(
            f"'{profile_path}': 'min_category_counts' must be an object"
        )
    for category, minimum in category_counts.items():
        if not isinstance(category, str) or not category.strip():
            raise EvalValidationError(
                f"'{profile_path}': 'min_category_counts' keys must be nonempty "
                "strings"
            )
        if type(minimum) is not int or minimum < 1:
            raise EvalValidationError(
                f"'{profile_path}': min_category_counts[{category!r}] must be an "
                "integer >= 1"
            )

    for field in ("require_min_answerable", "require_min_unanswerable"):
        value = payload.get(field, 0)
        if type(value) is not int or value < 0:
            raise EvalValidationError(
                f"'{profile_path}': '{field}' must be an integer >= 0"
            )

    return {
        "require_unique_questions": unique_questions,
        "min_category_counts": dict(category_counts),
        "require_min_answerable": payload.get("require_min_answerable", 0),
        "require_min_unanswerable": payload.get("require_min_unanswerable", 0),
    }


def validate_dataset_profile(
    dataset: GoldenDataset,
    profile: dict[str, Any],
) -> None:
    """Validate a dataset against a profile, failing before any retrieval."""
    problems: list[str] = []

    if profile.get("require_unique_questions"):
        seen: dict[str, str] = {}
        for record in dataset.records:
            normalized = " ".join(record.question.lower().split())
            if normalized in seen:
                problems.append(
                    f"duplicate question text shared by '{seen[normalized]}' and "
                    f"'{record.id}'"
                )
            else:
                seen[normalized] = record.id

    counts: dict[str, int] = {}
    for record in dataset.records:
        counts[record.category] = counts.get(record.category, 0) + 1

    for category, minimum in profile.get("min_category_counts", {}).items():
        actual = counts.get(category, 0)
        if actual < minimum:
            problems.append(
                f"category '{category}' has {actual} record(s); at least "
                f"{minimum} required"
            )

    minimum_answerable = profile.get("require_min_answerable", 0)
    if len(dataset.answerable_records) < minimum_answerable:
        problems.append(
            f"dataset has {len(dataset.answerable_records)} answerable record(s); "
            f"at least {minimum_answerable} required"
        )

    minimum_unanswerable = profile.get("require_min_unanswerable", 0)
    if len(dataset.unanswerable_records) < minimum_unanswerable:
        problems.append(
            f"dataset has {len(dataset.unanswerable_records)} unanswerable "
            f"record(s); at least {minimum_unanswerable} required"
        )

    if problems:
        raise EvalValidationError(
            f"Dataset '{dataset.dataset_id}' failed its quality profile: "
            + "; ".join(problems)
        )


def load_source_map(path: Path | str, dataset: GoldenDataset) -> SourceMap:
    """Load and validate the explicit expected-source to database-source mapping."""
    source_map_path = Path(path)
    payload, raw = _read_json(source_map_path)

    if not isinstance(payload, dict):
        raise EvalValidationError(f"'{source_map_path}' must contain a JSON object")

    required = {"mapping_version", "dataset_id", "sources"}
    missing = sorted(required - set(payload))
    if missing:
        raise EvalValidationError(
            f"'{source_map_path}': missing required field(s): {', '.join(missing)}"
        )

    unknown = sorted(set(payload) - required)
    if unknown:
        raise EvalValidationError(
            f"'{source_map_path}': unknown field(s): {', '.join(unknown)}"
        )

    mapping_version = payload["mapping_version"]
    if (
        type(mapping_version) is not int
        or mapping_version < 1
    ):
        raise EvalValidationError(
            f"'{source_map_path}': 'mapping_version' must be an integer >= 1"
        )

    mapped_dataset_id = _require_nonempty_string(
        payload["dataset_id"],
        field="dataset_id",
        location=str(source_map_path),
    )
    if mapped_dataset_id != dataset.dataset_id:
        raise EvalValidationError(
            f"'{source_map_path}': dataset_id '{mapped_dataset_id}' does not match "
            f"dataset '{dataset.dataset_id}'"
        )

    source_entries = payload["sources"]
    if not isinstance(source_entries, dict):
        raise EvalValidationError(
            f"'{source_map_path}': 'sources' must be an object"
        )

    sources: dict[str, SourceTarget] = {}
    source_keys: dict[str, str] = {}

    for expected_source_id, entry in source_entries.items():
        if not isinstance(expected_source_id, str) or not expected_source_id.strip():
            raise EvalValidationError(
                f"'{source_map_path}': source map keys must be nonempty strings"
            )

        location = f"{source_map_path}: sources[{expected_source_id!r}]"
        if not isinstance(entry, dict):
            raise EvalValidationError(f"{location}: mapping must be an object")

        entry_fields = set(entry)
        allowed = {"source_key", "version_number"}
        unknown_entry_fields = sorted(entry_fields - allowed)
        missing_entry_fields = sorted({"source_key"} - entry_fields)
        if unknown_entry_fields:
            raise EvalValidationError(
                f"{location}: unknown field(s): {', '.join(unknown_entry_fields)}"
            )
        if missing_entry_fields:
            raise EvalValidationError(
                f"{location}: missing required field(s): "
                f"{', '.join(missing_entry_fields)}"
            )

        source_key = _require_nonempty_string(
            entry["source_key"],
            field="source_key",
            location=location,
        )

        version_number = entry.get("version_number")
        if version_number is not None and (
            type(version_number) is not int or version_number < 1
        ):
            raise EvalValidationError(
                f"{location}: 'version_number' must be an integer >= 1 when present"
            )

        if source_key in source_keys:
            raise EvalValidationError(
                f"{location}: source_key '{source_key}' is already mapped to "
                f"'{source_keys[source_key]}'"
            )

        source_keys[source_key] = expected_source_id
        sources[expected_source_id] = SourceTarget(
            source_key=source_key,
            version_number=version_number,
        )

    expected_ids = {
        expected_source_id
        for record in dataset.records
        if record.answerable
        for expected_source_id in record.expected_source_ids
    }
    missing_mappings = sorted(expected_ids - set(sources))
    unknown_mappings = sorted(set(sources) - expected_ids)

    if missing_mappings or unknown_mappings:
        details: list[str] = []
        if missing_mappings:
            details.append(
                "missing explicit mappings for: " + ", ".join(missing_mappings)
            )
        if unknown_mappings:
            details.append(
                "unknown source IDs not present in the dataset: "
                + ", ".join(unknown_mappings)
            )
        raise EvalValidationError(
            f"Source map '{source_map_path}' cannot be used with dataset "
            f"'{dataset.dataset_id}': {'; '.join(details)}. Populate each expected "
            "source ID with its real source_key and optional version_number before "
            "running a live evaluation. No retrieval was executed and no scores "
            "were produced."
        )

    unpinned_version_records = [
        record.id
        for record in dataset.answerable_records
        if record.category == "current_version_selection"
        and any(
            sources[expected_source_id].version_number is None
            for expected_source_id in record.expected_source_ids
        )
    ]
    if unpinned_version_records:
        raise EvalValidationError(
            f"Source map '{source_map_path}' cannot evaluate current-version "
            "records without pinned versions. Add 'version_number' to every "
            "mapped source used by: " + ", ".join(unpinned_version_records) + ". "
            "No retrieval was executed and no scores were produced."
        )

    return SourceMap(
        mapping_version=mapping_version,
        dataset_id=mapped_dataset_id,
        path=source_map_path,
        sha256=_sha256_bytes(raw),
        sources=sources,
    )


def recall_at_k(
    ranked_source_ids: list[str] | tuple[str, ...],
    expected_source_ids: list[str] | tuple[str, ...],
    *,
    k: int = K,
) -> float:
    """Fraction of expected sources present in the top-k ranked sources."""
    expected = tuple(expected_source_ids)
    if not expected:
        raise ValueError("expected_source_ids must not be empty")

    retrieved = set(ranked_source_ids[:k])
    return len(retrieved.intersection(expected)) / len(expected)


def reciprocal_rank_at_k(
    ranked_source_ids: list[str] | tuple[str, ...],
    expected_source_ids: list[str] | tuple[str, ...],
    *,
    k: int = K,
) -> float:
    """Reciprocal rank of the first expected source, or 0.0 if none is found."""
    expected = set(expected_source_ids)
    if not expected:
        raise ValueError("expected_source_ids must not be empty")

    for rank, source_id in enumerate(ranked_source_ids[:k], start=1):
        if source_id in expected:
            return 1.0 / rank

    return 0.0


def any_hit_at_k(
    ranked_source_ids: list[str] | tuple[str, ...],
    expected_source_ids: list[str] | tuple[str, ...],
    *,
    k: int = K,
) -> int:
    """Return 1 when at least one expected source appears in the top-k results."""
    expected = set(expected_source_ids)
    if not expected:
        raise ValueError("expected_source_ids must not be empty")

    return int(bool(expected.intersection(ranked_source_ids[:k])))


def all_expected_at_k(
    ranked_source_ids: list[str] | tuple[str, ...],
    expected_source_ids: list[str] | tuple[str, ...],
    *,
    k: int = K,
) -> int:
    """Return 1 when every expected source appears in the top-k results."""
    expected = set(expected_source_ids)
    if not expected:
        raise ValueError("expected_source_ids must not be empty")

    return int(expected.issubset(set(ranked_source_ids[:k])))


def ndcg_at_k(
    ranked_source_ids: list[str] | tuple[str, ...],
    relevance_grades: dict[str, float],
    *,
    k: int = K,
) -> float | None:
    """Graded nDCG@k for datasets that provide explicit relevance grades.

    This helper is intentionally separate from the golden_v0 path, whose labels
    are binary and ungraded. Returns None when no positive graded relevance exists.
    """
    for source_id, grade in relevance_grades.items():
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError("relevance grade keys must be nonempty strings")
        if isinstance(grade, bool) or not isinstance(grade, (int, float)):
            raise TypeError("relevance grades must be numeric")
        if not math.isfinite(grade) or grade < 0:
            raise ValueError("relevance grades must be finite and nonnegative")

    def gain(grade: float) -> float:
        return (2.0**grade) - 1.0

    dcg = 0.0
    for rank, source_id in enumerate(ranked_source_ids[:k], start=1):
        grade = relevance_grades.get(source_id)
        if grade is not None and grade > 0:
            dcg += gain(grade) / math.log2(rank + 1)

    ideal_grades = sorted(
        (grade for grade in relevance_grades.values() if grade > 0),
        reverse=True,
    )[:k]
    idcg = sum(
        gain(grade) / math.log2(rank + 1)
        for rank, grade in enumerate(ideal_grades, start=1)
    )

    if idcg <= 0:
        return None

    return dcg / idcg


@dataclass(frozen=True)
class RankedSource:
    rank: int
    source_key: str
    version_number: int
    mapped_source_id: str | None
    mapping_status: str


def _validated_result_row(row: Any, *, index: int) -> tuple[Any, Any, Any, str, int]:
    if not isinstance(row, dict):
        raise EvaluationRunError(
            f"Retriever result at position {index} must be an object"
        )

    for field in ("chunk_id", "document_id", "version_id"):
        if row.get(field) is None:
            raise EvaluationRunError(
                f"Retriever result at position {index} is missing '{field}'"
            )

    source_key = row.get("source_key")
    if not isinstance(source_key, str) or not source_key.strip():
        raise EvaluationRunError(
            f"Retriever result at position {index} has an invalid 'source_key'"
        )

    version_number = row.get("version_number")
    if type(version_number) is not int or version_number < 1:
        raise EvaluationRunError(
            f"Retriever result at position {index} has an invalid 'version_number'"
        )

    return (
        row["chunk_id"],
        row["document_id"],
        row["version_id"],
        source_key,
        version_number,
    )


def _mapping_for_result(
    source_key: str,
    version_number: int,
    source_map: SourceMap,
) -> tuple[str | None, str]:
    target = source_map.by_source_key.get(source_key)
    if target is None:
        return None, "unmapped"

    expected_source_id, source_target = target
    if (
        source_target.version_number is not None
        and source_target.version_number != version_number
    ):
        return expected_source_id, "version_mismatch"

    return expected_source_id, "matched"


def map_ranked_sources(
    results: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    source_map: SourceMap,
    *,
    k: int = K,
) -> list[RankedSource]:
    """Collapse chunk results into unique ranked source units.

    A source unit is identified by (source_key, version_number). Relevance is
    assigned only through the explicit source map; title, document_id, and
    chunk_id never participate in matching.
    """
    seen: set[tuple[str, int]] = set()
    ranked: list[RankedSource] = []

    for index, row in enumerate(results, start=1):
        _, _, _, source_key, version_number = _validated_result_row(row, index=index)
        source_unit = (source_key, version_number)
        if source_unit in seen:
            continue

        seen.add(source_unit)
        mapped_source_id, mapping_status = _mapping_for_result(
            source_key,
            version_number,
            source_map,
        )
        ranked.append(
            RankedSource(
                rank=len(ranked) + 1,
                source_key=source_key,
                version_number=version_number,
                mapped_source_id=mapped_source_id,
                mapping_status=mapping_status,
            )
        )

        if len(ranked) >= k:
            break

    return ranked


def serialize_results(
    results: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    source_map: SourceMap,
    *,
    score_field: str,
    k: int = K,
) -> list[dict[str, Any]]:
    """Serialize top-k chunk results without including document content."""
    serialized: list[dict[str, Any]] = []

    for rank, row in enumerate(results[:k], start=1):
        chunk_id, document_id, version_id, source_key, version_number = (
            _validated_result_row(row, index=rank)
        )
        mapped_source_id, mapping_status = _mapping_for_result(
            source_key,
            version_number,
            source_map,
        )

        score = row.get(score_field)
        if score is not None and (
            isinstance(score, bool) or not isinstance(score, (int, float))
        ):
            raise EvaluationRunError(
                f"Retriever result at position {rank} has a non-numeric "
                f"'{score_field}' score"
            )

        serialized.append(
            {
                "rank": rank,
                "chunk_id": str(chunk_id),
                "document_id": str(document_id),
                "version_id": str(version_id),
                "source_key": source_key,
                "version_number": version_number,
                "mapped_source_id": mapped_source_id,
                "mapping_status": mapping_status,
                "score": None if score is None else float(score),
                "score_field": score_field,
            }
        )

    return serialized


def serialize_units(units: list[RankedSource] | tuple[RankedSource, ...]) -> list[dict[str, Any]]:
    """Serialize collapsed source units without document content."""
    return [
        {
            "rank": unit.rank,
            "source_key": unit.source_key,
            "version_number": unit.version_number,
            "mapped_source_id": unit.mapped_source_id,
            "mapping_status": unit.mapping_status,
            "relevant": (
                unit.mapping_status == "matched" and unit.mapped_source_id is not None
            ),
        }
        for unit in units
    ]


def _slot_ids(units: list[RankedSource] | tuple[RankedSource, ...]) -> list[str]:
    """Return rank-aligned IDs where irrelevant units keep their position.

    Unmapped and version-mismatched results must consume a rank slot and count as
    irrelevant. Removing them would compress the ranking and inflate MRR whenever
    a distractor or stale version outranks the relevant source.
    """
    ids: list[str] = []
    for unit in units:
        if unit.mapping_status == "matched" and unit.mapped_source_id is not None:
            ids.append(unit.mapped_source_id)
        else:
            ids.append(f"{IRRELEVANT_SLOT_PREFIX}{unit.rank}")
    return ids


def _first_expected_rank(
    ranked_ids: list[str] | tuple[str, ...],
    expected_source_ids: list[str] | tuple[str, ...],
    *,
    k: int,
) -> int | None:
    expected = set(expected_source_ids)
    for rank, source_id in enumerate(ranked_ids[:k], start=1):
        if source_id in expected:
            return rank
    return None


def _unit_metrics(
    ranked_ids: list[str],
    expected_source_ids: list[str],
    *,
    k: int,
    rank_unit: str,
    relevance_grades: dict[str, int] | None = None,
) -> dict[str, Any]:
    ndcg: float | None = None
    ndcg_status = "not_applicable_binary_relevance_only"
    if relevance_grades and rank_unit == "source":
        ndcg = ndcg_at_k(ranked_ids, relevance_grades, k=k)
        ndcg_status = "computed_graded_relevance"

    return {
        "rank_unit": rank_unit,
        "recall_at_5": recall_at_k(ranked_ids, expected_source_ids, k=k),
        "mrr": reciprocal_rank_at_k(ranked_ids, expected_source_ids, k=k),
        "any_hit_at_5": any_hit_at_k(ranked_ids, expected_source_ids, k=k),
        "all_expected_at_5": all_expected_at_k(
            ranked_ids,
            expected_source_ids,
            k=k,
        ),
        "first_expected_rank": _first_expected_rank(
            ranked_ids,
            expected_source_ids,
            k=k,
        ),
        "ndcg_at_5": ndcg,
        "ndcg_status": ndcg_status,
    }


def evaluate_query_result(
    record: GoldenRecord,
    results: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    source_map: SourceMap,
    *,
    score_field: str,
    k: int = K,
    fetch_limit: int | None = None,
) -> dict[str, Any]:
    """Evaluate one answerable query against one retriever's returned chunks.

    Two rankings are reported from the same result pool:

    - ``metrics`` / ``results`` use the first ``k`` chunks (chunk-level).
    - ``source_metrics`` / ``source_units`` collapse duplicate chunks from one
      source before truncating to ``k`` unique source units.

    Both rankings retain unmapped and version-mismatched units in their original
    positions as irrelevant. This makes MRR able to penalise a distractor or a
    stale version that outranks the expected source.
    """
    if not record.answerable:
        raise ValueError("unanswerable records cannot be scored for retrieval quality")

    result_pool = list(results)
    if fetch_limit is not None:
        result_pool = result_pool[:fetch_limit]

    chunk_units = map_ranked_sources(result_pool[:k], source_map, k=k)
    source_units = map_ranked_sources(result_pool, source_map, k=k)
    serialized = serialize_results(result_pool[:k], source_map, score_field=score_field, k=k)

    chunk_slot_ids = _slot_ids(chunk_units)
    source_slot_ids = _slot_ids(source_units)

    expected = list(record.expected_source_ids)
    matched = [source_id for source_id in expected if source_id in set(chunk_slot_ids)]
    missing = [source_id for source_id in expected if source_id not in set(chunk_slot_ids)]
    source_matched = [
        source_id for source_id in expected if source_id in set(source_slot_ids)
    ]
    source_missing = [
        source_id for source_id in expected if source_id not in set(source_slot_ids)
    ]

    unmapped_source_keys = sorted(
        {
            result["source_key"]
            for result in serialized
            if result["mapping_status"] == "unmapped"
        }
    )
    version_mismatch_source_keys = sorted(
        {
            result["source_key"]
            for result in serialized
            if result["mapping_status"] == "version_mismatch"
        }
    )

    return {
        "result_count": len(serialized),
        "results": serialized,
        "ranked_source_ids": [
            unit.mapped_source_id
            for unit in chunk_units
            if unit.mapping_status == "matched" and unit.mapped_source_id is not None
        ],
        "ranked_units": serialize_units(chunk_units),
        "source_units": serialize_units(source_units),
        "matched_source_ids": matched,
        "missing_source_ids": missing,
        "source_matched_source_ids": source_matched,
        "source_missing_source_ids": source_missing,
        "unmapped_source_keys": unmapped_source_keys,
        "version_mismatch_source_keys": version_mismatch_source_keys,
        "metrics": _unit_metrics(
            chunk_slot_ids,
            expected,
            k=k,
            rank_unit="chunk",
        ),
        "source_metrics": _unit_metrics(
            source_slot_ids,
            expected,
            k=k,
            rank_unit="source",
            relevance_grades=record.relevance_grades,
        ),
    }


def _mean(values: list[float | int]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _aggregate_retriever_metrics(
    query_reports: list[dict[str, Any]],
    retriever_name: str,
    *,
    rank_unit: str,
) -> dict[str, Any]:
    key = "metrics" if rank_unit == "chunk" else "source_metrics"
    metrics = [
        query["retrievers"][retriever_name][key]
        for query in query_reports
    ]
    ndcg_values = [
        metric["ndcg_at_5"]
        for metric in metrics
        if metric.get("ndcg_at_5") is not None
    ]

    return {
        "query_count": len(metrics),
        "metric_scope": "answerable queries only",
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


def _aggregate_category_metrics(
    query_reports: list[dict[str, Any]],
    retriever_names: list[str],
) -> dict[str, Any]:
    categories: dict[str, list[dict[str, Any]]] = {}
    for query in query_reports:
        categories.setdefault(query["category"], []).append(query)

    aggregated: dict[str, Any] = {}
    for category in sorted(categories):
        category_queries = categories[category]
        aggregated[category] = {
            "query_count": len(category_queries),
            "retrievers": {
                retriever_name: {
                    rank_unit: _aggregate_retriever_metrics(
                        category_queries,
                        retriever_name,
                        rank_unit=rank_unit,
                    )
                    for rank_unit in RANK_UNITS
                }
                for retriever_name in retriever_names
            },
        }
    return aggregated


def _metric_delta(hybrid_value: float | None, semantic_value: float | None) -> float | None:
    if hybrid_value is None or semantic_value is None:
        return None
    return hybrid_value - semantic_value


def _metric_set_delta(
    hybrid_metrics: dict[str, Any],
    semantic_metrics: dict[str, Any],
) -> dict[str, float | None]:
    return {
        "recall_at_5_delta": _metric_delta(
            hybrid_metrics["recall_at_5"],
            semantic_metrics["recall_at_5"],
        ),
        "mrr_delta": _metric_delta(hybrid_metrics["mrr"], semantic_metrics["mrr"]),
        "any_hit_at_5_delta": _metric_delta(
            hybrid_metrics["any_hit_at_5"],
            semantic_metrics["any_hit_at_5"],
        ),
        "all_expected_at_5_delta": _metric_delta(
            hybrid_metrics["all_expected_at_5"],
            semantic_metrics["all_expected_at_5"],
        ),
        "ndcg_at_5_delta": None,
    }


def _compare_query_retrievers(query: dict[str, Any]) -> dict[str, Any]:
    semantic = query["retrievers"]["semantic"]
    hybrid = query["retrievers"]["hybrid"]

    chunk_delta = _metric_set_delta(hybrid["metrics"], semantic["metrics"])
    source_delta = _metric_set_delta(
        hybrid["source_metrics"],
        semantic["source_metrics"],
    )

    # Chunk-level deltas stay at the top level for continuity with report v1.
    return {
        **chunk_delta,
        "by_rank_unit": {
            "chunk": chunk_delta,
            "source": source_delta,
        },
    }



def run_evaluation(
    *,
    dataset: GoldenDataset,
    source_map: SourceMap,
    tenant_id: Any,
    retrievers: dict[str, Callable[..., list[dict[str, Any]]]],
    retrieval_config: dict[str, Any],
    fetch_limit: int = K,
    corpus_metadata: dict[str, Any] | None = None,
    baseline_report: dict[str, Any] | None = None,
    repo_root: Path | str | None = None,
) -> dict[str, Any]:
    """Run semantic and hybrid retrieval on answerable records and build a report.

    ``fetch_limit`` controls how many chunks each retriever may return. Chunk-level
    metrics still use the first ``K`` chunks; source-level metrics collapse
    duplicate chunks first, so a larger fetch limit lets a source-level ranking
    reach ``K`` unique sources.
    """
    required_retrievers = {"semantic", "hybrid"}
    missing_retrievers = sorted(required_retrievers - set(retrievers))
    if missing_retrievers:
        raise ValueError(
            "retrievers must include: " + ", ".join(missing_retrievers)
        )
    unknown_retrievers = sorted(set(retrievers) - set(RETRIEVER_SCORE_FIELDS))
    if unknown_retrievers:
        raise ValueError(
            "unknown retriever(s): " + ", ".join(unknown_retrievers)
        )
    retriever_names = [
        name for name in RETRIEVER_ORDER if name in retrievers
    ]
    if not 1 <= fetch_limit <= DEFAULT_FETCH_LIMIT:
        raise ValueError(
            f"fetch_limit must be between 1 and {DEFAULT_FETCH_LIMIT}"
        )

    query_reports: list[dict[str, Any]] = []

    for record in dataset.answerable_records:
        retriever_results: dict[str, Any] = {}

        for retriever_name in retriever_names:
            retriever = retrievers[retriever_name]
            try:
                retriever_results[retriever_name] = retriever(
                    tenant_id=tenant_id,
                    query=record.question,
                    limit=fetch_limit,
                )
            except Exception as exc:  # noqa: BLE001 - fail the run, never leak details
                raise EvaluationRunError(
                    f"Retriever '{retriever_name}' failed for query id "
                    f"'{record.id}' ({type(exc).__name__}). No evaluation report "
                    "was written. Check the database and embedding service, then "
                    "rerun the evaluation."
                ) from None

        query_report = {
            "id": record.id,
            "question": record.question,
            "category": record.category,
            "answerable": True,
            "expected_source_ids": list(record.expected_source_ids),
            "retrievers": {
                retriever_name: evaluate_query_result(
                    record,
                    retriever_results[retriever_name],
                    source_map,
                    score_field=RETRIEVER_SCORE_FIELDS[retriever_name],
                    fetch_limit=fetch_limit,
                )
                for retriever_name in retriever_names
            },
        }
        query_report["comparison"] = _compare_query_retrievers(query_report)
        query_report["diagnostics"] = _ranking_diagnostics(
            query_report["retrievers"]["semantic"],
            query_report["retrievers"]["hybrid"],
            retriever_results.get("keyword"),
            retriever_results.get("lexical"),
            fetch_limit=fetch_limit,
        )
        query_report["diagnostics"]["fusion_contributions"] = _fusion_contributions(
            query_report["retrievers"],
            retriever_results,
            fetch_limit=fetch_limit,
        )
        query_reports.append(query_report)

    by_rank_unit: dict[str, Any] = {}
    for rank_unit in RANK_UNITS:
        semantic_metrics = _aggregate_retriever_metrics(
            query_reports,
            "semantic",
            rank_unit=rank_unit,
        )
        hybrid_metrics = _aggregate_retriever_metrics(
            query_reports,
            "hybrid",
            rank_unit=rank_unit,
        )
        per_retriever = {
            retriever_name: _aggregate_retriever_metrics(
                query_reports,
                retriever_name,
                rank_unit=rank_unit,
            )
            for retriever_name in retriever_names
        }
        by_rank_unit[rank_unit] = {
            **per_retriever,
            "delta": _metric_set_delta(hybrid_metrics, semantic_metrics),
        }

    semantic_metrics = by_rank_unit["chunk"]["semantic"]
    hybrid_metrics = by_rank_unit["chunk"]["hybrid"]

    unanswerable = [
        {
            "id": record.id,
            "question": record.question,
            "category": record.category,
            "answerable": False,
            "scored": False,
            "reason": (
                "Excluded from retrieval-quality metrics because the record has no "
                "relevant source set. This harness does not evaluate abstention, "
                "secret handling, citations, faithfulness, or prompt-injection "
                "resistance."
            ),
        }
        for record in dataset.unanswerable_records
    ]

    config = dict(retrieval_config)
    config["tenant_id"] = str(tenant_id)
    config["fetch_limit"] = fetch_limit

    corpus = (
        dict(corpus_metadata)
        if corpus_metadata is not None
        else {
            "status": "not_computed",
            "sha256": None,
            "document_count": None,
            "current_version_count": None,
            "chunk_count": None,
        }
    )
    corpus.setdefault("status", "computed")

    diagnostics = _aggregate_diagnostics(query_reports)
    saturation_warning = _saturation_warning(
        query_reports,
        by_rank_unit,
        retriever_names,
    )

    report: dict[str, Any] = {
        "report_version": REPORT_VERSION,
        "harness_version": HARNESS_VERSION,
        "dataset": {
            "id": dataset.dataset_id,
            "path": _display_path(dataset.path, repo_root),
            "sha256": dataset.sha256,
            "record_count": len(dataset.records),
            "answerable_count": len(dataset.answerable_records),
            "unanswerable_count": len(dataset.unanswerable_records),
        },
        "source_map": {
            "mapping_version": source_map.mapping_version,
            "path": _display_path(source_map.path, repo_root),
            "sha256": source_map.sha256,
        },
        "relevance_grading": {
            "provided": any(
                bool(record.relevance_grades) for record in dataset.records
            ),
            "scale": (
                "3 = directly answers a required part of the question; "
                "2 = required supporting detail but not sufficient alone; "
                "1 = background context, not used as expected relevance; "
                "absent = not a relevant source"
            ),
            "policy": (
                "nDCG@5 is computed over source-level units only. Binary "
                "Recall@5, MRR, and AllExpected@5 remain the primary metrics."
            ),
        },
        "corpus": corpus,
        "retrieval_configuration": config,
        "metric_definitions": {
            "rank_units": (
                "Chunk-level metrics take the first 5 returned chunks. Source-level "
                "metrics collapse duplicate chunks from one (source_key, "
                "version_number) unit first, then take the first 5 unique units. "
                "Both rankings keep unmapped and version-mismatched units in their "
                "original positions as irrelevant."
            ),
            "recall_at_5": (
                "Macro-average over answerable queries of the number of expected "
                "sources present in the top-5 rank units divided by the number of "
                "expected sources."
            ),
            "mrr": (
                "Macro-average over answerable queries of 1 / rank of the first "
                "expected source in the top-5 rank units; 0 if no expected source "
                "is present. Distractors and version mismatches occupy rank slots."
            ),
            "any_hit_at_5": (
                "Macro-average over answerable queries of 1 when at least one "
                "expected source is in the top 5 rank units, else 0."
            ),
            "all_expected_at_5": (
                "Macro-average over answerable queries of 1 when every expected "
                "source is in the top 5 rank units, else 0."
            ),
            "ndcg_at_5": (
                "Reported only when every expected source has an explicit "
                "integer relevance grade. Datasets without graded labels keep "
                "this metric null, and no graded relevance is inferred."
            ),
            "unanswerable_records": (
                "Excluded from all retrieval-quality metrics and never passed to a "
                "retriever. This is not an abstention evaluation."
            ),
        },
        "aggregate": {
            "semantic": semantic_metrics,
            "hybrid": hybrid_metrics,
            "delta": _metric_set_delta(hybrid_metrics, semantic_metrics),
            "retrievers": retriever_names,
            "by_retriever": {
                retriever_name: {
                    rank_unit: by_rank_unit[rank_unit][retriever_name]
                    for rank_unit in RANK_UNITS
                }
                for retriever_name in retriever_names
            },
            "by_rank_unit": by_rank_unit,
            "category_metrics": _aggregate_category_metrics(
                query_reports,
                retriever_names,
            ),
            "diagnostics": diagnostics,
            "saturation_warning": saturation_warning,
            "scored_query_count": len(query_reports),
        },
        "queries": query_reports,
        "unanswerable": unanswerable,
        "limitations": [
            "This report evaluates retrieval rankings only.",
            (
                "Relevance labels are binary for Recall@5, MRR, and "
                "AllExpected@5. nDCG@5 is reported only when every expected "
                "source has an explicit integer grade; otherwise it is null and "
                "no graded relevance is inferred."
            ),
            (
                "Retrieval is current-version-only. Only current-version selection "
                "is evaluated; historical-version ranking is not supported."
            ),
            (
                "Chunk-level and source-level rankings are reported separately. "
                "A chunk-level cutoff can under-cover unique sources when a source "
                "contributes more than one chunk."
            ),
            (
                "No abstention, secret-protection, prompt-injection resistance, "
                "citation-correctness, or faithfulness evaluation is performed."
            ),
            (
                "Unanswerable records are unscored and are not evidence that the "
                "system refuses or abstains correctly."
            ),
            "No retrieved document content is included in this report.",
            (
                "Semantic similarity scores and RRF hybrid scores use different "
                "scales and are never compared numerically; only rank-based "
                "metrics and rank deltas are reported."
            ),
        ],
    }

    if baseline_report is not None:
        report["baseline_comparison"] = compare_reports(report, baseline_report)

    return report


def _display_path(path: Path, repo_root: Path | str | None) -> str:
    if repo_root is None:
        return str(path)
    try:
        return Path(path).resolve().relative_to(Path(repo_root).resolve()).as_posix()
    except ValueError:
        return str(path)


def compute_corpus_fingerprint(
    rows: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    """Fingerprint the current corpus without storing any document content."""
    normalized: list[list[Any]] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise TypeError(f"corpus row {index} must be an object")
        source_key = row.get("source_key")
        version_number = row.get("version_number")
        chunk_index = row.get("chunk_index")
        content_hash = row.get("content_hash")

        if not isinstance(source_key, str) or not source_key.strip():
            raise ValueError(f"corpus row {index} has an invalid 'source_key'")
        if type(version_number) is not int or version_number < 1:
            raise ValueError(f"corpus row {index} has an invalid 'version_number'")
        if type(chunk_index) is not int or chunk_index < 0:
            raise ValueError(f"corpus row {index} has an invalid 'chunk_index'")
        if (
            not isinstance(content_hash, str)
            or len(content_hash) != 64
            or any(character not in "0123456789abcdef" for character in content_hash)
        ):
            raise ValueError(f"corpus row {index} has an invalid 'content_hash'")

        normalized.append([source_key, version_number, chunk_index, content_hash])

    normalized.sort()
    payload = json.dumps(normalized, separators=(",", ":"), ensure_ascii=True)
    return {
        "status": "computed",
        "sha256": _sha256_bytes(payload.encode("utf-8")),
        "document_count": len({row[0] for row in normalized}),
        "current_version_count": len({(row[0], row[1]) for row in normalized}),
        "chunk_count": len(normalized),
    }


def preflight_source_map(
    source_map: SourceMap,
    inventory: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> list[str]:
    """Return actionable problems before a live run touches the corpus."""
    by_source_key: dict[str, list[dict[str, Any]]] = {}
    for index, row in enumerate(inventory, start=1):
        if not isinstance(row, dict):
            raise TypeError(f"inventory row {index} must be an object")
        source_key = row.get("source_key")
        version_number = row.get("version_number")
        if not isinstance(source_key, str) or not source_key.strip():
            raise ValueError(f"inventory row {index} has an invalid 'source_key'")
        if type(version_number) is not int or version_number < 1:
            raise ValueError(f"inventory row {index} has an invalid 'version_number'")
        by_source_key.setdefault(source_key, []).append(row)

    issues: list[str] = []
    for expected_source_id, target in sorted(source_map.sources.items()):
        versions = by_source_key.get(target.source_key, [])
        if not versions:
            issues.append(
                f"'{expected_source_id}' maps to '{target.source_key}', which does "
                "not exist for this tenant"
            )
            continue

        current_versions = [
            version
            for version in versions
            if version.get("is_current") is True
            and version.get("status", "active") == "active"
        ]
        if not current_versions:
            issues.append(
                f"'{expected_source_id}' maps to '{target.source_key}', which has "
                "no active current version"
            )
            continue

        current = max(current_versions, key=lambda version: version["version_number"])
        if (
            target.version_number is not None
            and current["version_number"] != target.version_number
        ):
            issues.append(
                f"'{expected_source_id}' pins version {target.version_number}, but "
                f"the current version of '{target.source_key}' is "
                f"{current['version_number']}"
            )
            continue

        if current.get("chunk_count", 0) < 1:
            issues.append(
                f"'{expected_source_id}' maps to '{target.source_key}' "
                f"v{current['version_number']}, which has no chunks"
            )

    return issues


def _rank_delta(hybrid_rank: int | None, semantic_rank: int | None) -> int | None:
    if hybrid_rank is None or semantic_rank is None:
        return None
    return hybrid_rank - semantic_rank


def _candidate_diagnostics(
    candidate_rows: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
    semantic_eval: dict[str, Any],
    hybrid_eval: dict[str, Any],
    *,
    fetch_limit: int,
) -> dict[str, Any]:
    """Describe one candidate list and how much of it reached the fused top-k.

    The same function is used for the AND keyword baseline and for the OR+IDF
    lexical strategy, so a report can show whether fusion actually received a
    non-empty lexical candidate list.
    """
    if candidate_rows is None:
        return {"provided": False}

    rows = list(candidate_rows)[:fetch_limit]
    candidate_chunk_ids: set[Any] = set()
    candidate_source_keys: set[str] = set()
    for index, row in enumerate(rows, start=1):
        chunk_id, _, _, source_key, _ = _validated_result_row(row, index=index)
        candidate_chunk_ids.add(str(chunk_id))
        candidate_source_keys.add(source_key)

    semantic_chunk_ids = {result["chunk_id"] for result in semantic_eval["results"]}
    hybrid_chunk_ids = {result["chunk_id"] for result in hybrid_eval["results"]}
    semantic_source_keys = {
        result["source_key"] for result in semantic_eval["results"]
    }

    return {
        "provided": True,
        "result_count": len(rows),
        "unique_source_count": len(candidate_source_keys),
        "top_source_keys": sorted(candidate_source_keys),
        "chunks_contributing_to_semantic_top_k": len(
            candidate_chunk_ids & semantic_chunk_ids
        ),
        "chunks_contributing_to_hybrid_top_k": len(
            candidate_chunk_ids & hybrid_chunk_ids
        ),
        "source_keys_not_returned_by_semantic_top_k": sorted(
            candidate_source_keys - semantic_source_keys
        ),
    }


def _fusion_contributions(
    retriever_evals: dict[str, dict[str, Any]],
    retriever_results: dict[str, Any],
    *,
    fetch_limit: int,
) -> dict[str, Any]:
    """Describe which candidate lists supplied each fusion strategy's top-k.

    Chunk IDs are compared against the semantic and OR+IDF lexical candidate
    pools, not against only the top-k of either list. This makes a lexical-only
    promotion visible even when the chunk entered fusion from rank 7.
    """
    lexical_rows = list(retriever_results.get("lexical") or [])[:fetch_limit]
    semantic_rows = list(retriever_results.get("semantic") or [])[:fetch_limit]
    lexical_ids = {str(row["chunk_id"]) for row in lexical_rows}
    semantic_ids = {str(row["chunk_id"]) for row in semantic_rows}

    contributions: dict[str, Any] = {}
    for name, evaluation in retriever_evals.items():
        if not name.startswith("hybrid"):
            continue
        top_ids = {result["chunk_id"] for result in evaluation["results"]}
        origin_counts: dict[str, int] = {}
        for row in list(retriever_results.get(name) or [])[:K]:
            origin = row.get("ranking_origin", "fused")
            origin_counts[origin] = origin_counts.get(origin, 0) + 1
        contributions[name] = {
            "top_k_count": len(top_ids),
            "lexical_candidate_matches": len(top_ids & lexical_ids),
            "semantic_candidate_matches": len(top_ids & semantic_ids),
            "lexical_only_matches": len((top_ids & lexical_ids) - semantic_ids),
            "semantic_only_matches": len((top_ids & semantic_ids) - lexical_ids),
            "shared_matches": len(top_ids & lexical_ids & semantic_ids),
            "origin_counts": origin_counts,
        }

    return contributions


def _ranking_diagnostics(
    semantic_eval: dict[str, Any],
    hybrid_eval: dict[str, Any],
    keyword_rows: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
    lexical_rows: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
    *,
    fetch_limit: int,
) -> dict[str, Any]:
    semantic_chunks = [result["chunk_id"] for result in semantic_eval["results"]]
    hybrid_chunks = [result["chunk_id"] for result in hybrid_eval["results"]]
    semantic_sources = [
        (unit["source_key"], unit["version_number"])
        for unit in semantic_eval["source_units"]
    ]
    hybrid_sources = [
        (unit["source_key"], unit["version_number"])
        for unit in hybrid_eval["source_units"]
    ]

    return {
        "identical_chunk_ranking": semantic_chunks == hybrid_chunks,
        "identical_source_ranking": semantic_sources == hybrid_sources,
        "semantic_first_expected_rank_chunk": semantic_eval["metrics"][
            "first_expected_rank"
        ],
        "hybrid_first_expected_rank_chunk": hybrid_eval["metrics"][
            "first_expected_rank"
        ],
        "first_expected_rank_delta_chunk": _rank_delta(
            hybrid_eval["metrics"]["first_expected_rank"],
            semantic_eval["metrics"]["first_expected_rank"],
        ),
        "semantic_first_expected_rank_source": semantic_eval["source_metrics"][
            "first_expected_rank"
        ],
        "hybrid_first_expected_rank_source": hybrid_eval["source_metrics"][
            "first_expected_rank"
        ],
        "first_expected_rank_delta_source": _rank_delta(
            hybrid_eval["source_metrics"]["first_expected_rank"],
            semantic_eval["source_metrics"]["first_expected_rank"],
        ),
        "keyword": _candidate_diagnostics(
            keyword_rows,
            semantic_eval,
            hybrid_eval,
            fetch_limit=fetch_limit,
        ),
        "lexical": _candidate_diagnostics(
            lexical_rows,
            semantic_eval,
            hybrid_eval,
            fetch_limit=fetch_limit,
        ),
    }


def _aggregate_diagnostics(query_reports: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(query_reports)
    identical_chunk = sum(
        1
        for query in query_reports
        if query["diagnostics"]["identical_chunk_ranking"]
    )
    identical_source = sum(
        1
        for query in query_reports
        if query["diagnostics"]["identical_source_ranking"]
    )
    rank_difference = sum(
        1
        for query in query_reports
        if query["diagnostics"]["first_expected_rank_delta_chunk"] not in (0, None)
        or query["diagnostics"]["first_expected_rank_delta_source"] not in (0, None)
    )
    keyword_provided = total > 0 and all(
        query["diagnostics"]["keyword"]["provided"] for query in query_reports
    )
    lexical_provided = total > 0 and all(
        query["diagnostics"]["lexical"]["provided"] for query in query_reports
    )
    keyword_contribution = sum(
        1
        for query in query_reports
        if query["diagnostics"]["keyword"].get(
            "chunks_contributing_to_hybrid_top_k", 0
        )
        > 0
    )
    lexical_contribution = sum(
        1
        for query in query_reports
        if query["diagnostics"]["lexical"].get(
            "chunks_contributing_to_hybrid_top_k", 0
        )
        > 0
    )
    queries_with_keyword_candidates = sum(
        1
        for query in query_reports
        if query["diagnostics"]["keyword"].get("result_count", 0) > 0
    )
    queries_with_lexical_candidates = sum(
        1
        for query in query_reports
        if query["diagnostics"]["lexical"].get("result_count", 0) > 0
    )
    fusion_contributions: dict[str, Any] = {}
    if query_reports:
        for name in query_reports[0]["diagnostics"].get("fusion_contributions", {}):
            entries = [
                query["diagnostics"]["fusion_contributions"][name]
                for query in query_reports
            ]
            origin_totals: dict[str, int] = {}
            for entry in entries:
                for origin, count in entry.get("origin_counts", {}).items():
                    origin_totals[origin] = origin_totals.get(origin, 0) + count
            fusion_contributions[name] = {
                "queries_with_lexical_contribution": sum(
                    1
                    for entry in entries
                    if entry["lexical_candidate_matches"] > 0
                ),
                "queries_with_semantic_contribution": sum(
                    1
                    for entry in entries
                    if entry["semantic_candidate_matches"] > 0
                ),
                "queries_with_lexical_only_contribution": sum(
                    1 for entry in entries if entry["lexical_only_matches"] > 0
                ),
                "queries_with_semantic_only_contribution": sum(
                    1 for entry in entries if entry["semantic_only_matches"] > 0
                ),
                "total_lexical_candidate_matches": sum(
                    entry["lexical_candidate_matches"] for entry in entries
                ),
                "total_semantic_candidate_matches": sum(
                    entry["semantic_candidate_matches"] for entry in entries
                ),
                "total_lexical_only_matches": sum(
                    entry["lexical_only_matches"] for entry in entries
                ),
                "total_semantic_only_matches": sum(
                    entry["semantic_only_matches"] for entry in entries
                ),
                "origin_counts": origin_totals,
            }

    return {
        "scored_query_count": total,
        "queries_with_identical_chunk_ranking": identical_chunk,
        "queries_with_identical_source_ranking": identical_source,
        "queries_with_rank_difference": rank_difference,
        "keyword_retriever_provided": keyword_provided,
        "lexical_retriever_provided": lexical_provided,
        "queries_with_keyword_candidates": queries_with_keyword_candidates,
        "queries_with_lexical_candidates": queries_with_lexical_candidates,
        "queries_with_keyword_contribution": keyword_contribution,
        "queries_with_lexical_contribution": lexical_contribution,
        "fusion_contributions": fusion_contributions,
    }


def _saturation_warning(
    query_reports: list[dict[str, Any]],
    by_rank_unit: dict[str, Any],
    retriever_names: list[str] | None = None,
) -> str | None:
    if not query_reports:
        return None

    names = retriever_names or ["semantic", "hybrid"]
    metric_names = ("recall_at_5", "mrr", "any_hit_at_5", "all_expected_at_5")
    for rank_unit in RANK_UNITS:
        for retriever_name in names:
            metrics = by_rank_unit[rank_unit][retriever_name]
            if any(metrics[name] != 1.0 for name in metric_names):
                return None

    return (
        f"All retrievers scored 1.0 on every retrieval metric across "
        f"{len(query_reports)} answerable queries. This dataset does not "
        "discriminate between methods. Add harder negatives, distractor "
        "documents, or multi-source queries before drawing conclusions, and do "
        "not treat a perfect score as evidence of production readiness."
    )


def compare_reports(
    current: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    """Compare a report with a baseline, guarded by the corpus fingerprint."""
    reasons: list[str] = []
    current_fingerprint = (current.get("corpus") or {}).get("sha256")
    baseline_fingerprint = (baseline.get("corpus") or {}).get("sha256")

    if not current_fingerprint or not baseline_fingerprint:
        reasons.append(
            "corpus fingerprint missing in at least one report; metric deltas are "
            "informational only"
        )
        comparable = False
    elif current_fingerprint != baseline_fingerprint:
        reasons.append(
            "corpus fingerprint differs; metric deltas mix dataset and corpus "
            "changes and are informational only"
        )
        comparable = False
    else:
        comparable = True

    current_dataset = (current.get("dataset") or {}).get("id")
    baseline_dataset = (baseline.get("dataset") or {}).get("id")
    if current_dataset != baseline_dataset:
        reasons.append(
            f"dataset id differs ('{baseline_dataset}' -> '{current_dataset}')"
        )

    metric_names = (
        "recall_at_5",
        "mrr",
        "any_hit_at_5",
        "all_expected_at_5",
    )
    deltas: dict[str, Any] | None = {}
    for report_name, report in (("current", current), ("baseline", baseline)):
        if not isinstance(
            (report.get("aggregate") or {}).get("by_rank_unit"), dict
        ):
            reasons.append(
                f"{report_name} report has no chunk/source metric breakdown"
            )
            deltas = None
            break

    if deltas is not None:
        for retriever_name in ("semantic", "hybrid"):
            deltas[retriever_name] = {}
            for rank_unit in RANK_UNITS:
                current_metrics = current["aggregate"]["by_rank_unit"][rank_unit][
                    retriever_name
                ]
                baseline_metrics = baseline["aggregate"]["by_rank_unit"][rank_unit][
                    retriever_name
                ]
                deltas[retriever_name][rank_unit] = {
                    name: _metric_delta(
                        current_metrics.get(name),
                        baseline_metrics.get(name),
                    )
                    for name in metric_names
                }

    return {
        "baseline_dataset_id": baseline_dataset,
        "baseline_dataset_sha256": (baseline.get("dataset") or {}).get("sha256"),
        "baseline_corpus_sha256": baseline_fingerprint,
        "current_corpus_sha256": current_fingerprint,
        "comparable": comparable,
        "informational_only": not comparable,
        "reasons": reasons,
        "metric_deltas": deltas,
    }


def write_report(report: dict[str, Any], path: Path | str) -> None:
    """Write a deterministically ordered JSON report."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
