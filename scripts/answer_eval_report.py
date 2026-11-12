"""Merge two independent scorings, require manual adjudication, and build the report.

The report script never fills disagreements automatically. If any metric differs
between scorer A and scorer B, it writes an adjudication template and exits. A
human or reviewer must fill ``adjudicated`` and ``note`` for every row, then rerun.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.main import app as fastapi_app
from evals.answer_eval import METRIC_DEFINITIONS, PROVIDER_FAILURE_ERROR_TYPES

METRICS = tuple(METRIC_DEFINITIONS)
VALID_SCORES = (0, 0.5, 1, None)
FAILURE_LABELS = (
    "provider_failure",
    "retrieval_failure",
    "context_budget_omission",
    "evidence_assembly_failure",
    "abstention_failure",
    "conflict_handling_failure",
    "citation_mismatch",
    "incomplete_generation",
    "generation_hallucination",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def load_scores(path: Path) -> dict[str, dict[str, Any]]:
    return {item["record_id"]: item for item in load_jsonl(path)}


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def metric_block(records: list[dict[str, Any]], key: str = "final_scores") -> dict[str, Any]:
    block: dict[str, Any] = {}
    for metric in METRICS:
        values = [
            record[key][metric]
            for record in records
            if record.get(key, {}).get(metric) is not None
        ]
        block[metric] = mean(values)
    block["n"] = len(records)
    return block


def claim_coverage(score_entries: list[dict[str, Any]]) -> dict[str, Any]:
    total = supported = unsupported = uncited = 0
    for entry in score_entries:
        for claim in entry.get("claims", []):
            total += 1
            if claim.get("supported") is True:
                supported += 1
            elif claim.get("supported") is False:
                unsupported += 1
            else:
                uncited += 1
    return {
        "claims": total,
        "supported": supported,
        "unsupported": unsupported,
        "uncited": uncited,
        "coverage": round(supported / total, 4) if total else None,
    }


def scorer_agreement(
    predictions: list[dict[str, Any]],
    scores_a: dict[str, dict[str, Any]],
    scores_b: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Exact agreement, null mismatches, and mean absolute difference per metric.

    Exact agreement is computed over every record, including pairs where both
    scorers returned null. Mean absolute difference is computed only where both
    scorers returned a numeric value. This is descriptive, not a reliability
    coefficient.
    """

    per_metric: dict[str, dict[str, Any]] = {}
    total_exact = 0
    total_records = 0
    for metric in METRICS:
        exact = 0
        both_scored = 0
        null_mismatch = 0
        differences: list[float] = []
        for prediction in predictions:
            record_id = prediction["record_id"]
            value_a = scores_a[record_id]["scores"].get(metric)
            value_b = scores_b[record_id]["scores"].get(metric)
            if value_a == value_b:
                exact += 1
            if value_a is not None and value_b is not None:
                both_scored += 1
                differences.append(abs(float(value_a) - float(value_b)))
            elif (value_a is None) != (value_b is None):
                null_mismatch += 1
        records = len(predictions)
        total_exact += exact
        total_records += records
        per_metric[metric] = {
            "records": records,
            "exact": exact,
            "exact_rate": round(exact / records, 4) if records else None,
            "both_scored": both_scored,
            "null_mismatch": null_mismatch,
            "mean_absolute_difference": (
                round(statistics.fmean(differences), 4) if differences else None
            ),
        }
    return {
        "records": total_records,
        "exact": total_exact,
        "exact_rate": (
            round(total_exact / total_records, 4) if total_records else None
        ),
        "per_metric": per_metric,
    }


def expected_evidence_matches(prediction: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return (missing expected citations, dropped-by-budget expected citations)."""

    evidence = prediction.get("evidence", [])
    dropped = prediction.get("dropped_evidence", [])
    missing: list[str] = []
    budget_dropped: list[str] = []
    for expected in prediction.get("expected_citations", []):
        if not expected.get("must_include", True):
            continue
        source_key = expected["source_key"]
        version = expected.get("version_number")
        matched = any(
            item["source_key"] == source_key
            and (version is None or item["version_number"] == version)
            for item in evidence
        )
        if matched:
            continue
        label = f"{source_key}@v{version}" if version is not None else source_key
        missing.append(label)
        if any(
            item.get("source_key") == source_key
            and (version is None or item.get("version_number") == version)
            and item.get("reason") == "context_budget"
            for item in dropped
        ):
            budget_dropped.append(label)
    return missing, budget_dropped


def failure_attribution(
    prediction: dict[str, Any],
    final_scores: dict[str, Any],
) -> dict[str, Any]:
    failures: list[str] = []
    notes: list[str] = []
    missing, budget_dropped = expected_evidence_matches(prediction)
    diagnostics = prediction.get("diagnostics") or {}
    retrieval = prediction.get("retrieval") or {}

    if prediction["status"] == "error":
        error_type = (prediction.get("error") or {}).get("type", "")
        if error_type in PROVIDER_FAILURE_ERROR_TYPES:
            failures.append("provider_failure")
            notes.append(f"provider error: {error_type}")
        elif error_type == "CitationValidationError":
            failures.append("citation_mismatch")
            notes.append(f"citation contract error: {error_type}")
        else:
            # The provider returned, but the payload could not be parsed or
            # validated. This is an incomplete generation, not a fabricated
            # claim, and not a provider transport failure.
            failures.append("incomplete_generation")
            notes.append(f"generation contract error: {error_type}")

    if missing:
        if budget_dropped:
            failures.append("context_budget_omission")
            notes.append("expected evidence dropped by budget: " + ", ".join(budget_dropped))
        else:
            failures.append("retrieval_failure")
            notes.append("expected evidence not supplied: " + ", ".join(missing))
    if (
        retrieval.get("result_count", 0) > 0
        and diagnostics.get("evidence_count", 0) == 0
        and "context_budget_omission" not in failures
        and prediction["status"] != "error"
    ):
        failures.append("evidence_assembly_failure")
        notes.append("retrieval returned rows but no evidence survived assembly")

    if prediction["status"] == "error":
        # A provider failure is the cause; downstream answer-quality metrics do
        # not exist, so do not also label the record as a hallucination or a
        # citation error.
        ordered = [label for label in FAILURE_LABELS if label in failures]
        return {
            "primary_failure": ordered[0] if ordered else "provider_failure",
            "all_failures": ordered or ["provider_failure"],
            "notes": " ".join(notes),
        }

    if final_scores.get("abstention_correctness") == 0:
        failures.append("abstention_failure")
        notes.append("abstention did not match the label")
    if (
        final_scores.get("conflict_handling") is not None
        and final_scores["conflict_handling"] < 1
    ):
        failures.append("conflict_handling_failure")
        notes.append("conflict handling was incomplete")
    if (
        final_scores.get("citation_correctness") is not None
        and final_scores["citation_correctness"] < 1
    ):
        failures.append("citation_mismatch")
        notes.append("citation correctness was incomplete")
    if (
        final_scores.get("faithfulness") is not None
        and final_scores["faithfulness"] < 1
    ):
        failures.append("generation_hallucination")
        notes.append("faithfulness was incomplete")
    if (
        final_scores.get("answer_relevance") is not None
        and final_scores["answer_relevance"] < 1
        and "generation_hallucination" not in failures
    ):
        # A faithful but incomplete answer is a different defect from a
        # fabricated claim: it omitted a required fact or ended early.
        failures.append("incomplete_generation")
        notes.append(
            "answer omitted required content without an unsupported claim"
        )

    # De-duplicate while preserving the canonical label order.
    ordered = [label for label in FAILURE_LABELS if label in failures]
    return {
        "primary_failure": ordered[0] if ordered else "none",
        "all_failures": ordered,
        "notes": " ".join(notes),
    }


def build_adjudication_template(
    predictions: list[dict[str, Any]],
    scores_a: dict[str, dict[str, Any]],
    scores_b: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prediction in predictions:
        record_id = prediction["record_id"]
        a = scores_a[record_id]
        b = scores_b[record_id]
        for metric in METRICS:
            value_a = a["scores"].get(metric)
            value_b = b["scores"].get(metric)
            if value_a == value_b:
                continue
            rows.append(
                {
                    "record_id": record_id,
                    "metric": metric,
                    "scorer_a": value_a,
                    "scorer_b": value_b,
                    "question": prediction["question"],
                    "category": prediction["category"],
                    "answer_excerpt": (prediction.get("answer") or "")[:500],
                    "rationale_a": a.get("rationale", ""),
                    "rationale_b": b.get("rationale", ""),
                    "adjudicated": None,
                    "resolved": False,
                    "note": "",
                }
            )
    return rows


def resolve_scores(
    predictions: list[dict[str, Any]],
    scores_a: dict[str, dict[str, Any]],
    scores_b: dict[str, dict[str, Any]],
    adjudication: dict[tuple[str, str], dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    final: dict[str, dict[str, Any]] = {}
    disagreements: list[dict[str, Any]] = []
    for prediction in predictions:
        record_id = prediction["record_id"]
        entry_a = scores_a[record_id]
        entry_b = scores_b[record_id]
        final_scores: dict[str, Any] = {}
        for metric in METRICS:
            value_a = entry_a["scores"].get(metric)
            value_b = entry_b["scores"].get(metric)
            if value_a == value_b:
                final_scores[metric] = value_a
                continue
            row = adjudication.get((record_id, metric))
            if row is None:
                raise SystemExit(
                    f"Missing adjudication for {record_id}/{metric}: "
                    f"A={value_a!r}, B={value_b!r}"
                )
            if row.get("resolved") is not True:
                raise SystemExit(
                    f"Unfilled adjudication for {record_id}/{metric}: set "
                    "resolved=true and provide adjudicated plus note"
                )
            adjudicated = row.get("adjudicated")
            if adjudicated not in VALID_SCORES:
                raise SystemExit(
                    f"Invalid adjudicated score for {record_id}/{metric}: "
                    f"{adjudicated!r}"
                )
            if not str(row.get("note", "")).strip():
                raise SystemExit(
                    f"Adjudication note is required for {record_id}/{metric}"
                )
            final_scores[metric] = adjudicated
            disagreements.append(
                {
                    "record_id": record_id,
                    "metric": metric,
                    "scorer_a": value_a,
                    "scorer_b": value_b,
                    "adjudicated": adjudicated,
                    "note": row["note"],
                }
            )
        final[record_id] = final_scores
    return final, disagreements


def record_scores(
    predictions: list[dict[str, Any]],
    final: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prediction in predictions:
        record_id = prediction["record_id"]
        rows.append(
            {
                "prediction": prediction,
                "final_scores": final[record_id],
            }
        )
    return rows


def build_report(
    *,
    config: dict[str, Any],
    predictions: list[dict[str, Any]],
    scores_a: dict[str, dict[str, Any]],
    scores_b: dict[str, dict[str, Any]],
    final: dict[str, dict[str, Any]],
    disagreements: list[dict[str, Any]],
    adjudication_template_count: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    rows = record_scores(predictions, final)
    ok = [item for item in rows if item["prediction"]["status"] == "ok"]
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in rows:
        by_category[item["prediction"]["category"]].append(item)

    failure_rows: list[dict[str, Any]] = []
    failure_counts: Counter[str] = Counter()
    for item in rows:
        prediction = item["prediction"]
        attribution = failure_attribution(prediction, item["final_scores"])
        if attribution["primary_failure"] != "none":
            failure_counts[attribution["primary_failure"]] += 1
        missing, budget_dropped = expected_evidence_matches(prediction)
        failure_rows.append(
            {
                "record_id": prediction["record_id"],
                "category": prediction["category"],
                "question": prediction["question"],
                "status": prediction["status"],
                "error": prediction.get("error"),
                "primary_failure": attribution["primary_failure"],
                "all_failures": attribution["all_failures"],
                "failure_notes": attribution["notes"],
                "missing_expected_evidence": missing,
                "budget_dropped_expected_evidence": budget_dropped,
                "final_scores": item["final_scores"],
                "scorer_a": scores_a[prediction["record_id"]]["scores"],
                "scorer_b": scores_b[prediction["record_id"]]["scores"],
                "scorer_a_rationale": scores_a[prediction["record_id"]].get("rationale", ""),
                "scorer_b_rationale": scores_b[prediction["record_id"]].get("rationale", ""),
            }
        )

    report_records: list[dict[str, Any]] = []
    for item in rows:
        prediction = item["prediction"]
        attribution = failure_attribution(prediction, item["final_scores"])
        notes = {
            "status": prediction["status"],
            "error": prediction.get("error"),
            "primary_failure": attribution["primary_failure"],
            "all_failures": attribution["all_failures"],
            "failure_notes": attribution["notes"],
            "scorer_a_rationale": scores_a[prediction["record_id"]].get("rationale", ""),
            "scorer_b_rationale": scores_b[prediction["record_id"]].get("rationale", ""),
        }
        report_records.append(
            {
                "record_id": prediction["record_id"],
                "category": prediction["category"],
                "question": prediction["question"],
                "answerable": prediction["answerable"],
                "prediction": {
                    "answer": prediction.get("answer") or "",
                    "abstained": bool(prediction.get("abstained")),
                    "abstention_reason": prediction.get("abstention_reason"),
                    "citations": [
                        {
                            "source_key": citation["source_key"],
                            "version_number": citation["version_number"],
                            "chunk_id": str(citation["chunk_id"]),
                            "excerpt": citation["excerpt"],
                        }
                        for citation in prediction.get("citations", [])
                    ],
                    "conflicts": [
                        {
                            "evidence_ids": list(conflict["evidence_ids"]),
                            "description": conflict["description"],
                        }
                        for conflict in prediction.get("conflicts", [])
                    ],
                    "version_analysis": prediction.get("version_analysis"),
                    "uncertainty": prediction.get("uncertainty"),
                },
                "scores": item["final_scores"],
                "scoring_notes": json.dumps(notes, ensure_ascii=False),
            }
        )

    provider_failures = sum(
        1
        for item in rows
        if item["prediction"]["status"] == "error"
        and (item["prediction"].get("error") or {}).get("type")
        in PROVIDER_FAILURE_ERROR_TYPES
    )
    generation_contract_failures = sum(
        1
        for item in rows
        if item["prediction"]["status"] == "error"
        and (item["prediction"].get("error") or {}).get("type")
        not in PROVIDER_FAILURE_ERROR_TYPES
    )
    aggregate = {
        "status": "completed" if not any(
            scores_a[item["prediction"]["record_id"]].get("scoring_error")
            or scores_b[item["prediction"]["record_id"]].get("scoring_error")
            for item in rows
        ) else "partial",
        "sample_counts": {
            "records": len(rows),
            "scored": sum(1 for item in rows if item["final_scores"]),
            "abstained": sum(1 for item in ok if item["prediction"].get("abstained")),
            "answerable": sum(1 for item in rows if item["prediction"]["answerable"]),
            "unanswerable": sum(1 for item in rows if not item["prediction"]["answerable"]),
            "provider_failures": provider_failures,
            "generation_contract_failures": generation_contract_failures,
        },
        "provider_failure_rate": (
            round(provider_failures / len(rows), 4) if rows else None
        ),
        "overall": metric_block(rows),
        "by_category": {
            category: metric_block(category_rows)
            for category, category_rows in sorted(by_category.items())
        },
        "caveats": [
            "Metric values are means over non-null scores. The block n is the record count, not the per-metric denominator.",
            "Scores are independent LLM-assisted rubric judgments, not human scoring of every claim.",
            "The corpus is synthetic and small; these results do not establish production quality.",
            "The OpenCode CLI provider cannot enforce temperature or a hard output-token limit.",
        ],
    }

    scorer_metrics = {
        "scorer_a": {
            "model": scores_a[next(iter(scores_a))]["model"] if scores_a else None,
            "model_verification": (
                scores_a[next(iter(scores_a))].get("model_verification")
                if scores_a
                else None
            ),
            "aggregate": metric_block(
                [
                    {"scores": scores_a[item["prediction"]["record_id"]]["scores"]}
                    for item in rows
                ],
                key="scores",
            ),
            "claim_coverage": claim_coverage(
                [scores_a[item["prediction"]["record_id"]] for item in rows]
            ),
        },
        "scorer_b": {
            "model": scores_b[next(iter(scores_b))]["model"] if scores_b else None,
            "model_verification": (
                scores_b[next(iter(scores_b))].get("model_verification")
                if scores_b
                else None
            ),
            "aggregate": metric_block(
                [
                    {"scores": scores_b[item["prediction"]["record_id"]]["scores"]}
                    for item in rows
                ],
                key="scores",
            ),
            "claim_coverage": claim_coverage(
                [scores_b[item["prediction"]["record_id"]] for item in rows]
            ),
        },
        "inter_scorer_agreement": scorer_agreement(predictions, scores_a, scores_b),
    }

    failure_analysis = {
        "counts": dict(sorted(failure_counts.items())),
        "records": failure_rows,
        "method": (
            "Primary failure order: provider, retrieval/context, evidence assembly, "
            "abstention, conflict, citation, incomplete generation, hallucination. "
            "A record can have multiple failures; the report records all of them "
            "and counts the first. Incomplete generation means required content was "
            "omitted without an unsupported claim; hallucination means at least one "
            "claim was not supported by the supplied evidence."
        ),
    }

    report = {
        "report_version": 1,
        "dataset_id": "answer_eval_v1",
        "status": aggregate["status"],
        "run": {
            "system_name": "OpsAtlas",
            "system_version": fastapi_app.version,
            "retrieval_config": config["retrieval"],
            "generation_config": {
                **config["generation"],
                "observed_models": sorted(
                    {
                        item["generation"]["model"]
                        for item in predictions
                        if item.get("generation", {}).get("model")
                    }
                ),
                "observed_providers": sorted(
                    {
                        item["generation"]["provider"]
                        for item in predictions
                        if item.get("generation", {}).get("provider")
                    }
                ),
            },
            "prompt_version": config["generation"]["prompt_version"],
            "started_at": min(
                item["started_at"] for item in predictions if item.get("started_at")
            ),
            "completed_at": max(
                item["completed_at"] for item in predictions if item.get("completed_at")
            ),
        },
        "failure_taxonomy": list(FAILURE_LABELS),
        "metric_definitions": dict(METRIC_DEFINITIONS),
        "records": report_records,
        "aggregate": aggregate,
        "limitations": [
            "The corpus is synthetic; no production validation is claimed.",
            "Scoring was performed by two language models using a fixed rubric, not by two human domain experts.",
            "Claim-level citation coverage is reported per scorer; disagreements were adjudicated at the record-metric level.",
            "The evaluation provider is an evaluation-only OpenCode CLI adapter.",
            "Provider token usage was not exposed by the CLI; token counts are character-based estimates.",
            "Scorer model identity is the requested model unless the CLI event stream reports a different one.",
            "Repeated runs characterize this provider and model only; they do not establish stability across provider versions or model updates.",
        ],
    }

    return report, scorer_metrics, failure_analysis, {
        "disagreements": disagreements,
        "template_rows": adjudication_template_count,
    }


def markdown_report(
    report: dict[str, Any],
    scorer_metrics: dict[str, Any],
    failure_analysis: dict[str, Any],
    config: dict[str, Any],
) -> str:
    overall = report["aggregate"]["overall"]
    counts = report["aggregate"]["sample_counts"]
    latency = config.get("latency_summary", {})
    latency_text = " / ".join(
        f"{latency[key]:.1f}"
        if isinstance(latency.get(key), (int, float))
        else "n/a"
        for key in ("p50", "p95", "max")
    )
    lines = [
        "# Answer Evaluation v1 - Run Report",
        "",
        f"- Dataset: `{report['dataset_id']}`",
        f"- Status: `{report['status']}`",
        (
            f"- Records: {counts['records']} "
            f"({counts['answerable']} answerable, {counts['unanswerable']} unanswerable)"
        ),
        f"- Provider/model: `{config['generation']['provider']}` / `{config['generation']['model']}`",
        f"- Prompt version: `{config['generation']['prompt_version']}`",
        f"- Dataset SHA-256: `{config['dataset']['sha256']}`",
        f"- Corpus fingerprint: `{config['corpus']['fingerprint']}`",
        f"- Pipeline SHA-256: `{config['code']['pipeline_sha256']}`",
        "",
        "## Overall metrics",
        "",
        "| Metric | Score |",
        "| --- | ---: |",
    ]
    for metric in METRICS:
        value = overall[metric]
        lines.append(f"| `{metric}` | {'n/a' if value is None else f'{value:.3f}'} |")
    lines.extend(
        [
            "",
            "`n` is the number of records in the block. Metric values average only non-null scores.",
            "",
            "## Per-category metrics",
            "",
            "| Category | n | Relevance | Faithfulness | Citation correctness | Citation completeness | Abstention | Conflict |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for category, block in report["aggregate"]["by_category"].items():
        values = [
            "n/a" if block[metric] is None else f"{block[metric]:.2f}"
            for metric in METRICS
        ]
        lines.append(
            f"| `{category}` | {block['n']} | " + " | ".join(values) + " |"
        )
    lines.extend(
        [
            "",
            "## Independent scorers",
            "",
            "| Scorer | Model | Relevance | Faithfulness | Citation correctness | Citation completeness | Abstention | Conflict | Claim coverage |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for scorer_id in ("scorer_a", "scorer_b"):
        scorer = scorer_metrics[scorer_id]
        block = scorer["aggregate"]
        values = [
            "n/a" if block[metric] is None else f"{block[metric]:.2f}"
            for metric in METRICS
        ]
        coverage = scorer["claim_coverage"]["coverage"]
        lines.append(
            f"| {scorer_id} | `{scorer['model']}` | "
            + " | ".join(values)
            + f" | {'n/a' if coverage is None else f'{coverage:.2f}'} |"
        )
    agreement = scorer_metrics["inter_scorer_agreement"]
    lines.extend(
        [
            "",
            (
                "Inter-scorer exact agreement: "
                f"{agreement['exact']}/{agreement['records']} "
                f"({agreement['exact_rate'] if agreement['exact_rate'] is not None else 'n/a'}). "
                "Disagreements are adjudicated in `disagreements.json`."
            ),
        ]
    )
    lines.extend(
        [
            "",
            "## Failure attribution",
            "",
            "| Primary failure | Records |",
            "| --- | ---: |",
        ]
    )
    if failure_analysis["counts"]:
        for label, count in failure_analysis["counts"].items():
            lines.append(f"| `{label}` | {count} |")
    else:
        lines.append("| `none` | 0 |")
    lines.extend(
        [
            "",
            "## Provider latency and usage",
            "",
            (
                f"- Latency p50/p95/max: {latency_text} ms"
            ),
            (
                f"- Provider attempts: {config.get('provider_attempts_total', 'n/a')}; "
                f"retries: {config.get('provider_retries', 'n/a')}"
            ),
            (
                f"- Provider failure rate: "
                f"{report['aggregate'].get('provider_failure_rate', 'n/a')}"
            ),
            (
                "- Generation-contract failures: "
                f"{report['aggregate'].get('sample_counts', {}).get('generation_contract_failures', 'n/a')}"
            ),
            (
                f"- Token estimates (input/output): {config.get('tokens_estimate', {}).get('input', 'n/a')} / "
                f"{config.get('tokens_estimate', {}).get('output', 'n/a')}"
            ),
            "",
            "## Per-question results",
            "",
            "| Record | Category | Status | Abstained | Primary failure | Relevance | Faithfulness | Citation corr. | Citation comp. | Abstention | Conflict |",
            "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    failures_by_id = {
        item["record_id"]: item for item in failure_analysis["records"]
    }
    for record in report["records"]:
        scores = record["scores"]
        failure = failures_by_id[record["record_id"]]
        values = [
            "n/a" if scores[metric] is None else f"{scores[metric]:.1f}"
            for metric in METRICS
        ]
        lines.append(
            f"| `{record['record_id']}` | `{record['category']}` | "
            f"{failure['status']} | {record['prediction']['abstained']} | "
            f"`{failure['primary_failure']}` | " + " | ".join(values) + " |"
        )
    lines.extend(
        [
            "",
            "## Limitations and recommendations",
            "",
        ]
    )
    for limitation in report["limitations"]:
        lines.append(f"- {limitation}")
    lines.extend(
        [
            "",
            "Recommendations are recorded in `docs/ANSWER_EVALUATION_V1.md` after reviewing this report.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--scores-a", required=True)
    parser.add_argument("--scores-b", required=True)
    parser.add_argument("--adjudication", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    predictions_path = Path(args.predictions)
    output_dir = Path(args.output_dir) if args.output_dir else predictions_path.parent
    predictions = load_jsonl(predictions_path)
    scores_a = load_scores(Path(args.scores_a))
    scores_b = load_scores(Path(args.scores_b))
    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))

    missing_a = [item["record_id"] for item in predictions if item["record_id"] not in scores_a]
    missing_b = [item["record_id"] for item in predictions if item["record_id"] not in scores_b]
    if missing_a or missing_b:
        raise SystemExit(
            f"Missing scores: scorer_a={missing_a}, scorer_b={missing_b}"
        )

    first_prediction = predictions[0]["record_id"]
    actual_models = {
        "scorer_a": scores_a[first_prediction]["model"],
        "scorer_b": scores_b[first_prediction]["model"],
    }
    actual_requested_models = {
        "scorer_a": scores_a[first_prediction].get("requested_model"),
        "scorer_b": scores_b[first_prediction].get("requested_model"),
    }
    actual_verifications = {
        "scorer_a": scores_a[first_prediction].get("model_verification"),
        "scorer_b": scores_b[first_prediction].get("model_verification"),
    }
    configured_scorers = config["scoring"].get(
        "configured_scorers", config["scoring"]["scorers"]
    )
    configured_models = {
        item["id"]: item["model"] for item in configured_scorers
    }
    config["scoring"]["configured_scorers"] = configured_scorers
    config["scoring"]["scorers"] = [
        {
            "id": scorer_id,
            "model": actual_models[scorer_id],
            "requested_model": actual_requested_models[scorer_id],
            "model_verification": actual_verifications[scorer_id],
        }
        for scorer_id in ("scorer_a", "scorer_b")
    ]
    config["scoring"]["model_deviations"] = [
        {
            "scorer_id": scorer_id,
            "configured": configured_models.get(scorer_id),
            "used": actual_models[scorer_id],
        }
        for scorer_id in ("scorer_a", "scorer_b")
        if configured_models.get(scorer_id) != actual_models[scorer_id]
    ]

    template = build_adjudication_template(predictions, scores_a, scores_b)
    adjudication_path = Path(args.adjudication)
    if not adjudication_path.exists():
        with adjudication_path.open("w", encoding="utf-8") as handle:
            for row in template:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(
            f"Wrote {len(template)} disagreement rows to {adjudication_path}. "
            "Fill adjudicated and note for every row, then rerun.",
            flush=True,
        )
        return 2

    rows = load_jsonl(adjudication_path)
    adjudication = {
        (row["record_id"], row["metric"]): row
        for row in rows
    }
    final, disagreements = resolve_scores(predictions, scores_a, scores_b, adjudication)
    report, scorer_metrics, failure_analysis, adjudication_summary = build_report(
        config=config,
        predictions=predictions,
        scores_a=scores_a,
        scores_b=scores_b,
        final=final,
        disagreements=disagreements,
        adjudication_template_count=len(template),
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    config["latency_summary"] = summary["latency_ms"]
    config["provider_attempts_total"] = summary["provider_attempts_total"]
    config["provider_retries"] = summary["provider_retries"]
    config["tokens_estimate"] = summary["tokens_estimate"]
    config["scoring"]["status"] = "completed" if report["status"] == "completed" else "partial"
    config["scoring"]["template_disagreements"] = adjudication_summary["template_rows"]
    config["scoring"]["adjudicated_disagreements"] = len(disagreements)

    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "scorer_metrics.json").write_text(
        json.dumps(scorer_metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "failure_analysis.json").write_text(
        json.dumps(failure_analysis, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "disagreements.json").write_text(
        json.dumps(disagreements, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        markdown_report(report, scorer_metrics, failure_analysis, config),
        encoding="utf-8",
    )
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {output_dir / 'report.json'}", flush=True)
    print(f"Status: {report['status']}", flush=True)
    print(json.dumps(report["aggregate"]["overall"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
