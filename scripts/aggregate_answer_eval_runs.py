"""Aggregate repeated ``answer_eval_v1`` runs without changing any labels.

Each input directory must contain the artifacts written by
``run_answer_eval.py``, ``score_answer_eval.py``, and
``answer_eval_report.py``. The script reports per-run and cross-run metrics,
provider failure rates, latency, token counts, failure categories, and scorer
disagreements. It also records whether the frozen configuration was actually
held constant across runs.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

METRICS = (
    "answer_relevance",
    "faithfulness",
    "citation_correctness",
    "citation_completeness",
    "abstention_correctness",
    "conflict_handling",
)
REQUIRED_FILES = (
    "config.json",
    "summary.json",
    "report.json",
    "failure_analysis.json",
    "scorer_metrics.json",
    "disagreements.json",
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_summary(run_dir: Path) -> dict[str, Any]:
    missing = [name for name in REQUIRED_FILES if not (run_dir / name).exists()]
    if missing:
        raise SystemExit(f"{run_dir} is missing: {', '.join(missing)}")

    config = load_json(run_dir / "config.json")
    summary = load_json(run_dir / "summary.json")
    report = load_json(run_dir / "report.json")
    failure_analysis = load_json(run_dir / "failure_analysis.json")
    scorer_metrics = load_json(run_dir / "scorer_metrics.json")
    disagreements = load_json(run_dir / "disagreements.json")

    actual_scorers = {
        item["id"]: item["model"]
        for item in config.get("scoring", {}).get("scorers", [])
    }
    configured_scorers = {
        item["id"]: item["model"]
        for item in config.get("scoring", {}).get("configured_scorers", [])
    }
    generation = config.get("generation", {})
    report_generation = report.get("run", {}).get("generation_config", {})
    aggregate = report.get("aggregate", {})
    overall = aggregate.get("overall", {})

    return {
        "run_id": config.get("run_id") or run_dir.name,
        "run_directory": str(run_dir),
        "status": report.get("status"),
        "dataset_sha256": config.get("dataset", {}).get("sha256"),
        "corpus_fingerprint": config.get("corpus", {}).get("fingerprint"),
        "prompt_version": config.get("generation", {}).get("prompt_version"),
        "pipeline_sha256": config.get("code", {}).get("pipeline_sha256"),
        "resume": config.get("resume"),
        "retrieval_config": config.get("retrieval"),
        "generation": {
            "provider": generation.get("provider"),
            "model": generation.get("model"),
            "requested_model": generation.get("requested_model"),
            "observed_models": report_generation.get(
                "observed_models", [generation.get("model")]
            ),
            "observed_providers": report_generation.get(
                "observed_providers", [generation.get("provider")]
            ),
            "temperature": generation.get("temperature"),
            "temperature_enforced": generation.get("temperature_enforced"),
            "max_output_tokens": generation.get("max_output_tokens"),
            "max_output_tokens_enforced": generation.get(
                "max_output_tokens_enforced"
            ),
            "provider_capabilities": generation.get("provider_capabilities"),
        },
        "records": aggregate.get("sample_counts", {}).get("records"),
        "records_ok": summary.get("records_ok"),
        "records_error": summary.get("records_error"),
        "provider_failures": summary.get(
            "provider_failures",
            summary.get("records_error"),
        ),
        "generation_contract_failures": summary.get(
            "generation_contract_failures",
            0,
        ),
        "provider_failure_rate": aggregate.get(
            "provider_failure_rate",
            summary.get("provider_failure_rate"),
        ),
        "provider_attempts_total": summary.get("provider_attempts_total"),
        "provider_retries": summary.get("provider_retries"),
        "metrics": {metric: overall.get(metric) for metric in METRICS},
        "latency_ms": summary.get("latency_ms", {}),
        "tokens_estimate": summary.get("tokens_estimate", {}),
        "provider_usage_reported": summary.get("provider_usage_reported"),
        "failure_counts": failure_analysis.get("counts", {}),
        "scorer_models": actual_scorers,
        "configured_scorer_models": configured_scorers,
        "scorer_model_deviations": config.get("scoring", {}).get(
            "model_deviations", []
        ),
        "scorer_disagreements": len(disagreements)
        if isinstance(disagreements, list)
        else None,
        "inter_scorer_agreement": scorer_metrics.get("inter_scorer_agreement", {}),
    }


def _numeric(values: list[Any]) -> list[float]:
    return [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]


def _stat_block(values: list[Any]) -> dict[str, Any]:
    numeric = _numeric(values)
    if not numeric:
        return {"n": 0, "mean": None, "min": None, "max": None, "stdev": None}
    return {
        "n": len(numeric),
        "mean": round(statistics.fmean(numeric), 4),
        "min": round(min(numeric), 4),
        "max": round(max(numeric), 4),
        "stdev": round(statistics.stdev(numeric), 4) if len(numeric) > 1 else None,
    }


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    metric_blocks = {
        metric: _stat_block([run["metrics"].get(metric) for run in runs])
        for metric in METRICS
    }
    failure_labels = sorted(
        {label for run in runs for label in run["failure_counts"]}
    )
    failure_counts = {
        label: {
            "total": sum(run["failure_counts"].get(label, 0) for run in runs),
            "per_run": {
                run["run_id"]: run["failure_counts"].get(label, 0) for run in runs
            },
        }
        for label in failure_labels
    }

    frozen_fields = {
        "dataset_sha256": {run["dataset_sha256"] for run in runs},
        "corpus_fingerprint": {run["corpus_fingerprint"] for run in runs},
        "prompt_version": {run["prompt_version"] for run in runs},
        "retrieval_config": {
            json.dumps(run["retrieval_config"], sort_keys=True) for run in runs
        },
    }
    deviations = {
        field: sorted(value for value in values if value is not None)
        for field, values in frozen_fields.items()
        if len(values) > 1
    }

    total_records = sum(run["records"] or 0 for run in runs)
    total_errors = sum(run["records_error"] or 0 for run in runs)
    total_provider_failures = sum(run["provider_failures"] or 0 for run in runs)
    total_contract_failures = sum(
        run["generation_contract_failures"] or 0 for run in runs
    )
    return {
        "run_count": len(runs),
        "comparability": {
            "frozen_configuration_held": not deviations,
            "deviations": deviations,
            "note": (
                "Runs with different dataset hashes, corpus fingerprints, prompt "
                "versions, or retrieval configurations are reported but are not "
                "directly comparable."
            ),
        },
        "metric_summary": metric_blocks,
        "provider_failure": {
            "total_records": total_records,
            "total_errors": total_errors,
            "total_provider_failures": total_provider_failures,
            "total_generation_contract_failures": total_contract_failures,
            "overall_rate": (
                round(total_provider_failures / total_records, 4)
                if total_records
                else None
            ),
            "generation_contract_rate": (
                round(total_contract_failures / total_records, 4)
                if total_records
                else None
            ),
            "per_run": {
                run["run_id"]: {
                    "records": run["records"],
                    "errors": run["records_error"],
                    "provider_failures": run["provider_failures"],
                    "generation_contract_failures": run[
                        "generation_contract_failures"
                    ],
                    "rate": run["provider_failure_rate"],
                    "attempts": run["provider_attempts_total"],
                    "retries": run["provider_retries"],
                }
                for run in runs
            },
        },
        "latency_ms": {
            "p50": _stat_block([run["latency_ms"].get("p50") for run in runs]),
            "p95": _stat_block([run["latency_ms"].get("p95") for run in runs]),
            "mean": _stat_block([run["latency_ms"].get("mean") for run in runs]),
            "max": _stat_block([run["latency_ms"].get("max") for run in runs]),
        },
        "tokens_estimate": {
            "input_total": sum(
                run["tokens_estimate"].get("input") or 0 for run in runs
            ),
            "output_total": sum(
                run["tokens_estimate"].get("output") or 0 for run in runs
            ),
            "per_run": {
                run["run_id"]: run["tokens_estimate"] for run in runs
            },
        },
        "failure_counts": failure_counts,
        "scorer_disagreements": {
            "total": sum(run["scorer_disagreements"] or 0 for run in runs),
            "per_run": {
                run["run_id"]: run["scorer_disagreements"] for run in runs
            },
        },
        "scorer_models": {
            run["run_id"]: {
                "actual": run["scorer_models"],
                "configured": run["configured_scorer_models"],
                "deviations": run["scorer_model_deviations"],
            }
            for run in runs
        },
        "runs": runs,
    }


def markdown_report(aggregate: dict[str, Any]) -> str:
    runs = aggregate["runs"]
    lines = [
        "# Answer Evaluation v1 - Repeated-Run Aggregate",
        "",
        f"- Runs: {aggregate['run_count']}",
        (
            "- Frozen configuration held: "
            f"`{aggregate['comparability']['frozen_configuration_held']}`"
        ),
        "",
        "## Per-run metrics",
        "",
        "| Run | Status | Records | Provider failure rate | Relevance | Faithfulness | Citation corr. | Citation comp. | Abstention | Conflict |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in runs:
        values = [
            "n/a" if run["metrics"].get(metric) is None else f"{run['metrics'][metric]:.3f}"
            for metric in METRICS
        ]
        rate = run["provider_failure_rate"]
        lines.append(
            f"| `{run['run_id']}` | {run['status']} | {run['records']} | "
            f"{'n/a' if rate is None else f'{rate:.3f}'} | "
            + " | ".join(values)
            + " |"
        )

    lines.extend(
        [
            "",
            "## Cross-run metric summary",
            "",
            "| Metric | n | Mean | Min | Max | Stdev |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for metric in METRICS:
        block = aggregate["metric_summary"][metric]
        lines.append(
            f"| `{metric}` | {block['n']} | "
            + " | ".join(
                "n/a" if block[key] is None else f"{block[key]:.4f}"
                for key in ("mean", "min", "max", "stdev")
            )
            + " |"
        )

    failure = aggregate["provider_failure"]
    lines.extend(
        [
            "",
            "## Provider reliability",
            "",
            f"- Records: {failure['total_records']}",
            f"- Errored generations: {failure['total_errors']}",
            f"- Provider transport/envelope failures: {failure['total_provider_failures']}",
            (
                "- Generation-contract failures (invalid model output, not "
                f"provider downtime): {failure['total_generation_contract_failures']}"
            ),
            f"- Overall provider failure rate: {failure['overall_rate']}",
            (
                "- Overall generation-contract failure rate: "
                f"{failure['generation_contract_rate']}"
            ),
            "",
            "| Run | Records | Errors | Provider failures | Contract failures | Provider rate | Attempts | Retries |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for run_id, block in failure["per_run"].items():
        lines.append(
            f"| `{run_id}` | {block['records']} | {block['errors']} | "
            f"{block['provider_failures']} | {block['generation_contract_failures']} | "
            f"{block['rate']} | {block['attempts']} | {block['retries']} |"
        )

    lines.extend(
        [
            "",
            "## Latency and tokens",
            "",
            "| Measure | Mean | Min | Max | Stdev |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for label in ("p50", "p95", "mean", "max"):
        block = aggregate["latency_ms"][label]
        lines.append(
            f"| Latency {label} (ms) | "
            + " | ".join(
                "n/a" if block[key] is None else f"{block[key]:.1f}"
                for key in ("mean", "min", "max", "stdev")
            )
            + " |"
        )
    tokens = aggregate["tokens_estimate"]
    lines.append(f"| Token estimate input (total) | {tokens['input_total']} | | | |")
    lines.append(f"| Token estimate output (total) | {tokens['output_total']} | | | |")

    lines.extend(
        [
            "",
            "## Failure categories",
            "",
            "| Failure | Total |",
            "| --- | ---: |",
        ]
    )
    if aggregate["failure_counts"]:
        for label, block in sorted(aggregate["failure_counts"].items()):
            lines.append(f"| `{label}` | {block['total']} |")
    else:
        lines.append("| `none` | 0 |")

    lines.extend(
        [
            "",
            "## Scorer disagreement",
            "",
            f"- Total adjudicated disagreements: {aggregate['scorer_disagreements']['total']}",
            "",
            "| Run | Disagreements |",
            "| --- | ---: |",
        ]
    )
    for run_id, count in aggregate["scorer_disagreements"]["per_run"].items():
        lines.append(f"| `{run_id}` | {count} |")

    lines.extend(["", "## Run metadata", "", "| Run | Model | Prompt | Dataset SHA-256 | Corpus fingerprint |", "| --- | --- | --- | --- | --- |"])
    for run in runs:
        observed = ", ".join(
            f"`{model}`" for model in run["generation"]["observed_models"]
        )
        lines.append(
            f"| `{run['run_id']}` | {observed} | "
            f"`{run['prompt_version']}` | `{run['dataset_sha256']}` | "
            f"`{run['corpus_fingerprint']}` |"
        )

    lines.extend(
        [
            "",
            "## Scorer identity",
            "",
            "| Run | Scorer A actual | Scorer A configured | Scorer B actual | Scorer B configured | Deviations |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for run in runs:
        actual = run["scorer_models"]
        configured = run["configured_scorer_models"]
        deviations = (
            "; ".join(
                f"{item['scorer_id']}: configured {item['configured']}, used {item['used']}"
                for item in run["scorer_model_deviations"]
            )
            or "none"
        )
        lines.append(
            f"| `{run['run_id']}` | `{actual.get('scorer_a')}` | "
            f"`{configured.get('scorer_a')}` | `{actual.get('scorer_b')}` | "
            f"`{configured.get('scorer_b')}` | {deviations} |"
        )

    resumed = [run for run in runs if run.get("resume")]
    if resumed:
        lines.extend(["", "## Resumed runs", ""])
        for run in resumed:
            resume = run["resume"]
            lines.append(
                f"- `{run['run_id']}`: reused {resume.get('reused_records')} "
                f"records, reran {resume.get('rerun_records')} records "
                f"(previous run `{resume.get('previous_run_id')}`)."
            )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        dest="runs",
        required=True,
        help="Run directory. Repeat for each run.",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    summaries = [run_summary(Path(path)) for path in args.runs]
    aggregate = aggregate_runs(summaries)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "aggregate_runs.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "aggregate_runs.md").write_text(
        markdown_report(aggregate),
        encoding="utf-8",
    )

    print(f"Wrote {output_dir / 'aggregate_runs.json'}", flush=True)
    for metric, block in aggregate["metric_summary"].items():
        print(f"{metric}: mean={block['mean']} range={block['min']}..{block['max']}", flush=True)
    print(
        "provider failure rate: "
        f"{aggregate['provider_failure']['overall_rate']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
