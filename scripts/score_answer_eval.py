"""Score answer_eval_v1 predictions with one independent scorer model.

Each scorer receives the fixed rubric, labels, and one batch of predictions. It
never receives the other scorer's output. Raw model responses and invalid output
are preserved for audit.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.opencode_provider import run_opencode_task_result

RUBRIC_PATH = REPO_ROOT / "evals" / "answer_eval_v1.scoring_rubric.md"
METRICS = (
    "answer_relevance",
    "faithfulness",
    "citation_correctness",
    "citation_completeness",
    "abstention_correctness",
    "conflict_handling",
)
VALID_SCORES = (0, 0.5, 1, None)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def load_existing_scores(path: Path) -> dict[str, dict[str, Any]]:
    """Load scores already written by an interrupted scorer run."""

    if not path.exists():
        return {}
    return {
        item["record_id"]: item
        for item in load_jsonl(path)
        if isinstance(item.get("record_id"), str)
    }


def reusable_scores(
    existing: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Keep successful scores; scored-with-error records are regenerated."""

    return {
        record_id: item
        for record_id, item in existing.items()
        if not item.get("scoring_error")
    }


def write_scores_ordered(
    path: Path,
    predictions: list[dict[str, Any]],
    scores_by_id: dict[str, dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            item = scores_by_id.get(prediction["record_id"])
            if item is not None:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        handle.flush()


def strip_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def compact_prediction(prediction: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_id": prediction["record_id"],
        "category": prediction["category"],
        "question": prediction["question"],
        "answerable": prediction["answerable"],
        "expected_behavior": prediction["expected_behavior"],
        "expected_facts": prediction["expected_facts"],
        "expected_citations": prediction["expected_citations"],
        "min_required_citations": prediction["min_required_citations"],
        "conflict_expected": prediction["conflict_expected"],
        "abstention_reason_expected": prediction["abstention_reason_expected"],
        "notes": prediction["notes"],
        "prediction": {
            "status": prediction["status"],
            "error": prediction["error"],
            "answer": prediction["answer"],
            "abstained": prediction["abstained"],
            "abstention_reason": prediction["abstention_reason"],
            "insufficient_evidence": prediction["insufficient_evidence"],
            "citations": prediction["citations"],
            "conflicts": prediction["conflicts"],
            "version_analysis": prediction.get("version_analysis"),
            "uncertainty": prediction["uncertainty"],
            "evidence": [
                {
                    "evidence_id": item["evidence_id"],
                    "source_key": item["source_key"],
                    "version_number": item["version_number"],
                    "chunk_index": item["chunk_index"],
                    "content": item["content"],
                }
                for item in prediction["evidence"]
            ],
            "diagnostics": prediction["diagnostics"],
        },
    }


def build_task(rubric: str, batch: list[dict[str, Any]]) -> str:
    packet = [compact_prediction(item) for item in batch]
    return (
        "OpsAtlas answer-evaluation scoring task.\n"
        "You are an independent rubric scorer. Do not use tools. Read the rubric "
        "and score every record in the batch exactly once.\n\n"
        "=== RUBRIC ===\n"
        f"{rubric}\n\n"
        "=== BATCH ===\n"
        f"{json.dumps(packet, indent=2, ensure_ascii=False)}\n"
        "=== END BATCH ===\n\n"
        "Return exactly one JSON object with a records array. No Markdown fences, "
        "no commentary, and no omitted records."
    )


def parse_scores(raw: str, batch_ids: set[str]) -> list[dict[str, Any]]:
    payload = json.loads(strip_fence(raw))
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise TypeError("scorer output must be an object with a records list")
    scored: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in payload["records"]:
        if not isinstance(entry, dict):
            raise TypeError("each scored record must be an object")
        record_id = entry.get("record_id")
        if record_id not in batch_ids:
            raise ValueError(f"unknown or missing record_id {record_id!r}")
        if record_id in seen:
            raise ValueError(f"duplicate scored record {record_id!r}")
        seen.add(record_id)
        scores = entry.get("scores")
        if not isinstance(scores, dict):
            raise TypeError(f"{record_id}: scores must be an object")
        for metric in METRICS:
            if metric not in scores:
                raise ValueError(f"{record_id}: missing metric {metric}")
            value = scores[metric]
            if value not in VALID_SCORES:
                raise ValueError(f"{record_id}: invalid score for {metric}: {value!r}")
        claims = entry.get("claims", [])
        if not isinstance(claims, list):
            raise TypeError(f"{record_id}: claims must be a list")
        for claim in claims:
            if not isinstance(claim, dict) or not isinstance(claim.get("claim"), str):
                raise TypeError(
                    f"{record_id}: each claim must be an object with claim text"
                )
            if claim.get("citation_evidence_id") is not None and not isinstance(
                claim.get("citation_evidence_id"), str
            ):
                raise ValueError(f"{record_id}: invalid citation_evidence_id")
            if claim.get("supported") not in (True, False, None):
                raise ValueError(f"{record_id}: supported must be true, false, or null")
        rationale = entry.get("rationale", "")
        if not isinstance(rationale, str):
            raise TypeError(f"{record_id}: rationale must be a string")
        scored.append(
            {
                "record_id": record_id,
                "scores": {metric: scores[metric] for metric in METRICS},
                "claims": claims,
                "rationale": rationale,
            }
        )
    missing = batch_ids - seen
    if missing:
        raise ValueError("scorer omitted records: " + ", ".join(sorted(missing)))
    return scored


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--scorer-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=5.0,
        help="Pause before retrying a failed scoring attempt.",
    )
    parser.add_argument(
        "--inter-batch-delay-seconds",
        type=float,
        default=0.0,
        help="Pause between scoring batches to avoid free-tier rate limits.",
    )
    parser.add_argument("--executable", default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse successful scores in --output and regenerate failed batches.",
    )
    args = parser.parse_args()

    predictions_path = Path(args.predictions)
    output_dir = predictions_path.parent
    output_path = (
        Path(args.output)
        if args.output
        else output_dir / f"scores_{args.scorer_id}.jsonl"
    )
    raw_dir = output_dir / "scorer_raw" / args.scorer_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    rubric = RUBRIC_PATH.read_text(encoding="utf-8")
    predictions = load_jsonl(predictions_path)
    if not predictions:
        raise SystemExit("No predictions found")

    existing = reusable_scores(load_existing_scores(output_path)) if args.resume else {}
    existing_models = {item.get("model") for item in existing.values()}
    if existing_models and existing_models != {args.model}:
        raise SystemExit(
            "Refusing to mix scorer models: existing scores used "
            f"{sorted(existing_models)}, requested {args.model}"
        )
    if existing:
        print(
            f"Resume: reusing {len(existing)} successful scores; "
            f"regenerating {len(predictions) - len(existing)} records",
            flush=True,
        )
    pending = [
        (index, prediction)
        for index, prediction in enumerate(predictions)
        if prediction["record_id"] not in existing
    ]
    write_scores_ordered(output_path, predictions, existing)

    for start in range(0, len(pending), args.batch_size):
        if start > 0 and args.inter_batch_delay_seconds > 0:
            time.sleep(args.inter_batch_delay_seconds)
        batch_pairs = pending[start : start + args.batch_size]
        batch = [prediction for _, prediction in batch_pairs]
        batch_ids = {item["record_id"] for item in batch}
        task = build_task(rubric, batch)
        first_index = batch_pairs[0][0]
        last_index = batch_pairs[-1][0]
        batch_label = f"batch_{first_index:02d}_{last_index:02d}"
        (raw_dir / f"{batch_label}_task.json").write_text(
            json.dumps([compact_prediction(item) for item in batch], indent=2, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        print(
            f"[{args.scorer_id}] {batch_label}: {len(batch)} records",
            flush=True,
        )

        scored = None
        last_error = None
        attempts = 0
        actual_model = args.model
        requested_model = args.model
        model_verification = "requested_only"
        provider_usage = None
        for attempt in range(1, args.max_attempts + 1):
            attempts = attempt
            try:
                result = run_opencode_task_result(
                    model=args.model,
                    task_text=task,
                    timeout_seconds=args.timeout_seconds,
                    executable=args.executable,
                )
                raw = result.text
                actual_model = result.reported_model or args.model
                provider_usage = result.usage
                model_verification = (
                    "reported" if result.reported_model else "requested_only"
                )
                if result.reported_model and result.reported_model != args.model:
                    print(
                        "  model deviation: requested "
                        f"{args.model}, reported {result.reported_model}",
                        flush=True,
                    )
                (raw_dir / f"{batch_label}_attempt{attempt}.txt").write_text(
                    raw + "\n",
                    encoding="utf-8",
                )
                (raw_dir / f"{batch_label}_attempt{attempt}_metadata.json").write_text(
                    json.dumps(
                        {
                            "requested_model": requested_model,
                            "reported_model": result.reported_model,
                            "model_verification": model_verification,
                            "provider_usage": provider_usage,
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                scored = parse_scores(raw, batch_ids)
                break
            except Exception as exc:  # noqa: BLE001 - invalid output is an evaluation result
                last_error = f"{type(exc).__name__}: {exc}"
                print(f"  attempt {attempt} failed: {last_error}", flush=True)
                if attempt < args.max_attempts and args.retry_backoff_seconds > 0:
                    time.sleep(args.retry_backoff_seconds * attempt)

        if scored is None:
            for item in batch:
                existing[item["record_id"]] = {
                    "record_id": item["record_id"],
                    "scorer_id": args.scorer_id,
                    "model": actual_model,
                    "requested_model": requested_model,
                    "model_verification": model_verification,
                    "provider_usage": provider_usage,
                    "scores": dict.fromkeys(METRICS),
                    "claims": [],
                    "rationale": "",
                    "scoring_error": last_error,
                    "attempts": attempts,
                }
            write_scores_ordered(output_path, predictions, existing)
            continue

        for item in scored:
            existing[item["record_id"]] = {
                **item,
                "scorer_id": args.scorer_id,
                "model": actual_model,
                "requested_model": requested_model,
                "model_verification": model_verification,
                "provider_usage": provider_usage,
                "scoring_error": None,
                "attempts": attempts,
            }
        write_scores_ordered(output_path, predictions, existing)

    write_scores_ordered(output_path, predictions, existing)
    print(f"Wrote {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
