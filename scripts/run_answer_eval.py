"""Run answer_eval_v1 through the real answer-generation pipeline.

The script uses the same retrieval, evidence assembly, prompt, parser, and
citation validation as ``POST /api/v1/answer``. It uses the evaluation-only
OpenCode CLI provider because this environment has no paid provider key. Failed
generations are written to the predictions file instead of being dropped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.api.answers import RetrievalOptions, retrieve_for_answer
from app.core.config import Settings
from app.services.answer_generation import PROMPT_VERSION, generate_grounded_answer
from app.services.embedding_client import EXPECTED_DIMENSIONS, EXPECTED_MODEL
from app.services.evidence import assemble_evidence
from app.services.retrieval import IDENTIFIER_WEIGHT
from evals.answer_eval import (
    DATASET_PATH,
    PROFILE_PATH,
    PROVIDER_FAILURE_ERROR_TYPES,
    load_answer_eval_records,
    load_profile,
    validate_answer_eval_records,
)
from evals.opencode_provider import OPENCODE_CLI_CAPABILITIES, OpenCodeCLIProvider

RUBRIC_PATH = REPO_ROOT / "evals" / "answer_eval_v1.scoring_rubric.md"
PIPELINE_FILES = (
    "app/api/answers.py",
    "app/core/config.py",
    "app/services/answer_generation.py",
    "app/services/evidence.py",
    "app/services/llm_providers.py",
    "app/services/retrieval.py",
    "app/services/embedding_client.py",
    "evals/opencode_provider.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def pipeline_fingerprint() -> tuple[str, dict[str, str]]:
    digest = hashlib.sha256()
    file_hashes: dict[str, str] = {}
    for relative in PIPELINE_FILES:
        path = REPO_ROOT / relative
        file_hash = sha256_file(path)
        file_hashes[relative] = file_hash
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest(), file_hashes


def git_revision() -> dict[str, Any]:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.splitlines()
    except OSError:
        return {"head": None, "dirty": None, "status_lines": None}
    return {
        "head": head or None,
        "dirty": bool(status),
        "status_lines": len(status),
    }


def jsonable(value: Any) -> Any:
    if isinstance(value, UUID):
        return str(value)
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


class TracingProvider:
    """Record every provider attempt, including failed and invalid responses."""

    name = "opencode_cli"

    def __init__(self, inner: OpenCodeCLIProvider, *, max_raw_chars: int = 20_000) -> None:
        self.inner = inner
        self.model = inner.model
        self.max_raw_chars = max_raw_chars
        self.trace: list[dict[str, Any]] = []

    def generate(self, request):
        started = time.perf_counter()
        try:
            response = self.inner.generate(request)
        except Exception as exc:
            self.trace.append(
                {
                    "ok": False,
                    "latency_ms": int((time.perf_counter() - started) * 1000),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "retryable": getattr(exc, "retryable", None),
                }
            )
            raise
        raw_text = response.text
        truncated = len(raw_text) > self.max_raw_chars
        self.trace.append(
            {
                "ok": True,
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "raw_text": raw_text[: self.max_raw_chars],
                "raw_text_truncated": truncated,
                "provider": response.provider,
                "model": response.model,
                "requested_model": response.requested_model,
                "provider_usage": response.usage,
                "provider_capabilities": response.capabilities,
            }
        )
        return response


def build_config(args, records, profile) -> dict[str, Any]:
    pipeline_hash, file_hashes = pipeline_fingerprint()
    options = RetrievalOptions()
    return {
        "evaluation": "answer_eval_v1",
        "run_id": args.run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": {
            "path": str(DATASET_PATH.relative_to(REPO_ROOT)),
            "sha256": sha256_file(DATASET_PATH),
            "profile_path": str(PROFILE_PATH.relative_to(REPO_ROOT)),
            "profile_sha256": sha256_file(PROFILE_PATH),
            "rubric_path": str(RUBRIC_PATH.relative_to(REPO_ROOT)),
            "rubric_sha256": sha256_file(RUBRIC_PATH),
            "record_count": len(records),
            "limit_applied": args.limit,
        },
        "corpus": {
            "id": profile["corpus_id"],
            "fingerprint": profile["corpus_fingerprint"],
            "synthetic_only": profile["synthetic_only"],
        },
        "code": {
            "git": git_revision(),
            "pipeline_sha256": pipeline_hash,
            "files": file_hashes,
        },
        "retrieval": {
            "strategy": options.strategy,
            "keyword_strategy": options.keyword_strategy,
            "fusion_strategy": options.fusion_strategy,
            "semantic_weight": options.semantic_weight,
            "lexical_weight": options.lexical_weight,
            "rrf_k": options.rrf_k,
            "limit": options.limit,
            "embedding_model": EXPECTED_MODEL,
            "embedding_dimensions": EXPECTED_DIMENSIONS,
            "identifier_weight": IDENTIFIER_WEIGHT,
        },
        "evidence": {
            "context_token_budget": args.context_token_budget,
            "max_evidence": args.max_evidence,
            "citation_excerpt_chars": args.citation_excerpt_chars,
            "token_estimate": "ceil(len(text) / 4)",
        },
        "generation": {
            "provider": "opencode_cli",
            "model": args.model,
            "requested_model": args.model,
            "model_verification": "not_independently_verified",
            "temperature": args.temperature,
            "temperature_enforced": False,
            "max_output_tokens": args.max_output_tokens,
            "max_output_tokens_enforced": "soft prompt limit only",
            "timeout_seconds": args.timeout_seconds,
            "max_retries": args.max_retries,
            "retry_backoff_seconds": args.retry_backoff_seconds,
            "inter_record_delay_seconds": args.inter_record_delay_seconds,
            "prompt_version": PROMPT_VERSION,
            "provider_capabilities": dict(OPENCODE_CLI_CAPABILITIES),
            "model_substitution_policy": (
                "No substitution is permitted. The adapter reports the requested "
                "model and flags a mismatch if the CLI event stream exposes one."
            ),
        },
        "tenant_id": args.tenant_id,
        "scoring": {
            "status": "pending",
            "rubric": str(RUBRIC_PATH.relative_to(REPO_ROOT)),
            "scorers": [
                {"id": "scorer_a", "model": args.scorer_a_model},
                {"id": "scorer_b", "model": args.scorer_b_model},
            ],
            "configured_scorers": [
                {"id": "scorer_a", "model": args.scorer_a_model},
                {"id": "scorer_b", "model": args.scorer_b_model},
            ],
            "independence": (
                "Each scorer runs in a separate OpenCode session and receives only "
                "the rubric, labels, and prediction. Neither scorer receives the "
                "other's scores."
            ),
        },
        "notes": [
            "The OpenCode CLI provider is evaluation-only and is not a production provider.",
            (
                "Temperature and max output tokens cannot be enforced by the CLI adapter; "
                "max output tokens is a soft prompt instruction."
            ),
            "The corpus is synthetic.",
        ],
    }


def load_existing_predictions(path: Path) -> dict[str, dict[str, Any]]:
    """Load predictions already written by an interrupted run."""

    if not path.exists():
        return {}
    loaded: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            item = json.loads(stripped)
            if isinstance(item, dict) and isinstance(item.get("record_id"), str):
                loaded[item["record_id"]] = item
    return loaded


def reusable_predictions(
    existing: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Keep only successful predictions; failed records are regenerated."""

    return {
        record_id: item
        for record_id, item in existing.items()
        if item.get("status") == "ok"
    }


def write_predictions_ordered(
    path: Path,
    records: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            item = predictions.get(record["id"])
            if item is not None:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        handle.flush()


def summarize(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [item for item in predictions if item["status"] == "ok"]
    errors = [item for item in predictions if item["status"] == "error"]
    provider_errors = [
        item
        for item in errors
        if (item.get("error") or {}).get("type") in PROVIDER_FAILURE_ERROR_TYPES
    ]
    contract_errors = [item for item in errors if item not in provider_errors]
    latencies = [item["total_latency_ms"] for item in ok]
    attempts = [item["generation"]["attempts"] or 0 for item in predictions]
    return {
        "records_total": len(predictions),
        "records_ok": len(ok),
        "records_error": len(errors),
        "provider_failures": len(provider_errors),
        "generation_contract_failures": len(contract_errors),
        "provider_failure_rate": (
            round(len(provider_errors) / len(predictions), 4)
            if predictions
            else None
        ),
        "abstained": sum(1 for item in ok if item["abstained"]),
        "abstention_rate": (
            round(sum(1 for item in ok if item["abstained"]) / len(ok), 4)
            if ok
            else None
        ),
        "conflicts": sum(1 for item in ok if item["conflicts"]),
        "provider_attempts_total": sum(attempts),
        "provider_retries": sum(max(0, value - 1) for value in attempts),
        "latency_ms": {
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "mean": statistics.fmean(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "tokens_estimate": {
            "input": sum(item["generation"]["input_tokens_estimate"] or 0 for item in ok),
            "output": sum(item["generation"]["output_tokens_estimate"] or 0 for item in ok),
        },
        "provider_usage_reported": sum(
            1 for item in ok if item["generation"]["provider_usage"] is not None
        ),
        "error_types": sorted({item["error"]["type"] for item in errors}),
    }


def run_one(
    *,
    record: dict[str, Any],
    tenant_id: UUID,
    provider: OpenCodeCLIProvider,
    settings: Settings,
    args,
) -> dict[str, Any]:
    started_at = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    tracing_provider = TracingProvider(provider)
    prediction: dict[str, Any] = {
        "record_id": record["id"],
        "category": record["category"],
        "question": record["question"],
        "answerable": record["answerable"],
        "expected_behavior": record["expected_behavior"],
        "expected_facts": record["expected_facts"],
        "expected_citations": record["expected_citations"],
        "min_required_citations": record["min_required_citations"],
        "conflict_expected": record["conflict_expected"],
        "abstention_reason_expected": record["abstention_reason"],
        "notes": record["notes"],
        "started_at": started_at,
        "status": "error",
        "error": None,
        "retrieval": None,
        "evidence": [],
        "dropped_evidence": [],
        "diagnostics": None,
        "answer": None,
        "citations": [],
        "abstained": None,
        "abstention_reason": None,
        "insufficient_evidence": None,
        "conflicts": [],
        "version_analysis": None,
        "uncertainty": None,
        "generation": {
            "provider": tracing_provider.name,
            "model": provider.model,
            "requested_model": provider.model,
            "provider_capabilities": dict(OPENCODE_CLI_CAPABILITIES),
            "prompt_version": PROMPT_VERSION,
            "latency_ms": None,
            "attempts": 0,
            "input_tokens_estimate": None,
            "output_tokens_estimate": None,
            "provider_usage": None,
        },
        "provider_trace": tracing_provider.trace,
        "total_latency_ms": None,
        "completed_at": None,
    }

    try:
        results, retrieval_config = retrieve_for_answer(
            tenant_id=tenant_id,
            question=record["question"],
            options=RetrievalOptions(),
        )
        bundle = assemble_evidence(
            results,
            context_token_budget=args.context_token_budget,
            max_items=args.max_evidence,
            citation_excerpt_chars=args.citation_excerpt_chars,
        )
        prediction["retrieval"] = {
            "config": retrieval_config,
            "result_count": len(results),
            "results": jsonable(results),
        }
        prediction["evidence"] = jsonable(bundle.items)
        prediction["dropped_evidence"] = jsonable(bundle.dropped)
        prediction["diagnostics"] = {
            "evidence_count": len(bundle.items),
            "context_tokens_estimate": bundle.context_tokens_estimate,
            "context_token_budget": bundle.context_token_budget,
            "mixed_versions": bundle.mixed_versions,
            "version_pairs": jsonable(bundle.version_pairs),
        }

        generated = generate_grounded_answer(
            question=record["question"],
            bundle=bundle,
            provider=tracing_provider,
            settings=settings,
        )
        prediction.update(
            {
                "status": "ok",
                "answer": generated.answer,
                "citations": jsonable(generated.citations),
                "abstained": generated.abstained,
                "abstention_reason": generated.abstention_reason,
                "insufficient_evidence": generated.insufficient_evidence,
                "conflicts": jsonable(generated.conflicts),
                "version_analysis": jsonable(generated.version_analysis),
                "uncertainty": generated.uncertainty,
                "generation": {
                    "provider": generated.provider,
                    "model": generated.model,
                    "requested_model": generated.requested_model,
                    "provider_capabilities": generated.provider_capabilities,
                    "prompt_version": generated.prompt_version,
                    "latency_ms": generated.latency_ms,
                    "attempts": generated.attempts,
                    "input_tokens_estimate": generated.input_tokens_estimate,
                    "output_tokens_estimate": generated.output_tokens_estimate,
                    "provider_usage": generated.provider_usage,
                },
            }
        )
    except Exception as exc:  # noqa: BLE001 - every failure must be recorded
        prediction["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "retryable": getattr(exc, "retryable", None),
        }
        prediction["generation"]["attempts"] = len(tracing_provider.trace)
    finally:
        prediction["provider_trace"] = tracing_provider.trace
        prediction["total_latency_ms"] = int((time.perf_counter() - started) * 1000)
        prediction["completed_at"] = datetime.now(UTC).isoformat()

    return prediction


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--model", default="opencode/deepseek-v4.1-flash-free")
    parser.add_argument("--scorer-a-model", default="opencode/glm-5.3-flash-free")
    parser.add_argument("--scorer-b-model", default="opencode/flint-40b-free")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=5.0,
        help="Base exponential backoff between retryable provider attempts.",
    )
    parser.add_argument(
        "--inter-record-delay-seconds",
        type=float,
        default=2.0,
        help="Pause between records to avoid free-tier provider rate limits.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=700)
    parser.add_argument("--context-token-budget", type=int, default=4000)
    parser.add_argument("--max-evidence", type=int, default=8)
    parser.add_argument("--citation-excerpt-chars", type=int, default=600)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse successful predictions in --output-dir and regenerate failures.",
    )
    args = parser.parse_args()

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    args.run_id = run_id
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else REPO_ROOT / "evals" / "runs" / f"answer_eval_v1_{run_id}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_answer_eval_records()
    profile = load_profile()
    validate_answer_eval_records(records, profile=profile)
    if args.limit is not None:
        records = records[: args.limit]

    predictions_path = output_dir / "predictions.jsonl"
    previous_config: dict[str, Any] = {}
    previous_config_path = output_dir / "config.json"
    if args.resume and previous_config_path.exists():
        previous_config = json.loads(
            previous_config_path.read_text(encoding="utf-8")
        )
    existing = reusable_predictions(
        load_existing_predictions(predictions_path)
    ) if args.resume else {}

    config = build_config(args, records, profile)
    if args.resume:
        config["resume"] = {
            "enabled": True,
            "previous_run_id": previous_config.get("run_id"),
            "reused_records": len(existing),
            "rerun_records": len(records) - len(existing),
        }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    settings = Settings(
        llm_provider="opencode_cli",
        llm_model=args.model,
        llm_temperature=args.temperature,
        llm_max_output_tokens=args.max_output_tokens,
        llm_timeout_seconds=args.timeout_seconds,
        llm_max_retries=args.max_retries,
        llm_retry_backoff_seconds=args.retry_backoff_seconds,
        answer_context_token_budget=args.context_token_budget,
        answer_max_evidence=args.max_evidence,
        answer_citation_excerpt_chars=args.citation_excerpt_chars,
    )
    provider = OpenCodeCLIProvider(model=args.model)
    tenant_id = UUID(args.tenant_id)

    predictions_by_id: dict[str, dict[str, Any]] = dict(existing)
    if existing:
        print(
            f"Resume: reusing {len(existing)} successful predictions; "
            f"regenerating {len(records) - len(existing)} records",
            flush=True,
        )
    write_predictions_ordered(predictions_path, records, predictions_by_id)

    for index, record in enumerate(records, start=1):
        if record["id"] in predictions_by_id:
            print(
                f"[{index}/{len(records)}] {record['id']} ({record['category']}) "
                "reused",
                flush=True,
            )
            continue
        if index > 1 and args.inter_record_delay_seconds > 0:
            time.sleep(args.inter_record_delay_seconds)
        print(
            f"[{index}/{len(records)}] {record['id']} ({record['category']})",
            flush=True,
        )
        prediction = run_one(
            record=record,
            tenant_id=tenant_id,
            provider=provider,
            settings=settings,
            args=args,
        )
        predictions_by_id[record["id"]] = prediction
        write_predictions_ordered(predictions_path, records, predictions_by_id)
        if prediction["status"] == "ok":
            print(
                "  ok abstained="
                f"{prediction['abstained']} citations={len(prediction['citations'])} "
                f"latency={prediction['total_latency_ms']}ms "
                f"attempts={prediction['generation']['attempts']}",
                flush=True,
            )
        else:
            print(
                "  ERROR "
                f"{prediction['error']['type']}: {prediction['error']['message'][:200]}",
                flush=True,
            )

    predictions = [predictions_by_id[record["id"]] for record in records]
    summary = summarize(predictions)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {predictions_path}", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
