"""Evaluation-only LLM provider backed by the installed OpenCode CLI.

The production application keeps its provider-independent interface in
``app.services.llm_providers``. This adapter exists so the checked-in answer
evaluation can use a real model in an environment where OpenCode is already
authenticated, without adding an SDK or hardcoding a key.

It is not a production provider: it spawns a subprocess, depends on a local CLI,
and is only used by ``scripts/run_answer_eval.py`` and the scoring scripts.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.services.llm_providers import (
    GenerationConfigurationError,
    GenerationInput,
    GenerationInvalidResponseError,
    GenerationProviderError,
    GenerationRateLimitError,
    GenerationTimeoutError,
    ProviderResponse,
)

DEFAULT_AGENT = "general"
TASK_MESSAGE = (
    "Follow the attached instructions exactly. Return only the requested JSON "
    "object. Do not ask questions and do not add commentary."
)
OPENCODE_CLI_CAPABILITIES: dict[str, Any] = {
    "temperature": "not_enforced",
    "max_output_tokens": "soft_prompt_limit",
    "token_usage": "reported_when_available",
    "model_verification": "reported_when_available",
    "deterministic": False,
}

_RATE_LIMIT_MARKERS = (
    "rate limit",
    "rate-limit",
    "ratelimit",
    "429",
    "quota",
    "too many requests",
    "usage limit",
)
_SECRET_PATTERNS = (
    re.compile(r"(?i)bearer\s+[a-z0-9._~+/-]+=*"),
    re.compile(r"\bsk-[a-z0-9_-]{8,}\b", re.IGNORECASE),
    re.compile(r"(?i)(api[_-]?key|token|secret)\s*[:=]\s*\S+"),
)


def _sanitize_error_output(text: str, *, max_chars: int = 500) -> str:
    """Return a bounded diagnostic excerpt without credential-looking values."""

    cleaned = " ".join(text.split())
    for pattern in _SECRET_PATTERNS:
        cleaned = pattern.sub("[redacted]", cleaned)
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip() + "…"
    return cleaned


def _is_rate_limit_text(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _RATE_LIMIT_MARKERS)


@dataclass(frozen=True)
class OpenCodeTaskResult:
    """Assistant text plus any metadata the CLI event stream exposed."""

    text: str
    usage: dict[str, int] | None = None
    reported_model: str | None = None


def _resolve_executable(executable: str | None) -> str:
    resolved = executable or shutil.which("opencode") or shutil.which("opencode.cmd")
    if not resolved:
        raise GenerationConfigurationError(
            "The OpenCode CLI is required for the opencode_cli evaluation provider"
        )
    return resolved


def _first_int_mapping(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    parsed = {
        key: item
        for key, item in value.items()
        if isinstance(item, int) and not isinstance(item, bool)
    }
    cache = value.get("cache")
    if isinstance(cache, dict):
        for key in ("read", "write"):
            item = cache.get(key)
            if isinstance(item, int) and not isinstance(item, bool):
                parsed[f"cache_{key}_tokens"] = item
    return parsed


def _event_usage(event: dict[str, Any]) -> dict[str, int] | None:
    part = event.get("part")
    candidates = [
        event.get("usage"),
        event.get("tokens"),
        part.get("usage") if isinstance(part, dict) else None,
        part.get("tokens") if isinstance(part, dict) else None,
    ]
    for candidate in candidates:
        parsed = _first_int_mapping(candidate)
        if parsed:
            return parsed
    return None


def _event_model(event: dict[str, Any]) -> str | None:
    part = event.get("part")
    containers = [event, part] if isinstance(part, dict) else [event]
    for container in containers:
        for key in ("model", "modelID", "model_id"):
            value = container.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        provider_id = container.get("providerID") or container.get("provider_id")
        model_id = container.get("modelID") or container.get("model_id")
        if (
            isinstance(provider_id, str)
            and provider_id.strip()
            and isinstance(model_id, str)
            and model_id.strip()
        ):
            return f"{provider_id.strip()}/{model_id.strip()}"
    return None


def run_opencode_task_result(
    *,
    model: str,
    task_text: str,
    timeout_seconds: float,
    agent: str = DEFAULT_AGENT,
    executable: str | None = None,
) -> OpenCodeTaskResult:
    """Run one OpenCode completion and return text plus exposed metadata.

    The task is attached as a file because the CLI only forwards the first line
    of a multi-line positional message. No tool use is permitted; a tool event
    aborts the call rather than allowing the agent to touch the workspace.
    """

    if not model.strip():
        raise GenerationConfigurationError("An OpenCode model is required")
    if timeout_seconds <= 0:
        raise GenerationConfigurationError("timeout_seconds must be positive")

    binary = _resolve_executable(executable)
    with tempfile.TemporaryDirectory(prefix="opsatlas-answer-eval-") as tmp_dir:
        task_path = Path(tmp_dir) / "task.txt"
        task_path.write_text(task_text, encoding="utf-8")
        command = [
            binary,
            "run",
            "--model",
            model,
            "--format",
            "json",
            "--agent",
            agent,
            "--file",
            str(task_path),
            TASK_MESSAGE,
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise GenerationTimeoutError(
                "OpenCode evaluation provider timed out"
            ) from exc
        except OSError as exc:
            raise GenerationProviderError(
                "Could not start the OpenCode evaluation provider",
                retryable=True,
            ) from exc

    stderr_excerpt = _sanitize_error_output(completed.stderr or "")
    if completed.returncode != 0:
        detail = f": {stderr_excerpt}" if stderr_excerpt else ""
        message = (
            "OpenCode evaluation provider exited with code "
            f"{completed.returncode}{detail}"
        )
        if _is_rate_limit_text(stderr_excerpt):
            raise GenerationRateLimitError(message)
        raise GenerationProviderError(message, retryable=True)

    text_parts: list[str] = []
    usage: dict[str, int] | None = None
    reported_model: str | None = None
    tool_events = 0
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            event = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        event_type = event.get("type")
        if event_type == "tool":
            tool_events += 1
            continue
        if event_type == "text":
            part = event.get("part")
            text = part.get("text") if isinstance(part, dict) else None
            if isinstance(text, str):
                text_parts.append(text)
        usage = _event_usage(event) or usage
        reported_model = _event_model(event) or reported_model

    if tool_events:
        raise GenerationProviderError(
            "OpenCode evaluation provider attempted tool use; refusing to continue",
            retryable=False,
        )

    content = "".join(text_parts).strip()
    if not content:
        detail = f": {stderr_excerpt}" if stderr_excerpt else ""
        raise GenerationInvalidResponseError(
            f"OpenCode evaluation provider returned no text{detail}"
        )
    return OpenCodeTaskResult(
        text=content,
        usage=usage,
        reported_model=reported_model,
    )


def run_opencode_task(
    *,
    model: str,
    task_text: str,
    timeout_seconds: float,
    agent: str = DEFAULT_AGENT,
    executable: str | None = None,
) -> str:
    """Compatibility wrapper returning only assistant text."""

    return run_opencode_task_result(
        model=model,
        task_text=task_text,
        timeout_seconds=timeout_seconds,
        agent=agent,
        executable=executable,
    ).text


def _render_task(request: GenerationInput) -> str:
    return (
        "OpsAtlas grounded-answer evaluation task.\n"
        "You are a raw text-completion backend used by an automated evaluation. "
        "Do not use tools, do not read or write files, do not run commands, do "
        "not ask questions, and do not add commentary.\n\n"
        "=== SYSTEM INSTRUCTIONS ===\n"
        f"{request.system_prompt}\n\n"
        "=== USER REQUEST ===\n"
        f"{request.user_prompt}\n"
        "=== END USER REQUEST ===\n\n"
        f"Keep the response under {request.max_output_tokens} tokens.\n"
        "Return only the JSON object specified in the system instructions."
    )


class OpenCodeCLIProvider:
    """Evaluation-only provider implementing the production ``LLMProvider`` protocol."""

    name = "opencode_cli"

    def __init__(
        self,
        *,
        model: str,
        agent: str = DEFAULT_AGENT,
        executable: str | None = None,
    ) -> None:
        self.model = model
        self.agent = agent
        self._executable = executable

    def generate(self, request: GenerationInput) -> ProviderResponse:
        result = run_opencode_task_result(
            model=self.model,
            task_text=_render_task(request),
            timeout_seconds=request.timeout_seconds,
            agent=self.agent,
            executable=self._executable,
        )
        actual_model = result.reported_model or self.model
        if result.reported_model and result.reported_model != self.model:
            raise GenerationProviderError(
                "OpenCode evaluation provider used a different model than requested: "
                f"requested {self.model}, reported {result.reported_model}",
                retryable=False,
            )
        return ProviderResponse(
            text=result.text,
            provider=self.name,
            model=actual_model,
            usage=result.usage,
            requested_model=self.model,
            capabilities=dict(OPENCODE_CLI_CAPABILITIES),
        )
