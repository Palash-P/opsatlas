"""Provider-independent generation interface and concrete adapters.

Provider-specific HTTP calls live in this module. The grounded-answer business
logic in :mod:`app.services.answer_generation` only depends on the
:class:`LLMProvider` protocol, so tests and local development can run without a
paid API key.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.core.config import Settings
from app.services.evidence import EvidenceItem


class GenerationError(RuntimeError):
    """Base class for answer-generation failures."""


class GenerationConfigurationError(GenerationError):
    """Raised when the configured provider cannot be constructed."""


class GenerationTimeoutError(GenerationError):
    """Raised when a provider exceeds its configured timeout."""


class GenerationRateLimitError(GenerationError):
    """Raised when a provider rejects a request because of rate limits."""


class GenerationProviderError(GenerationError):
    """Raised when a provider returns an error response."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class GenerationInvalidResponseError(GenerationError):
    """Raised when a provider response envelope is structurally invalid."""


class InvalidModelOutputError(GenerationError):
    """Raised when the model text does not match the answer JSON contract."""


class CitationValidationError(GenerationError):
    """Raised when the model cites evidence that was not supplied."""


@dataclass(frozen=True)
class GenerationInput:
    """Everything a provider may use to produce one answer."""

    question: str
    system_prompt: str
    user_prompt: str
    evidence: tuple[EvidenceItem, ...]
    temperature: float
    max_output_tokens: int
    timeout_seconds: float


@dataclass(frozen=True)
class ProviderResponse:
    """Raw provider output plus non-sensitive call metadata."""

    text: str
    provider: str
    model: str
    usage: dict[str, int] | None = None
    requested_model: str | None = None
    capabilities: dict[str, Any] | None = None


EXTRACTIVE_CAPABILITIES: dict[str, Any] = {
    "temperature": "not_applicable",
    "max_output_tokens": "not_applicable",
    "token_usage": "not_reported",
    "model_verification": "declared",
    "deterministic": True,
}

OPENAI_COMPATIBLE_CAPABILITIES: dict[str, Any] = {
    "temperature": "enforced",
    "max_output_tokens": "enforced",
    "token_usage": "reported_when_available",
    "model_verification": "reported_when_available",
    "deterministic": False,
}


class LLMProvider(Protocol):
    """Minimal interface required by the grounded-answer workflow."""

    name: str
    model: str

    def generate(self, request: GenerationInput) -> ProviderResponse:
        """Generate one response from the supplied prompts and evidence."""


def _first_sentences(text: str, *, max_chars: int = 400) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    truncated = normalized[:max_chars].rstrip()
    if " " in truncated:
        truncated = truncated.rsplit(" ", 1)[0]
    return truncated + "…"


class ExtractiveProvider:
    """Deterministic, dependency-free local provider.

    This is not a language model. It returns a short extract from the top-ranked
    supplied evidence and labels the result as an extractive fallback. It exists
    so the complete answer path can run locally without a paid provider or API
    key. It must not be presented as a production-quality generator.
    """

    name = "extractive"
    model = "extractive-local-v1"

    def __init__(self, *, model: str | None = None) -> None:
        if model:
            self.model = model

    def generate(self, request: GenerationInput) -> ProviderResponse:
        if not request.evidence:
            payload = {
                "answer": "The available evidence is insufficient to answer this question.",
                "used_evidence_ids": [],
                "abstained": True,
                "abstention_reason": "insufficient_evidence",
                "insufficient_evidence": True,
                "conflicts": [],
                "uncertainty": "No evidence was supplied.",
            }
            return ProviderResponse(
                text=json.dumps(payload, ensure_ascii=False),
                provider=self.name,
                model=self.model,
                requested_model=self.model,
                capabilities=dict(EXTRACTIVE_CAPABILITIES),
            )

        primary = request.evidence[0]
        payload = {
            "answer": (
                f"{primary.title} (version {primary.version_number}) states: "
                f"{_first_sentences(primary.content)}"
            ),
            "used_evidence_ids": [primary.evidence_id],
            "abstained": False,
            "abstention_reason": None,
            "insufficient_evidence": False,
            "conflicts": [],
            "uncertainty": (
                "Local extractive fallback: this response copies the top-ranked "
                "evidence and does not synthesize across sources."
            ),
        }
        return ProviderResponse(
            text=json.dumps(payload, ensure_ascii=False),
            provider=self.name,
            model=self.model,
            requested_model=self.model,
            capabilities=dict(EXTRACTIVE_CAPABILITIES),
        )


class OpenAICompatibleProvider:
    """Adapter for an OpenAI-compatible chat-completions endpoint.

    The endpoint, model, timeout, token limit, and API key come from settings.
    The key is never logged or embedded in error messages.
    """

    name = "openai_compatible"

    def __init__(
        self,
        *,
        api_base: str,
        api_key: str,
        api_path: str,
        model: str,
    ) -> None:
        if not api_base.strip():
            raise GenerationConfigurationError("LLM_API_BASE is required")
        if not api_base.strip().lower().startswith(("http://", "https://")):
            raise GenerationConfigurationError(
                "LLM_API_BASE must start with http:// or https://"
            )
        if not api_key.strip():
            raise GenerationConfigurationError("LLM_API_KEY is required")
        if not model.strip():
            raise GenerationConfigurationError("LLM_MODEL is required")
        self.api_base = api_base.rstrip("/")
        self.api_path = "/" + api_path.lstrip("/")
        self.model = model
        self._api_key = api_key

    def generate(self, request: GenerationInput) -> ProviderResponse:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_output_tokens,
        }
        http_request = Request(
            f"{self.api_base}{self.api_path}",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(http_request, timeout=request.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            if exc.code == 429:
                raise GenerationRateLimitError(
                    "Answer generation provider rate limit exceeded"
                ) from exc
            if exc.code in (408, 504):
                raise GenerationTimeoutError(
                    "Answer generation provider timed out"
                ) from exc
            # Authentication/authorization failures are configuration problems,
            # not transient faults; retrying them only repeats the same failure.
            if exc.code in (401, 403):
                raise GenerationProviderError(
                    f"Answer generation provider rejected credentials (HTTP {exc.code})",
                    retryable=False,
                ) from exc
            raise GenerationProviderError(
                f"Answer generation provider returned HTTP {exc.code}",
                retryable=exc.code >= 500,
            ) from exc
        except TimeoutError as exc:
            raise GenerationTimeoutError(
                "Answer generation provider timed out"
            ) from exc
        except URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise GenerationTimeoutError(
                    "Answer generation provider timed out"
                ) from exc
            raise GenerationProviderError(
                "Could not reach answer generation provider",
                retryable=True,
            ) from exc
        except OSError as exc:
            raise GenerationProviderError(
                "Could not reach answer generation provider",
                retryable=True,
            ) from exc

        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise GenerationInvalidResponseError(
                "Provider returned invalid JSON"
            ) from exc

        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise GenerationInvalidResponseError(
                "Provider response did not contain a choice"
            )
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            parts = [
                part.get("text")
                for part in content
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            ]
            content = "".join(parts)
        if not isinstance(content, str) or not content.strip():
            raise GenerationInvalidResponseError(
                "Provider response did not contain text"
            )

        usage = payload.get("usage")
        parsed_usage = None
        if isinstance(usage, dict):
            parsed_usage = {
                key: value
                for key, value in usage.items()
                if isinstance(value, int) and not isinstance(value, bool)
            }
            if not parsed_usage:
                parsed_usage = None

        # Prefer the model the endpoint says it used. This makes a routing
        # substitution visible instead of silently attributing output to the
        # requested model.
        reported_model = payload.get("model")
        actual_model = (
            reported_model.strip()
            if isinstance(reported_model, str) and reported_model.strip()
            else self.model
        )

        return ProviderResponse(
            text=content,
            provider=self.name,
            model=actual_model,
            usage=parsed_usage,
            requested_model=self.model,
            capabilities=dict(OPENAI_COMPATIBLE_CAPABILITIES),
        )


def build_llm_provider(settings: Settings) -> LLMProvider:
    """Construct the configured provider without performing network I/O."""

    provider = settings.llm_provider.strip().lower()
    if provider == "extractive":
        return ExtractiveProvider(model=settings.llm_model)
    if provider == "openai_compatible":
        return OpenAICompatibleProvider(
            api_base=settings.llm_api_base or "",
            api_key=settings.llm_api_key or "",
            api_path=settings.llm_api_path,
            model=settings.llm_model,
        )
    raise GenerationConfigurationError(
        "LLM_PROVIDER must be one of: extractive, openai_compatible"
    )
