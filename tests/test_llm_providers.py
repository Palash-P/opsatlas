"""Tests for the provider-independent generation adapters."""

from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from uuid import uuid4

import pytest

from app.core.config import Settings
from app.services.evidence import EvidenceItem
from app.services.llm_providers import (
    ExtractiveProvider,
    GenerationConfigurationError,
    GenerationInput,
    GenerationInvalidResponseError,
    GenerationProviderError,
    GenerationRateLimitError,
    GenerationTimeoutError,
    OpenAICompatibleProvider,
    build_llm_provider,
)


def make_evidence() -> EvidenceItem:
    return EvidenceItem(
        evidence_id="E1",
        retrieval_rank=1,
        chunk_id=uuid4(),
        document_id=uuid4(),
        version_id=uuid4(),
        source_key="runbooks/payment-api.md",
        title="Payment API Runbook",
        version_number=3,
        chunk_index=0,
        authority_level=80,
        content="Check upstream timeouts before rolling back.",
        excerpt="Check upstream timeouts before rolling back.",
        estimated_tokens=13,
        score=0.8,
        score_field="hybrid_score",
        scores={"hybrid_score": 0.8},
        semantic_rank=1,
        keyword_rank=1,
        ranking_origin=None,
    )


def make_request(evidence: tuple[EvidenceItem, ...] | None = None) -> GenerationInput:
    return GenerationInput(
        question="What should I check first?",
        system_prompt="system",
        user_prompt="user",
        evidence=evidence if evidence is not None else (make_evidence(),),
        temperature=0.0,
        max_output_tokens=200,
        timeout_seconds=5.0,
    )


def test_build_provider_defaults_to_local_extractive() -> None:
    provider = build_llm_provider(Settings(llm_provider="extractive"))

    assert isinstance(provider, ExtractiveProvider)
    assert provider.name == "extractive"


def test_build_provider_requires_credentials_for_remote_provider() -> None:
    with pytest.raises(GenerationConfigurationError, match="LLM_API_BASE"):
        build_llm_provider(
            Settings(llm_provider="openai_compatible", llm_api_base=None, llm_api_key="key")
        )


def test_build_provider_rejects_unknown_provider() -> None:
    with pytest.raises(GenerationConfigurationError, match="LLM_PROVIDER"):
        build_llm_provider(Settings(llm_provider="unknown"))


def test_provider_rejects_non_http_api_base() -> None:
    with pytest.raises(GenerationConfigurationError, match="http"):
        OpenAICompatibleProvider(
            api_base="example.test/v1",
            api_key="secret-key",
            api_path="/chat/completions",
            model="test-model",
        )


def test_extractive_provider_abstains_without_evidence() -> None:
    response = ExtractiveProvider().generate(make_request(evidence=()))
    payload = json.loads(response.text)

    assert payload["abstained"] is True
    assert payload["used_evidence_ids"] == []
    assert payload["abstention_reason"] == "insufficient_evidence"
    assert response.provider == "extractive"


def test_extractive_provider_returns_valid_grounded_payload() -> None:
    response = ExtractiveProvider().generate(make_request())
    payload = json.loads(response.text)

    assert payload["abstained"] is False
    assert payload["used_evidence_ids"] == ["E1"]
    assert "Payment API Runbook" in payload["answer"]
    assert payload["uncertainty"]


def test_openai_compatible_provider_parses_choice_content(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def read(self):
            return json.dumps(
                {
                    "choices": [{"message": {"content": '{"answer": "ok"}'}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 4},
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("app.services.llm_providers.urlopen", fake_urlopen)

    provider = OpenAICompatibleProvider(
        api_base="https://example.test/v1",
        api_key="secret-key",
        api_path="/chat/completions",
        model="test-model",
    )
    response = provider.generate(make_request())

    assert response.text == '{"answer": "ok"}'
    assert response.model == "test-model"
    assert response.usage == {"prompt_tokens": 10, "completion_tokens": 4}
    assert captured["request"].full_url == "https://example.test/v1/chat/completions"
    assert captured["timeout"] == pytest.approx(5.0)
    assert captured["request"].get_header("Authorization") == "Bearer secret-key"


def test_openai_compatible_provider_maps_429_to_rate_limit(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        raise HTTPError(request.full_url, 429, "too many requests", {}, None)

    monkeypatch.setattr("app.services.llm_providers.urlopen", fake_urlopen)
    provider = OpenAICompatibleProvider(
        api_base="https://example.test/v1",
        api_key="secret-key",
        api_path="/chat/completions",
        model="test-model",
    )

    with pytest.raises(GenerationRateLimitError):
        provider.generate(make_request())


def test_openai_compatible_provider_maps_timeout(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        raise URLError(TimeoutError("timed out"))

    monkeypatch.setattr("app.services.llm_providers.urlopen", fake_urlopen)
    provider = OpenAICompatibleProvider(
        api_base="https://example.test/v1",
        api_key="secret-key",
        api_path="/chat/completions",
        model="test-model",
    )

    with pytest.raises(GenerationTimeoutError):
        provider.generate(make_request())


def test_openai_compatible_provider_marks_server_errors_retryable(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        raise HTTPError(request.full_url, 500, "server error", {}, None)

    monkeypatch.setattr("app.services.llm_providers.urlopen", fake_urlopen)
    provider = OpenAICompatibleProvider(
        api_base="https://example.test/v1",
        api_key="secret-key",
        api_path="/chat/completions",
        model="test-model",
    )

    with pytest.raises(GenerationProviderError) as excinfo:
        provider.generate(make_request())

    assert excinfo.value.retryable is True
    assert "secret-key" not in str(excinfo.value)


def test_openai_compatible_provider_rejects_invalid_json(monkeypatch) -> None:
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def read(self):
            return b"not-json"

    monkeypatch.setattr(
        "app.services.llm_providers.urlopen",
        lambda request, timeout: FakeResponse(),
    )
    provider = OpenAICompatibleProvider(
        api_base="https://example.test/v1",
        api_key="secret-key",
        api_path="/chat/completions",
        model="test-model",
    )

    with pytest.raises(GenerationInvalidResponseError):
        provider.generate(make_request())


def test_openai_compatible_provider_rejects_missing_choices(monkeypatch) -> None:
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def read(self):
            return json.dumps({"choices": []}).encode("utf-8")

    monkeypatch.setattr(
        "app.services.llm_providers.urlopen",
        lambda request, timeout: FakeResponse(),
    )
    provider = OpenAICompatibleProvider(
        api_base="https://example.test/v1",
        api_key="secret-key",
        api_path="/chat/completions",
        model="test-model",
    )

    with pytest.raises(GenerationInvalidResponseError):
        provider.generate(make_request())


def test_openai_compatible_provider_records_the_reported_model(monkeypatch) -> None:
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def read(self):
            return json.dumps(
                {
                    "model": "served-model-v2",
                    "choices": [{"message": {"content": '{"answer": "ok"}'}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 4, "junk": "x"},
                }
            ).encode("utf-8")

    monkeypatch.setattr(
        "app.services.llm_providers.urlopen",
        lambda request, timeout: FakeResponse(),
    )
    provider = OpenAICompatibleProvider(
        api_base="https://example.test/v1",
        api_key="secret-key",
        api_path="/chat/completions",
        model="test-model",
    )

    response = provider.generate(make_request())

    assert response.model == "served-model-v2"
    assert response.requested_model == "test-model"
    assert response.usage == {"prompt_tokens": 10, "completion_tokens": 4}
    assert response.capabilities["temperature"] == "enforced"
    assert response.capabilities["max_output_tokens"] == "enforced"


def test_openai_compatible_provider_does_not_retry_auth_failures(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        raise HTTPError(request.full_url, 401, "unauthorized", {}, None)

    monkeypatch.setattr("app.services.llm_providers.urlopen", fake_urlopen)
    provider = OpenAICompatibleProvider(
        api_base="https://example.test/v1",
        api_key="secret-key",
        api_path="/chat/completions",
        model="test-model",
    )

    with pytest.raises(GenerationProviderError) as excinfo:
        provider.generate(make_request())

    assert excinfo.value.retryable is False
    assert "secret-key" not in str(excinfo.value)


def test_extractive_provider_marks_limits_not_applicable() -> None:
    response = ExtractiveProvider().generate(make_request())

    assert response.capabilities["deterministic"] is True
    assert response.capabilities["token_usage"] == "not_reported"
    assert response.requested_model == response.model


def test_opencode_provider_records_requested_model_and_capabilities(monkeypatch) -> None:
    from evals.opencode_provider import OpenCodeCLIProvider, OpenCodeTaskResult

    monkeypatch.setattr(
        "evals.opencode_provider.run_opencode_task_result",
        lambda **kwargs: OpenCodeTaskResult(
            text='{"answer": "ok"}',
            usage={"input": 12, "output": 3},
            reported_model=kwargs["model"],
        ),
    )
    provider = OpenCodeCLIProvider(model="opencode/test-model")

    response = provider.generate(make_request())

    assert response.model == "opencode/test-model"
    assert response.requested_model == "opencode/test-model"
    assert response.usage == {"input": 12, "output": 3}
    assert response.capabilities["temperature"] == "not_enforced"
    assert response.capabilities["max_output_tokens"] == "soft_prompt_limit"


def test_opencode_provider_rejects_silent_model_substitution(monkeypatch) -> None:
    from evals.opencode_provider import OpenCodeCLIProvider, OpenCodeTaskResult

    monkeypatch.setattr(
        "evals.opencode_provider.run_opencode_task_result",
        lambda **kwargs: OpenCodeTaskResult(
            text='{"answer": "ok"}',
            reported_model="opencode/other-model",
        ),
    )
    provider = OpenCodeCLIProvider(model="opencode/test-model")

    with pytest.raises(GenerationProviderError, match="different model"):
        provider.generate(make_request())


def _completed(returncode: int, *, stdout: str = "", stderr: str = ""):
    import subprocess

    return subprocess.CompletedProcess(
        args=["opencode"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def test_opencode_provider_classifies_rate_limit_and_sanitizes_stderr(monkeypatch) -> None:
    from evals.opencode_provider import run_opencode_task_result

    monkeypatch.setattr(
        "evals.opencode_provider.subprocess.run",
        lambda *args, **kwargs: _completed(
            1,
            stderr=(
                "HTTP 429 too many requests; api_key=super-secret-value "
                "Authorization: Bearer abc.def.ghi"
            ),
        ),
    )

    with pytest.raises(GenerationRateLimitError) as excinfo:
        run_opencode_task_result(
            model="opencode/test-model",
            task_text="hello",
            timeout_seconds=5,
        )

    message = str(excinfo.value)
    assert "429" in message
    assert "super-secret-value" not in message
    assert "abc.def.ghi" not in message
    assert "[redacted]" in message


def test_opencode_provider_marks_non_rate_limit_exit_retryable(monkeypatch) -> None:
    from evals.opencode_provider import run_opencode_task_result

    monkeypatch.setattr(
        "evals.opencode_provider.subprocess.run",
        lambda *args, **kwargs: _completed(1, stderr="connection reset by peer"),
    )

    with pytest.raises(GenerationProviderError) as excinfo:
        run_opencode_task_result(
            model="opencode/test-model",
            task_text="hello",
            timeout_seconds=5,
        )

    assert excinfo.value.retryable is True
    assert "connection reset by peer" in str(excinfo.value)
