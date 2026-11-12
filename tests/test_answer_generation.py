"""Tests for grounded answer generation, parsing, and citation validation."""

from __future__ import annotations

import json
from uuid import uuid4

import pytest

from app.core.config import Settings
from app.services.answer_generation import (
    build_prompts,
    generate_grounded_answer,
    parse_model_answer,
)
from app.services.evidence import EvidenceBundle, EvidenceItem, assemble_evidence
from app.services.llm_providers import (
    CitationValidationError,
    GenerationConfigurationError,
    GenerationInput,
    GenerationProviderError,
    GenerationTimeoutError,
    InvalidModelOutputError,
    ProviderResponse,
)


def make_item(
    evidence_id: str,
    *,
    chunk_index: int = 0,
    content: str = "Check upstream timeouts.",
    source_key: str = "runbooks/payment-api.md",
    version_number: int = 1,
) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        retrieval_rank=chunk_index + 1,
        chunk_id=uuid4(),
        document_id=uuid4(),
        version_id=uuid4(),
        source_key=source_key,
        title="Payment API Runbook",
        version_number=version_number,
        chunk_index=chunk_index,
        authority_level=80,
        content=content,
        excerpt=content,
        estimated_tokens=6,
        score=0.8,
        score_field="hybrid_score",
        scores={"hybrid_score": 0.8},
        semantic_rank=1,
        keyword_rank=1,
        ranking_origin=None,
    )


def make_bundle(*items: EvidenceItem) -> EvidenceBundle:
    items = items or (make_item("E1"),)
    context_text = "\n".join(f'<evidence id="{item.evidence_id}">{item.content}</evidence>' for item in items)
    return EvidenceBundle(
        items=items,
        dropped=(),
        context_text=context_text,
        context_tokens_estimate=len(context_text) // 4,
        context_token_budget=4000,
        mixed_versions=False,
        version_pairs=tuple((item.source_key, item.version_number) for item in items),
    )


def make_empty_bundle() -> EvidenceBundle:
    return EvidenceBundle(
        items=(),
        dropped=(),
        context_text="",
        context_tokens_estimate=0,
        context_token_budget=4000,
        mixed_versions=False,
        version_pairs=(),
    )


def answer_payload(**overrides) -> str:
    payload = {
        "answer": "Check upstream timeouts.",
        "used_evidence_ids": ["E1"],
        "abstained": False,
        "abstention_reason": None,
        "insufficient_evidence": False,
        "conflicts": [],
        "uncertainty": None,
    }
    payload.update(overrides)
    return json.dumps(payload)


class FakeProvider:
    name = "fake"
    model = "fake-model"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[GenerationInput] = []

    def generate(self, request):
        self.calls.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return ProviderResponse(text=response, provider=self.name, model=self.model)


def test_prompts_include_evidence_and_allowed_ids() -> None:
    system_prompt, user_prompt = build_prompts("Why is checkout failing?", make_bundle())

    assert "ONLY the evidence" in system_prompt
    assert "abstain" in system_prompt.lower()
    assert "conflict" in system_prompt.lower()
    assert "Why is checkout failing?" in user_prompt
    assert "<evidence id=\"E1\">" in user_prompt
    assert "Allowed evidence IDs: E1" in user_prompt


def test_parse_accepts_fenced_json() -> None:
    bundle = make_bundle()
    raw = "```json\n" + answer_payload() + "\n```"

    parsed = parse_model_answer(raw, bundle)

    assert parsed.answer == "Check upstream timeouts."
    assert parsed.used_evidence_ids == ("E1",)
    assert parsed.abstained is False


def test_parse_rejects_invalid_json() -> None:
    with pytest.raises(InvalidModelOutputError, match="invalid JSON"):
        parse_model_answer("not json", make_bundle())


def test_parse_rejects_missing_answer() -> None:
    with pytest.raises(InvalidModelOutputError, match="answer"):
        parse_model_answer(answer_payload(answer="  "), make_bundle())


def test_parse_rejects_non_boolean_abstained() -> None:
    with pytest.raises(InvalidModelOutputError, match="abstained"):
        parse_model_answer(answer_payload(abstained="yes"), make_bundle())


def test_parse_rejects_fabricated_evidence_id() -> None:
    with pytest.raises(CitationValidationError, match="E9"):
        parse_model_answer(answer_payload(used_evidence_ids=["E9"]), make_bundle())


def test_parse_rejects_fabricated_conflict_evidence_id() -> None:
    payload = answer_payload(
        conflicts=[{"evidence_ids": ["E1", "E9"], "description": "versions conflict"}]
    )

    with pytest.raises(CitationValidationError, match="E9"):
        parse_model_answer(payload, make_bundle())


def test_parse_requires_conflict_description() -> None:
    payload = answer_payload(conflicts=[{"evidence_ids": ["E1"], "description": " "}])

    with pytest.raises(InvalidModelOutputError, match="description"):
        parse_model_answer(payload, make_bundle())


def test_parse_rejects_non_abstaining_answer_without_citations() -> None:
    with pytest.raises(InvalidModelOutputError, match="cite"):
        parse_model_answer(answer_payload(used_evidence_ids=[]), make_bundle())


def test_parse_accepts_abstention_without_citations() -> None:
    payload = answer_payload(
        answer="Insufficient evidence.",
        used_evidence_ids=[],
        abstained=True,
        abstention_reason="insufficient_evidence",
        insufficient_evidence=True,
    )

    parsed = parse_model_answer(payload, make_bundle())

    assert parsed.abstained is True
    assert parsed.used_evidence_ids == ()


def test_parse_rejects_abstention_with_citations() -> None:
    payload = answer_payload(
        used_evidence_ids=["E1"],
        abstained=True,
        abstention_reason="insufficient_evidence",
    )

    with pytest.raises(InvalidModelOutputError, match="must not cite"):
        parse_model_answer(payload, make_bundle())


def test_parse_rejects_abstention_without_reason() -> None:
    payload = answer_payload(
        used_evidence_ids=[],
        abstained=True,
        abstention_reason=None,
    )

    with pytest.raises(InvalidModelOutputError, match="abstention_reason"):
        parse_model_answer(payload, make_bundle())


def test_generate_returns_validated_citations() -> None:
    bundle = assemble_evidence(
        [
            {
                "chunk_id": uuid4(),
                "document_id": uuid4(),
                "version_id": uuid4(),
                "source_key": "runbooks/payment-api.md",
                "title": "Payment API Runbook",
                "version_number": 4,
                "authority_level": 80,
                "chunk_index": 2,
                "content": "Check upstream timeouts before rolling back.",
                "hybrid_score": 0.7,
            }
        ],
        context_token_budget=4000,
    )
    provider = FakeProvider([answer_payload()])

    generated = generate_grounded_answer(
        question="What should I check?",
        bundle=bundle,
        provider=provider,
        settings=Settings(llm_max_retries=0),
    )

    assert len(generated.citations) == 1
    citation = generated.citations[0]
    assert citation.evidence_id == "E1"
    assert citation.chunk_id == bundle.items[0].chunk_id
    assert citation.document_id == bundle.items[0].document_id
    assert citation.version_number == 4
    assert citation.source_key == "runbooks/payment-api.md"
    assert generated.attempts == 1
    assert generated.prompt_version


def test_generate_abstains_without_evidence_and_skips_provider() -> None:
    provider = FakeProvider([])

    generated = generate_grounded_answer(
        question="Anything?",
        bundle=make_empty_bundle(),
        provider=provider,
        settings=Settings(),
    )

    assert generated.abstained is True
    assert generated.abstention_reason == "no_evidence"
    assert generated.citations == ()
    assert provider.calls == []


def test_generate_retries_transient_failure_with_bounded_attempts(monkeypatch) -> None:
    monkeypatch.setattr("app.services.answer_generation.time.sleep", lambda _: None)
    provider = FakeProvider(
        [GenerationTimeoutError("timed out"), answer_payload()]
    )

    generated = generate_grounded_answer(
        question="What should I check?",
        bundle=make_bundle(),
        provider=provider,
        settings=Settings(llm_max_retries=1),
    )

    assert generated.attempts == 2
    assert len(provider.calls) == 2
    assert generated.answer == "Check upstream timeouts."


def test_generate_stops_after_bounded_retries(monkeypatch) -> None:
    monkeypatch.setattr("app.services.answer_generation.time.sleep", lambda _: None)
    provider = FakeProvider(
        [
            GenerationTimeoutError("timed out"),
            GenerationTimeoutError("timed out"),
            answer_payload(),
        ]
    )

    with pytest.raises(GenerationTimeoutError):
        generate_grounded_answer(
            question="What should I check?",
            bundle=make_bundle(),
            provider=provider,
            settings=Settings(llm_max_retries=1),
        )

    assert len(provider.calls) == 2


def test_generate_does_not_retry_invalid_model_output() -> None:
    provider = FakeProvider(["not json", answer_payload()])

    with pytest.raises(InvalidModelOutputError):
        generate_grounded_answer(
            question="What should I check?",
            bundle=make_bundle(),
            provider=provider,
            settings=Settings(llm_max_retries=2),
        )

    assert len(provider.calls) == 1


def test_generate_does_not_retry_non_retryable_provider_error() -> None:
    provider = FakeProvider(
        [GenerationProviderError("bad request", retryable=False), answer_payload()]
    )

    with pytest.raises(GenerationProviderError):
        generate_grounded_answer(
            question="What should I check?",
            bundle=make_bundle(),
            provider=provider,
            settings=Settings(llm_max_retries=2),
        )

    assert len(provider.calls) == 1


def test_generate_retries_retryable_provider_error(monkeypatch) -> None:
    monkeypatch.setattr("app.services.answer_generation.time.sleep", lambda _: None)
    provider = FakeProvider(
        [GenerationProviderError("server error", retryable=True), answer_payload()]
    )

    generated = generate_grounded_answer(
        question="What should I check?",
        bundle=make_bundle(),
        provider=provider,
        settings=Settings(llm_max_retries=1),
    )

    assert generated.attempts == 2


def test_generate_preserves_conflict_and_uncertainty_signals() -> None:
    bundle = make_bundle(make_item("E1"), make_item("E2", version_number=2))
    payload = answer_payload(
        used_evidence_ids=["E1", "E2"],
        conflicts=[
            {
                "evidence_ids": ["E1", "E2"],
                "description": "Version 1 restarts workers; version 2 rolls them.",
            }
        ],
        uncertainty="The current version is not stated.",
    )
    provider = FakeProvider([payload])

    generated = generate_grounded_answer(
        question="How do I restart workers?",
        bundle=bundle,
        provider=provider,
        settings=Settings(llm_max_retries=0),
    )

    assert generated.conflicts[0].evidence_ids == ("E1", "E2")
    assert "rolls" in generated.conflicts[0].description
    assert generated.uncertainty == "The current version is not stated."
    assert len(generated.citations) == 2


def test_generate_rejects_fabricated_citation() -> None:
    provider = FakeProvider([answer_payload(used_evidence_ids=["E7"])])

    with pytest.raises(CitationValidationError):
        generate_grounded_answer(
            question="What should I check?",
            bundle=make_bundle(),
            provider=provider,
            settings=Settings(llm_max_retries=0),
        )


def test_generate_rejects_negative_retry_configuration() -> None:
    with pytest.raises(GenerationConfigurationError, match="LLM_MAX_RETRIES"):
        generate_grounded_answer(
            question="What should I check?",
            bundle=make_bundle(),
            provider=FakeProvider([answer_payload()]),
            settings=Settings(llm_max_retries=-1),
        )


def test_generate_rejects_invalid_timeout_configuration() -> None:
    with pytest.raises(GenerationConfigurationError, match="LLM_TIMEOUT_SECONDS"):
        generate_grounded_answer(
            question="What should I check?",
            bundle=make_bundle(),
            provider=FakeProvider([answer_payload()]),
            settings=Settings(llm_timeout_seconds=0),
        )


def version_analysis_payload(**overrides) -> dict:
    payload = {
        "versions_identified": [
            {
                "source_key": "runbooks/payment-api.md",
                "version_number": 1,
                "status": "historical",
                "evidence_ids": ["E1"],
            },
            {
                "source_key": "runbooks/payment-api.md",
                "version_number": 2,
                "status": "current",
                "evidence_ids": ["E2"],
            },
        ],
        "differences": [
            "Version 2 adds a database-connectivity verification step."
        ],
        "conflicting_instructions": [],
        "recommendation_applies_to": "version 2",
        "supporting_evidence_ids": ["E1", "E2"],
    }
    payload.update(overrides)
    return payload


def test_parse_accepts_mixed_version_analysis_with_supporting_versions() -> None:
    bundle = make_bundle(
        make_item("E1", version_number=1, content="Follow the rollback procedure."),
        make_item("E2", version_number=2, content="Verify database connectivity, then roll back."),
    )
    payload = answer_payload(
        used_evidence_ids=["E1", "E2"],
        version_analysis=version_analysis_payload(),
    )

    parsed = parse_model_answer(payload, bundle)

    assert parsed.version_analysis is not None
    assert [
        (reference.source_key, reference.version_number, reference.status)
        for reference in parsed.version_analysis.versions_identified
    ] == [
        ("runbooks/payment-api.md", 1, "historical"),
        ("runbooks/payment-api.md", 2, "current"),
    ]
    assert parsed.version_analysis.recommendation_applies_to == "version 2"
    assert parsed.version_analysis.supporting_evidence_ids == ("E1", "E2")
    assert "database-connectivity" in parsed.version_analysis.differences[0]


def test_parse_accepts_non_conflicting_version_analysis() -> None:
    bundle = make_bundle(
        make_item("E1", source_key="runbooks/payment-api.md", version_number=2),
        make_item("E2", source_key="api-specs/payments.md", version_number=2),
    )
    payload = answer_payload(
        used_evidence_ids=["E1", "E2"],
        version_analysis=version_analysis_payload(
            versions_identified=[
                {
                    "source_key": "runbooks/payment-api.md",
                    "version_number": 2,
                    "status": "current",
                    "evidence_ids": ["E1"],
                },
                {
                    "source_key": "api-specs/payments.md",
                    "version_number": 2,
                    "status": "current",
                    "evidence_ids": ["E2"],
                },
            ],
            differences=[],
            conflicting_instructions=[],
            recommendation_applies_to="version 2",
            supporting_evidence_ids=["E1", "E2"],
        ),
    )

    parsed = parse_model_answer(payload, bundle)

    assert parsed.version_analysis is not None
    assert parsed.version_analysis.differences == ()
    assert parsed.version_analysis.conflicting_instructions == ()
    assert len(parsed.version_analysis.versions_identified) == 2


def test_parse_version_analysis_requires_supplied_evidence_ids() -> None:
    bundle = make_bundle(make_item("E1"))
    payload = answer_payload(
        version_analysis=version_analysis_payload(
            versions_identified=[
                {
                    "source_key": "runbooks/payment-api.md",
                    "version_number": 2,
                    "status": "current",
                    "evidence_ids": ["E99"],
                }
            ]
        )
    )

    with pytest.raises(CitationValidationError, match="E99"):
        parse_model_answer(payload, bundle)


def test_parse_version_analysis_requires_evidence_per_identified_version() -> None:
    bundle = make_bundle(make_item("E1"))
    payload = answer_payload(
        version_analysis=version_analysis_payload(
            versions_identified=[
                {
                    "source_key": "runbooks/payment-api.md",
                    "version_number": 2,
                    "status": "current",
                    "evidence_ids": [],
                }
            ]
        )
    )

    with pytest.raises(InvalidModelOutputError, match="at least one supplied evidence"):
        parse_model_answer(payload, bundle)


def test_parse_version_analysis_rejects_unsupported_differences() -> None:
    bundle = make_bundle(make_item("E1"), make_item("E2", version_number=2))
    payload = answer_payload(
        used_evidence_ids=["E1", "E2"],
        version_analysis=version_analysis_payload(supporting_evidence_ids=[]),
    )

    with pytest.raises(InvalidModelOutputError, match="supporting evidence"):
        parse_model_answer(payload, bundle)


def test_parse_version_analysis_rejects_unrecognized_status() -> None:
    bundle = make_bundle(make_item("E1"))
    payload = answer_payload(
        version_analysis=version_analysis_payload(
            versions_identified=[
                {
                    "source_key": "runbooks/payment-api.md",
                    "version_number": 2,
                    "status": "imaginary",
                    "evidence_ids": ["E1"],
                }
            ]
        )
    )

    with pytest.raises(InvalidModelOutputError, match="status"):
        parse_model_answer(payload, bundle)


def test_parse_version_analysis_normalizes_status_synonyms() -> None:
    bundle = make_bundle(make_item("E1"))
    payload = answer_payload(
        version_analysis=version_analysis_payload(
            versions_identified=[
                {
                    "source_key": "runbooks/payment-api.md",
                    "version_number": 1,
                    "status": "superseded",
                    "evidence_ids": ["E1"],
                }
            ],
            differences=[],
            conflicting_instructions=[],
            supporting_evidence_ids=[],
        )
    )

    parsed = parse_model_answer(payload, bundle)

    assert parsed.version_analysis is not None
    assert parsed.version_analysis.versions_identified[0].status == "historical"


def test_parse_version_analysis_rejects_empty_identification() -> None:
    bundle = make_bundle(make_item("E1"))
    payload = answer_payload(
        version_analysis=version_analysis_payload(
            versions_identified=[],
            differences=[],
            conflicting_instructions=[],
            supporting_evidence_ids=[],
        )
    )

    with pytest.raises(InvalidModelOutputError, match="at least one supplied version"):
        parse_model_answer(payload, bundle)


def test_generate_carries_version_analysis_into_the_result() -> None:
    bundle = make_bundle(
        make_item("E1", version_number=1),
        make_item("E2", version_number=2),
    )
    provider = FakeProvider(
        [
            answer_payload(
                used_evidence_ids=["E1", "E2"],
                version_analysis=version_analysis_payload(),
            )
        ]
    )

    generated = generate_grounded_answer(
        question="How did rollback change between versions?",
        bundle=bundle,
        provider=provider,
        settings=Settings(llm_max_retries=0),
    )

    assert generated.version_analysis is not None
    assert generated.version_analysis.recommendation_applies_to == "version 2"
    assert generated.requested_model == "fake-model"


