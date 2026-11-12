"""Read-only live checks for the grounded answer endpoint.

These tests need the seeded synthetic corpus and embedding service. They are
skipped unless ``OPSATLAS_LIVE_TESTS=1``, ``OPSATLAS_TEST_TENANT_ID``, and
``OPSATLAS_TEST_API_TOKEN`` are set. They use the deterministic local extractive
provider, so they verify wiring, authentication, and citation integrity, not
answer quality.
"""

from __future__ import annotations

import os
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from app.core import auth as auth_module
from app.core.auth import require_identity
from app.core.config import Settings
from app.main import app

LIVE = os.environ.get("OPSATLAS_LIVE_TESTS") == "1"
TENANT_VALUE = os.environ.get("OPSATLAS_TEST_TENANT_ID")
TOKEN_VALUE = os.environ.get("OPSATLAS_TEST_API_TOKEN")

pytestmark = [
    pytest.mark.skipif(not LIVE, reason="OPSATLAS_LIVE_TESTS is not 1"),
    pytest.mark.skipif(
        not TENANT_VALUE,
        reason="OPSATLAS_TEST_TENANT_ID is not set",
    ),
    pytest.mark.skipif(
        not TOKEN_VALUE,
        reason="OPSATLAS_TEST_API_TOKEN is not set",
    ),
]

client = TestClient(app)


@pytest.fixture(autouse=True)
def _live_authentication(monkeypatch):
    app.dependency_overrides.pop(require_identity, None)
    settings = Settings(
        auth_tokens={
            TOKEN_VALUE: {
                "tenant_id": TENANT_VALUE,
                "subject": "live-test",
                "roles": ["member"],
            }
        }
    )
    monkeypatch.setattr(auth_module, "get_settings", lambda: settings)
    yield


def post_answer(question: str, *, strategy: str = "hybrid") -> dict:
    response = client.post(
        "/api/v1/answer",
        headers={"Authorization": f"Bearer {TOKEN_VALUE}"},
        json={
            "tenant_id": str(UUID(TENANT_VALUE)),
            "question": question,
            "retrieval": {"strategy": strategy},
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_live_answer_citations_resolve_to_supplied_evidence() -> None:
    body = post_answer(
        "When PAY-503 is raised, which initial checks should happen before escalation?"
    )

    evidence_by_id = {
        item["evidence_id"]: item for item in body["evidence"]
    }
    assert evidence_by_id
    for citation in body["citations"]:
        evidence = evidence_by_id[citation["evidence_id"]]
        assert citation["chunk_id"] == evidence["chunk_id"]
        assert citation["document_id"] == evidence["document_id"]
        assert citation["version_id"] == evidence["version_id"]
        assert citation["source_key"] == evidence["source_key"]
        assert citation["version_number"] == evidence["version_number"]
        normalized_content = " ".join(evidence["content"].split())
        excerpt = citation["excerpt"]
        if excerpt.endswith("…"):
            excerpt = excerpt[:-1].rstrip()
        assert excerpt in normalized_content

    assert body["generation"]["provider"] == "extractive"
    assert body["generation"]["attempts"] == 1


def test_live_answer_preserves_source_and_version_metadata() -> None:
    body = post_answer(
        "Which revision of the payment troubleshooting runbook adds a "
        "database-connectivity check before retrying?"
    )

    assert body["evidence"]
    for item in body["evidence"]:
        assert item["source_key"]
        assert item["title"]
        assert item["version_number"] >= 1
        assert item["chunk_id"]
        assert item["document_id"]
        assert item["version_id"]

    version_pairs = {
        (item["source_key"], item["version_number"]) for item in body["evidence"]
    }
    assert version_pairs == {
        tuple(pair) for pair in body["diagnostics"]["version_pairs"]
    }


def test_live_lexical_answer_works_without_the_embedding_service() -> None:
    body = post_answer("What does the error-code runbook say about PAY-503?", strategy="lexical")

    assert body["retrieval"]["strategy"] == "lexical"
    assert body["evidence"]


def test_live_answer_requires_a_bearer_token() -> None:
    response = client.post(
        "/api/v1/answer",
        json={
            "tenant_id": str(UUID(TENANT_VALUE)),
            "question": "What does PAY-503 mean?",
            "retrieval": {"strategy": "lexical"},
        },
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Unauthorized"}


def test_live_answer_rejects_a_foreign_tenant_body() -> None:
    response = client.post(
        "/api/v1/answer",
        headers={"Authorization": f"Bearer {TOKEN_VALUE}"},
        json={
            "tenant_id": "00000000-0000-4000-8000-000000000000",
            "question": "What does PAY-503 mean?",
            "retrieval": {"strategy": "lexical"},
        },
    )

    assert response.status_code == 403
    assert response.json() == {"detail": "Forbidden"}
