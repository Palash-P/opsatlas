"""Optional live cross-tenant isolation checks.

These tests run only when ``OPSATLAS_LIVE_TESTS=1``,
``OPSATLAS_TEST_TENANT_ID``, and ``OPSATLAS_TEST_API_TOKEN`` are set. They seed a
throwaway tenant B document, exercise retrieval and the answer endpoint as
tenant A, then delete only the rows they created (the tenant delete cascades to
its documents, versions, and chunks).

The corpus of tenant A is never modified.
"""

from __future__ import annotations

import os
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from app.core import auth as auth_module
from app.core.auth import Principal, require_identity
from app.core.config import Settings
from app.db.connections import get_engine
from app.main import app
from app.services.ingestion import ingest_document
from app.services.retrieval import keyword_search, lexical_search

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


def tenant_a() -> UUID:
    return UUID(TENANT_VALUE)


@pytest.fixture
def tenant_b_document():
    """Seed one document owned by a fresh tenant B and remove it afterwards."""

    tenant_b = uuid4()
    canary = "zqxcanary" + uuid4().hex[:10]
    source_key = f"security-live/{canary}.md"

    engine = get_engine()
    try:
        with engine.begin() as connection:
            connection.execute(
                text("INSERT INTO tenants (id, name) VALUES (:id, :name)"),
                {"id": tenant_b, "name": f"security-live-{tenant_b}"},
            )
        ingest_document(
            tenant_id=tenant_b,
            source_key=source_key,
            title="Tenant B canary runbook",
            source_type="runbook",
            content=f"{canary} must never be visible to another tenant.",
        )
        yield {
            "tenant_id": tenant_b,
            "source_key": source_key,
            "canary": canary,
        }
    finally:
        # Targeted cleanup: deleting only this tenant cascades to the rows
        # created above. No existing tenant data is touched.
        with engine.begin() as connection:
            connection.execute(
                text("DELETE FROM tenants WHERE id = :id"),
                {"id": tenant_b},
            )


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


def test_live_tenant_a_retrieval_cannot_see_tenant_b_document(
    tenant_b_document,
) -> None:
    principal = Principal(tenant_id=tenant_a(), subject="live-test")
    canary = tenant_b_document["canary"]

    for search in (lexical_search, keyword_search):
        results = search(
            tenant_id=tenant_a(),
            query=canary,
            limit=20,
            principal=principal,
        )
        assert all(
            result["source_key"] != tenant_b_document["source_key"]
            for result in results
        )


def test_live_tenant_a_answer_cannot_retrieve_or_cite_tenant_b_document(
    tenant_b_document,
) -> None:
    canary = tenant_b_document["canary"]

    response = client.post(
        "/api/v1/answer",
        headers={"Authorization": f"Bearer {TOKEN_VALUE}"},
        json={
            "tenant_id": str(tenant_a()),
            "question": f"What does {canary} mean?",
            "retrieval": {"strategy": "lexical"},
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert all(
        item["source_key"] != tenant_b_document["source_key"]
        for item in body["evidence"]
    )
    assert all(
        citation["source_key"] != tenant_b_document["source_key"]
        for citation in body["citations"]
    )
    assert tenant_b_document["source_key"] not in response.text
    # The response echoes the question, so assert on supplied evidence and
    # citation text rather than the whole body.
    assert all(canary not in item["content"] for item in body["evidence"])
    assert all(canary not in citation["excerpt"] for citation in body["citations"])


def test_live_tenant_a_cannot_claim_tenant_b_in_the_body(
    tenant_b_document,
) -> None:
    response = client.post(
        "/api/v1/answer",
        headers={"Authorization": f"Bearer {TOKEN_VALUE}"},
        json={
            "tenant_id": str(tenant_b_document["tenant_id"]),
            "question": f"What does {tenant_b_document['canary']} mean?",
            "retrieval": {"strategy": "lexical"},
        },
    )

    assert response.status_code == 403
    assert response.json() == {"detail": "Forbidden"}
