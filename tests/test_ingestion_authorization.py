"""Phase F, Step 7: ingestion write authorization and job access control.

These tests use the real PostgreSQL schema because the policy depends on the
``document_acl`` rows and on transaction boundaries that prevent unauthorized
document, version, chunk, and job writes.

Coverage:

- a configured writer can create through both sync and async paths;
- tenant membership without the writer role cannot create;
- a ``read``-only ACL cannot write;
- a ``manage`` grant authorizes an existing document without the writer role;
- a ``manage`` grant on another document does not authorize the target;
- cross-tenant submission and cross-tenant job status;
- duplicate submission and historical-version reactivation after revocation;
- sync/async parity;
- the worker completing an already-authorized job without user credentials;
- same-tenant job status with and without document read permission;
- sanitized status responses and no writes on denial.
"""

from __future__ import annotations

import time
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import inspect, text

from app.api import documents
from app.core.auth import AuthorizationError, Principal, require_identity
from app.db.connections import check_postgres, get_engine
from app.main import app
from app.services import ingestion as ingestion_module
from app.services.ingestion import (
    IngestionJobNotFoundError,
    get_ingestion_job_status,
    resume_ingestion_job,
)

client = TestClient(app)
SYNC_URL = "/api/v1/documents/ingest"
ASYNC_URL = "/api/v1/documents/ingest/async"
JOB_STATUS_URL = "/api/v1/documents/jobs"
TEST_USER_ID = UUID("22222222-2222-4222-8222-222222222222")


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def db_engine():
    """Skip these tests when the local PostgreSQL schema is unavailable."""

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            check_postgres()
        except Exception as exc:  # noqa: BLE001 - absence must skip, not error
            last_error = exc
            if attempt < 2:
                time.sleep(0.25)
            continue

        engine = get_engine()
        job_columns = {column["name"] for column in inspect(engine).get_columns("ingestion_jobs")}
        document_columns = {column["name"] for column in inspect(engine).get_columns("documents")}
        missing = {"version_id", "activation_generation"} - job_columns
        missing |= {
            "next_dispatch_at",
            "dispatch_claimed_until",
            "dispatch_attempts",
            "execution_token",
            "execution_lease_expires_at",
        } - job_columns
        if "last_activation_generation" not in document_columns:
            missing.add("documents.last_activation_generation")
        if missing:
            pytest.fail(
                "Authorization tests require sql/003_ingestion_jobs_version.sql, "
                "sql/004_activation_generation.sql, and "
                "sql/005_job_reconciliation.sql; missing: " + ", ".join(sorted(missing)),
                pytrace=False,
            )
        return engine

    assert last_error is not None
    pytest.skip(f"PostgreSQL is not available for authorization tests: {type(last_error).__name__}")


@pytest.fixture(autouse=True)
def _deterministic_embedding(deterministic_embedding):
    return deterministic_embedding


def _insert_tenant(engine) -> UUID:
    tenant = uuid4()
    with engine.begin() as connection:
        connection.execute(
            text("INSERT INTO tenants (id, name) VALUES (:id, :name)"),
            {"id": tenant, "name": f"phase-f-authz-{tenant}"},
        )
    return tenant


def _delete_tenant(engine, tenant: UUID) -> None:
    with engine.begin() as connection:
        connection.execute(text("DELETE FROM tenants WHERE id = :id"), {"id": tenant})


@pytest.fixture
def tenant(db_engine) -> UUID:
    value = _insert_tenant(db_engine)
    try:
        yield value
    finally:
        _delete_tenant(db_engine, value)


@pytest.fixture
def other_tenant(db_engine) -> UUID:
    value = _insert_tenant(db_engine)
    try:
        yield value
    finally:
        _delete_tenant(db_engine, value)


def _principal(
    tenant: UUID,
    *,
    subject: str = "test-user",
    user_id: UUID | None = TEST_USER_ID,
    roles: frozenset[str] = frozenset({"member", "document_writer"}),
    groups: frozenset[str] = frozenset({"test-group"}),
) -> Principal:
    return Principal(
        tenant_id=tenant,
        subject=subject,
        user_id=user_id,
        roles=roles,
        groups=groups,
    )


def _payload(
    source_key: str,
    content: str = "Investigate PAY-503 before escalation.",
    **overrides,
) -> dict:
    payload = {
        "source_key": source_key,
        "title": "Authorization runbook",
        "source_type": "runbook",
        "content": content,
    }
    payload.update(overrides)
    return payload


def _counts(tenant: UUID) -> dict[str, int]:
    with get_engine().connect() as connection:
        row = (
            connection.execute(
                text(
                    """
                    SELECT
                        (SELECT count(*) FROM documents
                         WHERE tenant_id = :tenant_id) AS documents,
                        (SELECT count(*) FROM document_versions AS dv
                         JOIN documents AS d ON d.id = dv.document_id
                         WHERE d.tenant_id = :tenant_id) AS versions,
                        (SELECT count(*) FROM chunks
                         WHERE tenant_id = :tenant_id) AS chunks,
                        (SELECT count(*) FROM ingestion_jobs
                         WHERE tenant_id = :tenant_id) AS jobs
                    """
                ),
                {"tenant_id": tenant},
            )
            .mappings()
            .one()
        )
    return {key: int(row[key]) for key in row}


def _jobs(tenant: UUID) -> list[dict]:
    with get_engine().connect() as connection:
        rows = (
            connection.execute(
                text(
                    """
                    SELECT id, tenant_id, document_id, version_id, idempotency_key,
                           status, attempts, error_code, error_summary,
                           created_at, started_at, finished_at
                    FROM ingestion_jobs
                    WHERE tenant_id = :tenant_id
                    ORDER BY created_at, id
                    """
                ),
                {"tenant_id": tenant},
            )
            .mappings()
            .all()
        )
    return [dict(row) for row in rows]


def _job_row(job_id: UUID) -> dict:
    with get_engine().connect() as connection:
        row = (
            connection.execute(
                text(
                    """
                    SELECT id, tenant_id, document_id, version_id, status,
                           attempts, error_code, error_summary
                    FROM ingestion_jobs
                    WHERE id = :job_id
                    """
                ),
                {"job_id": job_id},
            )
            .mappings()
            .one()
        )
    return dict(row)


def _document(tenant: UUID, source_key: str) -> dict | None:
    with get_engine().connect() as connection:
        row = (
            connection.execute(
                text(
                    """
                    SELECT id, tenant_id, source_key, title, current_version_id,
                           status, last_activation_generation
                    FROM documents
                    WHERE tenant_id = :tenant_id AND source_key = :source_key
                    """
                ),
                {"tenant_id": tenant, "source_key": source_key},
            )
            .mappings()
            .first()
        )
    return dict(row) if row is not None else None


def _version_count(tenant: UUID, source_key: str) -> int:
    with get_engine().connect() as connection:
        return int(
            connection.execute(
                text(
                    """
                    SELECT count(*)
                    FROM document_versions AS dv
                    JOIN documents AS d ON d.id = dv.document_id
                    WHERE d.tenant_id = :tenant_id AND d.source_key = :source_key
                    """
                ),
                {"tenant_id": tenant, "source_key": source_key},
            ).scalar_one()
        )


def _add_acl(
    document_id: UUID,
    principal_type: str,
    principal_id: str,
    permission: str,
) -> None:
    from app.db.advisory_locks import acquire_document_acl_lock

    with get_engine().begin() as connection:
        acquire_document_acl_lock(connection, document_id)
        connection.execute(
            text(
                """
                INSERT INTO document_acl (
                    document_id, principal_type, principal_id, permission
                )
                VALUES (:document_id, :principal_type, :principal_id, :permission)
                ON CONFLICT (document_id, principal_type, principal_id, permission)
                DO NOTHING
                """
            ),
            {
                "document_id": document_id,
                "principal_type": principal_type,
                "principal_id": principal_id,
                "permission": permission,
            },
        )


def _allow_publish(monkeypatch) -> None:
    monkeypatch.setattr(documents, "publish_ingestion_job", lambda *args, **kwargs: "task-id")


# ---------------------------------------------------------------------------
# Write authorization: creation
# ---------------------------------------------------------------------------


def test_configured_writer_can_create_via_sync_and_async(
    tenant: UUID,
    set_identity,
    monkeypatch,
) -> None:
    set_identity(tenant_id=tenant)
    _allow_publish(monkeypatch)

    sync_response = client.post(SYNC_URL, json=_payload("runbooks/authz-sync.md"))
    async_response = client.post(ASYNC_URL, json=_payload("runbooks/authz-async.md"))

    assert sync_response.status_code == 201
    assert async_response.status_code == 202
    assert _counts(tenant)["documents"] == 2


@pytest.mark.parametrize("url", [SYNC_URL, ASYNC_URL])
def test_tenant_membership_without_writer_role_cannot_create(
    tenant: UUID,
    set_identity,
    monkeypatch,
    url: str,
) -> None:
    # A normal tenant member: authenticated, but with no document-write role.
    set_identity(tenant_id=tenant, roles=frozenset({"member"}))
    _allow_publish(monkeypatch)
    before = _counts(tenant)

    response = client.post(url, json=_payload("runbooks/authz-denied.md"))

    assert response.status_code == 403
    assert response.json() == {"detail": "Forbidden"}
    assert _counts(tenant) == before


def test_cross_tenant_submission_is_forbidden_and_writes_nothing(
    tenant: UUID,
    other_tenant: UUID,
    set_identity,
) -> None:
    set_identity(tenant_id=tenant)
    before = _counts(other_tenant)

    response = client.post(
        SYNC_URL,
        json=_payload("runbooks/authz-cross-tenant.md", tenant_id=str(other_tenant)),
    )

    assert response.status_code == 403
    assert response.json() == {"detail": "Forbidden"}
    assert _counts(other_tenant) == before


def test_service_rejects_a_principal_from_another_tenant(
    tenant: UUID,
    other_tenant: UUID,
) -> None:
    before = _counts(tenant)

    with pytest.raises(AuthorizationError):
        ingestion_module.ingest_document(
            principal=_principal(other_tenant),
            tenant_id=tenant,
            source_key="runbooks/authz-mismatched-principal.md",
            title="Mismatched principal",
            source_type="runbook",
            content="This must never be written.",
        )

    assert _counts(tenant) == before


# ---------------------------------------------------------------------------
# Write authorization: existing documents
# ---------------------------------------------------------------------------


def test_read_only_acl_cannot_write_an_existing_document(
    tenant: UUID,
    set_identity,
) -> None:
    source_key = "runbooks/authz-read-only.md"
    set_identity(tenant_id=tenant)
    created = client.post(SYNC_URL, json=_payload(source_key, "version one"))
    assert created.status_code == 201
    document_id = UUID(created.json()["document_id"])
    _add_acl(document_id, "user", "reader", "read")

    set_identity(
        tenant_id=tenant,
        subject="reader",
        user_id=None,
        roles=frozenset({"member"}),
    )
    before = _counts(tenant)
    denied = client.post(SYNC_URL, json=_payload(source_key, "version two"))

    assert denied.status_code == 403
    assert _counts(tenant) == before
    assert _version_count(tenant, source_key) == 1


def test_manage_grant_authorizes_an_existing_document_without_writer_role(
    tenant: UUID,
    set_identity,
) -> None:
    source_key = "runbooks/authz-manage.md"
    set_identity(tenant_id=tenant)
    created = client.post(SYNC_URL, json=_payload(source_key, "version one"))
    assert created.status_code == 201
    document_id = UUID(created.json()["document_id"])
    _add_acl(document_id, "user", "editor", "manage")

    set_identity(
        tenant_id=tenant,
        subject="editor",
        user_id=None,
        roles=frozenset({"member"}),
    )
    allowed = client.post(SYNC_URL, json=_payload(source_key, "version two"))

    assert allowed.status_code == 201
    assert allowed.json()["version_number"] == 2


def test_manage_on_another_document_does_not_authorize_the_target(
    tenant: UUID,
    set_identity,
) -> None:
    set_identity(tenant_id=tenant)
    doc_a = client.post(SYNC_URL, json=_payload("runbooks/authz-doc-a.md", "a"))
    doc_b = client.post(SYNC_URL, json=_payload("runbooks/authz-doc-b.md", "b"))
    assert doc_a.status_code == 201
    assert doc_b.status_code == 201
    _add_acl(UUID(doc_a.json()["document_id"]), "user", "editor", "manage")

    set_identity(
        tenant_id=tenant,
        subject="editor",
        user_id=None,
        roles=frozenset({"member"}),
    )

    # The editor may write the granted document.
    allowed = client.post(SYNC_URL, json=_payload("runbooks/authz-doc-a.md", "a-two"))
    assert allowed.status_code == 201

    before = _counts(tenant)
    denied_existing = client.post(
        SYNC_URL,
        json=_payload("runbooks/authz-doc-b.md", "b-two"),
    )
    denied_new = client.post(SYNC_URL, json=_payload("runbooks/authz-doc-c.md", "c"))

    assert denied_existing.status_code == 403
    assert denied_new.status_code == 403
    assert _counts(tenant) == before


# ---------------------------------------------------------------------------
# Revocation, duplicates, and historical reactivation
# ---------------------------------------------------------------------------


def test_duplicate_submission_after_permission_revocation_is_denied(
    tenant: UUID,
    set_identity,
) -> None:
    source_key = "runbooks/authz-revoke-duplicate.md"
    set_identity(tenant_id=tenant)
    first = client.post(SYNC_URL, json=_payload(source_key, "unchanged content"))
    assert first.status_code == 201
    document_id = UUID(first.json()["document_id"])

    # Adding any ACL row restricts the document. The tenant-wide writer role
    # must not override an explicit document restriction.
    _add_acl(document_id, "user", "someone-else", "read")
    set_identity(
        tenant_id=tenant,
        roles=frozenset({"member", "document_writer"}),
    )
    before_counts = _counts(tenant)
    before_jobs = _jobs(tenant)

    duplicate = client.post(SYNC_URL, json=_payload(source_key, "unchanged content"))

    assert duplicate.status_code == 403
    assert _counts(tenant) == before_counts
    assert _jobs(tenant) == before_jobs


def test_historical_version_reactivation_after_revocation_is_denied(
    tenant: UUID,
    set_identity,
) -> None:
    source_key = "runbooks/authz-revoke-history.md"
    set_identity(tenant_id=tenant)
    first = client.post(SYNC_URL, json=_payload(source_key, "historical content"))
    second = client.post(SYNC_URL, json=_payload(source_key, "current content"))
    assert first.status_code == 201
    assert second.status_code == 201
    assert second.json()["version_number"] == 2

    document_id = UUID(second.json()["document_id"])
    _add_acl(document_id, "user", "someone-else", "read")
    set_identity(
        tenant_id=tenant,
        roles=frozenset({"member", "document_writer"}),
    )
    before_counts = _counts(tenant)
    before_jobs = _jobs(tenant)

    reactivate = client.post(SYNC_URL, json=_payload(source_key, "historical content"))

    assert reactivate.status_code == 403
    assert _counts(tenant) == before_counts
    assert _jobs(tenant) == before_jobs
    document = _document(tenant, source_key)
    assert document is not None
    assert document["current_version_id"] == UUID(second.json()["version_id"])


def test_sync_and_async_share_manage_authorization(
    tenant: UUID,
    set_identity,
    monkeypatch,
) -> None:
    source_key = "runbooks/authz-manage-parity.md"
    set_identity(tenant_id=tenant)
    created = client.post(SYNC_URL, json=_payload(source_key, "version one"))
    assert created.status_code == 201
    _add_acl(UUID(created.json()["document_id"]), "user", "editor", "manage")
    set_identity(
        tenant_id=tenant,
        subject="editor",
        user_id=None,
        roles=frozenset({"member"}),
    )
    _allow_publish(monkeypatch)

    sync_response = client.post(SYNC_URL, json=_payload(source_key, "version two"))
    async_response = client.post(ASYNC_URL, json=_payload(source_key, "version three"))

    assert sync_response.status_code == 201
    assert async_response.status_code == 202


# ---------------------------------------------------------------------------
# Worker authorization model
# ---------------------------------------------------------------------------


def test_worker_completes_a_previously_authorized_job_without_credentials(
    tenant: UUID,
    set_identity,
    monkeypatch,
) -> None:
    set_identity(tenant_id=tenant)
    _allow_publish(monkeypatch)
    submitted = client.post(
        ASYNC_URL,
        json=_payload("runbooks/authz-worker.md", "worker completion content"),
    )
    assert submitted.status_code == 202
    job_id = UUID(submitted.json()["job_id"])

    # Remove the request identity entirely. The worker entry point takes only the
    # durable job ID and must not require or receive user credentials.
    app.dependency_overrides.pop(require_identity, None)
    result = resume_ingestion_job(job_id)

    assert result.outcome == "succeeded"
    assert result.is_current is True
    assert _job_row(job_id)["status"] == "succeeded"


# ---------------------------------------------------------------------------
# Job status authorization and non-disclosure
# ---------------------------------------------------------------------------


def _submit_async_job(
    tenant: UUID,
    set_identity,
    monkeypatch,
    source_key: str,
    content: str = "Job status content.",
) -> dict:
    set_identity(tenant_id=tenant)
    _allow_publish(monkeypatch)
    response = client.post(ASYNC_URL, json=_payload(source_key, content))
    assert response.status_code == 202
    return response.json()


def test_cross_tenant_job_status_is_not_found(
    tenant: UUID,
    other_tenant: UUID,
    set_identity,
    monkeypatch,
) -> None:
    body = _submit_async_job(
        tenant,
        set_identity,
        monkeypatch,
        "runbooks/authz-status-cross.md",
    )

    set_identity(tenant_id=other_tenant)
    response = client.get(f"{JOB_STATUS_URL}/{body['job_id']}")

    assert response.status_code == 404
    assert response.json() == {"detail": "Ingestion job not found"}


def test_same_tenant_job_status_requires_document_read_permission(
    tenant: UUID,
    set_identity,
    monkeypatch,
) -> None:
    body = _submit_async_job(
        tenant,
        set_identity,
        monkeypatch,
        "runbooks/authz-status-read.md",
    )
    job = _job_row(UUID(body["job_id"]))
    assert job["document_id"] is not None
    _add_acl(job["document_id"], "user", "reader", "read")
    _add_acl(job["document_id"], "user", "test-user", "manage")

    set_identity(
        tenant_id=tenant,
        subject="reader",
        user_id=None,
        roles=frozenset({"member"}),
    )
    allowed = client.get(f"{JOB_STATUS_URL}/{body['job_id']}")
    assert allowed.status_code == 200

    set_identity(
        tenant_id=tenant,
        subject="stranger",
        user_id=None,
        roles=frozenset({"member"}),
    )
    denied = client.get(f"{JOB_STATUS_URL}/{body['job_id']}")
    assert denied.status_code == 404
    assert denied.json() == {"detail": "Ingestion job not found"}

    # The submitter keeps access through the explicit manage grant.
    set_identity(tenant_id=tenant)
    owner = client.get(f"{JOB_STATUS_URL}/{body['job_id']}")
    assert owner.status_code == 200


def test_job_status_response_is_sanitized(
    tenant: UUID,
    set_identity,
    monkeypatch,
) -> None:
    source_key = "runbooks/authz-status-sanitized.md"
    canary = "UNIQUE_CANARY_CONTENT_42"
    body = _submit_async_job(
        tenant,
        set_identity,
        monkeypatch,
        source_key,
        f"{canary} investigate PAY-503 before escalation.",
    )

    response = client.get(f"{JOB_STATUS_URL}/{body['job_id']}")

    assert response.status_code == 200
    assert set(response.json()) == {
        "job_id",
        "status",
        "attempts",
        "document_id",
        "version_id",
        "error_code",
        "error_summary",
        "created_at",
        "started_at",
        "finished_at",
    }
    assert "execution_token" not in response.text
    assert canary not in response.text
    assert source_key not in response.text


def test_unlinked_job_status_is_tenant_scoped(
    tenant: UUID,
    other_tenant: UUID,
    set_identity,
) -> None:
    job_id = uuid4()
    with get_engine().begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO ingestion_jobs (id, tenant_id, idempotency_key, status)
                VALUES (:id, :tenant_id, :idempotency_key, 'queued')
                """
            ),
            {
                "id": job_id,
                "tenant_id": tenant,
                "idempotency_key": f"manual:{job_id}",
            },
        )

    set_identity(tenant_id=tenant)
    allowed = client.get(f"{JOB_STATUS_URL}/{job_id}")
    assert allowed.status_code == 200
    assert allowed.json()["document_id"] is None

    set_identity(tenant_id=other_tenant)
    denied = client.get(f"{JOB_STATUS_URL}/{job_id}")
    assert denied.status_code == 404


def test_job_status_service_enforces_the_same_acl(
    tenant: UUID,
    set_identity,
    monkeypatch,
) -> None:
    body = _submit_async_job(
        tenant,
        set_identity,
        monkeypatch,
        "runbooks/authz-status-service.md",
    )
    job = _job_row(UUID(body["job_id"]))
    _add_acl(job["document_id"], "user", "reader", "read")

    with pytest.raises(IngestionJobNotFoundError):
        get_ingestion_job_status(
            UUID(body["job_id"]),
            principal=_principal(
                tenant,
                subject="stranger",
                user_id=None,
                roles=frozenset({"member"}),
            ),
            tenant_id=tenant,
        )
