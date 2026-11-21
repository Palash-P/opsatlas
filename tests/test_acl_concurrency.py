"""Document ACL concurrency and authorization serialization tests (F-03).

These tests use real PostgreSQL transactions and the real advisory lock helper.
Ordering is controlled with explicit events and observed through ``pg_locks``;
no test relies on a sleep to decide which side won.

Every test uses a throwaway tenant whose deletion cascades to its documents,
versions, chunks, jobs, and ACL rows. Celery is never involved: the worker entry
point is called directly.

Like the other database-backed live suites (``test_celery_live``,
``test_api_ingestion_live``, ``test_security_live``), this module runs only with
``OPSATLAS_LIVE_TESTS=1``. The default offline run must never write to whatever
``DATABASE_URL`` happens to point at on a developer machine; CI sets the gate and
points ``DATABASE_URL`` at its disposable PostgreSQL service.
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from uuid import UUID, uuid4

import pytest
from sqlalchemy import inspect, text

from app.core.auth import AuthorizationError
from app.db.advisory_locks import (
    acquire_document_acl_lock,
    acquire_document_acl_locks,
    document_acl_lock_key,
)
from app.db.connections import check_postgres, get_engine
from app.services import ingestion as ingestion_module
from app.services.ingestion import (
    IngestionJobNotFoundError,
    _authorize_document_write,
    get_ingestion_job_status,
    ingest_document,
    resume_ingestion_job,
    submit_ingestion_job,
)
from app.services.retrieval import keyword_search
from tests.conftest import deterministic_embeddings, principal_for

LIVE = os.environ.get("OPSATLAS_LIVE_TESTS") == "1"

pytestmark = pytest.mark.skipif(
    not LIVE,
    reason="OPSATLAS_LIVE_TESTS is not 1",
)

REQUIRED_JOB_COLUMNS = {
    "version_id",
    "activation_generation",
    "execution_token",
    "execution_lease_expires_at",
}
POLL_TIMEOUT_SECONDS = 10.0
POLL_INTERVAL_SECONDS = 0.01


@pytest.fixture(scope="module")
def db_engine():
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
        acl_columns = {column["name"] for column in inspect(engine).get_columns("document_acl")}
        missing = REQUIRED_JOB_COLUMNS - job_columns
        if "last_activation_generation" not in document_columns:
            missing.add("documents.last_activation_generation")
        if not {"document_id", "principal_type", "principal_id", "permission"} <= acl_columns:
            missing.add("document_acl columns")
        if missing:
            pytest.fail(
                "ACL concurrency tests require the Phase F schema; missing: "
                + ", ".join(sorted(missing)),
                pytrace=False,
            )
        return engine

    assert last_error is not None
    pytest.skip(
        f"PostgreSQL is not available for ACL concurrency tests: {type(last_error).__name__}"
    )


@pytest.fixture(autouse=True)
def _deterministic_embedding(deterministic_embedding):
    return deterministic_embedding


@pytest.fixture
def tenant_id(db_engine) -> UUID:
    tenant = uuid4()
    with db_engine.begin() as connection:
        connection.execute(
            text("INSERT INTO tenants (id, name) VALUES (:id, :name)"),
            {"id": tenant, "name": f"phase-f-acl-concurrency-{tenant}"},
        )
    try:
        yield tenant
    finally:
        with db_engine.begin() as connection:
            connection.execute(text("DELETE FROM tenants WHERE id = :id"), {"id": tenant})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _content(prefix: str, lines: int = 60) -> str:
    return "\n".join(
        f"{prefix} step {index}: verify dependency {index} before escalation."
        for index in range(lines)
    )


def _writer(tenant_id: UUID):
    return principal_for(tenant_id, subject="test-user")


def _manager(tenant_id: UUID, *, groups: frozenset[str] = frozenset({"test-group"})):
    return principal_for(
        tenant_id,
        subject="acl-manager",
        user_id=None,
        roles=frozenset({"member"}),
        groups=groups,
    )


def _ingest(
    tenant_id: UUID,
    *,
    source_key: str,
    content: str,
    principal=None,
) -> dict:
    return ingest_document(
        principal=principal or _writer(tenant_id),
        tenant_id=tenant_id,
        source_key=source_key,
        title="ACL concurrency runbook",
        source_type="runbook",
        content=content,
    )


def _submit(
    tenant_id: UUID,
    *,
    source_key: str,
    content: str,
    principal=None,
):
    return submit_ingestion_job(
        principal=principal or _writer(tenant_id),
        tenant_id=tenant_id,
        source_key=source_key,
        title="ACL concurrency runbook",
        source_type="runbook",
        content=content,
    )


def _document(tenant_id: UUID, source_key: str) -> dict | None:
    with get_engine().connect() as connection:
        row = (
            connection.execute(
                text(
                    """
                    SELECT id, tenant_id, source_key, current_version_id
                    FROM documents
                    WHERE tenant_id = :tenant_id AND source_key = :source_key
                    """
                ),
                {"tenant_id": tenant_id, "source_key": source_key},
            )
            .mappings()
            .first()
        )
    return dict(row) if row is not None else None


def _acl_rows(document_id: UUID) -> list[dict]:
    with get_engine().connect() as connection:
        rows = (
            connection.execute(
                text(
                    """
                    SELECT principal_type, principal_id, permission
                    FROM document_acl
                    WHERE document_id = :document_id
                    ORDER BY permission, principal_type, principal_id
                    """
                ),
                {"document_id": document_id},
            )
            .mappings()
            .all()
        )
    return [dict(row) for row in rows]


def _counts(tenant_id: UUID) -> dict[str, int]:
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
                {"tenant_id": tenant_id},
            )
            .mappings()
            .one()
        )
    return {key: int(row[key]) for key in row}


def _job_row(job_id: UUID) -> dict:
    with get_engine().connect() as connection:
        row = (
            connection.execute(
                text(
                    """
                    SELECT id, status, attempts, document_id, version_id,
                           execution_token, execution_lease_expires_at,
                           (
                               execution_lease_expires_at IS NOT NULL
                               AND execution_lease_expires_at > now()
                           ) AS execution_lease_active
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


def _grant(
    document_id: UUID,
    principal_type: str,
    principal_id: str,
    permission: str,
) -> None:
    """Grant one ACL row using the same lock every production writer must use."""

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


def _revoke(
    document_id: UUID,
    principal_type: str,
    principal_id: str,
    permission: str,
) -> None:
    """Revoke one ACL row under the document ACL lock."""

    with get_engine().begin() as connection:
        acquire_document_acl_lock(connection, document_id)
        connection.execute(
            text(
                """
                DELETE FROM document_acl
                WHERE document_id = :document_id
                  AND principal_type = :principal_type
                  AND principal_id = :principal_id
                  AND permission = :permission
                """
            ),
            {
                "document_id": document_id,
                "principal_type": principal_type,
                "principal_id": principal_id,
                "permission": permission,
            },
        )


def _lock_parts(document_id: UUID) -> tuple[int, int]:
    key = document_acl_lock_key(document_id) & 0xFFFFFFFFFFFFFFFF
    return key >> 32, key & 0xFFFFFFFF


def _advisory_lock_state(engine, document_id: UUID, *, granted: bool) -> int:
    high, low = _lock_parts(document_id)
    with engine.connect() as connection:
        return int(
            connection.execute(
                text(
                    """
                    SELECT count(*)
                    FROM pg_locks
                    WHERE locktype = 'advisory'
                      AND granted = :granted
                      AND classid = :high
                      AND objid = :low
                    """
                ),
                {"granted": granted, "high": high, "low": low},
            ).scalar_one()
        )


def _wait_until_locked(engine, document_id: UUID, *, granted: bool) -> None:
    deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if _advisory_lock_state(engine, document_id, granted=granted):
            return
        time.sleep(POLL_INTERVAL_SECONDS)
    raise AssertionError(f"timed out waiting for advisory lock granted={granted} on {document_id}")


def _lock_is_held(engine, document_id: UUID) -> bool:
    """Return whether another session holds the ACL lock for this document."""

    key = document_acl_lock_key(document_id)
    with engine.connect() as connection:
        transaction = connection.begin()
        try:
            acquired = connection.execute(
                text("SELECT pg_try_advisory_xact_lock(:key)"),
                {"key": key},
            ).scalar_one()
        finally:
            transaction.rollback()
    return not acquired


class _ThreadResult:
    def __init__(self) -> None:
        self.value = None
        self.error: BaseException | None = None
        self.done = threading.Event()

    def run(self, function) -> None:
        try:
            self.value = function()
        except BaseException as exc:  # noqa: BLE001 - surfaced by the test
            self.error = exc
        finally:
            self.done.set()


# ---------------------------------------------------------------------------
# Lock identity and ordering
# ---------------------------------------------------------------------------


def test_document_acl_lock_key_is_stable_and_document_scoped() -> None:
    first = uuid4()
    second = uuid4()
    assert document_acl_lock_key(first) == document_acl_lock_key(UUID(str(first)))
    assert document_acl_lock_key(first) != document_acl_lock_key(second)


def test_bulk_acl_locks_are_acquired_in_sorted_uuid_order(monkeypatch) -> None:
    documents = [uuid4(), uuid4(), uuid4()]
    recorded: list[UUID] = []

    def recorder(_connection, document_id: UUID) -> None:
        recorded.append(document_id)

    monkeypatch.setattr(
        "app.db.advisory_locks.acquire_document_acl_lock",
        recorder,
    )

    acquire_document_acl_locks(object(), [documents[2], documents[0], documents[1]])

    assert recorded == sorted(documents, key=str)


def test_bulk_acl_locks_cover_every_document_until_commit(db_engine, tenant_id: UUID) -> None:
    first = _ingest(tenant_id, source_key="acl/bulk-a.md", content=_content("Bulk A"))
    second = _ingest(tenant_id, source_key="acl/bulk-b.md", content=_content("Bulk B"))
    document_ids = [first["document_id"], second["document_id"]]

    with db_engine.connect() as connection:
        transaction = connection.begin()
        try:
            acquire_document_acl_locks(connection, document_ids)
            for document_id in document_ids:
                assert _advisory_lock_state(db_engine, document_id, granted=True) == 1
        finally:
            transaction.commit()

    with db_engine.begin() as connection:
        acquire_document_acl_locks(connection, document_ids)
        # Re-acquiring in a new transaction succeeds, proving the first commit
        # released every lock.
        for document_id in document_ids:
            assert _advisory_lock_state(db_engine, document_id, granted=True) == 1


# ---------------------------------------------------------------------------
# Revocation races
# ---------------------------------------------------------------------------


def test_revocation_waits_for_an_in_flight_authorization(db_engine, tenant_id: UUID) -> None:
    source_key = "acl/revocation-waits.md"
    document = _ingest(tenant_id, source_key=source_key, content=_content("Revocation"))
    document_id = document["document_id"]
    manager = _manager(tenant_id)
    _grant(document_id, "user", "acl-manager", "manage")

    with db_engine.connect() as authorization:
        authorization.begin()
        try:
            _authorize_document_write(
                authorization,
                manager,
                tenant_id=tenant_id,
                source_key=source_key,
            )
            assert _advisory_lock_state(db_engine, document_id, granted=True) == 1

            revocation = _ThreadResult()
            thread = threading.Thread(
                target=revocation.run,
                args=(lambda: _revoke(document_id, "user", "acl-manager", "manage"),),
                daemon=True,
            )
            thread.start()
            _wait_until_locked(db_engine, document_id, granted=False)

            assert not revocation.done.is_set()
            assert _acl_rows(document_id) == [
                {"principal_type": "user", "principal_id": "acl-manager", "permission": "manage"}
            ]
        finally:
            authorization.commit()

    assert revocation.done.wait(timeout=POLL_TIMEOUT_SECONDS)
    assert revocation.error is None
    assert _acl_rows(document_id) == []

    before = _counts(tenant_id)
    with pytest.raises(AuthorizationError):
        _ingest(tenant_id, source_key=source_key, content=_content("Denied"), principal=manager)
    assert _counts(tenant_id) == before


def test_ingestion_waits_for_revocation_and_then_denies(db_engine, tenant_id: UUID) -> None:
    source_key = "acl/ingestion-waits.md"
    document = _ingest(tenant_id, source_key=source_key, content=_content("Ingestion waits"))
    document_id = document["document_id"]
    manager = _manager(tenant_id)
    _grant(document_id, "user", "acl-manager", "manage")

    locked = threading.Event()
    release = threading.Event()

    def revoke_transaction() -> None:
        with db_engine.begin() as connection:
            acquire_document_acl_lock(connection, document_id)
            connection.execute(
                text(
                    """
                    DELETE FROM document_acl
                    WHERE document_id = :document_id
                      AND principal_type = 'user'
                      AND principal_id = 'acl-manager'
                      AND permission = 'manage'
                    """
                ),
                {"document_id": document_id},
            )
            locked.set()
            assert release.wait(timeout=POLL_TIMEOUT_SECONDS)

    revoker = _ThreadResult()
    revoke_thread = threading.Thread(
        target=revoker.run,
        args=(revoke_transaction,),
        daemon=True,
    )
    revoke_thread.start()
    assert locked.wait(timeout=POLL_TIMEOUT_SECONDS)

    before = _counts(tenant_id)
    ingest_result = _ThreadResult()
    ingest_thread = threading.Thread(
        target=ingest_result.run,
        args=(
            lambda: _ingest(
                tenant_id,
                source_key=source_key,
                content=_content("Must be denied"),
                principal=manager,
            ),
        ),
        daemon=True,
    )
    ingest_thread.start()
    try:
        _wait_until_locked(db_engine, document_id, granted=False)

        assert not ingest_result.done.is_set()
        assert _counts(tenant_id) == before
    finally:
        # A failed assertion must not leave the revoker holding the document
        # lock (and therefore blocking tenant teardown) forever.
        release.set()

    assert revoker.done.wait(timeout=POLL_TIMEOUT_SECONDS)
    assert revoker.error is None
    assert ingest_result.done.wait(timeout=POLL_TIMEOUT_SECONDS)
    assert isinstance(ingest_result.error, AuthorizationError)
    assert _counts(tenant_id) == before


def test_manage_grant_deleted_concurrently_with_synchronous_ingestion(
    db_engine,
    tenant_id: UUID,
    monkeypatch,
) -> None:
    source_key = "acl/sync-revocation.md"
    document = _ingest(tenant_id, source_key=source_key, content=_content("Stable version"))
    document_id = document["document_id"]
    manager = _manager(tenant_id)
    _grant(document_id, "user", "acl-manager", "manage")

    inside_tx2 = threading.Event()
    release_tx2 = threading.Event()
    original = ingestion_module._find_existing_version

    def gated_find(*args, **kwargs):
        inside_tx2.set()
        assert release_tx2.wait(timeout=POLL_TIMEOUT_SECONDS)
        return original(*args, **kwargs)

    monkeypatch.setattr(ingestion_module, "_find_existing_version", gated_find)

    ingestion = _ThreadResult()
    ingestion_thread = threading.Thread(
        target=ingestion.run,
        args=(
            lambda: _ingest(
                tenant_id,
                source_key=source_key,
                content=_content("Authorized replacement"),
                principal=manager,
            ),
        ),
        daemon=True,
    )
    ingestion_thread.start()
    try:
        assert inside_tx2.wait(timeout=POLL_TIMEOUT_SECONDS)

        revocation = _ThreadResult()
        revoke_thread = threading.Thread(
            target=revocation.run,
            args=(lambda: _revoke(document_id, "user", "acl-manager", "manage"),),
            daemon=True,
        )
        revoke_thread.start()
        _wait_until_locked(db_engine, document_id, granted=False)
        assert not revocation.done.is_set()
        assert _acl_rows(document_id) != []
    finally:
        # Release a gated ingestion transaction so a failed assertion cannot
        # hold the ACL/job/document locks through tenant teardown.
        release_tx2.set()

    assert ingestion.done.wait(timeout=POLL_TIMEOUT_SECONDS)
    assert ingestion.error is None
    assert ingestion.value["is_current"] is True
    assert revocation.done.wait(timeout=POLL_TIMEOUT_SECONDS)
    assert revocation.error is None
    assert _acl_rows(document_id) == []

    before = _counts(tenant_id)
    with pytest.raises(AuthorizationError):
        _ingest(
            tenant_id,
            source_key=source_key,
            content=_content("Denied after revocation"),
            principal=manager,
        )
    assert _counts(tenant_id) == before


def test_manage_grant_deleted_concurrently_with_asynchronous_submission(
    db_engine,
    tenant_id: UUID,
    monkeypatch,
) -> None:
    source_key = "acl/async-revocation.md"
    document = _ingest(tenant_id, source_key=source_key, content=_content("Async stable"))
    document_id = document["document_id"]
    manager = _manager(tenant_id)
    _grant(document_id, "user", "acl-manager", "manage")

    inside_tx2 = threading.Event()
    release_tx2 = threading.Event()
    original = ingestion_module._find_existing_version

    def gated_find(*args, **kwargs):
        inside_tx2.set()
        assert release_tx2.wait(timeout=POLL_TIMEOUT_SECONDS)
        return original(*args, **kwargs)

    monkeypatch.setattr(ingestion_module, "_find_existing_version", gated_find)

    submission = _ThreadResult()
    submit_thread = threading.Thread(
        target=submission.run,
        args=(
            lambda: _submit(
                tenant_id,
                source_key=source_key,
                content=_content("Async replacement"),
                principal=manager,
            ),
        ),
        daemon=True,
    )
    submit_thread.start()
    try:
        assert inside_tx2.wait(timeout=POLL_TIMEOUT_SECONDS)

        revocation = _ThreadResult()
        revoke_thread = threading.Thread(
            target=revocation.run,
            args=(lambda: _revoke(document_id, "user", "acl-manager", "manage"),),
            daemon=True,
        )
        revoke_thread.start()
        _wait_until_locked(db_engine, document_id, granted=False)
        assert not revocation.done.is_set()
    finally:
        release_tx2.set()

    assert submission.done.wait(timeout=POLL_TIMEOUT_SECONDS)
    assert submission.error is None
    assert revocation.done.wait(timeout=POLL_TIMEOUT_SECONDS)
    assert revocation.error is None

    # The durable target was authorized before the revocation committed; the
    # worker completes it without user credentials, matching synchronous work.
    completion = resume_ingestion_job(submission.value.job_id)
    assert completion.status == "succeeded"

    before = _counts(tenant_id)
    with pytest.raises(AuthorizationError):
        _submit(
            tenant_id,
            source_key=source_key,
            content=_content("Denied async after revocation"),
            principal=manager,
        )
    assert _counts(tenant_id) == before


# ---------------------------------------------------------------------------
# Grant races
# ---------------------------------------------------------------------------


def test_restrictive_acl_insert_waits_for_in_flight_authorization(
    db_engine,
    tenant_id: UUID,
) -> None:
    source_key = "acl/restrictive-insert.md"
    document = _ingest(tenant_id, source_key=source_key, content=_content("Restrictive insert"))
    document_id = document["document_id"]
    writer = _writer(tenant_id)

    with db_engine.connect() as authorization:
        authorization.begin()
        try:
            _authorize_document_write(
                authorization,
                writer,
                tenant_id=tenant_id,
                source_key=source_key,
            )
            grant = _ThreadResult()
            thread = threading.Thread(
                target=grant.run,
                args=(lambda: _grant(document_id, "group", "other-group", "read"),),
                daemon=True,
            )
            thread.start()
            _wait_until_locked(db_engine, document_id, granted=False)
            assert not grant.done.is_set()
            assert _acl_rows(document_id) == []
        finally:
            authorization.commit()

    assert grant.done.wait(timeout=POLL_TIMEOUT_SECONDS)
    assert grant.error is None
    assert _acl_rows(document_id) == [
        {"principal_type": "group", "principal_id": "other-group", "permission": "read"}
    ]

    before = _counts(tenant_id)
    with pytest.raises(AuthorizationError):
        _ingest(
            tenant_id,
            source_key=source_key,
            content=_content("Now restricted"),
            principal=writer,
        )
    assert _counts(tenant_id) == before


def test_new_manage_grant_authorizes_the_next_request_after_commit(
    db_engine,
    tenant_id: UUID,
) -> None:
    source_key = "acl/new-grant.md"
    document = _ingest(tenant_id, source_key=source_key, content=_content("New grant"))
    document_id = document["document_id"]
    manager = _manager(tenant_id)
    _grant(document_id, "group", "other-group", "read")

    with pytest.raises(AuthorizationError):
        _ingest(
            tenant_id,
            source_key=source_key,
            content=_content("Denied before grant"),
            principal=manager,
        )

    with db_engine.connect() as authorization:
        authorization.begin()
        try:
            acquire_document_acl_lock(authorization, document_id)
            grant = _ThreadResult()
            thread = threading.Thread(
                target=grant.run,
                args=(lambda: _grant(document_id, "user", "acl-manager", "manage"),),
                daemon=True,
            )
            thread.start()
            _wait_until_locked(db_engine, document_id, granted=False)
            assert not grant.done.is_set()
        finally:
            authorization.commit()

    assert grant.done.wait(timeout=POLL_TIMEOUT_SECONDS)
    assert grant.error is None

    result = _ingest(
        tenant_id,
        source_key=source_key,
        content=_content("Allowed after grant"),
        principal=manager,
    )
    assert result["is_current"] is True


# ---------------------------------------------------------------------------
# No writes on denial, idempotency, fencing, isolation, retrieval
# ---------------------------------------------------------------------------


def test_denied_write_produces_no_document_version_chunk_or_job(
    db_engine,
    tenant_id: UUID,
) -> None:
    source_key = "acl/denied-new.md"

    before = _counts(tenant_id)
    with pytest.raises(AuthorizationError):
        _ingest(
            tenant_id,
            source_key=source_key,
            content=_content("Denied new document"),
            principal=principal_for(
                tenant_id,
                roles=frozenset({"member"}),
            ),
        )
    assert _counts(tenant_id) == before
    assert _document(tenant_id, source_key) is None

    document = _ingest(tenant_id, source_key=source_key, content=_content("Allowed first"))
    document_id = document["document_id"]
    _grant(document_id, "group", "different-group", "read")

    before = _counts(tenant_id)
    with pytest.raises(AuthorizationError):
        _ingest(tenant_id, source_key=source_key, content=_content("Denied update"))
    assert _counts(tenant_id) == before
    with pytest.raises(AuthorizationError):
        _submit(tenant_id, source_key=source_key, content=_content("Denied async"))
    assert _counts(tenant_id) == before

    # The writer's own ACL row is still readable for the original version.
    assert _acl_rows(document_id) == [
        {"principal_type": "group", "principal_id": "different-group", "permission": "read"}
    ]


def test_duplicate_submissions_remain_idempotent(tenant_id: UUID) -> None:
    source_key = "acl/idempotent.md"
    content = _content("Idempotent")
    manager = _manager(tenant_id)

    first = _ingest(tenant_id, source_key=source_key, content=content)
    document_id = first["document_id"]
    _grant(document_id, "user", "acl-manager", "manage")

    second = _ingest(tenant_id, source_key=source_key, content=content, principal=manager)

    assert second["version_id"] == first["version_id"]
    assert second["deduplicated"] is True
    assert second["is_current"] is True

    with get_engine().connect() as connection:
        row = (
            connection.execute(
                text(
                    """
                    SELECT status, attempts
                    FROM ingestion_jobs
                    WHERE tenant_id = :tenant_id
                    ORDER BY created_at DESC
                    LIMIT 1
                    """
                ),
                {"tenant_id": tenant_id},
            )
            .mappings()
            .one()
        )
    assert row["status"] == "succeeded"
    assert int(row["attempts"]) == 1


def test_f02_execution_fencing_still_applies_during_acl_authorized_ingestion(
    tenant_id: UUID,
    monkeypatch,
) -> None:
    source_key = "acl/f02-fencing.md"
    entered = threading.Event()
    release = threading.Event()

    def gated_embedder(texts: list[str]) -> list[list[float]]:
        entered.set()
        assert release.wait(timeout=POLL_TIMEOUT_SECONDS)
        return deterministic_embeddings(texts)

    monkeypatch.setattr(ingestion_module, "embed_texts", gated_embedder)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            _ingest,
            tenant_id,
            source_key=source_key,
            content=_content("Fenced", lines=120),
        )
        assert entered.wait(timeout=POLL_TIMEOUT_SECONDS)

        with get_engine().connect() as connection:
            job = (
                connection.execute(
                    text(
                        """
                        SELECT id FROM ingestion_jobs
                        WHERE tenant_id = :tenant_id
                        ORDER BY created_at DESC
                        LIMIT 1
                        """
                    ),
                    {"tenant_id": tenant_id},
                )
                .mappings()
                .one()
            )
        row = _job_row(job["id"])
        assert row["execution_token"] is not None
        assert row["execution_lease_active"] is True

        release.set()
        result = future.result(timeout=30)

    assert result["is_current"] is True
    final = _job_row(job["id"])
    assert final["status"] == "succeeded"
    assert final["execution_token"] is None
    assert final["execution_lease_expires_at"] is None


def test_tenant_isolation_survives_concurrent_acl_scope(
    db_engine,
    tenant_id: UUID,
) -> None:
    source_key = "acl/tenant-isolation.md"
    document = _ingest(tenant_id, source_key=source_key, content=_content("Tenant A"))
    _grant(document["document_id"], "group", "tenant-a-group", "read")

    other_tenant = uuid4()
    with db_engine.begin() as connection:
        connection.execute(
            text("INSERT INTO tenants (id, name) VALUES (:id, :name)"),
            {"id": other_tenant, "name": f"phase-f-acl-other-{other_tenant}"},
        )
    try:
        other = _ingest(
            other_tenant,
            source_key=source_key,
            content=_content("Tenant B"),
            principal=_writer(other_tenant),
        )
        assert other["is_current"] is True
        assert other["document_id"] != document["document_id"]

        submission = _submit(
            other_tenant,
            source_key="acl/tenant-isolation-job.md",
            content=_content("Tenant B job"),
        )
        with pytest.raises(IngestionJobNotFoundError):
            get_ingestion_job_status(
                submission.job_id,
                principal=_writer(tenant_id),
                tenant_id=tenant_id,
            )
    finally:
        with db_engine.begin() as connection:
            connection.execute(
                text("DELETE FROM tenants WHERE id = :id"),
                {"id": other_tenant},
            )


def test_retrieval_read_authorization_semantics_are_unchanged(
    db_engine,
    tenant_id: UUID,
) -> None:
    source_key = "acl/retrieval.md"
    document = _ingest(
        tenant_id,
        source_key=source_key,
        content="RestrictedMarker procurement escalation runbook.",
    )
    document_id = document["document_id"]
    member = principal_for(tenant_id, roles=frozenset({"member"}), groups=frozenset({"test-group"}))

    visible = keyword_search(
        tenant_id=tenant_id,
        query="RestrictedMarker",
        principal=member,
    )
    assert any(row["document_id"] == document_id for row in visible)

    _grant(document_id, "group", "other-group", "read")

    hidden = keyword_search(
        tenant_id=tenant_id,
        query="RestrictedMarker",
        principal=member,
    )
    assert not any(row["document_id"] == document_id for row in hidden)

    allowed = keyword_search(
        tenant_id=tenant_id,
        query="RestrictedMarker",
        principal=principal_for(
            tenant_id,
            roles=frozenset({"member"}),
            groups=frozenset({"other-group"}),
        ),
    )
    assert any(row["document_id"] == document_id for row in allowed)


def test_last_known_good_survives_a_denied_replacement(
    db_engine,
    tenant_id: UUID,
) -> None:
    source_key = "acl/last-known-good.md"
    good = _ingest(tenant_id, source_key=source_key, content=_content("Good version"))
    document_id = good["document_id"]
    _grant(document_id, "group", "other-group", "read")

    before = _counts(tenant_id)
    with pytest.raises(AuthorizationError):
        _ingest(tenant_id, source_key=source_key, content=_content("Denied replacement"))
    assert _counts(tenant_id) == before

    stored = _document(tenant_id, source_key)
    assert stored is not None
    assert stored["current_version_id"] == good["version_id"]
    with get_engine().connect() as connection:
        versions = (
            connection.execute(
                text(
                    """
                    SELECT id, is_current
                    FROM document_versions
                    WHERE document_id = :document_id
                    ORDER BY version_number
                    """
                ),
                {"document_id": document_id},
            )
            .mappings()
            .all()
        )
    assert len(versions) == 1
    assert versions[0]["is_current"] is True
    assert versions[0]["id"] == good["version_id"]


def test_acl_mutation_after_commit_is_seen_by_the_next_authorization(
    db_engine,
    tenant_id: UUID,
) -> None:
    source_key = "acl/committed-revocation.md"
    manager = _manager(tenant_id)
    first = _ingest(tenant_id, source_key=source_key, content=_content("Stable first"))
    document_id = first["document_id"]
    _grant(document_id, "user", "acl-manager", "manage")

    second = _ingest(
        tenant_id,
        source_key=source_key,
        content=_content("Manager second"),
        principal=manager,
    )
    assert second["is_current"] is True

    _revoke(document_id, "user", "acl-manager", "manage")

    before = _counts(tenant_id)
    with pytest.raises(AuthorizationError):
        _ingest(
            tenant_id,
            source_key=source_key,
            content=_content("Manager third"),
            principal=manager,
        )
    assert _counts(tenant_id) == before


def test_authorization_uses_the_current_transaction_snapshot(
    db_engine,
    tenant_id: UUID,
) -> None:
    """A managed grant written in this transaction is visible; a later commit is serialized."""

    source_key = "acl/lock-order.md"
    document = _ingest(tenant_id, source_key=source_key, content=_content("Lock order"))
    document_id = document["document_id"]
    manager = _manager(tenant_id)
    _grant(document_id, "user", "acl-manager", "manage")

    with db_engine.connect() as connection:
        transaction = connection.begin()
        try:
            assert (
                _authorize_document_write(
                    connection,
                    manager,
                    tenant_id=tenant_id,
                    source_key=source_key,
                )
                == document_id
            )
            assert _lock_is_held(db_engine, document_id) is True
        finally:
            transaction.rollback()

    assert _lock_is_held(db_engine, document_id) is False
