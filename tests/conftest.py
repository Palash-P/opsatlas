"""Shared test identity.

Protected routers require a bearer token resolved by ``require_identity``. The
offline suite is not an authorization test, so the default fixture injects a
fixed principal through FastAPI dependency overrides. Tests that exercise the
real authentication path remove the override with the ``real_auth`` fixture.
"""

from __future__ import annotations

import hashlib
import math
import os
import uuid
from dataclasses import dataclass
from uuid import UUID

import psycopg
import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, make_url

from app.core.auth import Principal, require_identity
from app.core.limits import reset_limiters
from app.main import app
from app.services.embedding_client import expected_dimensions

TEST_TENANT_ID = UUID("11111111-1111-4111-8111-111111111111")
TEST_PRINCIPAL = Principal(
    tenant_id=TEST_TENANT_ID,
    subject="test-user",
    user_id=UUID("22222222-2222-4222-8222-222222222222"),
    roles=frozenset({"member", "document_writer"}),
    groups=frozenset({"test-group"}),
)


def principal_for(
    tenant_id: UUID,
    *,
    subject: str = "test-user",
    user_id: UUID | None = TEST_PRINCIPAL.user_id,
    roles: frozenset[str] = frozenset({"member", "document_writer"}),
    groups: frozenset[str] = frozenset({"test-group"}),
) -> Principal:
    """Build a principal for direct service calls that now require identity."""

    return Principal(
        tenant_id=tenant_id,
        subject=subject,
        user_id=user_id,
        roles=roles,
        groups=groups,
    )


def deterministic_vector(text: str, dimensions: int | None = None) -> list[float]:
    """Return a deterministic, nonzero unit vector for embedding tests.

    Tests must not depend on a paid provider or a running local model server.
    The value is stable for a given text, so ingestion and retrieval can share
    the same fake without any process-local registry.
    """

    size = dimensions if dimensions is not None else expected_dimensions()
    seed = hashlib.sha256(text.encode("utf-8")).digest()
    values: list[float] = []
    counter = 0
    while len(values) < size:
        digest = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
        for offset in range(0, 32, 4):
            raw = int.from_bytes(digest[offset : offset + 4], "big")
            values.append((raw / 2**32) * 2.0 - 1.0)
        counter += 1
    values = values[:size]
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


def deterministic_embeddings(texts: list[str]) -> list[list[float]]:
    return [deterministic_vector(text) for text in texts]


@pytest.fixture
def deterministic_embedding(monkeypatch):
    """Patch ingestion and retrieval to use deterministic local embeddings."""

    from app.services import ingestion, retrieval

    monkeypatch.setattr(ingestion, "embed_texts", deterministic_embeddings)
    monkeypatch.setattr(retrieval, "embed_texts", deterministic_embeddings)
    return deterministic_embeddings


def _clear_redis_rate_limits() -> None:
    """Remove shared limiter counters so tests cannot inherit a used window.

    The memory backend is reset by ``reset_limiters``. With the Redis backend
    (used by CI and Compose) counters live outside the process, so a test that
    expects a fresh one-request budget could see a 429 from an earlier test in
    the same tenant bucket. Only keys under the configured rate-limit prefix are
    removed; unrelated Redis data is never touched.
    """

    from app.core.config import get_settings

    settings = get_settings()
    if settings.rate_limit_backend != "redis":
        return

    try:
        import redis

        client = redis.Redis.from_url(
            settings.redis_url,
            socket_connect_timeout=0.25,
            socket_timeout=0.25,
        )
        try:
            prefix = settings.rate_limit_redis_key_prefix.strip().rstrip(":")
            keys = list(client.scan_iter(match=f"{prefix}:*", count=500))
            if keys:
                client.delete(*keys)
        finally:
            client.close()
    except Exception:  # noqa: BLE001 - outage tests exercise degradation explicitly
        return


@pytest.fixture(autouse=True)
def _default_identity() -> None:
    app.dependency_overrides[require_identity] = lambda: TEST_PRINCIPAL
    _clear_redis_rate_limits()
    reset_limiters()
    yield
    app.dependency_overrides.pop(require_identity, None)
    reset_limiters()


@pytest.fixture
def set_identity():
    """Return a helper that installs a specific principal for one test."""

    def _set(
        tenant_id: UUID = TEST_TENANT_ID,
        *,
        subject: str = "test-user",
        user_id: UUID | None = TEST_PRINCIPAL.user_id,
        roles: frozenset[str] = frozenset({"member", "document_writer"}),
        groups: frozenset[str] = frozenset({"test-group"}),
    ) -> Principal:
        principal = Principal(
            tenant_id=tenant_id,
            subject=subject,
            user_id=user_id,
            roles=roles,
            groups=groups,
        )
        app.dependency_overrides[require_identity] = lambda: principal
        return principal

    return _set


@pytest.fixture
def real_auth():
    """Remove the default identity override for authentication tests."""

    app.dependency_overrides.pop(require_identity, None)
    yield
    app.dependency_overrides[require_identity] = lambda: TEST_PRINCIPAL


# ---------------------------------------------------------------------------
# Disposable PostgreSQL databases
#
# Schema tests must never migrate, alter, or drop the shared development
# database. These fixtures create a uniquely named database on the configured
# PostgreSQL server, hand out a SQLAlchemy engine, and drop only that database.
# Set OPSATLAS_REQUIRE_POSTGRES=1 to turn an unavailable server into a failure
# instead of a skip (useful in CI).
# ---------------------------------------------------------------------------


def _admin_dsn() -> str:
    configured = (
        os.getenv("OPSATLAS_TEST_DATABASE_URL")
        or os.getenv("DATABASE_URL")
        or "postgresql+psycopg://opsatlas:opsatlas_dev_password@localhost:5433/opsatlas"
    )
    url = make_url(configured)
    return url.set(drivername="postgresql", database="postgres").render_as_string(
        hide_password=False
    )


def _postgres_unavailable(reason: str) -> None:
    if os.getenv("OPSATLAS_REQUIRE_POSTGRES", "").lower() in {"1", "true", "yes"}:
        pytest.fail(f"PostgreSQL is required but unavailable: {reason}", pytrace=False)
    pytest.skip(f"PostgreSQL is not available for migration tests: {reason}")


@dataclass
class DisposablePostgres:
    name: str
    url: str
    engine: Engine


@pytest.fixture
def disposable_postgres() -> DisposablePostgres:
    admin_dsn = _admin_dsn()
    name = f"opsatlas_test_{uuid.uuid4().hex}"

    try:
        admin = psycopg.connect(admin_dsn, autocommit=True)
    except Exception as exc:  # noqa: BLE001 - absence must skip, not error
        _postgres_unavailable(type(exc).__name__)

    try:
        admin.execute(f'CREATE DATABASE "{name}"')
    except Exception as exc:  # noqa: BLE001 - surface server capability clearly
        admin.close()
        _postgres_unavailable(f"could not create disposable database: {type(exc).__name__}")

    url = make_url(admin_dsn).set(drivername="postgresql+psycopg", database=name).render_as_string(
        hide_password=False
    )
    engine = create_engine(url, pool_pre_ping=True)

    try:
        yield DisposablePostgres(name=name, url=url, engine=engine)
    finally:
        engine.dispose()
        try:
            admin.execute(
                """
                SELECT pg_terminate_backend(pid)
                FROM pg_stat_activity
                WHERE datname = %s AND pid <> pg_backend_pid()
                """,
                (name,),
            )
            admin.execute(f'DROP DATABASE IF EXISTS "{name}"')
        finally:
            admin.close()


@pytest.fixture
def migrated_postgres(disposable_postgres: DisposablePostgres) -> DisposablePostgres:
    """A disposable database with the complete migration set applied."""

    from app.db.migrations import migrate
    from app.db.schema_validation import reset_schema_validation_cache

    reset_schema_validation_cache()
    try:
        migrate(engine=disposable_postgres.engine)
        yield disposable_postgres
    finally:
        reset_schema_validation_cache()
