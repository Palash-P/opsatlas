"""Shared test identity.

Protected routers require a bearer token resolved by ``require_identity``. The
offline suite is not an authorization test, so the default fixture injects a
fixed principal through FastAPI dependency overrides. Tests that exercise the
real authentication path remove the override with the ``real_auth`` fixture.
"""

from __future__ import annotations

from uuid import UUID

import pytest

from app.core.auth import Principal, require_identity
from app.core.limits import reset_limiters
from app.main import app

TEST_TENANT_ID = UUID("11111111-1111-4111-8111-111111111111")
TEST_PRINCIPAL = Principal(
    tenant_id=TEST_TENANT_ID,
    subject="test-user",
    user_id=UUID("22222222-2222-4222-8222-222222222222"),
    roles=frozenset({"member"}),
    groups=frozenset({"test-group"}),
)


@pytest.fixture(autouse=True)
def _default_identity() -> None:
    app.dependency_overrides[require_identity] = lambda: TEST_PRINCIPAL
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
        roles: frozenset[str] = frozenset({"member"}),
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
