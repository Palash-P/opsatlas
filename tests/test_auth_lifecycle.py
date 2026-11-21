"""Bearer-token configuration, rotation, revocation, and expiry.

The token map is deployment configuration. These tests exercise the real
resolution path (no dependency override) and the startup validator, because a
regression here would either reject a valid deployment or authenticate a token
that should have been revoked.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import UUID

import pytest

from app.core import auth as auth_module
from app.core.auth import Principal, resolve_token
from app.core.auth_tokens import (
    MIN_PRODUCTION_TOKEN_LENGTH,
    parse_expires_at,
    validate_token_map,
)
from app.core.config import Settings

TENANT = UUID("11111111-1111-4111-8111-111111111111")
OTHER_TENANT = UUID("33333333-3333-4333-8333-333333333333")
TOKEN = "t" * MIN_PRODUCTION_TOKEN_LENGTH
ROTATED_TOKEN = "r" * MIN_PRODUCTION_TOKEN_LENGTH
EXPIRED_TOKEN = "e" * MIN_PRODUCTION_TOKEN_LENGTH


def _settings(tokens: dict, *, app_env: str = "local") -> Settings:
    return Settings(
        _env_file=None,
        app_env=app_env,
        auth_tokens=tokens,
        rate_limit_backend="redis" if app_env == "production" else "memory",
        auth_log_tenant_salt="test-salt",
        database_url=(
            "postgresql+psycopg://opsatlas:not-the-local-default@db.internal:5432/opsatlas"
        ),
    )


def _entry(**overrides) -> dict:
    values = {
        "tenant_id": str(TENANT),
        "subject": "alice",
        "user_id": "22222222-2222-4222-8222-222222222222",
        "roles": ["member", "document_writer"],
        "groups": ["on-call"],
    }
    values.update(overrides)
    return values


def test_valid_token_resolves_the_configured_principal(monkeypatch) -> None:
    settings = _settings({TOKEN: _entry()})
    monkeypatch.setattr(auth_module, "get_settings", lambda: settings)

    principal = resolve_token(TOKEN)

    assert principal is not None
    assert principal.tenant_id == TENANT
    assert principal.subject == "alice"
    assert principal.roles == frozenset({"member", "document_writer"})
    assert principal.groups == frozenset({"on-call"})
    assert principal.acl_user_ids == ("alice", "22222222-2222-4222-8222-222222222222")
    assert principal.expires_at is None


def test_unknown_and_empty_tokens_never_authenticate(monkeypatch) -> None:
    settings = _settings({TOKEN: _entry()})
    monkeypatch.setattr(auth_module, "get_settings", lambda: settings)

    assert resolve_token("not-configured") is None
    assert resolve_token("") is None
    assert resolve_token("x" * 5000) is None


def test_expired_token_is_rejected_at_request_time(monkeypatch) -> None:
    past = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    settings = _settings({EXPIRED_TOKEN: _entry(expires_at=past)})
    monkeypatch.setattr(auth_module, "get_settings", lambda: settings)

    assert resolve_token(EXPIRED_TOKEN) is None


def test_token_stops_authenticating_after_its_expiry_instant(monkeypatch) -> None:
    soon = (datetime.now(UTC) + timedelta(seconds=1)).isoformat()
    settings = _settings({TOKEN: _entry(expires_at=soon)})
    monkeypatch.setattr(auth_module, "get_settings", lambda: settings)

    # The expiry is evaluated per request, so no restart is needed for the
    # transition from valid to invalid.
    assert resolve_token(TOKEN) is not None

    monkeypatch.setattr(
        auth_module,
        "token_is_expired",
        lambda expires_at: expires_at is not None,
    )
    assert resolve_token(TOKEN) is None


def test_expiry_accepts_z_suffix_and_naive_utc_timestamps() -> None:
    assert parse_expires_at("2030-01-02T03:04:05Z") == datetime(
        2030, 1, 2, 3, 4, 5, tzinfo=UTC
    )
    assert parse_expires_at("2030-01-02T03:04:05") == datetime(
        2030, 1, 2, 3, 4, 5, tzinfo=UTC
    )
    assert parse_expires_at(None) is None


@pytest.mark.parametrize(
    "value",
    ["", "   ", "tomorrow", 123, True, "2030-13-40T00:00:00Z"],
)
def test_malformed_expiry_never_authenticates(monkeypatch, value) -> None:
    # Startup validation rejects this configuration, so build the settings
    # object directly to prove request handling still fails closed.
    settings = SimpleNamespace(auth_tokens={TOKEN: _entry(expires_at=value)})
    monkeypatch.setattr(auth_module, "get_settings", lambda: settings)

    assert resolve_token(TOKEN) is None


def test_malformed_entry_never_authenticates(monkeypatch) -> None:
    settings = SimpleNamespace(
        auth_tokens={TOKEN: {"tenant_id": "not-a-uuid", "subject": "alice"}}
    )
    monkeypatch.setattr(auth_module, "get_settings", lambda: settings)

    assert resolve_token(TOKEN) is None


def test_revocation_by_removal_takes_effect_on_resolve(monkeypatch) -> None:
    before = _settings({TOKEN: _entry(), ROTATED_TOKEN: _entry(subject="bob")})
    monkeypatch.setattr(auth_module, "get_settings", lambda: before)

    assert resolve_token(TOKEN) is not None
    assert resolve_token(ROTATED_TOKEN) is not None

    after = _settings({ROTATED_TOKEN: _entry(subject="bob")})
    monkeypatch.setattr(auth_module, "get_settings", lambda: after)

    assert resolve_token(TOKEN) is None
    assert resolve_token(ROTATED_TOKEN) is not None


def test_rotation_overlap_keeps_both_tokens_valid(monkeypatch) -> None:
    settings = _settings(
        {
            TOKEN: _entry(subject="alice"),
            ROTATED_TOKEN: _entry(subject="alice"),
        }
    )
    monkeypatch.setattr(auth_module, "get_settings", lambda: settings)

    old = resolve_token(TOKEN)
    new = resolve_token(ROTATED_TOKEN)

    assert old is not None and new is not None
    assert old.tenant_id == new.tenant_id == TENANT
    assert old.subject == new.subject == "alice"


def test_tenant_identity_cannot_be_overridden_by_entry_order(monkeypatch) -> None:
    settings = _settings(
        {
            TOKEN: _entry(tenant_id=str(TENANT)),
            ROTATED_TOKEN: _entry(tenant_id=str(OTHER_TENANT)),
        }
    )
    monkeypatch.setattr(auth_module, "get_settings", lambda: settings)

    assert resolve_token(TOKEN).tenant_id == TENANT
    assert resolve_token(ROTATED_TOKEN).tenant_id == OTHER_TENANT


def test_validation_problems_never_include_token_values() -> None:
    secret_token = "super-secret-token-value-that-must-not-leak-123"
    problems = validate_token_map(
        {secret_token: {"tenant_id": "not-a-uuid", "subject": ""}},
        production=True,
    )

    assert problems
    combined = "\n".join(problems)
    assert secret_token not in combined
    assert str(TENANT) not in combined


def test_local_development_allows_short_documented_tokens() -> None:
    settings = _settings({"development-token": _entry()})

    assert settings.app_env == "local"


def test_local_rejects_structurally_invalid_entries() -> None:
    with pytest.raises(ValueError) as excinfo:
        _settings({"development-token": _entry(tenant_id="not-a-uuid")})

    assert "invalid AUTH_TOKENS configuration" in str(excinfo.value)
    assert "development-token" not in str(excinfo.value)


def test_local_rejects_unknown_entry_fields() -> None:
    with pytest.raises(ValueError) as excinfo:
        _settings({"development-token": _entry(permissions=["admin"])})

    assert "unsupported field" in str(excinfo.value)


def test_production_rejects_short_tokens() -> None:
    with pytest.raises(ValueError) as excinfo:
        _settings({"short-token": _entry()}, app_env="production")

    assert "shorter than the production minimum" in str(excinfo.value)


def test_production_rejects_documented_placeholder_tokens() -> None:
    placeholder = "replace-with-a-long-random-token"
    assert len(placeholder) >= MIN_PRODUCTION_TOKEN_LENGTH

    with pytest.raises(ValueError) as excinfo:
        _settings({placeholder: _entry()}, app_env="production")

    assert "documented placeholder token" in str(excinfo.value)
    assert placeholder not in str(excinfo.value)


def test_production_rejects_a_map_with_only_expired_tokens() -> None:
    past = (datetime.now(UTC) - timedelta(days=1)).isoformat()

    with pytest.raises(ValueError) as excinfo:
        _settings({TOKEN: _entry(expires_at=past)}, app_env="production")

    assert "no usable, unexpired token" in str(excinfo.value)


def test_production_requires_the_shared_redis_limiter() -> None:
    with pytest.raises(ValueError) as excinfo:
        Settings(
            _env_file=None,
            app_env="production",
            database_url=(
                "postgresql+psycopg://opsatlas:not-the-local-default@"
                "db.internal:5432/opsatlas"
            ),
            auth_tokens={TOKEN: _entry()},
            auth_log_tenant_salt="deployment-specific-random-salt",
            rate_limit_backend="memory",
        )

    assert "rate_limit_backend must be 'redis'" in str(excinfo.value)


def test_unknown_rate_limit_backend_is_rejected() -> None:
    with pytest.raises(ValueError) as excinfo:
        Settings(_env_file=None, rate_limit_backend="sidecar")

    assert "rate_limit_backend must be 'memory' or 'redis'" in str(excinfo.value)


def test_principal_defaults_preserve_existing_constructor_calls() -> None:
    principal = Principal(tenant_id=TENANT, subject="alice")

    assert principal.expires_at is None
    assert principal.roles == frozenset()
    assert principal.groups == frozenset()
