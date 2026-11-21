"""Deployment-grade configuration guardrails.

These tests only construct ``Settings`` objects; they never open a connection.
"""

from __future__ import annotations

import pytest

from app.core.config import (
    LOCAL_DEVELOPMENT_DATABASE_PASSWORD,
    LOCAL_DEVELOPMENT_TENANT_SALT,
    Settings,
)


def _production(**overrides) -> Settings:
    values = {
        "app_env": "production",
        "database_url": (
            "postgresql+psycopg://opsatlas:correct-horse-battery-staple@"
            "db.internal:5432/opsatlas"
        ),
        "auth_tokens": {
            "long-random-token-0123456789abcdef0123456789abcdef": {
                "tenant_id": "00000000-0000-4000-8000-000000000001",
                "subject": "deployer",
                "roles": ["member"],
                "groups": [],
            }
        },
        "auth_log_tenant_salt": "deployment-specific-random-salt",
        "rate_limit_backend": "redis",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_production_rejects_the_local_development_password() -> None:
    with pytest.raises(ValueError) as excinfo:
        _production(
            database_url=(
                "postgresql+psycopg://opsatlas:"
                f"{LOCAL_DEVELOPMENT_DATABASE_PASSWORD}@db.internal:5432/opsatlas"
            )
        )

    assert "DATABASE_URL still uses the local development password" in str(excinfo.value)


def test_production_rejects_an_empty_token_map() -> None:
    with pytest.raises(ValueError) as excinfo:
        _production(auth_tokens={})

    assert "AUTH_TOKENS is empty" in str(excinfo.value)


def test_production_rejects_the_default_tenant_log_salt() -> None:
    with pytest.raises(ValueError) as excinfo:
        _production(auth_log_tenant_salt=LOCAL_DEVELOPMENT_TENANT_SALT)

    assert "AUTH_LOG_TENANT_SALT still uses the local default" in str(excinfo.value)


def test_production_accepts_deployment_secrets() -> None:
    settings = _production()

    assert settings.app_env == "production"
    assert settings.auth_tokens


def test_local_environment_keeps_working_with_placeholders(monkeypatch) -> None:
    # ``_env_file=None`` disables the dotenv file only. pydantic-settings still
    # reads the process environment, so an ambient DATABASE_URL (for example from
    # CI or a developer shell) would override the local defaults this test is
    # asserting. Remove the inherited values so the test exercises the defaults.
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("AUTH_TOKENS", raising=False)

    settings = Settings(_env_file=None, app_env="local")

    assert settings.app_env == "local"
    assert LOCAL_DEVELOPMENT_DATABASE_PASSWORD in settings.database_url
    assert settings.auth_tokens == {}
