"""Shared Redis rate limiter, degradation behavior, and key hygiene.

Tests that need a real Redis server skip when it is unreachable unless
``OPSATLAS_REQUIRE_REDIS=1`` is set (CI sets it). Every test uses a unique key
prefix/tenant and deletes only the keys it created.
"""

from __future__ import annotations

import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from redis import Redis

from app.core import limits as limits_module
from app.core.config import Settings
from app.core.limits import (
    InMemoryRateLimiter,
    RateLimitBackendUnavailableError,
    RedisRateLimiter,
    _tenant_key,
    get_rate_limiter,
    reset_limiters,
)
from app.main import app

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
TENANT_A = UUID("11111111-1111-4111-8111-111111111111")
TENANT_B = UUID("44444444-4444-4444-8444-444444444444")
client = TestClient(app)


def _redis_or_skip() -> Redis:
    redis_client = Redis.from_url(
        REDIS_URL, socket_connect_timeout=0.5, socket_timeout=0.5
    )
    try:
        redis_client.ping()
    except Exception as exc:  # noqa: BLE001 - absence skips, requirement fails
        redis_client.close()
        reason = f"Redis is not available: {type(exc).__name__}"
        if os.getenv("OPSATLAS_REQUIRE_REDIS", "").lower() in {"1", "true", "yes"}:
            pytest.fail(f"Redis is required but unavailable: {reason}", pytrace=False)
        pytest.skip(reason)
    return redis_client


def _settings(**overrides) -> Settings:
    values = {
        "redis_url": REDIS_URL,
        "auth_log_tenant_salt": "rate-limit-test-salt",
        "rate_limit_backend": "redis",
        "rate_limit_per_minute": 2,
        "rate_limit_window_seconds": 60.0,
        "rate_limit_redis_key_prefix": f"opsatlas:test:ratelimit:{uuid.uuid4().hex}",
        "rate_limit_redis_timeout_seconds": 0.5,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.fixture
def redis_server():
    redis_client = _redis_or_skip()
    created: list[str] = []

    def track(key: str) -> str:
        created.append(key)
        return key

    yield redis_client, track

    if created:
        redis_client.delete(*created)
    redis_client.close()
    reset_limiters()


def test_shared_limit_is_enforced_across_replicas(redis_server) -> None:
    redis_client, track = redis_server
    settings = _settings()
    key = track(_tenant_key(settings, str(TENANT_A)))

    first = RedisRateLimiter(
        client=redis_client, limit=2, window_seconds=60, fallback=None
    )
    second = RedisRateLimiter(
        client=redis_client, limit=2, window_seconds=60, fallback=None
    )

    assert first.allow(key) == (True, 0)
    assert second.allow(key) == (True, 0)
    allowed, retry_after = first.allow(key)
    assert allowed is False
    assert retry_after >= 1


def test_tenants_have_independent_shared_buckets(redis_server) -> None:
    redis_client, track = redis_server
    settings = _settings()
    key_a = track(_tenant_key(settings, str(TENANT_A)))
    key_b = track(_tenant_key(settings, str(TENANT_B)))

    limiter = RedisRateLimiter(
        client=redis_client, limit=1, window_seconds=60, fallback=None
    )

    assert limiter.allow(key_a)[0] is True
    assert limiter.allow(key_a)[0] is False
    assert limiter.allow(key_b)[0] is True
    assert limiter.allow(key_b)[0] is False
    assert key_a != key_b


def test_concurrent_requests_cannot_exceed_the_shared_limit(redis_server) -> None:
    redis_client, track = redis_server
    settings = _settings()
    key = track(_tenant_key(settings, str(TENANT_A)))

    first = RedisRateLimiter(
        client=redis_client, limit=5, window_seconds=60, fallback=None
    )
    second = RedisRateLimiter(
        client=redis_client, limit=5, window_seconds=60, fallback=None
    )

    def attempt(index: int) -> bool:
        limiter = first if index % 2 == 0 else second
        return limiter.allow(key)[0]

    with ThreadPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(attempt, range(40)))

    assert sum(results) == 5


def test_redis_key_contains_neither_tenant_nor_token() -> None:
    settings = _settings()
    token = "super-secret-bearer-token-value"
    key = _tenant_key(settings, str(TENANT_A))

    assert key.startswith(settings.rate_limit_redis_key_prefix)
    assert str(TENANT_A) not in key
    assert str(TENANT_A).replace("-", "") not in key
    assert token not in key
    assert len(key.rsplit(":", 1)[-1]) == 64


def test_redis_outage_falls_back_to_a_bounded_local_limit() -> None:
    unreachable = Redis.from_url(
        "redis://127.0.0.1:1/0", socket_connect_timeout=0.05, socket_timeout=0.05
    )
    limiter = RedisRateLimiter(
        client=unreachable,
        limit=1,
        window_seconds=60,
        fallback=InMemoryRateLimiter(limit=1, window_seconds=60),
    )
    try:
        assert limiter.allow("opaque-key")[0] is True
        # Degradation must not become an unlimited path.
        assert limiter.allow("opaque-key")[0] is False
    finally:
        limiter.close()


def test_redis_outage_fails_closed_when_fallback_is_disabled() -> None:
    unreachable = Redis.from_url(
        "redis://127.0.0.1:1/0", socket_connect_timeout=0.05, socket_timeout=0.05
    )
    limiter = RedisRateLimiter(
        client=unreachable, limit=10, window_seconds=60, fallback=None
    )
    try:
        with pytest.raises(RateLimitBackendUnavailableError):
            limiter.allow("opaque-key")
    finally:
        limiter.close()


def test_backend_selection_uses_memory_without_redis() -> None:
    reset_limiters()
    settings = _settings(rate_limit_backend="memory", rate_limit_per_minute=1)
    limiter = get_rate_limiter(settings)

    assert isinstance(limiter, InMemoryRateLimiter)


def test_endpoint_returns_503_when_shared_limiter_is_required_but_down(monkeypatch) -> None:
    reset_limiters()
    settings = _settings(
        redis_url="redis://127.0.0.1:1/0",
        rate_limit_redis_timeout_seconds=0.05,
        rate_limit_redis_fallback_to_local=False,
    )
    monkeypatch.setattr(limits_module, "get_settings", lambda: settings)
    try:
        response = client.post(
            "/api/v1/search",
            json={"tenant_id": str(TENANT_A), "query": "payment errors"},
        )
    finally:
        reset_limiters()

    assert response.status_code == 503
    assert response.json() == {"detail": "A required dependency is unavailable"}
    assert "127.0.0.1" not in response.text
    assert str(TENANT_A) not in response.text
