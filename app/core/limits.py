"""Request rate limiting and bounded concurrency.

Two rate-limit backends are supported:

``memory`` (default)
    A thread-safe sliding window per process. This is correct for a single API
    process, local development, and tests. Running N processes allows up to N
    times the configured rate per tenant.

``redis``
    A shared sliding window implemented atomically in Lua. Every API replica
    using the same Redis database enforces one per-tenant limit. The key is
    ``<prefix>:<salted SHA-256 of the tenant UUID>``; the raw tenant UUID and the
    bearer token never appear in Redis keys or logs.

A Redis outage must never turn limiting into an unlimited path. With
``rate_limit_redis_fallback_to_local=true`` (the default) the limiter falls back
to the in-process window, logs one sanitized degradation event, and keeps
enforcing a bounded (process-local) rate. With the flag disabled it fails closed
with :class:`RateLimitBackendUnavailableError` (HTTP 503).

The concurrency slot is intentionally process-local in both modes: it bounds
provider/DB fan-out inside one process, which is a different concern from the
cross-replica request quota.
"""

from __future__ import annotations

import hashlib
import logging
import math
import threading
import time
import uuid
from collections import deque
from collections.abc import Iterator
from contextlib import suppress
from typing import Annotated, Protocol

from fastapi import Depends
from redis import Redis
from redis.exceptions import RedisError

from app.core.auth import Principal, require_identity
from app.core.config import Settings, get_settings
from app.core.observability import SCHEMA_LOGGER_NAME, emit_operational_event, set_error_category

logger = logging.getLogger(SCHEMA_LOGGER_NAME)


class RateLimitExceededError(Exception):
    """Raised when a principal exceeds the configured request rate."""

    def __init__(self, retry_after: int) -> None:
        super().__init__("rate limit exceeded")
        self.retry_after = max(1, retry_after)


class RateLimitBackendUnavailableError(Exception):
    """Raised when the shared limiter is required but cannot be reached."""

    retry_after = 1


class ConcurrencyLimitExceededError(Exception):
    """Raised when no request slot is available within the acquire timeout."""

    retry_after = 1


class RateLimiter(Protocol):
    def allow(self, key: str) -> tuple[bool, int]:
        """Return ``(allowed, retry_after_seconds)`` for one opaque key."""


class InMemoryRateLimiter:
    """Thread-safe sliding-window limiter keyed by an opaque key."""

    def __init__(self, *, limit: int, window_seconds: float) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._lock = threading.Lock()
        self._hits: dict[str, deque[float]] = {}

    def allow(self, key: str) -> tuple[bool, int]:
        if self.limit <= 0:
            return True, 0
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if len(hits) >= self.limit:
                retry_after = math.ceil(self.window_seconds - (now - hits[0]))
                return False, max(1, retry_after)
            hits.append(now)
            return True, 0

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


# A single atomic script: prune the window, count, reject or append, and bound
# the key lifetime. Redis server time is used so replicas with skewed clocks
# still share one window. The member is a per-call random suffix so two requests
# that arrive in the same millisecond cannot overwrite each other's score.
_REDIS_SLIDING_WINDOW = """
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local window_ms = tonumber(ARGV[2])
local member = ARGV[3]
local clock = redis.call('TIME')
local now_ms = tonumber(clock[1]) * 1000 + math.floor(tonumber(clock[2]) / 1000)
local cutoff = now_ms - window_ms
redis.call('ZREMRANGEBYSCORE', key, '-inf', cutoff)
local count = redis.call('ZCARD', key)
if count >= limit then
    local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
    local retry = 1
    if oldest[2] then
        retry = math.ceil((tonumber(oldest[2]) + window_ms - now_ms) / 1000)
    end
    if retry < 1 then retry = 1 end
    redis.call('PEXPIRE', key, window_ms)
    return {0, retry}
end
redis.call('ZADD', key, now_ms, member)
redis.call('PEXPIRE', key, window_ms)
return {1, 0}
"""


def _tenant_key(settings: Settings, tenant: str) -> str:
    """Derive an opaque, salted key that reveals neither tenant nor token."""

    digest = hashlib.sha256(
        settings.auth_log_tenant_salt.encode("utf-8") + b"\x00" + tenant.encode("utf-8")
    ).hexdigest()
    prefix = settings.rate_limit_redis_key_prefix.strip().rstrip(":")
    return f"{prefix}:{digest}"


class RedisRateLimiter:
    """Shared sliding-window limiter with explicit degradation behavior."""

    def __init__(
        self,
        *,
        client: Redis,
        limit: int,
        window_seconds: float,
        fallback: InMemoryRateLimiter | None,
    ) -> None:
        self.client = client
        self.limit = limit
        self.window_seconds = window_seconds
        self.fallback = fallback
        self._lock = threading.Lock()
        self._degraded = False

    def allow(self, key: str) -> tuple[bool, int]:
        if self.limit <= 0:
            return True, 0
        if not key:
            # Never allow an empty key: it would collapse every tenant into one
            # bucket. The caller always passes a hashed tenant key.
            return True, 0
        try:
            result = self.client.eval(
                _REDIS_SLIDING_WINDOW,
                1,
                key,
                self.limit,
                int(self.window_seconds * 1000),
                uuid.uuid4().hex,
            )
            allowed, retry_after = int(result[0]), int(result[1])
        except (RedisError, OSError, ValueError) as exc:
            return self._degrade(key, exc)
        self._recover()
        return bool(allowed), retry_after

    def _degrade(self, key: str, exc: Exception) -> tuple[bool, int]:
        # Log only the exception class. The message can contain a Redis URL or
        # tenant-derived key, so it is deliberately not included.
        with self._lock:
            first_failure = not self._degraded
            self._degraded = True
        if first_failure:
            emit_operational_event(
                logger,
                "rate_limit_backend_degraded",
                level=logging.WARNING,
                message="shared rate limiter unavailable",
                outcome="degraded",
                error_category="rate_limit_backend_unavailable",
                error_type=type(exc).__name__,
                fallback="memory" if self.fallback is not None else "fail_closed",
            )
        if self.fallback is None:
            raise RateLimitBackendUnavailableError from exc
        return self.fallback.allow(key)

    def _recover(self) -> None:
        with self._lock:
            if not self._degraded:
                return
            self._degraded = False
        emit_operational_event(
            logger,
            "rate_limit_backend_recovered",
            message="shared rate limiter recovered",
            outcome="ok",
        )

    def close(self) -> None:
        with suppress(Exception):
            self.client.close()


_rate_limiters: dict[tuple, RateLimiter] = {}
_semaphores: dict[int, threading.BoundedSemaphore] = {}
_registry_lock = threading.Lock()


def _redis_client(settings: Settings) -> Redis:
    return Redis.from_url(
        settings.redis_url,
        socket_connect_timeout=settings.rate_limit_redis_timeout_seconds,
        socket_timeout=settings.rate_limit_redis_timeout_seconds,
        decode_responses=False,
    )


def get_rate_limiter(settings: Settings) -> RateLimiter:
    key = (
        settings.rate_limit_backend,
        settings.rate_limit_per_minute,
        settings.rate_limit_window_seconds,
        settings.rate_limit_redis_key_prefix,
    )
    with _registry_lock:
        limiter = _rate_limiters.get(key)
        if limiter is not None:
            return limiter

        if settings.rate_limit_backend == "redis":
            fallback: InMemoryRateLimiter | None = None
            if settings.rate_limit_redis_fallback_to_local:
                fallback = InMemoryRateLimiter(
                    limit=settings.rate_limit_per_minute,
                    window_seconds=settings.rate_limit_window_seconds,
                )
            limiter = RedisRateLimiter(
                client=_redis_client(settings),
                limit=settings.rate_limit_per_minute,
                window_seconds=settings.rate_limit_window_seconds,
                fallback=fallback,
            )
        else:
            limiter = InMemoryRateLimiter(
                limit=settings.rate_limit_per_minute,
                window_seconds=settings.rate_limit_window_seconds,
            )
        _rate_limiters[key] = limiter
        return limiter


def get_concurrency_semaphore(settings: Settings) -> threading.BoundedSemaphore | None:
    if settings.max_concurrent_requests <= 0:
        return None
    key = settings.max_concurrent_requests
    with _registry_lock:
        semaphore = _semaphores.get(key)
        if semaphore is None:
            semaphore = threading.BoundedSemaphore(key)
            _semaphores[key] = semaphore
        return semaphore


def reset_limiters() -> None:
    """Test helper: drop cached limiter and semaphore state."""

    with _registry_lock:
        limiters = list(_rate_limiters.values())
        _rate_limiters.clear()
        _semaphores.clear()
    for limiter in limiters:
        if isinstance(limiter, RedisRateLimiter):
            limiter.close()


def request_limits(
    identity: Annotated[Principal, Depends(require_identity)],
) -> Iterator[None]:
    """Dependency applied to protected routers.

    Authentication runs first. Rate limiting is keyed by the authenticated
    tenant UUID, never by an unauthenticated header or request-body value; the
    tenant is hashed before it becomes a Redis key.
    """

    settings = get_settings()
    limiter = get_rate_limiter(settings)
    key = _tenant_key(settings, str(identity.tenant_id))
    allowed, retry_after = limiter.allow(key)
    if not allowed:
        set_error_category("rate_limited")
        raise RateLimitExceededError(retry_after)

    semaphore = get_concurrency_semaphore(settings)
    if semaphore is None:
        yield
        return

    acquired = semaphore.acquire(timeout=settings.concurrency_acquire_timeout_seconds)
    if not acquired:
        set_error_category("concurrency_limited")
        raise ConcurrencyLimitExceededError
    try:
        yield
    finally:
        semaphore.release()
