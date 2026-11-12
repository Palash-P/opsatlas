"""In-process request rate limiting and bounded concurrency.

These limits are intentionally process-local. They protect one API process from
unbounded provider/DB fan-out and per-tenant request bursts. They are not a
distributed quota: running N processes allows up to N times the configured rate.
A distributed limiter would need Redis or another shared store, which this phase
does not add.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends

from app.core.auth import Principal, require_identity
from app.core.config import Settings, get_settings
from app.core.observability import set_error_category


class RateLimitExceededError(Exception):
    """Raised when a principal exceeds the configured request rate."""

    def __init__(self, retry_after: int) -> None:
        super().__init__("rate limit exceeded")
        self.retry_after = max(1, retry_after)


class ConcurrencyLimitExceededError(Exception):
    """Raised when no request slot is available within the acquire timeout."""

    retry_after = 1


class InMemoryRateLimiter:
    """Thread-safe sliding-window limiter keyed by tenant."""

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
                return False, retry_after
            hits.append(now)
            return True, 0

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


_rate_limiters: dict[tuple[int, float], InMemoryRateLimiter] = {}
_semaphores: dict[int, threading.BoundedSemaphore] = {}
_registry_lock = threading.Lock()


def get_rate_limiter(settings: Settings) -> InMemoryRateLimiter:
    key = (settings.rate_limit_per_minute, settings.rate_limit_window_seconds)
    with _registry_lock:
        limiter = _rate_limiters.get(key)
        if limiter is None:
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
        _rate_limiters.clear()
        _semaphores.clear()


def request_limits(
    identity: Annotated[Principal, Depends(require_identity)],
) -> Iterator[None]:
    """Dependency applied to protected routers.

    Authentication runs first. Rate limiting is keyed by the authenticated
    tenant UUID, never by an unauthenticated header or request-body value.
    """

    settings = get_settings()
    limiter = get_rate_limiter(settings)
    allowed, retry_after = limiter.allow(str(identity.tenant_id))
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
