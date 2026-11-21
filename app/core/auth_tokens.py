"""Validation and lifecycle helpers for the configured bearer-token map.

The token map is a deployment configuration boundary, not an identity provider.
This module keeps its rules in one place so startup validation, request-time
authentication, and tests cannot drift:

- a token is a nonempty opaque string and never appears in an error message;
- an entry maps to one authoritative tenant, subject, optional user ID, roles,
  groups, and an optional ISO-8601 expiry;
- malformed entries never authenticate;
- production rejects short, placeholder, or empty token configurations;
- revoking a token means removing its entry and restarting/redeploying;
- rotating a token means overlapping the old and new entry, then removing the
  old one.

There is deliberately no token database, OAuth, SSO, JWT refresh, or admin API.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

# A production bearer token should be long enough that it cannot be guessed or
# accidentally confused with a development placeholder. This is a configuration
# floor, not a cryptographic key-derivation mechanism.
MIN_PRODUCTION_TOKEN_LENGTH = 32

# Exact placeholder values that must never authenticate a production process.
UNSAFE_TOKEN_VALUES = frozenset(
    {
        "replace-with-a-long-random-token",
        "replace-me",
        "changeme",
        "change-me",
        "secret",
        "token",
    }
)

_ALLOWED_ENTRY_FIELDS = frozenset(
    {"tenant_id", "subject", "user_id", "roles", "groups", "expires_at"}
)


def parse_expires_at(value: Any) -> datetime | None:
    """Parse an optional ISO-8601 expiry into an aware UTC datetime.

    ``None`` means the token does not expire. A trailing ``Z`` is accepted.
    Naive timestamps are interpreted as UTC. This function raises ``ValueError``
    with a caller-safe message; it never includes the token value.
    """

    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("expires_at must be an ISO-8601 timestamp or null")
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError("expires_at must be a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def token_is_expired(expires_at: datetime | None, *, now: datetime | None = None) -> bool:
    if expires_at is None:
        return False
    current = now or datetime.now(UTC)
    return expires_at <= current


def _validate_string_set(value: Any, field: str, problems: list[str]) -> None:
    if value is None:
        return
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set, frozenset)):
        values = list(value)
    else:
        problems.append(f"{field} must be a string or a list of strings")
        return
    if any(not isinstance(item, str) or not item.strip() for item in values):
        problems.append(f"{field} must contain only nonempty strings")


def _validate_entry(raw: Any, label: str, problems: list[str], *, now: datetime) -> bool:
    """Validate one entry; return whether it is a usable, unexpired principal."""

    if not isinstance(raw, Mapping):
        problems.append(f"{label} must be a mapping")
        return False

    unknown = sorted(set(raw) - _ALLOWED_ENTRY_FIELDS)
    if unknown:
        problems.append(
            f"{label} contains unsupported field(s): {', '.join(str(item) for item in unknown)}"
        )

    tenant_value = raw.get("tenant_id")
    if not isinstance(tenant_value, str) or not tenant_value.strip():
        problems.append(f"{label}.tenant_id must be a UUID string")
    else:
        try:
            UUID(tenant_value)
        except (TypeError, ValueError):
            problems.append(f"{label}.tenant_id must be a valid UUID")

    subject = raw.get("subject", "api-key")
    if not isinstance(subject, str) or not subject.strip():
        problems.append(f"{label}.subject must be a nonempty string")

    user_id_value = raw.get("user_id")
    if user_id_value is not None:
        if not isinstance(user_id_value, str):
            problems.append(f"{label}.user_id must be a UUID string")
        else:
            try:
                UUID(user_id_value)
            except ValueError:
                problems.append(f"{label}.user_id must be a valid UUID")

    _validate_string_set(raw.get("roles"), f"{label}.roles", problems)
    _validate_string_set(raw.get("groups"), f"{label}.groups", problems)

    try:
        expires_at = parse_expires_at(raw.get("expires_at"))
    except ValueError as exc:
        problems.append(f"{label}.{exc}")
        return False

    return not token_is_expired(expires_at, now=now)


def validate_token_map(
    token_map: Mapping[Any, Any],
    *,
    production: bool,
    now: datetime | None = None,
) -> list[str]:
    """Return safe, non-secret configuration problems for ``AUTH_TOKENS``.

    Error text includes the entry position and field name but never the token
    value, tenant UUID, subject, or other configured identity data.
    """

    current = now or datetime.now(UTC)
    problems: list[str] = []
    usable = 0

    for index, (token, raw) in enumerate(token_map.items()):
        label = f"AUTH_TOKENS entry {index}"
        if not isinstance(token, str) or not token.strip():
            problems.append(f"{label} has an empty token")
            continue
        if production:
            if len(token) < MIN_PRODUCTION_TOKEN_LENGTH:
                problems.append(
                    f"{label} is shorter than the production minimum of "
                    f"{MIN_PRODUCTION_TOKEN_LENGTH} characters"
                )
            if token.strip().lower() in UNSAFE_TOKEN_VALUES:
                problems.append(f"{label} uses a documented placeholder token")
        if _validate_entry(raw, label, problems, now=current):
            usable += 1

    if production and not problems and usable == 0:
        problems.append(
            "AUTH_TOKENS has no usable, unexpired token; every protected endpoint "
            "would reject requests"
        )
    return problems
