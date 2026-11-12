"""Bearer-token authentication, tenant resolution, and document ACL scope.

Trust model:

- A request is authenticated only by a configured bearer token.
- The token resolves to a :class:`Principal` whose tenant is authoritative.
- A caller-supplied ``tenant_id`` is a compatibility hint at most. It must
  match the authenticated tenant or the request is forbidden.
- Retrieval and citations are scoped to the authenticated tenant and to
  document ACL grants.

The token map is intentionally configuration-only. This is a local deployment
boundary, not an identity provider. It does not create users, rotate secrets,
or support SSO.
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from fastapi import Request

from app.core.config import Settings, get_settings
from app.core.observability import record_tenant_reference


class AuthenticationError(Exception):
    """Raised when a request has no valid authenticated identity."""


class AuthorizationError(Exception):
    """Raised when an authenticated identity may not perform the request."""


@dataclass(frozen=True)
class Principal:
    """Authenticated request identity used for tenant and ACL checks."""

    tenant_id: UUID
    subject: str
    user_id: UUID | None = None
    roles: frozenset[str] = frozenset()
    groups: frozenset[str] = frozenset()

    @property
    def acl_user_ids(self) -> tuple[str, ...]:
        values = [self.subject]
        if self.user_id is not None:
            values.append(str(self.user_id))
        return tuple(dict.fromkeys(value for value in values if value))


def _token_digest(token: str) -> bytes:
    return hashlib.sha256(token.encode("utf-8")).digest()


def _string_set(value: Any) -> frozenset[str]:
    if value is None:
        return frozenset()
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set, frozenset)):
        values = list(value)
    else:
        return frozenset()
    return frozenset(
        item.strip()
        for item in values
        if isinstance(item, str) and item.strip()
    )


def _parse_principal(raw: Any) -> Principal | None:
    """Convert one configured token entry into a principal.

    Malformed entries never authenticate. They are ignored rather than allowed
    to fail open or expose configuration details to the caller.
    """

    if not isinstance(raw, Mapping):
        return None
    tenant_value = raw.get("tenant_id")
    if not isinstance(tenant_value, str) or not tenant_value.strip():
        return None
    try:
        tenant_id = UUID(tenant_value)
    except (TypeError, ValueError):
        return None

    subject = raw.get("subject", "api-key")
    if not isinstance(subject, str) or not subject.strip():
        return None

    user_id_value = raw.get("user_id")
    user_id: UUID | None = None
    if user_id_value is not None:
        if not isinstance(user_id_value, str):
            return None
        try:
            user_id = UUID(user_id_value)
        except ValueError:
            return None

    return Principal(
        tenant_id=tenant_id,
        subject=subject.strip(),
        user_id=user_id,
        roles=_string_set(raw.get("roles")),
        groups=_string_set(raw.get("groups")),
    )


def resolve_token(token: str, settings: Settings | None = None) -> Principal | None:
    """Return the configured principal for a token, or ``None``.

    Every configured digest is compared so a match does not short-circuit the
    loop. Tokens are compared by SHA-256 digest with ``hmac.compare_digest``.
    """

    if not token or len(token) > 4096:
        return None

    settings = settings or get_settings()
    presented = _token_digest(token)
    matched: Any = None
    for configured_token, configured_principal in settings.auth_tokens.items():
        if not isinstance(configured_token, str) or not configured_token:
            continue
        candidate = _token_digest(configured_token)
        if hmac.compare_digest(presented, candidate):
            matched = configured_principal

    if matched is None:
        return None
    return _parse_principal(matched)


def _extract_bearer_token(header_value: str | None) -> str | None:
    if not header_value:
        return None
    parts = header_value.strip().split()
    if len(parts) != 2:
        return None
    scheme, token = parts
    if scheme.lower() != "bearer" or not token:
        return None
    return token


def require_identity(request: Request) -> Principal:
    """FastAPI dependency: authenticate the bearer token and record a safe ref."""

    token = _extract_bearer_token(request.headers.get("authorization"))
    if token is None:
        raise AuthenticationError("missing bearer credentials")

    principal = resolve_token(token)
    if principal is None:
        raise AuthenticationError("invalid bearer credentials")

    # Record the privacy-safe reference as soon as identity is known so that
    # authorization and rate-limit failures are still attributable.
    record_tenant_reference(principal.tenant_id)
    return principal


def resolve_request_tenant(
    requested_tenant_id: UUID | None,
    identity: Principal,
) -> UUID:
    """Return the authoritative tenant for a request.

    The authenticated tenant always wins. A request-body tenant is honored only
    as an equality assertion, and never as proof of authorization.
    """

    if (
        requested_tenant_id is not None
        and requested_tenant_id != identity.tenant_id
    ):
        raise AuthorizationError("requested tenant does not match identity")

    record_tenant_reference(identity.tenant_id)
    return identity.tenant_id


def acl_parameters(principal: Principal) -> dict[str, Any]:
    """Parameters consumed by the document ACL SQL fragment."""

    return {
        "acl_user_ids": list(principal.acl_user_ids),
        "acl_groups": sorted(principal.groups),
        "acl_roles": sorted(principal.roles),
    }


def acl_filter_sql(document_alias: str = "d") -> str:
    """Return the document ACL predicate.

    Precedence is explicit and intentionally simple:

    1. The tenant predicate is always applied first by every retrieval query.
    2. A document with no ACL rows is visible to authenticated members of its
       tenant (the existing tenant-wide default).
    3. A document with any ACL row is visible only through a ``read`` or
       ``manage`` grant matching the principal's subject/user ID, group, or role.
    4. The schema has no deny rows, so there is no deny-overrides-allow branch.

    This is an allow-list augmentation, not a default-deny ACL system. Making
    documents default-deny requires a schema and migration decision.
    """

    return f"""
          AND (
              NOT EXISTS (
                  SELECT 1
                  FROM document_acl AS acl
                  WHERE acl.document_id = {document_alias}.id
              )
              OR EXISTS (
                  SELECT 1
                  FROM document_acl AS acl
                  WHERE acl.document_id = {document_alias}.id
                    AND acl.permission IN ('read', 'manage')
                    AND (
                        (
                            acl.principal_type = 'user'
                            AND acl.principal_id = ANY(
                                CAST(:acl_user_ids AS text[])
                            )
                        )
                        OR (
                            acl.principal_type = 'group'
                            AND acl.principal_id = ANY(
                                CAST(:acl_groups AS text[])
                            )
                        )
                        OR (
                            acl.principal_type = 'role'
                            AND acl.principal_id = ANY(
                                CAST(:acl_roles AS text[])
                            )
                        )
                    )
              )
          )
    """
