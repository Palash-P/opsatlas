"""Document ingestion API with authenticated tenant scoping."""

from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, Field

from app.core.auth import Principal, require_identity, resolve_request_tenant
from app.core.limits import request_limits
from app.core.observability import set_error_category
from app.services.ingestion import EmptyDocumentError, TenantNotFoundError, ingest_document

router = APIRouter(
    prefix="/api/v1/documents",
    tags=["documents"],
    dependencies=[Depends(request_limits)],
)


class DocumentIngestRequest(BaseModel):
    # Deprecated compatibility field. When supplied it must match the tenant
    # resolved from the bearer token.
    tenant_id: UUID | None = None
    source_key: str = Field(min_length=1, max_length=500)
    title: str = Field(min_length=1, max_length=500)
    source_type: Literal["markdown", "text", "runbook", "incident_report", "api_spec"] = "markdown"
    content: str = Field(min_length=1, max_length=2_000_000)
    source_uri: str | None = Field(default=None, max_length=2000)
    authority_level: int = Field(default=50, ge=0, le=100)


@router.post("/ingest", status_code=201)
def ingest_document_endpoint(
    request: DocumentIngestRequest,
    response: Response,
    identity: Annotated[Principal, Depends(require_identity)],
) -> dict:
    """Ingest a text document and create a new immutable version when content changes."""
    tenant_id = resolve_request_tenant(request.tenant_id, identity)
    try:
        result = ingest_document(
            tenant_id=tenant_id,
            source_key=request.source_key,
            title=request.title,
            source_type=request.source_type,
            content=request.content,
            source_uri=request.source_uri,
            authority_level=request.authority_level,
        )
        # A duplicate submission is an idempotent read of an existing version, not a new create.
        response.status_code = 200 if result["deduplicated"] else 201
        return result
    except TenantNotFoundError:
        set_error_category("tenant_not_found")
        raise HTTPException(status_code=404, detail="Tenant not found") from None
    except EmptyDocumentError as exc:
        set_error_category("empty_document")
        raise HTTPException(status_code=422, detail=str(exc)) from None
