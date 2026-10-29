"""Document ingestion API."""

from typing import Literal
from uuid import UUID

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field

from app.services.ingestion import EmptyDocumentError, TenantNotFoundError, ingest_document

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


class DocumentIngestRequest(BaseModel):
    tenant_id: UUID
    source_key: str = Field(min_length=1, max_length=500)
    title: str = Field(min_length=1, max_length=500)
    source_type: Literal["markdown", "text", "runbook", "incident_report", "api_spec"] = "markdown"
    content: str = Field(min_length=1, max_length=2_000_000)
    source_uri: str | None = Field(default=None, max_length=2000)
    authority_level: int = Field(default=50, ge=0, le=100)


@router.post("/ingest", status_code=201)
def ingest_document_endpoint(
    request: DocumentIngestRequest, response: Response
) -> dict:
    """Ingest a text document and create a new immutable version when content changes."""
    try:
        result = ingest_document(**request.model_dump())
        # A duplicate submission is an idempotent read of an existing version, not a new create.
        response.status_code = 200 if result["deduplicated"] else 201
        return result
    except TenantNotFoundError:
        raise HTTPException(status_code=404, detail="Tenant not found") from None
    except EmptyDocumentError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
