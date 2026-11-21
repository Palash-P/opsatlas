"""Document ingestion API with authenticated tenant scoping."""

from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, Field

from app.core.auth import Principal, require_identity, resolve_request_tenant
from app.core.limits import request_limits
from app.core.observability import (
    emit_operational_event,
    set_error_category,
    tenant_reference,
)
from app.core.observability import (
    logger as request_logger,
)
from app.db.embedding_schema import EmbeddingSchemaError
from app.services.embedding_client import (
    EmbeddingServiceError,
    EmbeddingUnavailableError,
    EmbeddingValidationError,
)
from app.services.ingestion import (
    EmbeddingCompletenessError,
    EmptyDocumentError,
    IdempotencyKeyCollisionError,
    IngestionExecutionLeaseLostError,
    IngestionJobBusyError,
    IngestionJobNotFoundError,
    IngestionQueueUnavailableError,
    SubmittedIngestion,
    TenantNotFoundError,
    get_ingestion_job_status,
    ingest_document,
    submit_ingestion_job,
)
from app.tasks.ingestion import publish_ingestion_job

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
            principal=identity,
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
    except IdempotencyKeyCollisionError:
        # A derived key is bound to other content. This should be unreachable
        # with the SHA-256 key, but never reuse or overwrite the unrelated job.
        set_error_category("idempotency_conflict")
        raise HTTPException(status_code=409, detail="Ingestion idempotency key conflict") from None
    except (IngestionJobBusyError, IngestionExecutionLeaseLostError):
        # Another execution owned this job while the synchronous request was
        # working. Nothing was corrupted: the live owner finished or the request
        # may be retried, which resolves the job idempotently.
        set_error_category("ingestion_busy")
        raise HTTPException(
            status_code=409,
            detail="Ingestion job is owned by another execution; retry the request",
        ) from None
    except EmbeddingUnavailableError:
        # Transient dependency failure. The inactive target version can be
        # retried; the previous current version was never touched.
        set_error_category("embedding_unavailable")
        raise HTTPException(
            status_code=503,
            detail="Embedding service is unavailable",
        ) from None
    except (EmbeddingValidationError, EmbeddingCompletenessError):
        # Provider output violated the embedding contract. Do not echo the
        # provider response or any internal exception text.
        set_error_category("embedding_invalid_response")
        raise HTTPException(
            status_code=502,
            detail="Embedding generation failed validation",
        ) from None
    except EmbeddingSchemaError:
        # The configured dimension and the live vector column are incompatible.
        # This is a server deployment error, not a client request problem.
        set_error_category("embedding_schema_mismatch")
        raise HTTPException(
            status_code=500,
            detail="Embedding configuration is incompatible with the database schema",
        ) from None
    except EmbeddingServiceError:
        set_error_category("provider_error")
        raise HTTPException(
            status_code=502,
            detail="Embedding generation failed",
        ) from None


@router.post("/ingest/async", status_code=202)
def submit_document_endpoint(
    request: DocumentIngestRequest,
    identity: Annotated[Principal, Depends(require_identity)],
) -> dict:
    """Queue ingestion after durably committing a non-current target version.

    The synchronous ``/ingest`` contract is unchanged. This endpoint only adds
    an asynchronous path: the request transaction creates or resolves the job
    and persists the target/version/chunks, then publishes a task containing only
    the job identifier. Document content never enters Redis.
    """

    tenant_id = resolve_request_tenant(request.tenant_id, identity)
    try:
        submission = submit_ingestion_job(
            principal=identity,
            tenant_id=tenant_id,
            source_key=request.source_key,
            title=request.title,
            source_type=request.source_type,
            content=request.content,
            source_uri=request.source_uri,
            authority_level=request.authority_level,
        )
        if submission.needs_publish:
            try:
                publish_ingestion_job(submission.job_id, tenant_id=tenant_id)
            except IngestionQueueUnavailableError:
                set_error_category("queue_unavailable")
                raise HTTPException(
                    status_code=503,
                    detail="Background ingestion queue is unavailable",
                ) from None
        # One sanitized event per accepted submission. The durable job row and
        # the worker task event carry the rest of the lifecycle; document text
        # and the bearer token are never included.
        emit_operational_event(
            request_logger,
            "ingestion_submission",
            job_id=str(submission.job_id),
            tenant_ref=tenant_reference(tenant_id),
            status=submission.status,
            deduplicated=not submission.created,
            published=submission.needs_publish,
        )
        return _async_ingestion_response(submission)
    except TenantNotFoundError:
        set_error_category("tenant_not_found")
        raise HTTPException(status_code=404, detail="Tenant not found") from None
    except EmptyDocumentError as exc:
        set_error_category("empty_document")
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except IdempotencyKeyCollisionError:
        set_error_category("idempotency_conflict")
        raise HTTPException(status_code=409, detail="Ingestion idempotency key conflict") from None
    except EmbeddingSchemaError:
        set_error_category("embedding_schema_mismatch")
        raise HTTPException(
            status_code=500,
            detail="Embedding configuration is incompatible with the database schema",
        ) from None


def _async_ingestion_response(submission: SubmittedIngestion) -> dict:
    return {
        "job_id": str(submission.job_id),
        "status": submission.status,
        "status_url": f"{router.prefix}/jobs/{submission.job_id}",
        "deduplicated": not submission.created,
    }


@router.get("/jobs/{job_id}")
def get_ingestion_job_endpoint(
    job_id: UUID,
    identity: Annotated[Principal, Depends(require_identity)],
) -> dict:
    """Return tenant-scoped job state without leaking other tenants' jobs."""

    try:
        return get_ingestion_job_status(
            job_id,
            principal=identity,
            tenant_id=identity.tenant_id,
        )
    except IngestionJobNotFoundError:
        # The same response is used for a missing job and a job owned by a
        # different tenant, so existence is not disclosed across tenants.
        set_error_category("job_not_found")
        raise HTTPException(status_code=404, detail="Ingestion job not found") from None
