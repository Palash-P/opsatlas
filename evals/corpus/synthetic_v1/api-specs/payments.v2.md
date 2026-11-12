# Payments Migration Guide

Synthetic engineering reference for OpsAtlas retrieval evaluation. This is not a real production document.

## Endpoint
The /api/v1/payments endpoint is deprecated. New clients must use:

POST /api/v3/payments

Existing clients should follow their migration plan rather than assuming endpoint compatibility.

## Request behavior
- Send a valid payment request according to the client's configured schema.
- Use the documented idempotency mechanism when retrying operations that might create a payment.
- Validate required fields before submitting a request.

## Retry behavior
- Retry only operations that are safe to retry.
- Use the documented idempotency mechanism instead of retrying blindly.
- Do not repeatedly retry non-idempotent payment operations.

## Error handling
- PAY-503 indicates that the payment service is unavailable or unable to complete the request.

## Versioning guidance
- New clients must target /api/v3/payments.
- Verify the configured API version before troubleshooting endpoint-related errors.

This document is synthetic and is not an approved production specification.
