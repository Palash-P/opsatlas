# Payment API v3 Specification
Synthetic API reference for OpsAtlas retrieval evaluation. This is not a real external API contract.

## Current endpoint
New payment clients should use the v3 endpoint:
POST /api/v3/payments

## Request behavior
- Send a valid payment request according to the client's configured schema.
- Use the documented idempotency mechanism when retrying operations that might create a payment.
- Validate required fields before submitting a request.

## Error handling
- PAY-400 indicates an invalid payment request.
- PAY-409 indicates a conflict with the current payment state.
- PAY-503 indicates that the payment service is unavailable or unable to complete the request.

## Versioning guidance
- New clients should target /api/v3/payments.
- Clients using older API versions should follow their migration plan rather than assuming endpoint compatibility.
- Verify the configured API version before troubleshooting endpoint-related errors.

This document is synthetic and does not define a real production API.
