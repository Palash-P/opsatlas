# Payments Migration Guide

Synthetic engineering reference for OpsAtlas retrieval evaluation. This is not a real production document.

## Endpoint
The payments endpoint for all clients is:

POST /api/v1/payments

## Request behavior
- Send a valid payment request according to the client's configured schema.
- Validate required fields before submitting a request.

## Retry behavior
- Retry requests that fail with a timeout.
- Use the payment identifier when reconciling a failed request.

## Error handling
- PAY-503 indicates that the payment service is unavailable or unable to complete the request.

This document is synthetic and is not an approved production specification.
