# Payment API Error Codes Runbook
Synthetic training document for OpsAtlas retrieval evaluation.

## PAY-503: Payment Service Unavailable
Meaning: The payment service could not complete the request because a required downstream service was unavailable or the request could not be processed within its availability limits.

Initial checks:
1. Check the payment service health endpoint.
2. Inspect recent deployment events and error-rate metrics.
3. Check downstream gateway availability and connection-pool saturation.
4. Correlate the request timestamp with service logs using the request ID.

Response:
- If a recent deployment correlates with the errors, follow the approved rollback procedure.
- If the downstream dependency is unavailable, follow its incident runbook.
- Do not repeatedly retry non-idempotent payment operations without checking idempotency guarantees.
- Escalate to the payment on-call engineer if errors persist.

## PAY-400: Invalid Payment Request
Meaning: The request is invalid or required fields are missing.
Response: Verify the request schema and validation error details before retrying.

## PAY-409: Payment Conflict
Meaning: The request conflicts with the current payment state.
Response: Retrieve the payment state and reconcile before submitting another operation.

This document is synthetic and does not describe a real production incident or actual company procedure.
