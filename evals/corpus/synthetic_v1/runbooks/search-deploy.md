# Search Service Deployment Runbook
Synthetic operational reference for OpsAtlas retrieval evaluation.

## Pre-deployment checks
1. Confirm the release artifact and configuration are correct.
2. Verify database migrations are compatible with the running application.
3. Check readiness and dependency health.
4. Confirm monitoring and rollback procedures are available.

## Deployment verification
1. Deploy using the approved release procedure.
2. Check the service health endpoint.
3. Check readiness and confirm database and Redis connectivity.
4. Run a representative search request.
5. Inspect error rate, latency, and application logs.

## Troubleshooting
- If readiness fails, inspect dependency connectivity and configuration.
- If search requests fail after a release, inspect application logs and embedding-service availability.
- If database queries fail, verify schema migrations and database connectivity.
- Compare behavior against the previous known-good release.

## Rollback
If the deployment causes sustained errors and the rollback criteria are met:
1. Follow the approved rollback procedure.
2. Verify the previous release is healthy.
3. Run a representative search request.
4. Record the failure symptoms and relevant logs for investigation.

## Important distinction
A successful health endpoint alone does not prove that retrieval works. Validate readiness and execute a representative search request.

This document is synthetic and is not an approved production deployment procedure.
