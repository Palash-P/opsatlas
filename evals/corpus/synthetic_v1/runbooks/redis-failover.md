# Redis Failover Runbook

Synthetic operational reference for OpsAtlas retrieval evaluation. This is not a real production procedure.

## Symptom
Readiness checks fail and requests time out after a Redis failover. Payment session cache lookups return errors.

## Investigation
1. Check the Redis health endpoint and replication status.
2. Confirm the application is connected to the promoted primary.
3. Inspect timeout and connection-refused errors in application logs.
4. Verify that readiness probes depend on Redis and database connectivity.
5. Compare behavior with the previous known-good release.

## Recovery
- Fail over to the healthy replica using the approved procedure.
- Clear stale connection state in the application.
- Confirm that session cache lookups succeed before restoring traffic.

## Important distinction
A successful Redis ping does not prove that the application can serve payment requests.

This document is synthetic and is not an approved production procedure.
