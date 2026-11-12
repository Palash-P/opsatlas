# PostgreSQL Connection Pool Operations Runbook
Synthetic engineering reference for retrieval evaluation.

## Purpose
Diagnose database connection-pool saturation and reduce connection acquisition timeouts.

## Symptoms
- Application requests wait for an available database connection.
- Connection acquisition timeouts increase.
- Pool utilization remains near its configured maximum.
- PostgreSQL connection usage approaches the available connection budget.

## Diagnostic procedure
1. Inspect application pool size, checked-out connections, idle connections, and acquisition wait time.
2. Check PostgreSQL active sessions and long-running transactions.
3. Compare total potential connections across all application instances against the database connection budget.
4. Look for connection leaks and code paths that fail to release sessions.
5. Check whether a recent increase in concurrency or a deployment changed database demand.

## Mitigation
- Reduce excessive incoming concurrency if database demand is overloaded.
- Resolve leaked connections or long-running transactions using the approved operational procedure.
- Adjust pool configuration only after calculating the aggregate connection budget.
- Roll back a recent change when evidence links it to the incident and rollback is approved.
- Verify connection wait time, error rate, and latency recover after mitigation.

## Prevention
- Alert on pool saturation and connection acquisition timeouts.
- Load-test expected concurrent workloads.
- Set timeouts and ensure database sessions are released in cleanup paths.
- Document pool sizing assumptions for every application instance.

## Important distinction
Increasing each instance's pool size can make the problem worse if aggregate connections exceed database capacity. Pool configuration must be evaluated across all instances, not in isolation.

This document is synthetic and is not an approved production procedure.
