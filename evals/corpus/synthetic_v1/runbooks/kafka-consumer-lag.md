# Kafka Consumer Lag Runbook

Synthetic operational reference for OpsAtlas retrieval evaluation. This is not a real production procedure.

## Symptom
Consumer lag grows after a deployment and downstream payment events arrive late.

## Investigation
1. Check consumer group state and partition assignment.
2. Inspect broker health and network connectivity.
3. Review the latest deployment for changes to consumer concurrency.
4. Compare produce rate with consume rate per partition.
5. Check whether the connection pool for the admin client is exhausted.

## Recovery
- Reduce producer load if the backlog threatens downstream payment processing.
- Restart unhealthy consumers one at a time.
- Scale consumers only after confirming partition count and broker capacity.

## Important distinction
A growing connection pool for the admin client is not the same as database connection-pool exhaustion.

This document is synthetic and is not an approved production procedure.
