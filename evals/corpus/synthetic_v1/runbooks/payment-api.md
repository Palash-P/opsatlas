# Payment API Troubleshooting

## Symptom
The payment API returns HTTP 503 errors after deployment.

## Investigation
1. Check application logs for upstream timeout errors.
2. Check PostgreSQL connection pool saturation.
3. Review the latest deployment changes.

## Recovery
Follow the approved rollback procedure if the deployment caused the outage.
