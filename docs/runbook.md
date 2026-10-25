# Runbook

| Alert | What it means | What to do |
|---|---|---|
| **Batch quarantined** | A critical validation check failed (nulls, out-of-range values, missing columns, duplicates). | Inspect `data/quarantine/`, fix the upstream extract, then re-run the day: `python -m pipeline.orchestrate --day N`. |
| **Data drift: feature X** | The applicant population has shifted (PSI > 0.2). | Check whether it's a business change (new channel or campaign) or a data bug. Retraining is triggered automatically, and the candidate is promoted only if it's better. |
| **Performance drop** | The model ranks risk worse on recent outcomes than at validation. | Usually concept drift. Retraining with recency weighting is automatic. If the floor alert persists, escalate to the model owner. |
| **Candidate rejected** | A retrained model was not better than Production on the same holdout. | No action: Production keeps serving. Repeated rejections during drift are expected while new data accumulates. |
| **Task failed (Airflow)** | Code or infrastructure error. Airflow retries twice. | Check task logs in the Airflow UI, fix, then clear the task to re-run. |

**Rollback:** set `production` in `models/registry.json` to a previous version (or use `registry.promote(version)`). The API and dashboard pick it up on the next request.
