# Data

The pipeline's source system is simulated by `src/pipeline/ingest.py`: a UK consumer lender receiving ~800 personal-loan applications a day, with known outcomes. Running the pipeline creates:

| Path | Contents | In git? |
|---|---|---|
| `data/raw/` | one CSV per day, exactly as ingested | no (regenerated) |
| `data/processed/` | batches that passed validation | no |
| `data/quarantine/` | batches that failed validation | no |
| `data/runs.jsonl` | one record per daily pipeline run (task status, timings, decisions) | yes |
| `data/monitoring.jsonl` | daily drift and live-performance metrics | yes |

Planted events: data-quality incidents on days 17 and 33, data drift (new broker channel) from day 25, and concept drift (cost-of-living shock) from day 40.

To use your own data, replace `simulate_batch()` with a reader for your source (SQL query, S3 file, API) that returns the columns listed in `src/pipeline/config.py`.
