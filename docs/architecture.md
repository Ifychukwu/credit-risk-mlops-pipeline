# Architecture

```mermaid
flowchart LR
    subgraph Airflow DAG: credit_risk_ml_pipeline (daily 06:00)
        A[ingest] --> B[validate]
        B -->|critical check failed| Q[quarantine + alert]
        B -->|passed| M[monitor production model]
        M --> D{decide}
        D -->|drift / performance / schedule| T[train candidate]
        T --> R[register version]
        R --> C{champion vs challenger}
        C -->|better| P[promote to Production]
        C -->|not better| X[reject]
        D -->|no| N[notify]
    end
    P --> REG[(Model registry<br/>models/registry.json)]
    REG --> API[FastAPI /score]
    REG --> UI[Streamlit control room]
    M --> LOG[(monitoring.jsonl)]
    A --> RAW[(data/raw)]
    B --> PROC[(data/processed)]
```

## Components

| Component | File | Responsibility |
|---|---|---|
| Ingestion | `src/pipeline/ingest.py` | Pull the day's applications, land them as CSV, and record row count and content hash (lineage). |
| Validation | `src/pipeline/validate.py` | Schema, duplicates, null rates, value ranges, categorical domains, target sanity. Critical failures quarantine the batch. |
| Preprocessing + training | `src/pipeline/train.py` | One sklearn `Pipeline` (imputation, one-hot encoding, gradient boosting), so training and serving share transforms. Time-based holdout. Optional recency weighting. |
| Registry / versioning | `src/pipeline/registry.py` | Versioned artifacts, metrics, params, training period, **data fingerprint**, and stages (Staging → Production → Archived / Rejected). MLflow-compatible concepts. |
| Monitoring | `src/pipeline/monitor.py` | PSI per feature and on predictions; live AUROC/PR-AUC/KS on the latest 3 labelled batches. |
| Orchestration | `src/pipeline/orchestrate.py`, `dags/credit_risk_pipeline.py` | The same task functions run locally, in tests and as Airflow tasks (TaskFlow API, branching, retries). |
| Serving | `src/pipeline/serve.py` | FastAPI endpoint that always loads the current Production version. |
| UI | `app.py` | Control room: runs, quality, drift, registry, live scoring. |

## Retraining policy

| Trigger | Rule |
|---|---|
| Initial | 7 accepted daily batches available |
| Data drift | any feature PSI > 0.2 vs the production model's training sample |
| Performance drop | 3-day live AUROC more than 0.04 below the model's validation AUROC |
| Performance floor | 3-day live AUROC below 0.65 |
| Scheduled | 14 days since the last training |
| Cool-down | no retraining within 2 days of the previous attempt |

Candidates train on the last 21 accepted days. After a performance-based trigger, older days are down-weighted (4-day half-life). A candidate is promoted only if it beats Production by ≥ 0.002 AUROC **on the same recent holdout**.

## Kubeflow alternative

Each task function is a pure Python callable with JSON-serialisable inputs and outputs, so it maps one-to-one onto Kubeflow Pipelines components (`@dsl.component`). The registry swaps for MLflow or Vertex AI Model Registry without changing the task logic.
