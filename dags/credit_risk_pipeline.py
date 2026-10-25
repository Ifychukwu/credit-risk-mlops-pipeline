"""Airflow DAG: daily credit-risk ML pipeline.

ingest → validate ─┬─(failed)→ quarantine_alert
                   └─(passed)→ monitor → decide ─┬─(retrain)→ train_register → notify
                                                  └─(no)─────→ notify

Each task calls the same functions used locally and in tests (src/pipeline/orchestrate.py),
so the DAG only adds scheduling, retries, branching and XCom hand-offs.
Place this file in your Airflow `dags/` folder with `src/` on PYTHONPATH (see docker-compose.yaml).
"""
from __future__ import annotations

from datetime import datetime, timedelta

from airflow.decorators import dag, task
from airflow.operators.empty import EmptyOperator

START = datetime(2026, 1, 5)


@dag(
    dag_id="credit_risk_ml_pipeline",
    schedule="0 6 * * *",              # every morning at 06:00, after the nightly application extract
    start_date=START,
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "ml-platform", "retries": 2, "retry_delay": timedelta(minutes=5)},
    tags=["ml", "credit-risk", "monitoring"],
    doc_md=__doc__,
)
def credit_risk_ml_pipeline():

    @task
    def ingest(ds=None) -> dict:
        from pipeline.orchestrate import task_ingest
        day = (datetime.fromisoformat(ds) - START).days
        return task_ingest(day) | {"day": day}

    @task
    def validate(meta: dict) -> dict:
        from pipeline.orchestrate import task_validate
        report = task_validate(meta)
        return {"passed": report["passed"], "failed": [c["name"] for c in report["checks"] if not c["passed"]],
                "date": meta["date"], "day": meta["day"]}

    @task.branch
    def gate(report: dict) -> str:
        return "monitor" if report["passed"] else "quarantine_alert"

    @task
    def quarantine_alert(report: dict):
        # Swap for SlackWebhookOperator / email in production.
        print(f"Batch {report['date']} quarantined. Failed checks: {report['failed']}")

    @task
    def monitor(report: dict) -> dict:
        from pipeline.orchestrate import task_monitor
        rec = task_monitor(report["date"])
        return {"day": report["day"], "date": report["date"], "monitoring": rec}

    @task.branch
    def decide(ctx: dict) -> str:
        from pipeline.orchestrate import task_decide
        reason = task_decide(ctx["day"], ctx["monitoring"])
        return "train_register" if reason else "notify"

    @task
    def train_register(ctx: dict) -> dict:
        from pipeline.orchestrate import task_decide, task_train_and_register
        reason = task_decide(ctx["day"], ctx["monitoring"]) or "manual"
        return task_train_and_register(ctx["day"], ctx["date"], reason)

    notify = EmptyOperator(task_id="notify", trigger_rule="none_failed_min_one_success")

    meta = ingest()
    report = validate(meta)
    branch = gate(report)
    q = quarantine_alert(report)
    ctx = monitor(report)
    d = decide(ctx)
    t = train_register(ctx)
    branch >> [ctx, q]
    d >> [t, notify]
    t >> notify


credit_risk_ml_pipeline()
