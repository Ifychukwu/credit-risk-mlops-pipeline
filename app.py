"""ML Ops control room: pipeline runs, data quality, drift, model registry and live scoring.

Run:  streamlit run app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from pipeline import registry  # noqa: E402
from pipeline.config import MONITOR_LOG, PSI_ALERT, RUNS_LOG  # noqa: E402
from pipeline.ingest import CONCEPT_DRIFT_DAY, DATA_DRIFT_DAY  # noqa: E402
from pipeline.train import FEATURES  # noqa: E402

st.set_page_config(page_title="ML Ops Control Room", page_icon="🛠️", layout="wide")
st.markdown("""<style>
.block-container {padding-top: 1.6rem;}
div[data-testid="stMetric"] {background: rgba(120,120,140,0.08); border-radius: 10px; padding: 10px 14px;}
.badge {display:inline-block; padding:2px 10px; border-radius:12px; font-size:0.8rem; font-weight:600; color:white;}
</style>""", unsafe_allow_html=True)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


st.title("🛠️ ML Ops Control Room · Credit Risk")
st.caption("Daily pipeline: ingest → validate → monitor → retrain → register → promote · orchestrated by Airflow")

runs = read_jsonl(RUNS_LOG)
if not runs:
    st.info("No pipeline history yet. Replay 60 days of the pipeline to populate the dashboard (about 15 seconds).")
    if st.button("▶ Run 60-day simulation", type="primary"):
        from pipeline.orchestrate import simulate
        with st.spinner("Running pipeline day by day…"):
            simulate(60)
        st.rerun()
    st.stop()

mon = pd.DataFrame([{"date": m["date"], "model_version": m["model_version"], "live_auroc": m["live"]["auroc"],
                     "validation_auroc": m["validation_auroc"], "max_psi": m["max_psi"],
                     "prediction_psi": m.get("prediction_psi"), "alerts": "; ".join(m["alerts"]),
                     **{f"psi:{k}": v for k, v in m["feature_psi"].items()}} for m in read_jsonl(MONITOR_LOG)])
reg = registry.load_registry()
runs_df = pd.DataFrame([{"day": r["day"], "date": r["date"], "outcome": r["outcome"],
                         "validation": "passed" if r["validation"]["passed"] else "FAILED",
                         "failed_checks": ", ".join(r["validation"]["failed_checks"]),
                         "retrain_reason": r.get("retrain_reason", ""), "production": r["production_version"],
                         "seconds": r["seconds"], **{f"task:{k}": v["status"] for k, v in r["tasks"].items()}} for r in runs])

prod = next((v for v in reg["versions"] if v["version"] == reg["production"]), None)
last = mon.iloc[-1] if len(mon) else None
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Pipeline runs", len(runs_df), f"{(runs_df.validation == 'FAILED').sum()} quarantined", delta_color="off")
c2.metric("Production model", f"v{reg['production']}", f"trained {prod['trained_on']}" if prod else "", delta_color="off")
c3.metric("Live AUROC (3-day)", f"{last.live_auroc:.3f}" if last is not None else "—",
          f"{last.live_auroc - last.validation_auroc:+.3f} vs validation" if last is not None else None)
c4.metric("Max feature PSI today", f"{last.max_psi:.2f}" if last is not None else "—",
          "drift" if last is not None and last.max_psi > PSI_ALERT else "stable", delta_color="inverse")
c5.metric("Model versions", len(reg["versions"]),
          f"{sum(v['stage'] == 'Rejected' for v in reg['versions'])} rejected", delta_color="off")

tab_runs, tab_perf, tab_drift, tab_dq, tab_reg, tab_score = st.tabs(
    ["📅 Pipeline runs", "📈 Model performance", "🌊 Data drift", "🧪 Data quality", "🗂️ Model registry", "🎯 Live scoring"])

with tab_runs:
    st.subheader("Task status by day")
    task_cols = [c for c in runs_df.columns if c.startswith("task:")]
    status = runs_df.set_index("date")[task_cols].rename(columns=lambda c: c[5:])
    colour = {"success": "#2e7d32", "skipped": "#9e9e9e", "failed": "#c62828"}
    st.dataframe(status.T.style.map(lambda v: f"background-color: {colour.get(v, '#fff')}; color: {colour.get(v, '#fff')}"),
                 use_container_width=True, height=250)
    st.caption("Green = success · grey = skipped (no retrain needed, or batch quarantined) · red = failed")
    events = runs_df[runs_df.outcome != "no retraining needed"][["day", "date", "outcome", "retrain_reason", "failed_checks"]]
    st.subheader("Notable events")
    st.dataframe(events, hide_index=True, use_container_width=True)

with tab_perf:
    st.subheader("Live AUROC of the production model vs its validation AUROC")
    chart = mon.set_index("date")[["live_auroc", "validation_auroc"]]
    st.line_chart(chart, height=320)
    st.caption(f"Data drift begins day {DATA_DRIFT_DAY}; concept drift (relationship change) begins day {CONCEPT_DRIFT_DAY}. "
               "Vertical changes in the validation line = a new model was promoted.")
    st.subheader("Which version was serving each day")
    st.bar_chart(mon.set_index("date")["model_version"], height=200)

with tab_drift:
    st.subheader("Population Stability Index per feature (vs the production model's training data)")
    psi = mon.set_index("date")[[c for c in mon.columns if c.startswith("psi:")]].rename(columns=lambda c: c[4:])
    st.dataframe(psi.T.style.background_gradient(cmap="Reds", vmin=0, vmax=0.4).format("{:.2f}"),
                 use_container_width=True, height=470)
    st.caption("PSI < 0.1 stable · 0.1–0.2 moderate · > 0.2 significant drift (triggers retraining)")
    st.line_chart(mon.set_index("date")[["max_psi", "prediction_psi"]], height=220)

with tab_dq:
    st.subheader("Validation results per batch")
    st.dataframe(runs_df[["date", "validation", "failed_checks"]], hide_index=True, use_container_width=True, height=300)
    bad = runs_df[runs_df.validation == "FAILED"]
    for _, r in bad.iterrows():
        st.error(f"**{r.date}** quarantined: {r.failed_checks}")

with tab_reg:
    st.subheader("Model registry")
    vers = pd.DataFrame([{"version": f"v{v['version']}", "stage": v["stage"], "trained_on": v["trained_on"],
                          "reason": v["reason"], "auroc": v["metrics"]["auroc"], "pr_auc": v["metrics"]["pr_auc"],
                          "ks": v["metrics"]["ks"], "brier": v["metrics"]["brier"], "train_rows": v["train_rows"],
                          "data_fingerprint": v["data_fingerprint"]} for v in reg["versions"]])
    stage_colour = {"Production": "#c8e6c9", "Archived": "#eeeeee", "Rejected": "#ffcdd2", "Staging": "#fff9c4"}
    st.dataframe(vers.style.apply(lambda r: [f"background-color: {stage_colour.get(r.stage, '')}"] * len(r), axis=1),
                 hide_index=True, use_container_width=True)
    st.caption("Every version stores its metrics, parameters, training period and a fingerprint of the exact training data (data versioning).")
    st.bar_chart(vers.set_index("version")["auroc"], height=220)

with tab_score:
    st.subheader(f"Score an application with the production model (v{reg['production']})")
    model, version = registry.load_model()
    a, b, c = st.columns(3)
    x = {
        "age": a.slider("Age", 18, 85, 34), "income_k": a.slider("Income (£k)", 8.0, 150.0, 32.0),
        "loan_amount_k": a.slider("Loan (£k)", 1.0, 60.0, 12.0), "term_months": b.select_slider("Term (months)", [12, 24, 36, 48, 60], 48),
        "credit_score": b.slider("Credit score", 300, 999, 660), "employment_years": b.slider("Years employed", 0.0, 40.0, 3.0),
        "debt_to_income": c.slider("Debt-to-income", 0.0, 1.5, 0.40), "num_credit_lines": c.slider("Credit lines", 0, 20, 6),
        "recent_inquiries": c.slider("Recent credit searches", 0, 10, 2),
        "channel": a.selectbox("Channel", ["online", "branch", "broker"]), "purpose": b.selectbox(
            "Purpose", ["car", "home_improvement", "debt_consolidation", "holiday", "other"]),
        "home_ownership": c.selectbox("Home", ["own", "mortgage", "rent"]),
    }
    p = float(model.predict_proba(pd.DataFrame([x])[FEATURES])[:, 1][0])
    band, colour = ("Low", "#2e7d32") if p < 0.08 else ("Medium", "#f9a825") if p < 0.2 else ("High", "#c62828")
    st.markdown(f"### Default probability: **{p:.1%}** <span class='badge' style='background:{colour}'>{band} risk</span>",
                unsafe_allow_html=True)
    st.caption("Decision support only: lending decisions need affordability checks and human review (see the README's responsible-AI notes).")
