"""The daily pipeline as plain functions, so it runs the same locally, in tests and in Airflow.

Daily flow:  ingest → validate → (quarantine | accept) → monitor production model
             → decide whether to retrain → train candidate → register → promote or reject

Usage:
    python -m pipeline.orchestrate --simulate 60     # replay 60 days end to end
    python -m pipeline.orchestrate --day 12          # run a single day
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import pandas as pd

from pipeline import registry
from pipeline.config import (AUC_DROP_ALERT, AUC_FLOOR, RETRAIN_COOLDOWN_DAYS, MONITOR_LOG, MODELS_DIR, PROCESSED_DIR, PSI_ALERT, PROMOTION_MARGIN,
                             RAW_DIR, ROOT, RUNS_LOG, TARGET, TRAIN_WINDOW_DAYS)
from pipeline.ingest import ingest
from pipeline.monitor import drift_report
from pipeline.train import FEATURES, PARAMS, metrics, train_candidate
from pipeline.validate import validate

MIN_DAYS_FOR_FIRST_MODEL = 7
SCHEDULED_RETRAIN_EVERY = 14
RECENCY_HALF_LIFE_DAYS = 4


# --- individual tasks (each is one Airflow task) -----------------------------------

def task_ingest(day: int) -> dict:
    return ingest(day)


def task_validate(ingest_meta: dict) -> dict:
    df = pd.read_csv(ROOT / ingest_meta["path"])
    report = validate(df)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if report["passed"]:
        df.to_csv(PROCESSED_DIR / Path(ingest_meta["path"]).name, index=False)
    else:  # quarantine: keep the evidence, never train on it
        qdir = ROOT / "data" / "quarantine"
        qdir.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / ingest_meta["path"], qdir / Path(ingest_meta["path"]).name)
    return report


def accepted_batches() -> list[str]:
    return sorted(p.name.removeprefix("applications_").removesuffix(".csv") for p in PROCESSED_DIR.glob("applications_*.csv"))


def load_accepted(dates: list[str]) -> pd.DataFrame:
    return pd.concat([pd.read_csv(PROCESSED_DIR / f"applications_{d}.csv") for d in dates], ignore_index=True)


def task_monitor(date: str) -> dict | None:
    """Score today's batch with the production model; compare to its training reference."""
    reg = registry.load_registry()
    if reg["production"] is None or not (PROCESSED_DIR / f"applications_{date}.csv").exists():
        return None
    model, version = registry.load_model()
    current = pd.read_csv(PROCESSED_DIR / f"applications_{date}.csv")
    reference = pd.read_csv(MODELS_DIR / f"v{version:03d}" / "reference_sample.csv")
    cur_scores = model.predict_proba(current[FEATURES])[:, 1]
    ref_scores = model.predict_proba(reference[FEATURES])[:, 1]
    # Live performance over the last 3 accepted batches: one day (800 rows) is too noisy for AUROC.
    recent = load_accepted([d for d in accepted_batches() if d <= date][-3:])
    live = metrics(recent[TARGET], model.predict_proba(recent[FEATURES])[:, 1])
    meta = next(v for v in reg["versions"] if v["version"] == version)
    rec = {"date": date, "model_version": version, "live": live, "today": metrics(current[TARGET], cur_scores),
           "validation_auroc": meta["metrics"]["auroc"], **drift_report(reference, current, ref_scores, cur_scores)}
    rec["alerts"] = []
    if rec["max_psi"] > PSI_ALERT:
        rec["alerts"].append(f"data drift: {', '.join(rec['drifted_features'])}")
    if rec["live"]["auroc"] < rec["validation_auroc"] - AUC_DROP_ALERT:
        rec["alerts"].append(f"performance drop: live AUROC {rec['live']['auroc']:.3f} vs {rec['validation_auroc']:.3f}")
    elif rec["live"]["auroc"] < AUC_FLOOR:
        rec["alerts"].append(f"performance below floor: live AUROC {rec['live']['auroc']:.3f} < {AUC_FLOOR}")
    with open(MONITOR_LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def task_decide(day: int, monitoring: dict | None) -> str | None:
    """Return the reason to retrain today, or None."""
    reg = registry.load_registry()
    n = len(accepted_batches())
    if reg["production"] is None:
        return "initial model" if n >= MIN_DAYS_FOR_FIRST_MODEL else None
    last = max((v.get("trained_on_day", 0) for v in reg["versions"]), default=0)
    if monitoring and monitoring["alerts"]:
        if day - last <= RETRAIN_COOLDOWN_DAYS:
            return None  # a fresh candidate was just evaluated; give it time before trying again
        return "; ".join(monitoring["alerts"])
    if day - last >= SCHEDULED_RETRAIN_EVERY:
        return "scheduled refresh"
    return None


def task_train_and_register(day: int, date: str, reason: str) -> dict:
    # A performance problem suggests the relationship itself has changed (concept drift):
    # old data now teaches partly the wrong lesson, so recent days are weighted more heavily.
    half_life = RECENCY_HALF_LIFE_DAYS if "performance" in reason else None
    window = accepted_batches()[-TRAIN_WINDOW_DAYS:]
    data = load_accepted(window)
    model, cand_metrics, holdout = train_candidate(data, half_life_days=half_life)
    version = registry.register(model, cand_metrics, data, PARAMS, reason, trained_on=date)
    cand_metrics["recency_half_life_days"] = half_life
    data.sample(min(3000, len(data)), random_state=0).to_csv(MODELS_DIR / f"v{version:03d}" / "reference_sample.csv", index=False)
    reg = registry.load_registry()
    for v in reg["versions"]:
        if v["version"] == version:
            v["trained_on_day"] = day
    (MODELS_DIR / "registry.json").write_text(json.dumps(reg, indent=2))

    # Champion/challenger: compare on the *same* recent holdout.
    decision = {"candidate_version": version, "candidate_auroc": cand_metrics["auroc"]}
    if reg["production"] is None:
        registry.promote(version, day=date)
        decision["decision"] = "promoted (first model)"
    else:
        prod, prod_v = registry.load_model()
        prod_auroc = metrics(holdout[TARGET], prod.predict_proba(holdout[FEATURES])[:, 1])["auroc"]
        decision.update(production_version=prod_v, production_auroc=prod_auroc)
        if cand_metrics["auroc"] >= prod_auroc + PROMOTION_MARGIN:
            registry.promote(version, day=date)
            decision["decision"] = "promoted"
        else:
            registry.reject(version)
            decision["decision"] = "rejected (not better than production)"
    return decision


# --- one daily run ----------------------------------------------------------------

def run_day(day: int) -> dict:
    run = {"day": day, "tasks": {}}
    t0 = time.perf_counter()

    def timed(name, fn, *args):
        s = time.perf_counter()
        try:
            out = fn(*args)
            run["tasks"][name] = {"status": "success", "seconds": round(time.perf_counter() - s, 3)}
            return out
        except Exception as e:  # noqa: BLE001 - recorded as a failed task, like Airflow would
            run["tasks"][name] = {"status": "failed", "seconds": round(time.perf_counter() - s, 3), "error": str(e)}
            raise

    meta = timed("ingest", task_ingest, day)
    run["date"], run["ingest"] = meta["date"], meta
    report = timed("validate", task_validate, meta)
    run["validation"] = {k: report[k] for k in ("passed", "n_checks", "n_failed")}
    run["validation"]["failed_checks"] = [c["name"] for c in report["checks"] if not c["passed"]]
    if not report["passed"]:
        for t in ("monitor", "decide", "train_register"):
            run["tasks"][t] = {"status": "skipped"}
        run["outcome"] = "batch quarantined"
    else:
        mon = timed("monitor", task_monitor, meta["date"])
        run["alerts"] = mon["alerts"] if mon else []
        reason = timed("decide", task_decide, day, mon)
        if reason:
            run["retrain_reason"] = reason
            run["model_decision"] = timed("train_register", task_train_and_register, day, meta["date"], reason)
            run["outcome"] = f"model v{run['model_decision']['candidate_version']} {run['model_decision']['decision']}"
        else:
            run["tasks"]["train_register"] = {"status": "skipped"}
            run["outcome"] = "no retraining needed"
    run["production_version"] = registry.load_registry()["production"]
    run["seconds"] = round(time.perf_counter() - t0, 2)
    RUNS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(RUNS_LOG, "a") as f:
        f.write(json.dumps(run) + "\n")
    return run


def reset():
    for p in (RAW_DIR, PROCESSED_DIR, ROOT / "data" / "quarantine"):
        shutil.rmtree(p, ignore_errors=True)
    for p in MODELS_DIR.glob("v*"):
        shutil.rmtree(p, ignore_errors=True)
    for f in (RUNS_LOG, MONITOR_LOG, MODELS_DIR / "registry.json"):
        f.unlink(missing_ok=True)


def simulate(days: int = 60) -> list[dict]:
    reset()
    return [run_day(d) for d in range(days)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulate", type=int)
    ap.add_argument("--day", type=int)
    a = ap.parse_args()
    if a.simulate:
        for r in simulate(a.simulate):
            print(f"day {r['day']:>2} {r['date']}  {r['outcome']:<55} prod=v{r['production_version']}  {'; '.join(r.get('alerts', []))}")
    elif a.day is not None:
        print(json.dumps(run_day(a.day), indent=2))
