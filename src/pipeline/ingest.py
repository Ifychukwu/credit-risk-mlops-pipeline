"""Ingestion: pull a day's loan applications from the source system into the landing zone.

The source here is a simulator of a UK consumer lender. It deliberately
changes over time so the monitoring has something to catch:

* **Data drift from day 25:** a new broker partnership shifts the channel
  mix and brings lower-income applicants.
* **Concept drift from day 40:** after a cost-of-living shock, long-term loans
  become much riskier and credit score predicts default less well, so the
  same applicant data now means something different.
* **Occasional data-quality incidents:** a few batches arrive with nulls or
  out-of-range values, which validation must catch.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from pipeline.config import BATCH_SIZE, RAW_DIR, ROOT, START_DATE, TARGET

DATA_DRIFT_DAY, CONCEPT_DRIFT_DAY = 25, 40
INCIDENT_DAYS = {17: "nulls", 33: "out_of_range"}


def simulate_batch(day: int, n: int = BATCH_SIZE, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed * 10_000 + day)
    date = pd.Timestamp(START_DATE) + pd.Timedelta(days=day)
    drift = day >= DATA_DRIFT_DAY
    channel = rng.choice(["online", "branch", "broker"], n, p=[0.45, 0.25, 0.30] if drift else [0.6, 0.32, 0.08])
    broker = channel == "broker"
    income = np.round(rng.lognormal(np.where(broker & drift, 3.35, 3.6), 0.4), 1)
    age = np.clip(rng.normal(41, 12, n), 18, 85).round()
    credit = np.clip(rng.normal(np.where(broker, 660, 720), 70), 300, 999).round()
    loan = np.round(np.clip(rng.lognormal(2.3, 0.6, n), 1, 60), 1)
    dti = np.clip(rng.normal(0.32, 0.12, n) + 0.08 * broker, 0.01, 1.5).round(3)
    df = pd.DataFrame({
        "application_id": [f"A{day:03d}-{i:04d}" for i in range(n)],
        "application_date": date.strftime("%Y-%m-%d"),
        "age": age, "income_k": income, "loan_amount_k": loan,
        "term_months": rng.choice([12, 24, 36, 48, 60], n, p=[0.15, 0.25, 0.3, 0.15, 0.15]),
        "credit_score": credit, "employment_years": np.clip(rng.gamma(2, 3, n), 0, 45).round(1),
        "debt_to_income": dti, "num_credit_lines": rng.poisson(5, n), "recent_inquiries": rng.poisson(np.where(broker, 2.0, 0.8)),
        "channel": channel,
        "purpose": rng.choice(["car", "home_improvement", "debt_consolidation", "holiday", "other"], n,
                              p=[0.3, 0.2, 0.3, 0.1, 0.1]),
        "home_ownership": rng.choice(["own", "mortgage", "rent"], n, p=[0.2, 0.45, 0.35]),
    })
    concept = day >= CONCEPT_DRIFT_DAY
    # After the cost-of-living shock, long loans become much riskier and credit score predicts less.
    credit_weight = 0.004 if concept else 0.011
    long_loan = 1.6 * (df["term_months"] >= 48) if concept else 0.0
    logit = (-2.4 - credit_weight * (credit - 700) + 1.5 * (dti - 0.32) + long_loan - 0.25 * np.log(income / 35)
             + 0.25 * df["recent_inquiries"] + 0.012 * (loan / (income / 10)) + 0.3 * (df["purpose"] == "holiday")
             - 0.02 * df["employment_years"] + 0.2 * (df["home_ownership"] == "rent"))
    df[TARGET] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    incident = INCIDENT_DAYS.get(day)
    if incident == "nulls":
        df.loc[rng.choice(n, n // 4, replace=False), "income_k"] = np.nan
    elif incident == "out_of_range":
        df.loc[rng.choice(n, 40, replace=False), "credit_score"] = -1
    return df


def ingest(day: int, seed: int = 0) -> dict:
    """Land one day's batch as CSV and return lineage metadata (path, rows, content hash)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = simulate_batch(day, seed=seed)
    path = RAW_DIR / f"applications_{df['application_date'].iloc[0]}.csv"
    df.to_csv(path, index=False)
    # Lineage paths are relative to the project root so the audit trail survives a clone or a container move.
    return {"path": path.relative_to(ROOT).as_posix(), "rows": len(df), "date": df["application_date"].iloc[0],
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()[:16]}


def load_batches(dates: list[str]) -> pd.DataFrame:
    frames = [pd.read_csv(RAW_DIR / f"applications_{d}.csv") for d in dates if (RAW_DIR / f"applications_{d}.csv").exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
