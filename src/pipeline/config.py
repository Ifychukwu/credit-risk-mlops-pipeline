"""Paths and pipeline settings in one place (override with environment variables)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.getenv("PIPELINE_ROOT", Path(__file__).resolve().parents[2]))
RAW_DIR = ROOT / "data" / "raw"            # landing zone: one CSV per daily batch
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"               # versioned model artifacts + registry.json
RUNS_LOG = ROOT / "data" / "runs.jsonl"    # one line per pipeline run (audit trail)
MONITOR_LOG = ROOT / "data" / "monitoring.jsonl"

START_DATE = "2026-01-05"
BATCH_SIZE = 800                 # applications per day
TRAIN_WINDOW_DAYS = 21           # rolling training window
PSI_ALERT = 0.20                 # population stability index: > 0.2 = significant drift
AUC_DROP_ALERT = 0.04            # retrain if live AUROC falls this far below the model's validation AUROC
AUC_FLOOR = 0.65                 # absolute minimum acceptable live AUROC (business requirement)
RETRAIN_COOLDOWN_DAYS = 2        # don't retrain again within 2 days (avoids thrashing on noisy alerts)
PROMOTION_MARGIN = 0.002         # candidate must beat production AUROC by this margin on the same holdout
TARGET = "defaulted"

NUMERIC = ["age", "income_k", "loan_amount_k", "term_months", "credit_score", "employment_years",
           "debt_to_income", "num_credit_lines", "recent_inquiries"]
CATEGORICAL = ["channel", "purpose", "home_ownership"]
SCHEMA = {
    "application_id": "str", "application_date": "date",
    **{c: "num" for c in NUMERIC}, **{c: "cat" for c in CATEGORICAL}, TARGET: "binary",
}
RANGES = {"age": (18, 100), "income_k": (0, 1000), "loan_amount_k": (0.5, 100), "term_months": (6, 84),
          "credit_score": (300, 999), "employment_years": (0, 60), "debt_to_income": (0, 2),
          "num_credit_lines": (0, 60), "recent_inquiries": (0, 30)}
DOMAINS = {"channel": {"online", "branch", "broker"},
           "purpose": {"car", "home_improvement", "debt_consolidation", "holiday", "other"},
           "home_ownership": {"own", "mortgage", "rent"}}
