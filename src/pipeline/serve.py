"""Model serving: always uses the current Production version from the registry.

Run:  uvicorn pipeline.serve:app --app-dir src --reload   →  http://127.0.0.1:8000/docs
"""
from __future__ import annotations

from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pipeline import registry
from pipeline.train import FEATURES

app = FastAPI(title="Credit Risk Scoring API", version="1.0.0",
              description="Scores loan applications with the Production model from the registry.")


class Application(BaseModel):
    age: float = Field(..., ge=18, le=100, examples=[34])
    income_k: float = Field(..., gt=0, examples=[32.5])
    loan_amount_k: float = Field(..., gt=0, examples=[12])
    term_months: int = Field(..., examples=[48])
    credit_score: float = Field(..., ge=300, le=999, examples=[640])
    employment_years: float = Field(..., ge=0, examples=[3])
    debt_to_income: float = Field(..., ge=0, le=2, examples=[0.45])
    num_credit_lines: int = Field(..., ge=0, examples=[6])
    recent_inquiries: int = Field(..., ge=0, examples=[3])
    channel: Literal["online", "branch", "broker"] = "broker"
    purpose: Literal["car", "home_improvement", "debt_consolidation", "holiday", "other"] = "debt_consolidation"
    home_ownership: Literal["own", "mortgage", "rent"] = "rent"


@app.get("/model")
def model_info() -> dict:
    reg = registry.load_registry()
    if reg["production"] is None:
        raise HTTPException(503, "No production model yet: run the pipeline first")
    return next(v for v in reg["versions"] if v["version"] == reg["production"])


@app.post("/score")
def score(app_in: Application) -> dict:
    try:
        model, version = registry.load_model()
    except LookupError as e:
        raise HTTPException(503, str(e))
    p = float(model.predict_proba(pd.DataFrame([app_in.model_dump()])[FEATURES])[:, 1][0])
    band = "low" if p < 0.08 else "medium" if p < 0.2 else "high"
    return {"default_probability": round(p, 4), "risk_band": band, "model_version": version}
