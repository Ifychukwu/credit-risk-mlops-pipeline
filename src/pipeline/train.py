"""Preprocessing + training + evaluation for the credit-default model."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from pipeline.config import CATEGORICAL, NUMERIC, TARGET

FEATURES = NUMERIC + CATEGORICAL
PARAMS = {"learning_rate": 0.05, "max_iter": 300, "max_leaf_nodes": 15, "l2_regularization": 1.0}


def build_model(params: dict | None = None) -> Pipeline:
    """Preprocessing and model in one artifact, so serving applies identical transforms."""
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])
    return Pipeline([("pre", pre), ("clf", HistGradientBoostingClassifier(random_state=0, **(params or PARAMS)))])


def ks_statistic(y, p) -> float:
    """Kolmogorov–Smirnov separation between defaulters and non-defaulters (a credit-risk staple)."""
    y, p = np.asarray(y), np.asarray(p)
    grid = np.sort(np.unique(p))
    cdf1 = np.searchsorted(np.sort(p[y == 1]), grid, side="right") / max((y == 1).sum(), 1)
    cdf0 = np.searchsorted(np.sort(p[y == 0]), grid, side="right") / max((y == 0).sum(), 1)
    return float(np.max(np.abs(cdf1 - cdf0)))


def metrics(y, p) -> dict:
    return {"auroc": round(roc_auc_score(y, p), 4), "pr_auc": round(average_precision_score(y, p), 4),
            "brier": round(brier_score_loss(y, p), 4), "ks": round(ks_statistic(y, p), 4),
            "default_rate": round(float(np.mean(y)), 4), "n": int(len(y))}


def recency_weights(dates: pd.Series, half_life_days: float) -> np.ndarray:
    """Exponential decay: data from ``half_life_days`` ago counts half as much as today's."""
    age = (pd.to_datetime(dates).max() - pd.to_datetime(dates)).dt.days.to_numpy()
    return 0.5 ** (age / half_life_days)


def train_candidate(data: pd.DataFrame, holdout_days: int = 3, params: dict | None = None,
                    half_life_days: float | None = None):
    """Time-based split: train on the window minus its last ``holdout_days`` days, validate on those.

    ``half_life_days`` down-weights older rows, so after concept drift the model
    learns the new relationship quickly without throwing away all history.
    """
    dates = sorted(data["application_date"].unique())
    cut = dates[-holdout_days]
    tr, ho = data[data["application_date"] < cut], data[data["application_date"] >= cut]
    fit_kw = {}
    if half_life_days:
        fit_kw["clf__sample_weight"] = recency_weights(tr["application_date"], half_life_days)
    model = build_model(params).fit(tr[FEATURES], tr[TARGET], **fit_kw)
    return model, metrics(ho[TARGET], model.predict_proba(ho[FEATURES])[:, 1]), ho
