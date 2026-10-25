"""Monitoring: data drift (PSI), prediction drift and live performance."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.config import CATEGORICAL, NUMERIC


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population Stability Index. <0.1 stable, 0.1–0.2 moderate shift, >0.2 significant shift."""
    expected, actual = expected.dropna(), actual.dropna()
    if expected.dtype == object or isinstance(expected.dtype, pd.CategoricalDtype) or expected.nunique() < 15:
        cats = sorted(set(expected.astype(str)) | set(actual.astype(str)))
        e = expected.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
        a = actual.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    else:
        edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        e = np.histogram(expected, edges)[0] / len(expected)
        a = np.histogram(actual, edges)[0] / max(len(actual), 1)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, ref_scores=None, cur_scores=None) -> dict:
    features = {c: round(psi(reference[c], current[c]), 4) for c in NUMERIC + CATEGORICAL}
    out = {"feature_psi": features, "max_psi": max(features.values()),
           "drifted_features": [c for c, v in features.items() if v > 0.2]}
    if ref_scores is not None and cur_scores is not None:
        out["prediction_psi"] = round(psi(pd.Series(ref_scores), pd.Series(cur_scores)), 4)
    return out
