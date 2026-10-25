"""A small file-based model registry with versioning and stage transitions.

Mirrors the MLflow Model Registry workflow (versions, stages, lineage) without
a server: every model version is a folder with the pickled pipeline plus a
``metadata.json`` holding metrics, parameters, the training-data fingerprint
and the reason it was trained. ``registry.json`` records which version is in
Production. To use MLflow instead, swap these functions for
``mlflow.sklearn.log_model`` and ``MlflowClient().transition_model_version_stage``.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from pipeline.config import MODELS_DIR


def data_fingerprint(df: pd.DataFrame) -> str:
    """Content hash of the training data: the data version a model was trained on."""
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()[:16]


def _registry_path(root: Path) -> Path:
    return root / "registry.json"


def load_registry(root: Path = MODELS_DIR) -> dict:
    p = _registry_path(root)
    return json.loads(p.read_text()) if p.exists() else {"production": None, "versions": []}


def _save(reg: dict, root: Path):
    root.mkdir(parents=True, exist_ok=True)
    _registry_path(root).write_text(json.dumps(reg, indent=2))


def register(model, metrics: dict, train_data: pd.DataFrame, params: dict, reason: str, trained_on: str,
             root: Path = MODELS_DIR) -> int:
    reg = load_registry(root)
    version = len(reg["versions"]) + 1
    vdir = root / f"v{version:03d}"
    vdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, vdir / "model.joblib")
    meta = {"version": version, "stage": "Staging", "registered_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "trained_on": trained_on, "reason": reason, "metrics": metrics, "params": params,
            "data_fingerprint": data_fingerprint(train_data), "train_rows": len(train_data),
            "train_period": [train_data["application_date"].min(), train_data["application_date"].max()]}
    (vdir / "metadata.json").write_text(json.dumps(meta, indent=2))
    reg["versions"].append(meta)
    _save(reg, root)
    return version


def promote(version: int, root: Path = MODELS_DIR, day: str | None = None):
    reg = load_registry(root)
    for v in reg["versions"]:
        if v["version"] == reg["production"]:
            v["stage"] = "Archived"
        if v["version"] == version:
            v["stage"] = "Production"
            v["promoted_on"] = day
    reg["production"] = version
    _save(reg, root)


def reject(version: int, root: Path = MODELS_DIR):
    reg = load_registry(root)
    for v in reg["versions"]:
        if v["version"] == version:
            v["stage"] = "Rejected"
    _save(reg, root)


def load_model(version: int | None = None, root: Path = MODELS_DIR):
    reg = load_registry(root)
    version = version or reg["production"]
    if version is None:
        raise LookupError("No production model registered yet")
    return joblib.load(root / f"v{version:03d}" / "model.joblib"), version
