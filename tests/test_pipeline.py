import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def isolated(tmp_path):
    """Point every pipeline path at a temporary folder."""
    import importlib

    import pipeline.config as cfg
    old = {k: getattr(cfg, k) for k in ("ROOT", "RAW_DIR", "PROCESSED_DIR", "MODELS_DIR", "RUNS_LOG", "MONITOR_LOG")}
    cfg.ROOT, cfg.RAW_DIR, cfg.PROCESSED_DIR = tmp_path, tmp_path / "data/raw", tmp_path / "data/processed"
    cfg.MODELS_DIR, cfg.RUNS_LOG, cfg.MONITOR_LOG = tmp_path / "models", tmp_path / "data/runs.jsonl", tmp_path / "data/monitoring.jsonl"
    import pipeline.ingest, pipeline.registry, pipeline.orchestrate  # noqa: E401
    mods = [importlib.reload(m) for m in (pipeline.ingest, pipeline.registry, pipeline.orchestrate)]
    yield mods
    for k, v in old.items():
        setattr(cfg, k, v)
    for m in mods:
        importlib.reload(m)


def test_batches_are_reproducible_and_drift():
    from pipeline.ingest import simulate_batch
    a, b = simulate_batch(3), simulate_batch(3)
    pd.testing.assert_frame_equal(a, b)
    early, late = simulate_batch(5), simulate_batch(30)
    assert (late.channel == "broker").mean() > 2 * (early.channel == "broker").mean()


def test_validation_catches_incidents():
    from pipeline.ingest import simulate_batch
    from pipeline.validate import validate
    assert validate(simulate_batch(5))["passed"]
    nulls = validate(simulate_batch(17))
    assert not nulls["passed"] and any("nulls: income_k" in c["name"] and not c["passed"] for c in nulls["checks"])
    assert not validate(simulate_batch(33))["passed"]
    assert not validate(simulate_batch(5).drop(columns=["credit_score"]))["passed"]


def test_psi_detects_shift_and_ignores_noise():
    from pipeline.monitor import psi
    rng = np.random.default_rng(0)
    base = pd.Series(rng.normal(0, 1, 5000))
    assert psi(base, pd.Series(rng.normal(0, 1, 5000))) < 0.05
    assert psi(base, pd.Series(rng.normal(1, 1, 5000))) > 0.2
    cats = pd.Series(["a"] * 80 + ["b"] * 20)
    assert psi(cats, pd.Series(["a"] * 40 + ["b"] * 60)) > 0.2


def test_ks_statistic_bounds():
    from pipeline.train import ks_statistic
    assert ks_statistic([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert ks_statistic([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.0


def test_recency_weights_halve_per_half_life():
    from pipeline.train import recency_weights
    w = recency_weights(pd.Series(["2026-01-01", "2026-01-05", "2026-01-09"]), half_life_days=4)
    assert np.allclose(w, [0.25, 0.5, 1.0])


def test_registry_versioning_and_promotion(tmp_path):
    from pipeline import registry
    from pipeline.ingest import simulate_batch
    from pipeline.train import train_candidate
    data = pd.concat([simulate_batch(d) for d in range(6)])
    model, m, _ = train_candidate(data)
    v1 = registry.register(model, m, data, {}, "initial", "2026-01-10", root=tmp_path)
    v2 = registry.register(model, m, data, {}, "retrain", "2026-01-11", root=tmp_path)
    registry.promote(v1, root=tmp_path)
    registry.promote(v2, root=tmp_path)
    reg = registry.load_registry(tmp_path)
    assert reg["production"] == 2
    assert [v["stage"] for v in reg["versions"]] == ["Archived", "Production"]
    assert reg["versions"][0]["data_fingerprint"] == registry.data_fingerprint(data)
    loaded, version = registry.load_model(root=tmp_path)
    assert version == 2 and loaded.predict_proba(data.head(5)[model.feature_names_in_]).shape == (5, 2)


def test_end_to_end_simulation_recovers_from_drift(isolated):
    _, registry, orch = isolated
    runs = orch.simulate(50)
    outcomes = [r["outcome"] for r in runs]
    assert outcomes.count("batch quarantined") == 2           # both planted incidents caught
    assert any("promoted" in o for o in outcomes[:10])         # first model appears early
    reasons = " ".join(r.get("retrain_reason", "") for r in runs)
    assert "data drift" in reasons and "performance" in reasons
    reg = registry.load_registry()
    assert reg["production"] is not None and len(reg["versions"]) >= 4


def test_airflow_dag_is_valid_python_with_expected_tasks():
    tree = ast.parse((ROOT / "dags" / "credit_risk_pipeline.py").read_text())
    funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert {"ingest", "validate", "gate", "monitor", "decide", "train_register", "quarantine_alert"} <= funcs
