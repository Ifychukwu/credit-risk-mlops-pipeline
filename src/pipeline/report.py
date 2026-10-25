"""Static figures and a summary of a simulated pipeline history (used by the README).

Usage:  python -m pipeline.report
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from pipeline import registry  # noqa: E402
from pipeline.config import MONITOR_LOG, ROOT, RUNS_LOG  # noqa: E402
from pipeline.ingest import CONCEPT_DRIFT_DAY, DATA_DRIFT_DAY  # noqa: E402

FIGS = ROOT / "reports" / "figures"


def load():
    runs = [json.loads(line) for line in RUNS_LOG.read_text().splitlines()]
    mon = [json.loads(line) for line in MONITOR_LOG.read_text().splitlines()]
    return runs, mon


def main() -> dict:
    runs, mon = load()
    FIGS.mkdir(parents=True, exist_ok=True)
    m = pd.DataFrame([{"day": next(r["day"] for r in runs if r["date"] == x["date"]), "live": x["live"]["auroc"],
                       "validation": x["validation_auroc"], "max_psi": x["max_psi"], "version": x["model_version"],
                       **{k: v for k, v in x["feature_psi"].items()}} for x in mon])
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(m.day, m.live, marker="o", ms=3, label="live AUROC (3-day window)")
    axes[0].step(m.day, m.validation, where="post", ls="--", label="production model's validation AUROC")
    for r in runs:
        if "model_decision" in r and r["model_decision"]["decision"].startswith("promoted"):
            axes[0].axvline(r["day"], color="#2e7d32", alpha=0.35)
            axes[0].text(r["day"], 0.97, f"v{r['model_decision']['candidate_version']}", fontsize=7, color="#2e7d32",
                         ha="center", va="top", transform=axes[0].get_xaxis_transform())
        if not r["validation"]["passed"]:
            axes[0].axvline(r["day"], color="#c62828", ls=":", alpha=0.7)
    axes[0].set(ylabel="AUROC", title="Production model performance (green = new model promoted, red dotted = batch quarantined)")
    axes[0].legend(loc="lower left", fontsize=8)
    axes[1].plot(m.day, m.max_psi, color="#ef6c00"); axes[1].axhline(0.2, ls="--", color="grey")
    axes[1].set(ylabel="max feature PSI", xlabel="day")
    for ax in axes:
        ax.axvspan(DATA_DRIFT_DAY, CONCEPT_DRIFT_DAY, color="#ffe0b2", alpha=0.3)
        ax.axvspan(CONCEPT_DRIFT_DAY, m.day.max(), color="#e1bee7", alpha=0.25)
    axes[1].text(DATA_DRIFT_DAY + 0.5, axes[1].get_ylim()[1] * 0.8, "data drift", fontsize=8)
    axes[1].text(CONCEPT_DRIFT_DAY + 0.5, axes[1].get_ylim()[1] * 0.8, "concept drift", fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / "pipeline_timeline.png", dpi=130); plt.close(fig)

    feats = [c for c in m.columns if c not in ("day", "live", "validation", "max_psi", "version")]
    fig, ax = plt.subplots(figsize=(11, 4))
    im = ax.imshow(m[feats].T.values, aspect="auto", cmap="Reds", vmin=0, vmax=0.4)
    ax.set_yticks(range(len(feats)), feats, fontsize=8); ax.set_xticks(range(0, len(m), 5), m.day[::5])
    ax.set(xlabel="day", title="Feature drift (PSI vs production model's training data)")
    fig.colorbar(im, ax=ax, fraction=0.02); fig.tight_layout(); fig.savefig(FIGS / "psi_heatmap.png", dpi=130); plt.close(fig)

    reg = registry.load_registry()
    # Counterfactual: what if the pipeline had stopped after the first full-window model (no monitoring/retraining)?
    from pipeline.orchestrate import accepted_batches, load_accepted
    from pipeline.train import FEATURES, metrics
    from pipeline.config import TARGET
    late = [d for d in accepted_batches() if d >= runs[-11]["date"]]
    late_df = load_accepted(late)
    frozen_v = min(3, len(reg["versions"]))  # v3: first model trained on a full 21-day window
    frozen, _ = registry.load_model(frozen_v)
    final, final_v = registry.load_model()
    comparison = {
        "evaluation_days": [late[0], late[-1]],
        f"frozen_model_v{frozen_v}_auroc": metrics(late_df[TARGET], frozen.predict_proba(late_df[FEATURES])[:, 1])["auroc"],
        f"pipeline_model_v{final_v}_auroc": metrics(late_df[TARGET], final.predict_proba(late_df[FEATURES])[:, 1])["auroc"],
    }
    summary = {
        "counterfactual_last_10_days": comparison,
        "days": len(runs), "quarantined_batches": sum(not r["validation"]["passed"] for r in runs),
        "retraining_runs": sum("model_decision" in r for r in runs),
        "promoted": sum(r.get("model_decision", {}).get("decision", "").startswith("promoted") for r in runs),
        "rejected": sum(r.get("model_decision", {}).get("decision", "").startswith("rejected") for r in runs),
        "retrain_reasons": pd.Series([r["retrain_reason"].split(":")[0] for r in runs if "retrain_reason" in r]).value_counts().to_dict(),
        "production_version": reg["production"],
    }
    (ROOT / "reports" / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
