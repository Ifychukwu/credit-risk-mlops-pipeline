"""Data validation: schema, nulls, ranges and categorical domains (Great-Expectations style).

Each check returns pass/fail plus a severity. A *critical* failure stops the
pipeline for that batch and quarantines it, so bad data can never reach a model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from pipeline.config import DOMAINS, NUMERIC, RANGES, SCHEMA, TARGET


@dataclass
class Check:
    name: str
    passed: bool
    severity: str  # "critical" | "warning"
    detail: str = ""


def validate(df: pd.DataFrame, max_null_rate: float = 0.02) -> dict:
    checks: list[Check] = []
    missing = set(SCHEMA) - set(df.columns)
    checks.append(Check("schema: required columns present", not missing, "critical", ", ".join(sorted(missing))))
    if missing:
        return _report(checks, len(df))

    checks.append(Check("no duplicate application_id", not df["application_id"].duplicated().any(), "critical",
                        f"{df['application_id'].duplicated().sum()} duplicates"))
    for col in NUMERIC:
        rate = df[col].isna().mean()
        checks.append(Check(f"nulls: {col}", rate <= max_null_rate, "critical", f"{rate:.1%} null"))
        lo, hi = RANGES[col]
        bad = (~df[col].dropna().between(lo, hi)).sum()
        checks.append(Check(f"range: {col} in [{lo}, {hi}]", bad == 0, "critical", f"{bad} out of range"))
    for col, allowed in DOMAINS.items():
        unknown = set(df[col].dropna().unique()) - allowed
        checks.append(Check(f"domain: {col}", not unknown, "warning", ", ".join(map(str, unknown))))
    rate = df[TARGET].mean()
    checks.append(Check("target rate plausible (2%–40%)", 0.02 <= rate <= 0.40, "warning", f"{rate:.1%}"))
    checks.append(Check("batch size ≥ 100", len(df) >= 100, "warning", f"{len(df)} rows"))
    return _report(checks, len(df))


def _report(checks: list[Check], rows: int) -> dict:
    critical_fail = [c for c in checks if not c.passed and c.severity == "critical"]
    return {"passed": not critical_fail, "rows": rows, "n_checks": len(checks),
            "n_failed": sum(not c.passed for c in checks), "checks": [asdict(c) for c in checks]}
