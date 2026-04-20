"""
scripts/smoke/smoke_isd_vis_signal.py

After regenerating the WN dep 2025 feature parquet with the ISD visibility
join enabled, verify the new column actually carries signal. Prints:
  - population rate of origin_dep_visibility_isd_m (vs the Open-Meteo one)
  - univariate AUC for each threshold (ge15/30/60/120)
  - direct AUC delta vs Open-Meteo visibility

Decision gate per the R2 plan:
  * if new-column AUC < 0.52 on all thresholds -> DATA QUALITY ISSUE, abort.
  * if 0.52 <= AUC < 0.55 everywhere -> marginal, investigate but don't roll out.
  * if max AUC >= 0.55 on any threshold -> real signal, proceed to smoke train.

Usage:
  python scripts/smoke/smoke_isd_vis_signal.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
DATA = REPO.parent / "flightrightdata" / "data" / "processed"

PARQUET = DATA / "features_dep_WN_v11_2025_unbalanced.parquet"
THRESHOLDS = [15, 30, 60, 120]


def _auc(x: pd.Series, y: np.ndarray) -> float:
    mask = ~x.isna().values
    xv = x.values[mask]
    yv = y[mask]
    if len(np.unique(yv)) < 2 or len(xv) < 100:
        return float("nan")
    try:
        a = roc_auc_score(yv, xv)
    except ValueError:
        return float("nan")
    return a if a >= 0.5 else 1 - a


def main() -> None:
    if not PARQUET.exists():
        raise SystemExit(f"features parquet not found: {PARQUET}\n"
                         "Re-run features_dep.py on blueprint_dep_WN_v11_2025_pf030.json first.")

    df = pd.read_parquet(
        PARQUET,
        columns=[
            "origin_dep_visibility_m",
            "origin_dep_visibility_isd_m",
            "y_dep_ge15", "y_dep_ge30", "y_dep_ge60", "y_dep_ge120",
            "Origin",
        ],
    )
    print(f"rows: {len(df):,}")
    pop_isd = 1 - df["origin_dep_visibility_isd_m"].isna().mean()
    pop_om = 1 - df["origin_dep_visibility_m"].isna().mean() if "origin_dep_visibility_m" in df.columns else 0.0
    print(f"origin_dep_visibility_m    (Open-Meteo) populated: {pop_om * 100:.1f}%")
    print(f"origin_dep_visibility_isd_m (IEM ASOS)  populated: {pop_isd * 100:.1f}%")
    print(f"unique origins with ISD data: {df.loc[df['origin_dep_visibility_isd_m'].notna(), 'Origin'].nunique()}")

    print()
    print(f"{'threshold':<10} {'Open-Meteo AUC':<16} {'ISD AUC':<10} {'delta':<8}")
    print("-" * 50)
    any_pass_055 = False
    for thr in THRESHOLDS:
        y = df[f"y_dep_ge{thr}"].astype(int).values
        auc_om = _auc(df["origin_dep_visibility_m"], y) if "origin_dep_visibility_m" in df.columns else float("nan")
        auc_isd = _auc(df["origin_dep_visibility_isd_m"], y)
        delta = auc_isd - auc_om if not (np.isnan(auc_isd) or np.isnan(auc_om)) else float("nan")
        mark = ""
        if not np.isnan(auc_isd) and auc_isd >= 0.55:
            mark = "  ← signal"
            any_pass_055 = True
        print(f"ge{thr:<7} {auc_om:<16.3f} {auc_isd:<10.3f} {delta:+.3f}{mark}")

    print()
    if any_pass_055:
        print("[GATE] PASS — at least one threshold hits AUC >= 0.55. Proceed to smoke train.")
    else:
        print("[GATE] CAUTION — no threshold hits AUC >= 0.55. Investigate data quality before proceeding.")


if __name__ == "__main__":
    main()
