"""14 -- New feature candidates: engineer and test marginal AUC gain."""

import json, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, Pool

from utils import (load_full_df, train_catboost_full, set_plot_style, save_fig,
                   short_name, FIGURES,
                   DEP_FEATURE_ORDER, DEP_CAT_FEATURES, DEP_NUM_FEATURES)


def run(airline="WN", sample_n=None):
    set_plot_style()
    print("[14] Loading dep data ...")
    df = load_full_df(airline, "dep", sample_n,)
    summary = {}

    target = "y_dep_ge15"

    # -- Engineer candidate features --
    print("[14] Engineering candidate features ...")
    candidates = {}

    # 1. is_convective (CAPE > 500)
    if "origin_dep_cape_jkg" in df.columns:
        df["is_convective"] = (df["origin_dep_cape_jkg"] > 500).astype(int)
        candidates["is_convective"] = "CAPE > 500 J/kg flag"

    # 2. wind_x_precip
    if "origin_dep_windspeed_kmh" in df.columns and "origin_dep_precip_mm" in df.columns:
        df["wind_x_precip"] = df["origin_dep_windspeed_kmh"] * df["origin_dep_precip_mm"]
        candidates["wind_x_precip"] = "windspeed * precipitation"

    # 3. delay_trend_7d (recent trend)
    if "carrier_depdelay_mean_last1" in df.columns and "carrier_depdelay_mean_last7" in df.columns:
        df["delay_trend_7d"] = df["carrier_depdelay_mean_last1"] - df["carrier_depdelay_mean_last7"]
        candidates["delay_trend_7d"] = "carrier delay last1 - last7 (trend)"

    # 4. hub_max_delay_last1
    hub_delay_cols = [c for c in DEP_NUM_FEATURES if c.startswith("hub_") and c.endswith("depdelay_mean_last1")]
    if hub_delay_cols:
        df["hub_max_delay_last1"] = df[hub_delay_cols].max(axis=1)
        candidates["hub_max_delay_last1"] = "max hub delay across all hubs (last 1d)"

    # 5. hub_max_lateaircraft_last1
    hub_late_cols = [c for c in DEP_NUM_FEATURES if c.startswith("hub_") and c.endswith("lateaircraft_rate_last1")]
    if hub_late_cols:
        df["hub_max_lateaircraft_last1"] = df[hub_late_cols].max(axis=1)
        candidates["hub_max_lateaircraft_last1"] = "max hub late-aircraft rate (last 1d)"

    # 6. origin_delay_volatility
    if "origin_depdelay_mean_last7" in df.columns:
        std_col = "origin_depdelay_std_last7" if "origin_depdelay_std_last7" in df.columns else None
        if std_col:
            df["origin_delay_volatility"] = df[std_col] / df["origin_depdelay_mean_last7"].clip(lower=1)
            candidates["origin_delay_volatility"] = "origin std/mean delay ratio (volatility)"

    # 7. is_peak_hour
    if "sched_dep_hour" in df.columns:
        df["is_peak_hour"] = df["sched_dep_hour"].isin([16, 17, 18, 19, 20]).astype(int)
        candidates["is_peak_hour"] = "departure hour 16-20 flag"

    # 8. gust_factor
    if "origin_dep_windgusts_kmh" in df.columns and "origin_dep_windspeed_kmh" in df.columns:
        safe_ws = df["origin_dep_windspeed_kmh"].replace(0, np.nan)
        df["gust_factor"] = df["origin_dep_windgusts_kmh"] / safe_ws
        candidates["gust_factor"] = "windgusts / windspeed turbulence ratio"

    print(f"[14] Engineered {len(candidates)} candidate features")

    # -- Baseline model --
    print("[14] Training baseline model ...")
    model_base, X_test_base, y_test_base, y_prob_base, baseline_auc, _ = train_catboost_full(
        df, DEP_FEATURE_ORDER, DEP_CAT_FEATURES, target,
        iterations=500, depth=6, lr=0.05, verbose=0)
    print(f"[14] Baseline AUC = {baseline_auc:.4f}")
    summary["baseline_auc"] = round(baseline_auc, 4)

    # -- Test each candidate: add to full feature set, measure AUC delta --
    print("[14] Testing each candidate ...")
    results = []
    for feat_name, desc in candidates.items():
        if feat_name not in df.columns:
            continue
        extended_order = DEP_FEATURE_ORDER + [feat_name]
        extended_cat = DEP_CAT_FEATURES  # new features are all numeric
        try:
            _, _, _, _, new_auc, _ = train_catboost_full(
                df, extended_order, extended_cat, target,
                iterations=500, depth=6, lr=0.05, verbose=0)
            delta = new_auc - baseline_auc
            results.append({
                "feature": feat_name,
                "description": desc,
                "auc": round(new_auc, 5),
                "delta_auc": round(delta, 5),
                "pearson_r": round(float(df[feat_name].corr(df[target])), 4),
            })
            print(f"  {feat_name}: AUC={new_auc:.5f} (delta={delta:+.5f})")
        except Exception as e:
            print(f"  {feat_name}: FAILED ({e})")

    results.sort(key=lambda x: -x["delta_auc"])
    summary["candidate_results"] = results

    # -- Plot --
    print("[14] fig_14a -- Candidate feature AUC impact ...")
    fig, ax = plt.subplots(figsize=(12, 6))
    names = [r["feature"] for r in results]
    deltas = [r["delta_auc"] * 10000 for r in results]  # in basis points
    colors = ["#4CAF50" if d > 0 else "#D32F2F" for d in deltas]
    ax.barh(range(len(names)), deltas, color=colors, alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("AUC change (basis points, 1bp = 0.0001)")
    ax.axvline(0, color="black", lw=0.5)
    ax.set_title(f"Candidate Feature Marginal AUC Impact ({airline})\n"
                 f"Baseline AUC = {baseline_auc:.4f}")
    fig.tight_layout()
    save_fig(fig, "fig_14a_candidate_auc_impact")

    # -- Correlation comparison --
    print("[14] fig_14b -- Candidate correlations ...")
    fig, ax = plt.subplots(figsize=(10, 5))
    names = [r["feature"] for r in results]
    corrs = [r["pearson_r"] for r in results]
    colors = ["#D32F2F" if c > 0 else "#1976D2" for c in corrs]
    ax.barh(range(len(names)), corrs, color=colors, alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Pearson r with y_dep_ge15")
    ax.set_title(f"Candidate Feature Correlations ({airline})")
    ax.axvline(0, color="black", lw=0.5)
    fig.tight_layout()
    save_fig(fig, "fig_14b_candidate_correlations")

    with open(FIGURES / "14_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("[14] Done.")
    return summary


if __name__ == "__main__":
    airline = sys.argv[1] if len(sys.argv) > 1 else "WN"
    sample = int(sys.argv[2]) if len(sys.argv) > 2 else None
    run(airline, sample)
