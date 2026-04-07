"""10 -- Full v7g feature importance for departure and arrival models."""

import json, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import (load_full_df, train_catboost_full, set_plot_style, save_fig,
                   short_name, FIGURES,
                   DEP_FEATURE_ORDER, DEP_CAT_FEATURES, DEP_NUM_FEATURES,
                   ARR_FEATURE_ORDER, ARR_CAT_FEATURES, ARR_NUM_FEATURES)
from catboost import Pool


def _run_one(airline, target, feature_order, cat_features, target_col, sample_n):
    """Train model, get importance + SHAP for one target (dep or arr)."""
    print(f"[10] Loading {target} data for {airline} ...")
    df = load_full_df(airline, target, sample_n)
    print(f"[10] {target} {airline}: {len(df):,} rows, {len(feature_order)} features")

    model, X_test, y_test, y_prob, auc, cat_idx = train_catboost_full(
        df, feature_order, cat_features, target_col, iterations=1000, depth=6)
    print(f"[10] {target} {airline} AUC = {auc:.4f}")

    # Feature importance
    imp = model.get_feature_importance()
    imp_series = pd.Series(imp, index=feature_order).sort_values(ascending=False)

    # SHAP (subsample)
    shap_sample = X_test.sample(n=min(10_000, len(X_test)), random_state=42)
    shap_pool = Pool(shap_sample, cat_features=cat_idx)
    shap_vals = model.get_feature_importance(shap_pool, type="ShapValues")[:, :-1]
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    shap_series = pd.Series(mean_abs_shap, index=feature_order).sort_values(ascending=False)

    return {
        "auc": round(auc, 4),
        "importance": {k: round(v, 2) for k, v in imp_series.items()},
        "shap": {k: round(v, 5) for k, v in shap_series.items()},
        "imp_series": imp_series,
        "shap_series": shap_series,
    }


def run(airline="WN", sample_n=None):
    set_plot_style()
    summary = {}

    # -- Departure --
    dep_res = _run_one(airline, "dep", DEP_FEATURE_ORDER, DEP_CAT_FEATURES,
                       "y_dep_ge15", sample_n)
    summary["dep"] = {"auc": dep_res["auc"],
                      "importance": dep_res["importance"],
                      "shap": dep_res["shap"]}

    # -- Arrival --
    arr_res = _run_one(airline, "arr", ARR_FEATURE_ORDER, ARR_CAT_FEATURES,
                       "y_arr_ge15", sample_n)
    summary["arr"] = {"auc": arr_res["auc"],
                      "importance": arr_res["importance"],
                      "shap": arr_res["shap"]}

    # -- Plot: Top-30 SHAP for dep --
    print("[10] Plotting dep importance ...")
    top30 = dep_res["shap_series"].head(30).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.barh(range(len(top30)), top30.values, color="#FF7043", alpha=0.85)
    ax.set_yticks(range(len(top30)))
    ax.set_yticklabels([short_name(c) for c in top30.index], fontsize=8)
    ax.set_xlabel("Mean |SHAP|")
    ax.set_title(f"Departure Model: Top-30 Features by SHAP ({airline}, AUC={dep_res['auc']:.4f})")
    fig.tight_layout()
    save_fig(fig, "fig_10a_dep_shap_top30")

    # -- Plot: Top-30 SHAP for arr --
    print("[10] Plotting arr importance ...")
    top30a = arr_res["shap_series"].head(30).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.barh(range(len(top30a)), top30a.values, color="#7E57C2", alpha=0.85)
    ax.set_yticks(range(len(top30a)))
    ax.set_yticklabels([short_name(c) for c in top30a.index], fontsize=8)
    ax.set_xlabel("Mean |SHAP|")
    ax.set_title(f"Arrival Model: Top-30 Features by SHAP ({airline}, AUC={arr_res['auc']:.4f})")
    fig.tight_layout()
    save_fig(fig, "fig_10b_arr_shap_top30")

    # -- Plot: Feature importance by category --
    print("[10] Plotting feature group breakdown ...")
    def _group_importance(shap_dict, prefix_groups):
        result = {}
        for group, prefixes in prefix_groups.items():
            total = sum(v for k, v in shap_dict.items()
                        if any(k.startswith(p) for p in prefixes))
            result[group] = total
        return result

    dep_groups = {
        "Flight/OD History": ["flightnum_od_"],
        "Carrier Baselines": ["carrier_depdelay_", "carrier_origin_", "carrier_lateaircraft_"],
        "Origin Baselines": ["origin_depdelay_", "origin_lateaircraft_"],
        "Hub Spillover": ["hub_"],
        "Dest Baselines": ["dest_depdelay_", "dest_lateaircraft_"],
        "Congestion": ["origin_congestion_", "origin_airline_congestion_"],
        "Weather": ["origin_temp_", "origin_daily_", "origin_dep_"],
        "Schedule/Route": ["dep_dow", "sched_dep_hour", "is_holiday", "CRS", "Distance", "DepTimeBlk"],
        "Categorical IDs": ["Reporting_Airline", "Origin", "Dest"],
    }
    dep_grp = _group_importance(dep_res["shap"], dep_groups)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    for ax, grp, title, color in [
        (ax1, dep_grp, "Departure", "#FF7043"),
        (ax2, _group_importance(arr_res["shap"], {
            **dep_groups,
            "Arr History": ["arrdelay_"],
            "Airtime": ["airtime_"],
            "WO Slip": ["wo_slip_"],
            "Dest Congestion": ["dest_arrivals_", "dest_airline_arrivals_"],
            "Dest Weather": ["dest_arr_"],
        }), "Arrival", "#7E57C2"),
    ]:
        sorted_grp = dict(sorted(grp.items(), key=lambda x: -x[1]))
        ax.barh(range(len(sorted_grp)), list(sorted_grp.values()), color=color, alpha=0.85)
        ax.set_yticks(range(len(sorted_grp)))
        ax.set_yticklabels(list(sorted_grp.keys()), fontsize=9)
        ax.set_xlabel("Sum of Mean |SHAP|")
        ax.set_title(f"{title} Model Feature Groups")
    fig.tight_layout()
    save_fig(fig, "fig_10c_feature_group_breakdown")

    with open(FIGURES / "10_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("[10] Done.")
    return summary


if __name__ == "__main__":
    airline = sys.argv[1] if len(sys.argv) > 1 else "WN"
    sample = int(sys.argv[2]) if len(sys.argv) > 2 else None
    run(airline, sample)
