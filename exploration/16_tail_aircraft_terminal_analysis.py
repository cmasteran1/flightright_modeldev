"""16 -- Viability analysis for new-API features: tail number, aircraft type,
       aircraft age, terminal origin/dest, and tail-based operational features.

Analyzes each feature SEPARATELY for marginal AUC gain on dep and arr models.
"""

import json, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, Pool

from utils import (set_plot_style, save_fig, short_name, FIGURES, DATA_ROOT,
                   DEP_FEATURE_ORDER, DEP_CAT_FEATURES,
                   ARR_FEATURE_ORDER, ARR_CAT_FEATURES, train_catboost_full)


# ── features that are in the parquet but NOT in the v7g model ──────────
TAIL_EXTRA_COLS = [
    "Tail_Number", "aircraft_type", "Aircraft_Age_Bucket",
    "tail_depdelay_mean_last1", "tail_depdelay_mean_last7",
    "tail_depdelay_mean_last14",
    "tail_lateaircraft_rate_last1", "tail_lateaircraft_rate_last7",
    "tail_lateaircraft_rate_last14",
    "tail_leg_num_day", "turn_time_hours",
    "has_recent_arrival_turn_5h",
    "flightnum_hours_since_first_departure_today",
]

DEP_TARGETS = ["DepDelayMinutes", "y_dep_ge15", "y_dep_ge30", "y_dep_ge45", "y_dep_ge60"]
ARR_TARGETS = ["ArrDelayMinutes", "y_arr_ge15", "y_arr_ge30", "y_arr_ge45", "y_arr_ge60"]


def _load_extended(airline, target, sample_n):
    """Load v7g model features PLUS all tail/aircraft columns."""
    if target == "dep":
        base_cols = DEP_FEATURE_ORDER + DEP_TARGETS + ["FlightDate"]
    else:
        base_cols = ARR_FEATURE_ORDER + ARR_TARGETS + ["FlightDate"]
    all_cols = list(set(base_cols + TAIL_EXTRA_COLS))
    path = DATA_ROOT / f"features_{target}_{airline}_100_v7g_unbalanced.parquet"
    df = pd.read_parquet(path, columns=[c for c in all_cols])
    if sample_n and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=42)
    return df


def run(airline="WN", sample_n=None):
    set_plot_style()
    summary = {"airline": airline}

    print("[16] Loading dep data ...")
    df_dep = _load_extended(airline, "dep", sample_n)
    print(f"[16] {len(df_dep):,} dep rows")

    target_dep = "y_dep_ge15"

    # ── 1. Baseline dep model ─────────────────────────────────────────
    print("[16] Training baseline dep model ...")
    base_m, _, _, _, base_auc, _ = train_catboost_full(
        df_dep, DEP_FEATURE_ORDER, DEP_CAT_FEATURES, target_dep,
        iterations=800, depth=6, lr=0.05, verbose=0)
    print(f"[16] Baseline dep AUC = {base_auc:.5f}")
    summary["baseline_dep_auc"] = round(base_auc, 5)

    # ── 2. Test each feature group individually ───────────────────────
    feature_groups = {
        # --- Categorical features (need special handling) ---
        "aircraft_type": {
            "add_cat": ["aircraft_type"],
            "add_num": [],
            "desc": "Aircraft model (737-7H4, 737-8, etc)",
        },
        "Aircraft_Age_Bucket": {
            "add_cat": ["Aircraft_Age_Bucket"],
            "add_num": [],
            "desc": "Aircraft age bracket (0-5, 6-10, 11-15, 16+)",
        },
        # --- Numeric features ---
        "tail_depdelay_history": {
            "add_cat": [],
            "add_num": ["tail_depdelay_mean_last1", "tail_depdelay_mean_last7",
                        "tail_depdelay_mean_last14"],
            "desc": "Tail-specific mean dep delay (1/7/14d rolling)",
        },
        "tail_lateaircraft_history": {
            "add_cat": [],
            "add_num": ["tail_lateaircraft_rate_last1", "tail_lateaircraft_rate_last7",
                        "tail_lateaircraft_rate_last14"],
            "desc": "Tail-specific late-aircraft rate (1/7/14d rolling)",
        },
        "tail_leg_num_day": {
            "add_cat": [],
            "add_num": ["tail_leg_num_day"],
            "desc": "Which leg of the day this tail is on (1st, 2nd, 3rd...)",
        },
        "turn_time_hours": {
            "add_cat": [],
            "add_num": ["turn_time_hours"],
            "desc": "Hours since this tail's previous arrival (turnaround time)",
        },
        "has_recent_arrival_turn_5h": {
            "add_cat": [],
            "add_num": ["has_recent_arrival_turn_5h"],
            "desc": "Binary: tail arrived within last 5h (quick-turn flag)",
        },
        # --- Combined tail package ---
        "ALL_tail_features": {
            "add_cat": ["aircraft_type", "Aircraft_Age_Bucket"],
            "add_num": ["tail_depdelay_mean_last1", "tail_depdelay_mean_last7",
                        "tail_depdelay_mean_last14",
                        "tail_lateaircraft_rate_last1", "tail_lateaircraft_rate_last7",
                        "tail_lateaircraft_rate_last14",
                        "tail_leg_num_day", "turn_time_hours",
                        "has_recent_arrival_turn_5h"],
            "desc": "All tail/aircraft features combined",
        },
    }

    results = []
    for group_name, group_cfg in feature_groups.items():
        add_cat = [c for c in group_cfg["add_cat"] if c in df_dep.columns]
        add_num = [c for c in group_cfg["add_num"] if c in df_dep.columns]
        if not add_cat and not add_num:
            print(f"  {group_name}: no columns available, skipping")
            continue

        ext_order = DEP_FEATURE_ORDER + add_cat + add_num
        ext_cat = DEP_CAT_FEATURES + add_cat

        try:
            _, _, _, _, new_auc, _ = train_catboost_full(
                df_dep, ext_order, ext_cat, target_dep,
                iterations=800, depth=6, lr=0.05, verbose=0)
            delta = new_auc - base_auc
            bp = delta * 10000
            print(f"  {group_name}: AUC={new_auc:.5f} (delta={bp:+.1f} bp)")
            results.append({
                "group": group_name,
                "desc": group_cfg["desc"],
                "auc": round(new_auc, 5),
                "delta_auc": round(delta, 5),
                "delta_bp": round(bp, 1),
                "n_features_added": len(add_cat) + len(add_num),
            })
        except Exception as e:
            print(f"  {group_name}: FAILED ({e})")
    results.sort(key=lambda x: -x["delta_bp"])
    summary["dep_results"] = results

    # ── 3. Univariate analysis: delay rate by aircraft type / age ─────
    print("[16] Univariate analysis ...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # a) Delay rate by aircraft_type
    ax = axes[0, 0]
    if "aircraft_type" in df_dep.columns:
        grp = df_dep.groupby("aircraft_type")[target_dep].agg(["mean", "count"])
        grp = grp[grp["count"] > 1000].sort_values("mean", ascending=True)
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(grp)))
        ax.barh(range(len(grp)), grp["mean"], color=colors)
        ax.set_yticks(range(len(grp)))
        ax.set_yticklabels([t[:25] for t in grp.index], fontsize=7)
        ax.set_xlabel("P(dep delay >= 15 min)")
        ax.set_title("Delay Rate by Aircraft Type")

    # b) Delay rate by age bucket
    ax = axes[0, 1]
    if "Aircraft_Age_Bucket" in df_dep.columns:
        order = ["0-5", "6-10", "11-15", "16+"]
        grp = df_dep.groupby("Aircraft_Age_Bucket")[target_dep].agg(["mean", "count"])
        grp = grp.reindex([o for o in order if o in grp.index])
        ax.bar(range(len(grp)), grp["mean"], color="#FF7043", alpha=0.8)
        ax.set_xticks(range(len(grp)))
        ax.set_xticklabels(grp.index, fontsize=9)
        ax.set_ylabel("P(dep delay >= 15 min)")
        ax.set_title("Delay Rate by Aircraft Age")

    # c) Delay rate by leg number
    ax = axes[0, 2]
    if "tail_leg_num_day" in df_dep.columns:
        grp = df_dep.groupby("tail_leg_num_day")[target_dep].agg(["mean", "count"])
        grp = grp[grp["count"] > 500]
        ax.bar(grp.index, grp["mean"], color="#7E57C2", alpha=0.8)
        ax.set_xlabel("Leg # of day")
        ax.set_ylabel("P(dep delay >= 15 min)")
        ax.set_title("Delay Rate by Tail Leg # of Day")

    # d) Delay rate by turn time (binned)
    ax = axes[1, 0]
    if "turn_time_hours" in df_dep.columns:
        valid = df_dep[df_dep["turn_time_hours"].notna() & (df_dep["turn_time_hours"] > 0)]
        valid = valid.copy()
        valid["turn_bin"] = pd.cut(valid["turn_time_hours"],
                                   bins=[0, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0, 8.0, 48.0],
                                   labels=["<30m", "30-45m", "45-60m", "1-1.5h",
                                           "1.5-2h", "2-4h", "4-8h", "8h+"])
        grp = valid.groupby("turn_bin", observed=True)[target_dep].agg(["mean", "count"])
        grp = grp[grp["count"] > 200]
        ax.bar(range(len(grp)), grp["mean"], color="#26A69A", alpha=0.8)
        ax.set_xticks(range(len(grp)))
        ax.set_xticklabels(grp.index, fontsize=8, rotation=30)
        ax.set_ylabel("P(dep delay >= 15 min)")
        ax.set_title("Delay Rate by Turn Time")

    # e) tail_depdelay_mean_last1 vs delay rate
    ax = axes[1, 1]
    if "tail_depdelay_mean_last1" in df_dep.columns:
        valid = df_dep[df_dep["tail_depdelay_mean_last1"].notna()].copy()
        try:
            valid["tail_bin"] = pd.qcut(valid["tail_depdelay_mean_last1"], 10,
                                        labels=False, duplicates="drop")
            grp = valid.groupby("tail_bin").agg(
                delay_rate=(target_dep, "mean"),
                tail_delay=("tail_depdelay_mean_last1", "mean")
            )
            ax.plot(grp["tail_delay"], grp["delay_rate"], "o-", color="#D32F2F", markersize=8)
            ax.set_xlabel("Tail avg dep delay (last 1d, min)")
            ax.set_ylabel("P(dep delay >= 15 min)")
            ax.set_title("Tail Recent History vs Delay Risk")
        except Exception:
            ax.set_title("Tail Recent History (insufficient data)")

    # f) AUC impact bar chart
    ax = axes[1, 2]
    if results:
        names = [r["group"] for r in results]
        bps = [r["delta_bp"] for r in results]
        colors = ["#4CAF50" if b > 0 else "#D32F2F" for b in bps]
        ax.barh(range(len(names)), bps, color=colors, alpha=0.85)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([n.replace("_", " ") for n in names], fontsize=7)
        ax.axvline(0, color="black", lw=0.5)
        ax.set_xlabel("AUC change (basis points)")
        ax.set_title("Marginal AUC Impact by Feature Group")

    fig.suptitle(f"Tail/Aircraft Feature Viability ({airline})", fontsize=14, y=1.01)
    fig.tight_layout()
    save_fig(fig, "fig_16a_tail_aircraft_viability")

    # ── 4. Terminal analysis (data availability note) ─────────────────
    print("[16] Terminal analysis ...")
    summary["terminal_analysis"] = {
        "status": "NOT_IN_BTS_DATA",
        "note": ("BTS On-Time data does not include terminal or gate fields. "
                 "If the new API provides origin_terminal and dest_terminal, "
                 "these would need to be engineered from scratch. "
                 "Terminal-level congestion features (departures per terminal per hour) "
                 "could be powerful at hub airports where different terminals serve "
                 "different routes/airlines. Cannot evaluate from historical data alone."),
        "recommendation": ("WORTH TESTING when available. At major hubs (ATL, ORD, DFW, DEN), "
                           "terminal assignment correlates with operational patterns. "
                           "Suggested features: terminal_congestion_3h, terminal_delay_rate_last7d, "
                           "is_concourse_change (connection risk at dest)."),
    }

    # ── 5. Cross-airline validation (if WN, also sample AA) ──────────
    print("[16] Cross-airline check ...")
    cross_airline = {}
    for al in ["AA", "DL", "UA"]:
        try:
            path = DATA_ROOT / f"features_dep_{al}_100_v7g_unbalanced.parquet"
            if not path.exists():
                continue
            df_al = pd.read_parquet(path, columns=["aircraft_type", "Aircraft_Age_Bucket",
                                                     target_dep])
            if sample_n:
                df_al = df_al.sample(n=min(sample_n, len(df_al)), random_state=42)
            n_types = df_al["aircraft_type"].nunique()
            delay_by_type = df_al.groupby("aircraft_type")[target_dep].mean()
            spread = delay_by_type.max() - delay_by_type.min()
            cross_airline[al] = {
                "n_aircraft_types": int(n_types),
                "delay_rate_spread_by_type": round(float(spread), 4),
            }
            print(f"  {al}: {n_types} aircraft types, delay spread={spread:.4f}")
        except Exception as e:
            print(f"  {al}: {e}")
    summary["cross_airline_aircraft_type"] = cross_airline

    with open(FIGURES / "16_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("[16] Done.")
    return summary


if __name__ == "__main__":
    airline = sys.argv[1] if len(sys.argv) > 1 else "WN"
    sample = int(sys.argv[2]) if len(sys.argv) > 2 else None
    run(airline, sample)
