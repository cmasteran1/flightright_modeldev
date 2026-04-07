"""17 -- Deep dive into tail-based historical tracking features.

Analyzes which tail history signals are most useful, at which rolling windows,
and how tail history compares to flight-number and carrier-level history.
"""

import json, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from utils import (set_plot_style, save_fig, short_name, FIGURES, DATA_ROOT,
                   DEP_FEATURE_ORDER, DEP_CAT_FEATURES, train_catboost_full)

TAIL_EXTRA_COLS = [
    "Tail_Number", "aircraft_type", "Aircraft_Age_Bucket",
    "tail_depdelay_mean_last1", "tail_depdelay_mean_last7",
    "tail_depdelay_mean_last14",
    "tail_lateaircraft_rate_last1", "tail_lateaircraft_rate_last7",
    "tail_lateaircraft_rate_last14",
    "tail_leg_num_day", "turn_time_hours",
    "has_recent_arrival_turn_5h",
]


def _load(airline, sample_n):
    from utils import DEP_TARGETS
    base = DEP_FEATURE_ORDER + DEP_TARGETS + ["FlightDate"]
    all_cols = list(set(base + TAIL_EXTRA_COLS))
    path = DATA_ROOT / f"features_dep_{airline}_100_v7g_unbalanced.parquet"
    df = pd.read_parquet(path, columns=[c for c in all_cols])
    if sample_n and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=42)
    return df


def run(airline="WN", sample_n=None):
    set_plot_style()
    print("[17] Loading dep data ...")
    df = _load(airline, sample_n)
    target = "y_dep_ge15"
    summary = {}

    # ── 1. Compare tail vs flightnum vs carrier history predictive power ──
    print("[17] Comparing history signal levels ...")
    history_groups = {
        "tail_delay": ["tail_depdelay_mean_last1", "tail_depdelay_mean_last7",
                       "tail_depdelay_mean_last14"],
        "tail_lateaircraft": ["tail_lateaircraft_rate_last1", "tail_lateaircraft_rate_last7",
                               "tail_lateaircraft_rate_last14"],
        "flightnum_delay": ["flightnum_od_depdelay_mean_last1", "flightnum_od_depdelay_mean_last7",
                            "flightnum_od_depdelay_mean_last14"],
        "flightnum_median": ["flightnum_od_depdelay_median_last1", "flightnum_od_depdelay_median_last7",
                              "flightnum_od_depdelay_median_last14"],
        "carrier_delay": ["carrier_depdelay_mean_last1", "carrier_depdelay_mean_last7",
                          "carrier_depdelay_mean_last14"],
        "carrier_origin": ["carrier_origin_depdelay_mean_last1", "carrier_origin_depdelay_mean_last7",
                           "carrier_origin_depdelay_mean_last14"],
    }

    # Compute Pearson r with target for each
    corr_results = {}
    for name, cols in history_groups.items():
        valid = [c for c in cols if c in df.columns]
        corrs = {}
        for c in valid:
            r = df[c].corr(df[target])
            window = c.split("last")[-1] if "last" in c else c
            corrs[window] = round(float(r), 4) if not pd.isna(r) else 0.0
        corr_results[name] = corrs
    summary["correlation_with_target"] = corr_results

    print("[17] fig_17a -- History signal comparison ...")
    fig, ax = plt.subplots(figsize=(14, 7))
    groups = list(corr_results.keys())
    windows = ["1", "7", "14"]
    x = np.arange(len(groups))
    width = 0.25
    colors = {"1": "#D32F2F", "7": "#FF9800", "14": "#4CAF50"}
    for i, w in enumerate(windows):
        vals = [corr_results[g].get(w, 0) for g in groups]
        ax.bar(x + i * width, vals, width, label=f"last {w}d", color=colors[w], alpha=0.85)
    ax.set_xticks(x + width)
    ax.set_xticklabels([g.replace("_", " ") for g in groups], fontsize=9, rotation=20, ha="right")
    ax.set_ylabel("Pearson r with y_dep_ge15")
    ax.set_title(f"History Feature Correlation by Source & Window ({airline})")
    ax.legend()
    ax.axhline(0, color="black", lw=0.5)
    fig.tight_layout()
    save_fig(fig, "fig_17a_history_signal_comparison")

    # ── 2. Tail history uniqueness: how much signal does tail add BEYOND flightnum? ──
    print("[17] Testing marginal value of tail history beyond flightnum ...")

    # Model with just flightnum history
    fn_hist = [c for c in DEP_FEATURE_ORDER
               if c.startswith("flightnum_od_depdelay") or c.startswith("flightnum_od_depdelay_median")]
    fn_hist_set = set(fn_hist)

    # Baseline: full v7g (includes flightnum but not tail)
    _, _, _, _, base_auc, _ = train_catboost_full(
        df, DEP_FEATURE_ORDER, DEP_CAT_FEATURES, target,
        iterations=500, depth=6, lr=0.05, verbose=0)

    # Add tail history
    tail_all = ["tail_depdelay_mean_last1", "tail_depdelay_mean_last7",
                "tail_depdelay_mean_last14",
                "tail_lateaircraft_rate_last1", "tail_lateaircraft_rate_last7",
                "tail_lateaircraft_rate_last14"]
    tail_valid = [c for c in tail_all if c in df.columns]
    ext_order = DEP_FEATURE_ORDER + tail_valid
    _, _, _, _, tail_auc, _ = train_catboost_full(
        df, ext_order, DEP_CAT_FEATURES, target,
        iterations=500, depth=6, lr=0.05, verbose=0)
    tail_marginal = tail_auc - base_auc
    print(f"  Tail history marginal: AUC {base_auc:.5f} -> {tail_auc:.5f} ({tail_marginal*10000:+.1f} bp)")
    summary["tail_marginal_beyond_flightnum"] = {
        "base_auc": round(base_auc, 5),
        "with_tail_auc": round(tail_auc, 5),
        "delta_bp": round(tail_marginal * 10000, 1),
    }

    # ── 3. Which window matters most for tail? ──
    print("[17] Window-by-window tail importance ...")
    window_results = []
    for window_cols, label in [
        (["tail_depdelay_mean_last1", "tail_lateaircraft_rate_last1"], "last1"),
        (["tail_depdelay_mean_last7", "tail_lateaircraft_rate_last7"], "last7"),
        (["tail_depdelay_mean_last14", "tail_lateaircraft_rate_last14"], "last14"),
    ]:
        valid = [c for c in window_cols if c in df.columns]
        if not valid:
            continue
        ext = DEP_FEATURE_ORDER + valid
        _, _, _, _, w_auc, _ = train_catboost_full(
            df, ext, DEP_CAT_FEATURES, target,
            iterations=500, depth=6, lr=0.05, verbose=0)
        window_results.append({
            "window": label,
            "auc": round(w_auc, 5),
            "delta_bp": round((w_auc - base_auc) * 10000, 1),
        })
        print(f"  tail {label}: AUC={w_auc:.5f} ({(w_auc-base_auc)*10000:+.1f} bp)")
    summary["tail_by_window"] = window_results

    # ── 4. Tail history stability: do delay-prone tails stay delay-prone? ──
    print("[17] Tail delay persistence analysis ...")
    if "Tail_Number" in df.columns and "FlightDate" in df.columns:
        df_valid = df[df["Tail_Number"].notna()].copy()
        df_valid["FlightDate"] = pd.to_datetime(df_valid["FlightDate"])

        # Split into two halves by date
        dates = sorted(df_valid["FlightDate"].unique())
        mid = dates[len(dates) // 2]
        first_half = df_valid[df_valid["FlightDate"] <= mid]
        second_half = df_valid[df_valid["FlightDate"] > mid]

        # Mean delay per tail in each half
        t1 = first_half.groupby("Tail_Number")[target].mean()
        t2 = second_half.groupby("Tail_Number")[target].mean()
        both = pd.DataFrame({"first_half": t1, "second_half": t2}).dropna()
        both = both[both.index.map(lambda x: first_half[first_half["Tail_Number"]==x].shape[0] > 20)]

        if len(both) > 50:
            persistence_r = both["first_half"].corr(both["second_half"])
            summary["tail_persistence"] = {
                "n_tails": len(both),
                "pearson_r_first_vs_second_half": round(float(persistence_r), 4),
                "interpretation": (
                    "High r means tail delay propensity is stable over time "
                    "(individual aircraft have consistent delay patterns). "
                    "Low r means tail history is noisy/transient."
                ),
            }
            print(f"  Tail persistence r={persistence_r:.4f} across {len(both)} tails")

            # Plot
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(both["first_half"], both["second_half"], alpha=0.4, s=15, color="#1976D2")
            ax.plot([0, both.values.max()], [0, both.values.max()], "r--", alpha=0.5)
            ax.set_xlabel("Tail delay rate (first half of data)")
            ax.set_ylabel("Tail delay rate (second half of data)")
            ax.set_title(f"Tail Delay Persistence (r={persistence_r:.3f}, n={len(both)} tails)")
            fig.tight_layout()
            save_fig(fig, "fig_17b_tail_persistence")

    # ── 5. Tail leg progression: does delay risk increase through the day? ──
    print("[17] fig_17c -- Delay cascade through daily legs ...")
    if "tail_leg_num_day" in df.columns:
        legs = df.groupby("tail_leg_num_day").agg(
            delay_rate=(target, "mean"),
            mean_delay=("DepDelayMinutes", "mean"),
            n=("DepDelayMinutes", "count"),
        )
        legs = legs[legs["n"] > 500]
        summary["delay_by_leg"] = {
            k: {str(kk): vv for kk, vv in v.items()}
            for k, v in legs[["delay_rate", "mean_delay"]].to_dict().items()
        }

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.bar(legs.index, legs["delay_rate"], color="#FF7043", alpha=0.85)
        ax1.set_xlabel("Tail Leg # of Day")
        ax1.set_ylabel("P(dep delay >= 15 min)")
        ax1.set_title("Delay Probability by Leg Number")

        ax2.bar(legs.index, legs["mean_delay"], color="#7E57C2", alpha=0.85)
        ax2.set_xlabel("Tail Leg # of Day")
        ax2.set_ylabel("Mean DepDelayMinutes")
        ax2.set_title("Mean Delay by Leg Number")

        fig.suptitle(f"Delay Cascade Through Daily Tail Legs ({airline})", fontsize=13)
        fig.tight_layout()
        save_fig(fig, "fig_17c_leg_progression")

    with open(FIGURES / "17_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("[17] Done.")
    return summary


if __name__ == "__main__":
    airline = sys.argv[1] if len(sys.argv) > 1 else "WN"
    sample = int(sys.argv[2]) if len(sys.argv) > 2 else None
    run(airline, sample)
