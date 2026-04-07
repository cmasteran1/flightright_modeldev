"""11 -- Redundancy analysis: correlated features & rolling window assessment."""

import json, sys
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import (load_full_df, set_plot_style, save_fig, short_name, FIGURES,
                   DEP_NUM_FEATURES, DEP_FEATURE_ORDER, DEP_CAT_FEATURES,
                   ARR_NUM_FEATURES)


def run(airline="WN", sample_n=None):
    set_plot_style()
    print("[11] Loading dep data ...")
    df = load_full_df(airline, "dep", sample_n)
    summary = {}

    num_feats = [c for c in DEP_NUM_FEATURES if c in df.columns]

    # -- Spearman correlation matrix --
    print("[11] Computing Spearman correlation matrix ...")
    corr = df[num_feats].corr(method="spearman")

    # -- Find highly correlated pairs (|r| > 0.85) --
    high_corr_pairs = []
    for i in range(len(num_feats)):
        for j in range(i + 1, len(num_feats)):
            r = corr.iloc[i, j]
            if abs(r) > 0.85:
                high_corr_pairs.append({
                    "feat_a": num_feats[i],
                    "feat_b": num_feats[j],
                    "r": round(float(r), 4),
                })
    print(f"[11] Found {len(high_corr_pairs)} highly correlated pairs (|r| > 0.85)")
    summary["high_corr_pairs"] = high_corr_pairs

    # -- Dendrogram --
    print("[11] fig_11a -- Cluster dendrogram ...")
    dist_arr = (1 - corr.abs()).clip(lower=0).values.copy()
    np.fill_diagonal(dist_arr, 0)
    condensed = squareform(dist_arr, checks=False)
    Z = linkage(condensed, method="ward")

    fig, ax = plt.subplots(figsize=(20, 8))
    dendrogram(Z, labels=[short_name(c) for c in num_feats],
               leaf_rotation=90, leaf_font_size=7, ax=ax)
    ax.set_ylabel("Distance (1 - |Spearman r|)")
    ax.set_title(f"Feature Cluster Dendrogram -- Dep Model ({airline})")
    ax.axhline(0.15, color="red", ls="--", alpha=0.5, label="High redundancy threshold")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "fig_11a_feature_dendrogram")

    # -- Heatmap of top correlated cluster --
    print("[11] fig_11b -- Rolling window redundancy ...")
    # Focus on rolling window groups
    rolling_groups = {
        "flightnum_od_depdelay": ["flightnum_od_depdelay_mean_last1",
                                   "flightnum_od_depdelay_mean_last7",
                                   "flightnum_od_depdelay_mean_last14",
                                   "flightnum_od_depdelay_median_last1",
                                   "flightnum_od_depdelay_median_last7",
                                   "flightnum_od_depdelay_median_last14"],
        "carrier_depdelay": ["carrier_depdelay_mean_last1",
                             "carrier_depdelay_mean_last7",
                             "carrier_depdelay_mean_last14"],
        "carrier_origin": ["carrier_origin_depdelay_mean_last1",
                           "carrier_origin_depdelay_mean_last7",
                           "carrier_origin_depdelay_mean_last14"],
        "origin_depdelay": ["origin_depdelay_mean_last1",
                            "origin_depdelay_mean_last7",
                            "origin_depdelay_mean_last14"],
        "origin_lateaircraft": ["origin_lateaircraft_rate_last1",
                                "origin_lateaircraft_rate_last7",
                                "origin_lateaircraft_rate_last14"],
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()
    rolling_corr_summary = {}
    for idx, (name, cols) in enumerate(rolling_groups.items()):
        if idx >= len(axes):
            break
        ax = axes[idx]
        valid_cols = [c for c in cols if c in df.columns]
        if len(valid_cols) < 2:
            ax.set_visible(False)
            continue
        rc = df[valid_cols].corr(method="spearman")
        sn = [short_name(c) for c in valid_cols]
        rc.index = sn
        rc.columns = sn
        from utils import safe_imshow
        safe_imshow(ax, rc.values, cmap="RdYlBu_r", vmin=0.5, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(sn)))
        ax.set_yticks(range(len(sn)))
        ax.set_xticklabels(sn, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(sn, fontsize=7)
        for i in range(len(sn)):
            for j in range(len(sn)):
                ax.text(j, i, f"{rc.values[i,j]:.2f}", ha="center", va="center", fontsize=7)
        ax.set_title(name, fontsize=9)
        # store min cross-correlation
        min_r = rc.values[np.triu_indices_from(rc.values, k=1)].min()
        rolling_corr_summary[name] = {"min_r": round(float(min_r), 3),
                                      "features": valid_cols}
    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"Rolling Window Feature Redundancy ({airline})", fontsize=13, y=1.01)
    fig.tight_layout()
    save_fig(fig, "fig_11b_rolling_window_redundancy")
    summary["rolling_groups"] = rolling_corr_summary

    # -- Hub spillover redundancy --
    print("[11] fig_11c -- Hub spillover redundancy ...")
    hub_delay_cols = [c for c in num_feats if c.startswith("hub_") and "depdelay_mean" in c]
    hub_late_cols = [c for c in num_feats if c.startswith("hub_") and "lateaircraft" in c]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for ax, cols, title in [(ax1, hub_delay_cols, "Hub Delay Means"),
                             (ax2, hub_late_cols, "Hub Late-Aircraft Rates")]:
        valid = [c for c in cols if c in df.columns]
        if len(valid) < 2:
            continue
        rc = df[valid].corr(method="spearman")
        sn = [short_name(c) for c in valid]
        safe_imshow(ax, rc.values, cmap="RdYlBu_r", vmin=-0.2, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(sn)))
        ax.set_yticks(range(len(sn)))
        ax.set_xticklabels(sn, rotation=90, fontsize=5)
        ax.set_yticklabels(sn, fontsize=5)
        ax.set_title(title, fontsize=10)
    fig.suptitle(f"Hub Spillover Feature Correlation ({airline})")
    fig.tight_layout()
    save_fig(fig, "fig_11c_hub_spillover_redundancy")

    with open(FIGURES / "11_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("[11] Done.")
    return summary


if __name__ == "__main__":
    airline = sys.argv[1] if len(sys.argv) > 1 else "WN"
    sample = int(sys.argv[2]) if len(sys.argv) > 2 else None
    run(airline, sample)
