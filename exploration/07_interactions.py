"""07 -- Pairwise interaction effects between weather features."""

import json, sys
import numpy as np
import pandas as pd
from itertools import combinations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import (load_weather_df, NUMERIC_WEATHER,
                   set_plot_style, save_fig, short_name, FIGURES)

# top-8 features for interaction testing (can be updated from script 06 results)
TOP8 = [
    "origin_dep_visibility_m", "origin_dep_cape_jkg",
    "origin_dep_windspeed_kmh", "origin_dep_windgusts_kmh",
    "origin_daily_precip_sum_mm", "origin_dep_precip_mm",
    "origin_dep_cloudcover_pct", "origin_daily_windgusts_max_kmh",
]


def _heatmap_2d(df, col_x, col_y, target, ax, n_bins=5):
    """Compute and plot 2D delay rate heatmap."""
    valid = df[[col_x, col_y, target]].dropna()
    try:
        valid["bx"] = pd.qcut(valid[col_x], n_bins, labels=False, duplicates="drop")
        valid["by"] = pd.qcut(valid[col_y], n_bins, labels=False, duplicates="drop")
    except ValueError:
        return None
    ct = valid.groupby(["by", "bx"])[target].mean().unstack()

    # axis labels from medians
    xlabels = valid.groupby("bx")[col_x].median().round(1)
    ylabels = valid.groupby("by")[col_y].median().round(1)

    from utils import safe_imshow
    im = safe_imshow(ax, ct.values, cmap="YlOrRd", aspect="auto", origin="lower")
    ax.set_xticks(range(ct.shape[1]))
    ax.set_xticklabels([f"{xlabels.get(i, '?')}" for i in range(ct.shape[1])], fontsize=7)
    ax.set_yticks(range(ct.shape[0]))
    ax.set_yticklabels([f"{ylabels.get(i, '?')}" for i in range(ct.shape[0])], fontsize=7)
    ax.set_xlabel(short_name(col_x), fontsize=9)
    ax.set_ylabel(short_name(col_y), fontsize=9)
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            v = ct.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.35 else "black")
    return ct


def run(airline="WN", sample_n=None):
    set_plot_style()
    print("[07] Loading data ...")
    df = load_weather_df(airline, sample_n)
    summary = {}
    base_rate = df["y_dep_ge15"].mean()

    # ── 7a–c: Key interaction heatmaps ────────────────────────────────
    print("[07] fig_07a-c -- Key interaction heatmaps ...")
    pairs = [
        ("origin_dep_windspeed_kmh", "origin_dep_visibility_m", "wind_visibility"),
        ("origin_daily_precip_sum_mm", "origin_dep_windgusts_kmh", "precip_wind"),
        ("origin_dep_cape_jkg", "origin_dep_cloudcover_pct", "cape_cloudcover"),
    ]
    for col_x, col_y, name in pairs:
        fig, ax = plt.subplots(figsize=(9, 7))
        ct = _heatmap_2d(df, col_x, col_y, "y_dep_ge15", ax)
        if ct is not None:
            fig.colorbar(plt.cm.ScalarMappable(cmap="YlOrRd",
                         norm=plt.Normalize(ct.min().min(), ct.max().max())),
                         ax=ax, label="Delay Rate (>=15 min)", shrink=0.8)
        ax.set_title(f"{short_name(col_x)} x {short_name(col_y)} -> Delay Rate ({airline})")
        fig.tight_layout()
        letter = {"wind_visibility": "a", "precip_wind": "b", "cape_cloudcover": "c"}[name]
        save_fig(fig, f"fig_07{letter}_{name}_interaction")

    # ── 7d: Interaction strength for all top-8 pairs ──────────────────
    print("[07] fig_07d -- Interaction strength scores ...")
    available = [c for c in TOP8 if c in df.columns]
    pair_scores = {}
    for col_a, col_b in combinations(available, 2):
        valid = df[[col_a, col_b, "y_dep_ge15"]].dropna()
        try:
            valid["ba"] = pd.qcut(valid[col_a], 5, labels=False, duplicates="drop")
            valid["bb"] = pd.qcut(valid[col_b], 5, labels=False, duplicates="drop")
        except ValueError:
            continue
        # marginal rates in worst quintile (4)
        rate_a_worst = valid.loc[valid["ba"] == valid["ba"].max(), "y_dep_ge15"].mean()
        rate_b_worst = valid.loc[valid["bb"] == valid["bb"].max(), "y_dep_ge15"].mean()
        # for visibility, worst = lowest, so use min
        if "visibility" in col_a:
            rate_a_worst = valid.loc[valid["ba"] == 0, "y_dep_ge15"].mean()
        if "visibility" in col_b:
            rate_b_worst = valid.loc[valid["bb"] == 0, "y_dep_ge15"].mean()

        # joint worst
        mask_a = valid["ba"] == (0 if "visibility" in col_a else valid["ba"].max())
        mask_b = valid["bb"] == (0 if "visibility" in col_b else valid["bb"].max())
        joint_worst = valid.loc[mask_a & mask_b, "y_dep_ge15"].mean()

        # independence prediction
        expected = rate_a_worst * rate_b_worst / base_rate if base_rate > 0 else 0
        score = joint_worst / expected if expected > 0 else 1.0

        pair_name = f"{short_name(col_a)} x {short_name(col_b)}"
        pair_scores[pair_name] = round(score, 3)

    scores_sorted = dict(sorted(pair_scores.items(), key=lambda x: -x[1]))
    fig, ax = plt.subplots(figsize=(12, 7))
    names = list(scores_sorted.keys())
    vals  = list(scores_sorted.values())
    colors = ["#D32F2F" if v > 1.1 else "#4CAF50" if v < 0.9 else "#FFA726"
              for v in vals]
    ax.barh(range(len(names)), vals, color=colors, alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.axvline(1.0, color="black", ls="--", lw=1, label="Independence (1.0)")
    ax.set_xlabel("Interaction Score (observed/expected)")
    ax.set_title(f"Pairwise Weather Interaction Strength ({airline})")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "fig_07d_interaction_strength_bars")
    summary["interaction_scores"] = scores_sorted

    # ── 7e: Suggested engineered features ─────────────────────────────
    print("[07] fig_07e -- Candidate engineered features ...")
    candidates = {}
    if "origin_dep_windspeed_kmh" in df.columns and "origin_dep_precip_mm" in df.columns:
        df["wind_x_precip"] = df["origin_dep_windspeed_kmh"] * df["origin_dep_precip_mm"]
        candidates["wind_x_precip"] = df["wind_x_precip"].corr(df["y_dep_ge15"])
    if "origin_dep_visibility_m" in df.columns:
        df["is_ifr"] = (df["origin_dep_visibility_m"] < 5000).astype(int)
        candidates["is_ifr (<5km)"] = df["is_ifr"].corr(df["y_dep_ge15"])
    if "origin_dep_cape_jkg" in df.columns:
        df["is_convective"] = (df["origin_dep_cape_jkg"] > 500).astype(int)
        candidates["is_convective (CAPE>500)"] = df["is_convective"].corr(df["y_dep_ge15"])
    if "origin_dep_windgusts_kmh" in df.columns and "origin_dep_windspeed_kmh" in df.columns:
        safe_ws = df["origin_dep_windspeed_kmh"].replace(0, np.nan)
        df["gust_factor"] = df["origin_dep_windgusts_kmh"] / safe_ws
        candidates["gust_factor"] = df["gust_factor"].corr(df["y_dep_ge15"])
    if "origin_dep_temp_K" in df.columns:
        df["below_freezing"] = (df["origin_dep_temp_K"] < 273.15).astype(int)
        candidates["below_freezing"] = df["below_freezing"].corr(df["y_dep_ge15"])

    # compare with individual feature correlations
    indiv_corrs = {short_name(c): df[c].corr(df["y_dep_ge15"]) for c in NUMERIC_WEATHER}

    fig, ax = plt.subplots(figsize=(12, 6))
    all_feats = {**{f"[NEW] {k}": v for k, v in candidates.items()}, **indiv_corrs}
    all_sorted = dict(sorted(all_feats.items(), key=lambda x: abs(x[1]), reverse=True))
    names = list(all_sorted.keys())[:20]
    vals  = [all_sorted[n] for n in names]
    colors = ["#D32F2F" if "[NEW]" in n else "#1976D2" for n in names]
    ax.barh(range(len(names)), vals, color=colors, alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Pearson correlation with y_dep_ge15")
    ax.set_title(f"Candidate Engineered Features vs Existing ({airline})")
    ax.axvline(0, color="black", lw=0.5)
    fig.tight_layout()
    save_fig(fig, "fig_07e_suggested_engineered_features")
    summary["candidate_features"] = {k: round(v, 4) for k, v in candidates.items()}

    # ── summary ───────────────────────────────────────────────────────
    with open(FIGURES / "07_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("[07] Done.")
    return summary


if __name__ == "__main__":
    airline = sys.argv[1] if len(sys.argv) > 1 else "WN"
    sample  = int(sys.argv[2]) if len(sys.argv) > 2 else None
    run(airline, sample)
