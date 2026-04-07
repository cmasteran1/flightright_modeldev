"""02 -- Correlation analysis & non-linearity detection."""

import json, sys
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from utils import (load_weather_df, NUMERIC_WEATHER, TARGETS,
                   set_plot_style, save_fig, short_name, FIGURES)


def run(airline="WN", sample_n=None):
    set_plot_style()
    print("[02] Loading data ...")
    df = load_weather_df(airline, sample_n)
    summary = {}

    short_names = [short_name(c) for c in NUMERIC_WEATHER]

    # ── 2a: Pearson & Spearman heatmaps vs DepDelayMinutes ────────────
    print("[02] fig_02a -- Pearson & Spearman vs delay minutes ...")
    pearson_corr  = df[NUMERIC_WEATHER].corrwith(df["DepDelayMinutes"], method="pearson")
    spearman_corr = df[NUMERIC_WEATHER].corrwith(df["DepDelayMinutes"], method="spearman")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for ax, vals, title in [(ax1, pearson_corr, "Pearson"),
                             (ax2, spearman_corr, "Spearman")]:
        colors = ["#D32F2F" if v > 0 else "#1976D2" for v in vals]
        y_pos = np.arange(len(vals))
        ax.barh(y_pos, vals, color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(short_names, fontsize=8)
        ax.set_xlabel("Correlation")
        ax.set_title(f"{title} with DepDelayMinutes")
        ax.axvline(0, color="black", lw=0.5)
        for i, v in enumerate(vals):
            ax.text(v + 0.002 * np.sign(v), i, f"{v:.3f}", va="center", fontsize=7)
    fig.suptitle(f"Weather Feature Correlation with Delay Minutes ({airline})", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "fig_02a_pearson_spearman_heatmap")
    summary["pearson"]  = {short_name(c): round(v, 4) for c, v in pearson_corr.items()}
    summary["spearman"] = {short_name(c): round(v, 4) for c, v in spearman_corr.items()}

    # ── 2b: Inter-feature correlation matrix ──────────────────────────
    print("[02] fig_02b -- Inter-feature correlation matrix ...")
    corr_mat = df[NUMERIC_WEATHER].corr(method="spearman")
    corr_mat.index = short_names
    corr_mat.columns = short_names

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr_mat.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(short_names)))
    ax.set_yticks(range(len(short_names)))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short_names, fontsize=8)
    for i in range(len(short_names)):
        for j in range(len(short_names)):
            v = corr_mat.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=6, color="white" if abs(v) > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"Weather Feature Inter-Correlation (Spearman, {airline})")
    fig.tight_layout()
    save_fig(fig, "fig_02b_correlation_matrix_weather")

    # ── 2c: Point-biserial correlation with binary thresholds ─────────
    print("[02] fig_02c -- Point-biserial with binary thresholds ...")
    binary_targets = ["y_dep_ge15", "y_dep_ge30", "y_dep_ge45", "y_dep_ge60"]
    pb_data = {}
    for tgt in binary_targets:
        pb_data[tgt] = {}
        for col in NUMERIC_WEATHER:
            mask = df[col].notna() & df[tgt].notna()
            r, _ = sp_stats.pointbiserialr(df.loc[mask, tgt], df.loc[mask, col])
            pb_data[tgt][short_name(col)] = round(r, 4)

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(NUMERIC_WEATHER))
    width = 0.18
    colors_tgt = ["#66BB6A", "#FFA726", "#EF5350", "#AB47BC"]
    for k, tgt in enumerate(binary_targets):
        vals = [pb_data[tgt][short_name(c)] for c in NUMERIC_WEATHER]
        ax.bar(x + (k - 1.5) * width, vals, width,
               label=tgt.replace("y_dep_ge", ">=") + " min",
               color=colors_tgt[k], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Point-biserial r")
    ax.set_title(f"Weather Correlation by Delay Threshold ({airline})")
    ax.legend(fontsize=9)
    ax.axhline(0, color="black", lw=0.5)
    fig.tight_layout()
    save_fig(fig, "fig_02c_point_biserial_barplot")
    summary["point_biserial"] = pb_data

    # ── 2d: Decile delay-rate curves (non-linearity detector) ─────────
    print("[02] fig_02d -- Decile delay-rate curves ...")
    ncols = 4
    nrows = int(np.ceil(len(NUMERIC_WEATHER) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.5 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(NUMERIC_WEATHER):
        ax = axes[i]
        valid = df[[col, "y_dep_ge15"]].dropna()
        try:
            valid["decile"] = pd.qcut(valid[col], 10, labels=False, duplicates="drop")
        except ValueError:
            ax.set_title(short_name(col) + " (too few unique)", fontsize=9)
            continue
        grouped = valid.groupby("decile")["y_dep_ge15"]
        means = grouped.mean()
        sems  = grouped.sem()
        medians = valid.groupby("decile")[col].median()

        ax.plot(medians, means, "o-", color="#D32F2F", markersize=4)
        ax.fill_between(medians, means - 1.96 * sems, means + 1.96 * sems,
                        alpha=0.2, color="#D32F2F")
        ax.set_xlabel(short_name(col), fontsize=8)
        ax.set_ylabel("P(delay >=15)", fontsize=8)
        ax.set_title(short_name(col), fontsize=9)
        ax.tick_params(labelsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"Delay Rate by Weather Feature Decile ({airline})", fontsize=13, y=1.01)
    fig.tight_layout()
    save_fig(fig, "fig_02d_delay_by_weather_decile")

    # ── summary ───────────────────────────────────────────────────────
    with open(FIGURES / "02_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("[02] Done.")
    return summary


if __name__ == "__main__":
    airline = sys.argv[1] if len(sys.argv) > 1 else "WN"
    sample  = int(sys.argv[2]) if len(sys.argv) > 2 else None
    run(airline, sample)
