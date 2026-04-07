"""01 -- Univariate distribution analysis of weather features vs delay."""

import json, sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import (load_weather_df, NUMERIC_WEATHER, set_plot_style,
                   save_fig, short_name, FIGURES)


def run(airline="WN", sample_n=None):
    set_plot_style()
    print("[01] Loading data ...")
    df = load_weather_df(airline, sample_n)
    summary = {}

    # ── 1a: KDE overlays (on-time vs delayed >=15 min) ─────────────────
    print("[01] fig_01a -- KDE distributions ...")
    ncols = 4
    nrows = int(np.ceil(len(NUMERIC_WEATHER) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.2 * nrows))
    axes = axes.flatten()

    # subsample for KDE speed
    kde_df = df.sample(n=min(500_000, len(df)), random_state=42)
    ontime  = kde_df[kde_df["y_dep_ge15"] == 0]
    delayed = kde_df[kde_df["y_dep_ge15"] == 1]

    ks_scores = {}
    for i, col in enumerate(NUMERIC_WEATHER):
        ax = axes[i]
        v_on  = ontime[col].dropna()
        v_del = delayed[col].dropna()
        if len(v_on) == 0 or len(v_del) == 0:
            ax.set_title(short_name(col))
            continue
        ks_stat, ks_p = stats.ks_2samp(v_on, v_del)
        ks_scores[col] = ks_stat

        for vals, label, color in [(v_on, "On-time", "#1976D2"),
                                    (v_del, "Delayed >=15", "#D32F2F")]:
            lo, hi = vals.quantile(0.01), vals.quantile(0.99)
            bins = np.linspace(lo, hi, 60)
            ax.hist(vals.clip(lo, hi), bins=bins, density=True,
                    alpha=0.45, color=color, label=label)
        ax.set_title(f"{short_name(col)}\nKS={ks_stat:.3f}", fontsize=9)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"Weather Feature Distributions: On-time vs Delayed >=15 min ({airline})",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    save_fig(fig, "fig_01a_weather_distributions")

    # ── 1b: Boxplots by delay severity ────────────────────────────────
    print("[01] fig_01b -- Boxplots by delay severity ...")
    bins_edges = [0, 15, 30, 45, 60, 9999]
    labels_sev = ["<15", "15-30", "30-45", "45-60", "60+"]
    df["delay_bin"] = pd.cut(df["DepDelayMinutes"].clip(lower=0),
                             bins=bins_edges, labels=labels_sev, right=False)

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.5 * nrows))
    axes = axes.flatten()
    for i, col in enumerate(NUMERIC_WEATHER):
        ax = axes[i]
        data_groups = [df.loc[df["delay_bin"] == lab, col].dropna()
                       for lab in labels_sev]
        bp = ax.boxplot(data_groups, labels=labels_sev, patch_artist=True,
                        showfliers=False, medianprops=dict(color="black"))
        colors = ["#A5D6A7", "#FFF176", "#FFB74D", "#EF5350", "#B71C1C"]
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
        ax.set_title(short_name(col), fontsize=9)
        ax.tick_params(labelsize=7)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"Weather Features by Delay Severity ({airline})", fontsize=13, y=1.01)
    fig.tight_layout()
    save_fig(fig, "fig_01b_boxplots_by_delay")

    # ── 1c: Violin plots for top-6 by KS statistic ───────────────────
    print("[01] fig_01c -- Violin plots for top-6 features ...")
    top6 = sorted(ks_scores, key=ks_scores.get, reverse=True)[:6]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, col in enumerate(top6):
        ax = axes[i]
        vals_on  = ontime[col].dropna()
        vals_del = delayed[col].dropna()
        lo = min(vals_on.quantile(0.01), vals_del.quantile(0.01))
        hi = max(vals_on.quantile(0.99), vals_del.quantile(0.99))
        parts = ax.violinplot([vals_on.clip(lo, hi), vals_del.clip(lo, hi)],
                              showmedians=True, showextrema=False)
        for idx, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(["#1976D2", "#D32F2F"][idx])
            pc.set_alpha(0.6)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["On-time", "Delayed >=15"])
        ax.set_title(f"{short_name(col)}  (KS={ks_scores[col]:.3f})", fontsize=10)
    fig.suptitle(f"Top-6 Most Separating Weather Features ({airline})", fontsize=13, y=1.01)
    fig.tight_layout()
    save_fig(fig, "fig_01c_weather_delay_violin")

    # ── 1d: Zero-rate analysis (precip, CAPE) ─────────────────────────
    print("[01] fig_01d -- Zero-rate analysis ...")
    zero_cols = [c for c in NUMERIC_WEATHER
                 if "precip" in c or "cape" in c.lower() and "log" not in c]
    if not zero_cols:
        zero_cols = NUMERIC_WEATHER  # fallback
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(zero_cols))
    w = 0.35
    for offset, (mask, label, color) in enumerate([
        (df["y_dep_ge15"] == 0, "On-time", "#1976D2"),
        (df["y_dep_ge15"] == 1, "Delayed >=15", "#D32F2F"),
    ]):
        rates = [(df.loc[mask, c] == 0).mean() * 100 for c in zero_cols]
        ax.bar(x + (offset - 0.5) * w, rates, w, label=label, color=color, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([short_name(c) for c in zero_cols], rotation=30, ha="right")
    ax.set_ylabel("% of rows with value = 0")
    ax.set_title(f"Zero-Rate by Delay Status ({airline})")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "fig_01d_zero_rate_analysis")

    # ── summary ───────────────────────────────────────────────────────
    summary["ks_scores"] = {short_name(k): round(v, 4) for k, v in
                             sorted(ks_scores.items(), key=lambda x: -x[1])}
    summary["top6_by_ks"] = [short_name(c) for c in top6]
    summary["n_rows"] = len(df)
    with open(FIGURES / "01_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("[01] Done.")
    return summary


if __name__ == "__main__":
    airline = sys.argv[1] if len(sys.argv) > 1 else "WN"
    sample  = int(sys.argv[2]) if len(sys.argv) > 2 else None
    run(airline, sample)
