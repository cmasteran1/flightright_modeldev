"""04 -- Visibility & CAPE deep-dive (forecast-only variables)."""

import json, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import (load_weather_df, WMO_SEVERITY_MAP, SEVERITY_ORDER,
                   SEVERITY_COLORS, set_plot_style, save_fig, FIGURES)


def _sev(code):
    return WMO_SEVERITY_MAP.get(int(code), "Unknown") if pd.notna(code) else "Unknown"


def run(airline="WN", sample_n=None):
    set_plot_style()
    print("[04] Loading data ...")
    df = load_weather_df(airline, sample_n)
    summary = {}

    # ── 4a: Visibility delay curve with aviation thresholds ───────────
    print("[04] fig_04a -- Visibility delay curve ...")
    vis_bins = [0, 1000, 3000, 5000, 8000, 10000, 15000, 20000, 25001]
    vis_labels = ["0-1k", "1-3k", "3-5k", "5-8k", "8-10k", "10-15k", "15-20k", "20-25k"]
    df["vis_bin"] = pd.cut(df["origin_dep_visibility_m"], bins=vis_bins,
                           labels=vis_labels, right=False)

    grp = df.groupby("vis_bin", observed=True).agg(
        ge15=("y_dep_ge15", "mean"), ge30=("y_dep_ge30", "mean"),
        ge45=("y_dep_ge45", "mean"), ge60=("y_dep_ge60", "mean"),
        count=("y_dep_ge15", "count"),
    )

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    colors_thr = {"ge15": "#66BB6A", "ge30": "#FFA726", "ge45": "#EF5350", "ge60": "#AB47BC"}
    for thr, color in colors_thr.items():
        ax1.plot(range(len(grp)), grp[thr], "o-", color=color,
                 label=f">={thr[2:]} min", linewidth=2)
    ax2.bar(range(len(grp)), grp["count"], alpha=0.15, color="#90A4AE", label="Flight count")

    ax1.set_xticks(range(len(grp)))
    ax1.set_xticklabels(vis_labels, rotation=20)
    ax1.set_xlabel("Visibility (meters)")
    ax1.set_ylabel("Delay Rate")
    ax2.set_ylabel("Flight Count")
    ax1.legend(loc="upper right")

    # mark IFR / VFR thresholds
    ax1.axvline(2, color="red", linestyle="--", alpha=0.6, label="IFR <5km")
    ax1.axvline(3.5, color="green", linestyle="--", alpha=0.6, label="VFR >8km")
    ax1.annotate("IFR threshold", xy=(2, ax1.get_ylim()[1] * 0.9),
                 fontsize=8, color="red", ha="center")
    ax1.annotate("VFR threshold", xy=(3.5, ax1.get_ylim()[1] * 0.85),
                 fontsize=8, color="green", ha="center")

    ax1.set_title(f"Delay Rate by Visibility Bin ({airline})")
    fig.tight_layout()
    save_fig(fig, "fig_04a_visibility_delay_curve")

    summary["visibility_delay_rates"] = {
        k: {thr: round(grp.loc[k, thr], 4) for thr in colors_thr}
        for k in grp.index if k in grp.index
    }

    # ── 4b: CAPE delay analysis (raw vs log) ─────────────────────────
    print("[04] fig_04b -- CAPE delay analysis ...")
    cape_bins  = [-1, 0.01, 100, 500, 1000, 2500, 7000]
    cape_labels = ["0", "0-100", "100-500", "500-1k", "1k-2.5k", "2.5k+"]
    df["cape_bin"] = pd.cut(df["origin_dep_cape_jkg"], bins=cape_bins,
                            labels=cape_labels, right=False)

    log_bins = np.linspace(0, df["origin_dep_log1p_cape"].max() + 0.1, 8)
    df["log_cape_bin"] = pd.cut(df["origin_dep_log1p_cape"], bins=log_bins)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # raw CAPE bins
    grp1 = df.groupby("cape_bin", observed=True)["y_dep_ge15"].agg(["mean", "count"])
    ax1.bar(range(len(grp1)), grp1["mean"], color="#FF7043", alpha=0.8)
    ax1.set_xticks(range(len(grp1)))
    ax1.set_xticklabels(cape_labels, rotation=20)
    ax1.set_ylabel("Delay Rate (>=15 min)")
    ax1.set_title("Raw CAPE bins (J/kg)")
    for i, (m, c) in enumerate(zip(grp1["mean"], grp1["count"])):
        ax1.text(i, m + 0.005, f"n={c:,}", ha="center", fontsize=7)
    # thresholds
    ax1.axhline(df["y_dep_ge15"].mean(), color="gray", ls="--", lw=1, label="Baseline")
    ax1.legend(fontsize=8)

    # log1p CAPE bins
    grp2 = df.groupby("log_cape_bin", observed=True)["y_dep_ge15"].agg(["mean", "count"])
    ax2.bar(range(len(grp2)), grp2["mean"], color="#7E57C2", alpha=0.8)
    ax2.set_xticks(range(len(grp2)))
    ax2.set_xticklabels([f"{iv.left:.1f}-{iv.right:.1f}" for iv in grp2.index],
                         rotation=25, fontsize=8)
    ax2.set_ylabel("Delay Rate (>=15 min)")
    ax2.set_title("log(1+CAPE) equal-width bins")
    ax2.axhline(df["y_dep_ge15"].mean(), color="gray", ls="--", lw=1)

    fig.suptitle(f"CAPE vs Delay Rate ({airline})", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "fig_04b_cape_delay_analysis")

    # ── 4c: Visibility x CAPE 2D heatmap ─────────────────────────────
    print("[04] fig_04c -- Visibility x CAPE interaction ...")
    ct = df.groupby(["vis_bin", "cape_bin"], observed=True)["y_dep_ge15"].mean().unstack()

    fig, ax = plt.subplots(figsize=(10, 7))
    from utils import safe_imshow
    im = safe_imshow(ax, ct.values, cmap="YlOrRd", aspect="auto", origin="lower")
    ax.set_xticks(range(ct.shape[1]))
    ax.set_yticks(range(ct.shape[0]))
    ax.set_xticklabels(ct.columns, fontsize=9)
    ax.set_yticklabels(ct.index, fontsize=9)
    ax.set_xlabel("CAPE bin (J/kg)")
    ax.set_ylabel("Visibility bin (m)")
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            v = ct.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if v > 0.35 else "black")
    fig.colorbar(im, ax=ax, label="Delay Rate (>=15 min)", shrink=0.8)
    ax.set_title(f"Visibility x CAPE -> Delay Rate ({airline})")
    fig.tight_layout()
    save_fig(fig, "fig_04c_visibility_cape_interaction")

    # ── 4d: CAPE zero vs nonzero analysis ─────────────────────────────
    print("[04] fig_04d -- CAPE zero vs nonzero ...")
    df["cape_flag"] = np.where(df["origin_dep_cape_jkg"] > 0, "CAPE > 0", "CAPE = 0")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # delay distribution
    for flag, color in [("CAPE = 0", "#1976D2"), ("CAPE > 0", "#D32F2F")]:
        vals = df.loc[df["cape_flag"] == flag, "DepDelayMinutes"].clip(upper=120)
        ax1.hist(vals, bins=60, density=True, alpha=0.5, color=color, label=flag)
    ax1.set_xlabel("Departure Delay (min, clipped 120)")
    ax1.set_ylabel("Density")
    ax1.set_title("Delay Distribution: CAPE=0 vs CAPE>0")
    ax1.legend()

    # delay rate by hour for CAPE=0 vs >0
    hourly = df.groupby(["sched_dep_hour", "cape_flag"])["y_dep_ge15"].mean().unstack()
    for flag, color in [("CAPE = 0", "#1976D2"), ("CAPE > 0", "#D32F2F")]:
        if flag in hourly.columns:
            ax2.plot(hourly.index, hourly[flag], "o-", color=color, label=flag)
    ax2.set_xlabel("Scheduled Departure Hour")
    ax2.set_ylabel("Delay Rate (>=15 min)")
    ax2.set_title("Delay Rate by Hour: CAPE=0 vs CAPE>0")
    ax2.legend()

    fig.suptitle(f"CAPE Zero/Nonzero Analysis ({airline})", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "fig_04d_cape_zero_analysis")

    # ── 4e: Visibility by weather code group ──────────────────────────
    print("[04] fig_04e -- Visibility by weather code group ...")
    df["hourly_severity"] = df["origin_dep_hour_weathercode"].apply(_sev)
    present = [g for g in SEVERITY_ORDER if g in df["hourly_severity"].unique()]

    fig, ax = plt.subplots(figsize=(12, 6))
    data_groups = [df.loc[df["hourly_severity"] == g, "origin_dep_visibility_m"].dropna()
                   for g in present]
    bp = ax.boxplot(data_groups, labels=present, patch_artist=True,
                    showfliers=False, medianprops=dict(color="black"))
    for patch, g in zip(bp["boxes"], present):
        patch.set_facecolor(SEVERITY_COLORS.get(g, "#757575"))
        patch.set_alpha(0.7)
    ax.axhline(5000, color="red", ls="--", lw=1, label="IFR threshold (5 km)")
    ax.set_ylabel("Visibility (m)")
    ax.set_title(f"Visibility by Weather Severity Group ({airline})")
    ax.legend()
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    save_fig(fig, "fig_04e_visibility_by_weathercode")

    # ── summary ───────────────────────────────────────────────────────
    cape_zero_rate = (df["origin_dep_cape_jkg"] == 0).mean()
    cape_gt0_delay = df.loc[df["origin_dep_cape_jkg"] > 0, "y_dep_ge15"].mean()
    cape_eq0_delay = df.loc[df["origin_dep_cape_jkg"] == 0, "y_dep_ge15"].mean()
    summary["cape_zero_fraction"] = round(cape_zero_rate, 4)
    summary["cape_gt0_delay_rate"] = round(cape_gt0_delay, 4)
    summary["cape_eq0_delay_rate"] = round(cape_eq0_delay, 4)
    with open(FIGURES / "04_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("[04] Done.")
    return summary


if __name__ == "__main__":
    airline = sys.argv[1] if len(sys.argv) > 1 else "WN"
    sample  = int(sys.argv[2]) if len(sys.argv) > 2 else None
    run(airline, sample)
