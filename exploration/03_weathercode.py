"""03 -- WMO weather code deep-dive."""

import json, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import (load_weather_df, WMO_CODE_MAP, WMO_SEVERITY_MAP,
                   SEVERITY_ORDER, SEVERITY_COLORS,
                   set_plot_style, save_fig, FIGURES)


def _sev(code):
    return WMO_SEVERITY_MAP.get(int(code), "Unknown") if pd.notna(code) else "Unknown"


def run(airline="WN", sample_n=None):
    set_plot_style()
    print("[03] Loading data ...")
    df = load_weather_df(airline, sample_n)
    summary = {}

    # ── 3a: Delay rate by weather code ────────────────────────────────
    print("[03] fig_03a -- Delay rate by weather code ...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

    for ax, col, title in [
        (ax1, "origin_daily_weathercode", "Daily Weather Code"),
        (ax2, "origin_dep_hour_weathercode", "Hourly Departure Weather Code"),
    ]:
        grp = df.groupby(col).agg(
            delay_rate=("y_dep_ge15", "mean"),
            count=("y_dep_ge15", "count"),
        ).reset_index()
        grp = grp[grp["count"] >= 500].sort_values("delay_rate", ascending=True)
        grp["severity"] = grp[col].apply(_sev)
        grp["label"] = grp[col].apply(
            lambda c: f"{int(c)}: {WMO_CODE_MAP.get(int(c), '?')}")

        colors = [SEVERITY_COLORS.get(s, "#757575") for s in grp["severity"]]
        ax.barh(range(len(grp)), grp["delay_rate"], color=colors, alpha=0.85)
        ax.set_yticks(range(len(grp)))
        ax.set_yticklabels(grp["label"], fontsize=8)
        ax.set_xlabel("Delay Rate (>=15 min)")
        ax.set_title(title)
        # annotate counts
        for idx, (rate, cnt) in enumerate(zip(grp["delay_rate"], grp["count"])):
            ax.text(rate + 0.003, idx, f"n={cnt:,}", va="center", fontsize=7)

    fig.suptitle(f"Delay Rate by WMO Weather Code ({airline})", fontsize=13, y=1.01)
    fig.tight_layout()
    save_fig(fig, "fig_03a_weathercode_delay_rate")

    # ── 3b: Frequency vs impact scatter ───────────────────────────────
    print("[03] fig_03b -- Frequency vs impact scatter ...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    for ax, col, title in [
        (ax1, "origin_daily_weathercode", "Daily"),
        (ax2, "origin_dep_hour_weathercode", "Hourly at Departure"),
    ]:
        grp = df.groupby(col).agg(
            delay_rate=("y_dep_ge15", "mean"),
            count=("y_dep_ge15", "count"),
        ).reset_index()
        grp["freq_pct"] = grp["count"] / len(df) * 100
        grp["severity"] = grp[col].apply(_sev)
        colors = [SEVERITY_COLORS.get(s, "#757575") for s in grp["severity"]]
        sizes  = np.clip(grp["count"] / grp["count"].max() * 500, 30, 500)

        ax.scatter(grp["freq_pct"], grp["delay_rate"], s=sizes,
                   c=colors, alpha=0.7, edgecolors="black", linewidth=0.5)
        for _, row in grp.iterrows():
            ax.annotate(f"{int(row[col])}", (row["freq_pct"], row["delay_rate"]),
                        fontsize=7, ha="center")
        ax.set_xlabel("Frequency (% of flights)")
        ax.set_ylabel("Delay Rate (>=15 min)")
        ax.set_title(title)

    fig.suptitle(f"Weather Code: Frequency vs Impact ({airline})", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "fig_03b_weathercode_frequency_vs_impact")

    # ── 3c: Delay minutes boxplot by severity group ───────────────────
    print("[03] fig_03c -- Delay minutes by severity group ...")
    df["hourly_severity"] = df["origin_dep_hour_weathercode"].apply(_sev)
    present_groups = [g for g in SEVERITY_ORDER if g in df["hourly_severity"].unique()]

    fig, ax = plt.subplots(figsize=(12, 6))
    data_groups = []
    for g in present_groups:
        vals = df.loc[df["hourly_severity"] == g, "DepDelayMinutes"].clip(upper=120).dropna()
        data_groups.append(vals)
    bp = ax.boxplot(data_groups, labels=present_groups, patch_artist=True,
                    showfliers=False, medianprops=dict(color="black"))
    for patch, g in zip(bp["boxes"], present_groups):
        patch.set_facecolor(SEVERITY_COLORS.get(g, "#757575"))
        patch.set_alpha(0.7)
    means = [v.mean() for v in data_groups]
    ax.scatter(range(1, len(means) + 1), means, color="black", marker="D",
               zorder=5, s=30, label="Mean")
    ax.set_ylabel("Departure Delay (min, clipped 120)")
    ax.set_title(f"Delay Distribution by Weather Severity ({airline})")
    ax.legend()
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    save_fig(fig, "fig_03c_weathercode_severity_boxplot")

    # ── 3d: Daily vs hourly cross-tabulation heatmap ──────────────────
    print("[03] fig_03d -- Daily vs hourly code cross-tabulation ...")
    df["daily_severity"]  = df["origin_daily_weathercode"].apply(_sev)
    ct = df.groupby(["daily_severity", "hourly_severity"])["y_dep_ge15"].mean().unstack()
    # reorder by severity
    row_order = [g for g in SEVERITY_ORDER if g in ct.index]
    col_order = [g for g in SEVERITY_ORDER if g in ct.columns]
    ct = ct.reindex(index=row_order, columns=col_order).astype(float)

    fig, ax = plt.subplots(figsize=(10, 7))
    data = np.where(np.isnan(ct.values), 0, ct.values)
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(col_order)))
    ax.set_yticks(range(len(row_order)))
    ax.set_xticklabels(col_order, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(row_order, fontsize=9)
    ax.set_xlabel("Hourly (at departure)")
    ax.set_ylabel("Daily")
    for i in range(len(row_order)):
        for j in range(len(col_order)):
            v = ct.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if v > 0.35 else "black")
    fig.colorbar(im, ax=ax, label="Delay Rate (>=15 min)", shrink=0.8)
    ax.set_title(f"Daily vs Hourly Weather Severity -> Delay Rate ({airline})")
    fig.tight_layout()
    save_fig(fig, "fig_03d_weathercode_daily_vs_hourly")

    # ── summary ───────────────────────────────────────────────────────
    sev_rates = df.groupby("hourly_severity")["y_dep_ge15"].mean().to_dict()
    summary["delay_rate_by_severity"] = {k: round(v, 4) for k, v in sev_rates.items()}
    with open(FIGURES / "03_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("[03] Done.")
    return summary


if __name__ == "__main__":
    airline = sys.argv[1] if len(sys.argv) > 1 else "WN"
    sample  = int(sys.argv[2]) if len(sys.argv) > 2 else None
    run(airline, sample)
