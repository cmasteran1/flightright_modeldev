"""05 -- Seasonal and geographic patterns in weather-delay relationship."""

import json, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import (load_weather_df, NUMERIC_WEATHER, US_REGION_MAP,
                   set_plot_style, save_fig, short_name, FIGURES)

SEASONS = {12: "Winter", 1: "Winter", 2: "Winter",
           3: "Spring", 4: "Spring", 5: "Spring",
           6: "Summer", 7: "Summer", 8: "Summer",
           9: "Fall", 10: "Fall", 11: "Fall"}
SEASON_ORDER = ["Winter", "Spring", "Summer", "Fall"]


def run(airline="WN", sample_n=None):
    set_plot_style()
    print("[05] Loading data ...")
    df = load_weather_df(airline, sample_n)
    df["month"] = df["flight_month"] if "flight_month" in df.columns else pd.to_datetime(df["FlightDate"]).dt.month
    df["season"] = df["month"].map(SEASONS)
    df["region"] = df["Origin"].map(US_REGION_MAP).fillna("Other")
    summary = {}

    # ── 5a: Monthly co-movement: delay rate vs weather means ──────────
    print("[05] fig_05a -- Monthly co-movement ...")
    monthly = df.groupby("month").agg(
        delay_rate=("y_dep_ge15", "mean"),
        precip=("origin_daily_precip_sum_mm", "mean"),
        windspeed=("origin_dep_windspeed_kmh", "mean"),
        cape=("origin_dep_cape_jkg", "mean"),
    )

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    ax1.plot(monthly.index, monthly["delay_rate"], "o-", color="#D32F2F",
             linewidth=2, markersize=6, label="Delay Rate >=15", zorder=5)
    ax2.plot(monthly.index, monthly["precip"], "s--", color="#1565C0",
             alpha=0.7, label="Avg Precip (mm)")
    ax2.plot(monthly.index, monthly["cape"], "^--", color="#FF7043",
             alpha=0.7, label="Avg CAPE (J/kg)")
    ax2.plot(monthly.index, monthly["windspeed"], "d--", color="#4CAF50",
             alpha=0.7, label="Avg Windspeed (km/h)")

    ax1.set_xlabel("Month")
    ax1.set_ylabel("Delay Rate (>=15 min)", color="#D32F2F")
    ax2.set_ylabel("Weather Feature Mean")
    ax1.set_xticks(range(1, 13))
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax1.set_title(f"Monthly Delay Rate vs Weather Features ({airline})")
    fig.tight_layout()
    save_fig(fig, "fig_05a_monthly_delay_vs_weather")

    # ── 5b: Per-season correlation bars ───────────────────────────────
    print("[05] fig_05b -- Seasonal correlation change ...")
    fig, axes = plt.subplots(1, 4, figsize=(18, 6), sharey=True)
    season_corrs = {}
    for ax, season in zip(axes, SEASON_ORDER):
        sdf = df[df["season"] == season]
        corrs = sdf[NUMERIC_WEATHER].corrwith(sdf["y_dep_ge15"], method="spearman")
        season_corrs[season] = {short_name(c): round(v, 4) for c, v in corrs.items()}
        sn = [short_name(c) for c in NUMERIC_WEATHER]
        colors = ["#D32F2F" if v > 0 else "#1976D2" for v in corrs]
        ax.barh(range(len(corrs)), corrs, color=colors, alpha=0.8)
        ax.set_yticks(range(len(corrs)))
        ax.set_yticklabels(sn, fontsize=8)
        ax.set_title(f"{season} (n={len(sdf):,})")
        ax.axvline(0, color="black", lw=0.5)
        ax.set_xlabel("Spearman r")
    fig.suptitle(f"Weather-Delay Correlation by Season ({airline})", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "fig_05b_seasonal_correlation_change")
    summary["seasonal_correlations"] = season_corrs

    # ── 5c: Regional delay rates ──────────────────────────────────────
    print("[05] fig_05c -- Regional delay rates ...")
    regions = ["Northeast", "Southeast", "Midwest", "Southwest", "West"]
    reg_df = df[df["region"].isin(regions)]
    reg_grp = reg_df.groupby("region").agg(
        delay_rate=("y_dep_ge15", "mean"),
        weather_rate=("origin_weather_rate_last7", "mean") if "origin_weather_rate_last7" in df.columns else ("y_dep_ge15", "mean"),
        count=("y_dep_ge15", "count"),
    ).reindex(regions)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(regions))
    ax.bar(x, reg_grp["delay_rate"], color="#EF5350", alpha=0.8, label="Overall Delay Rate")
    if "origin_weather_rate_last7" in df.columns:
        ax.bar(x, reg_grp["weather_rate"], color="#42A5F5", alpha=0.6,
               label="BTS Weather-Attributed Rate (7d avg)")
    ax.set_xticks(x)
    ax.set_xticklabels(regions, rotation=15)
    ax.set_ylabel("Rate")
    ax.set_title(f"Delay Rate by Region ({airline})")
    ax.legend()
    for i, cnt in enumerate(reg_grp["count"]):
        ax.text(i, reg_grp["delay_rate"].iloc[i] + 0.005, f"n={cnt:,}",
                ha="center", fontsize=8)
    fig.tight_layout()
    save_fig(fig, "fig_05c_regional_delay_rates")

    # ── 5d: Regional weather sensitivity ──────────────────────────────
    print("[05] fig_05d -- Regional weather sensitivity slopes ...")
    key_features = ["origin_daily_precip_sum_mm", "origin_dep_windspeed_kmh",
                    "origin_dep_visibility_m", "origin_dep_cape_jkg"]
    slopes = {r: {} for r in regions}
    for region in regions:
        rdf = reg_df[reg_df["region"] == region]
        for feat in key_features:
            valid = rdf[[feat, "y_dep_ge15"]].dropna()
            if len(valid) < 100:
                slopes[region][short_name(feat)] = 0
                continue
            try:
                q = pd.qcut(valid[feat], 5, labels=False, duplicates="drop")
                med = valid.groupby(q)[feat].median()
                rate = valid.groupby(q)["y_dep_ge15"].mean()
                if len(med) >= 2:
                    slope = np.polyfit(med, rate, 1)[0]
                else:
                    slope = 0
            except Exception:
                slope = 0
            slopes[region][short_name(feat)] = round(slope * 1000, 3)  # per 1000 units

    fig, ax = plt.subplots(figsize=(12, 6))
    sn = [short_name(f) for f in key_features]
    x = np.arange(len(sn))
    w = 0.15
    region_colors = ["#1565C0", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]
    for k, (region, color) in enumerate(zip(regions, region_colors)):
        vals = [slopes[region].get(s, 0) for s in sn]
        ax.bar(x + (k - 2) * w, vals, w, label=region, color=color, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(sn, rotation=20, ha="right")
    ax.set_ylabel("Slope (delay rate change per 1000 units)")
    ax.set_title(f"Regional Weather Sensitivity ({airline})")
    ax.legend(fontsize=8)
    ax.axhline(0, color="black", lw=0.5)
    fig.tight_layout()
    save_fig(fig, "fig_05d_regional_weather_sensitivity")

    # ── 5e: Airport x month heatmap (top-20 airports) ────────────────
    print("[05] fig_05e -- Airport x month heatmap ...")
    top20 = df["Origin"].value_counts().head(20).index.tolist()
    sub = df[df["Origin"].isin(top20)]
    ct = sub.groupby(["Origin", "month"])["y_dep_ge15"].mean().unstack()
    ct = ct.reindex(columns=range(1, 13))
    # sort by overall delay rate
    ct = ct.loc[ct.mean(axis=1).sort_values(ascending=False).index]

    from utils import safe_imshow
    fig, ax = plt.subplots(figsize=(12, 8))
    im = safe_imshow(ax, ct.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(12))
    ax.set_xticklabels([f"{m}" for m in range(1, 13)])
    ax.set_yticks(range(len(ct)))
    ax.set_yticklabels(ct.index, fontsize=9)
    ax.set_xlabel("Month")
    ax.set_ylabel("Airport")
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            v = ct.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if v > 0.3 else "black")
    fig.colorbar(im, ax=ax, label="Delay Rate (>=15 min)", shrink=0.7)
    ax.set_title(f"Top-20 Airports: Monthly Delay Rate ({airline})")
    fig.tight_layout()
    save_fig(fig, "fig_05e_heatmap_airport_weatherdelay")

    # ── summary ───────────────────────────────────────────────────────
    summary["regional_slopes"] = slopes
    with open(FIGURES / "05_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("[05] Done.")
    return summary


if __name__ == "__main__":
    airline = sys.argv[1] if len(sys.argv) > 1 else "WN"
    sample  = int(sys.argv[2]) if len(sys.argv) > 2 else None
    run(airline, sample)
