"""08 -- Cross-airline weather sensitivity comparison."""

import json, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from utils import (load_weather_df, NUMERIC_WEATHER, CAT_WEATHER, ALL_WEATHER,
                   WMO_SEVERITY_MAP, SEVERITY_ORDER, SEVERITY_COLORS,
                   set_plot_style, save_fig, short_name, FIGURES)

try:
    from catboost import CatBoostClassifier, Pool
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

AIRLINES = ["WN", "AA", "DL", "UA"]
AIRLINE_COLORS = {"WN": "#F57C00", "AA": "#C62828", "DL": "#1565C0", "UA": "#2E7D32"}


def _sev(code):
    return WMO_SEVERITY_MAP.get(int(code), "Unknown") if pd.notna(code) else "Unknown"


def run(sample_n=None):
    set_plot_style()
    print("[08] Loading data for all airlines ...")
    dfs = {}
    for al in AIRLINES:
        try:
            dfs[al] = load_weather_df(al, sample_n)
            dfs[al]["airline"] = al
            print(f"  {al}: {len(dfs[al]):,} rows")
        except FileNotFoundError:
            print(f"  {al}: NOT FOUND, skipping")

    if not dfs:
        print("[08] No data found. Aborting.")
        return {}

    combined = pd.concat(dfs.values(), ignore_index=True)
    combined["hourly_severity"] = combined["origin_dep_hour_weathercode"].apply(_sev)
    summary = {}

    # ── 8a: Delay rate by airline x severity ─────────────────────────
    print("[08] fig_08a -- Delay rate by airline x weather severity ...")
    present_sev = [s for s in SEVERITY_ORDER if s in combined["hourly_severity"].unique()]
    grp = combined.groupby(["airline", "hourly_severity"])["y_dep_ge15"].mean().unstack()
    grp = grp.reindex(columns=present_sev)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(present_sev))
    w = 0.18
    for k, al in enumerate(AIRLINES):
        if al in grp.index:
            vals = [grp.loc[al, s] if s in grp.columns and pd.notna(grp.loc[al, s]) else 0
                    for s in present_sev]
            ax.bar(x + (k - 1.5) * w, vals, w, label=al,
                   color=AIRLINE_COLORS[al], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(present_sev, rotation=25, ha="right")
    ax.set_ylabel("Delay Rate (>=15 min)")
    ax.set_title("Delay Rate by Airline x Weather Severity")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "fig_08a_delay_rate_by_airline_weather")

    # ── 8b: Weather sensitivity slopes by airline ─────────────────────
    print("[08] fig_08b -- Weather sensitivity slopes ...")
    key_feats = ["origin_daily_precip_sum_mm", "origin_dep_windspeed_kmh",
                 "origin_dep_visibility_m", "origin_dep_cape_jkg"]
    slopes = {}
    for al in AIRLINES:
        if al not in dfs:
            continue
        adf = dfs[al]
        slopes[al] = {}
        for feat in key_feats:
            valid = adf[[feat, "y_dep_ge15"]].dropna()
            try:
                q = pd.qcut(valid[feat], 5, labels=False, duplicates="drop")
                med = valid.groupby(q)[feat].median()
                rate = valid.groupby(q)["y_dep_ge15"].mean()
                slope = np.polyfit(med, rate, 1)[0] if len(med) >= 2 else 0
            except Exception:
                slope = 0
            slopes[al][short_name(feat)] = round(slope * 1000, 3)

    fig, ax = plt.subplots(figsize=(12, 6))
    sn = [short_name(f) for f in key_feats]
    x = np.arange(len(sn))
    w = 0.18
    for k, al in enumerate(AIRLINES):
        if al in slopes:
            vals = [slopes[al].get(s, 0) for s in sn]
            ax.bar(x + (k - 1.5) * w, vals, w, label=al,
                   color=AIRLINE_COLORS[al], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(sn, rotation=20, ha="right")
    ax.set_ylabel("Slope (delay rate per 1000 units)")
    ax.set_title("Weather Sensitivity Slopes by Airline")
    ax.legend()
    ax.axhline(0, color="black", lw=0.5)
    fig.tight_layout()
    save_fig(fig, "fig_08b_weather_sensitivity_slopes")
    summary["sensitivity_slopes"] = slopes

    # ── 8c: Weather-only AUC by airline ───────────────────────────────
    if HAS_CATBOOST:
        print("[08] fig_08c -- Weather-only AUC by airline ...")
        aucs = {}
        feature_cols = NUMERIC_WEATHER + CAT_WEATHER
        cat_indices = [feature_cols.index(c) for c in CAT_WEATHER]

        for al in AIRLINES:
            if al not in dfs:
                continue
            adf = dfs[al].copy()
            sub = adf[feature_cols + ["y_dep_ge15", "FlightDate"]].dropna()
            for cc in CAT_WEATHER:
                sub[cc] = sub[cc].astype(int).astype(str)
            dates = sorted(sub["FlightDate"].unique())
            split = int(len(dates) * 0.8)
            train = sub[sub["FlightDate"].isin(set(dates[:split]))]
            test  = sub[sub["FlightDate"].isin(set(dates[split:]))]

            model = CatBoostClassifier(
                iterations=500, depth=6, learning_rate=0.05,
                loss_function="Logloss", cat_features=cat_indices,
                verbose=0, random_seed=42,
            )
            model.fit(train[feature_cols], train["y_dep_ge15"])
            prob = model.predict_proba(test[feature_cols])[:, 1]
            auc = roc_auc_score(test["y_dep_ge15"], prob)
            aucs[al] = round(auc, 4)
            print(f"  {al} weather-only AUC = {auc:.4f}")

        fig, ax = plt.subplots(figsize=(8, 5))
        als = list(aucs.keys())
        vals = [aucs[a] for a in als]
        bars = ax.bar(als, vals, color=[AIRLINE_COLORS[a] for a in als], alpha=0.85)
        ax.set_ylabel("ROC AUC")
        ax.set_title("Weather-Only Model AUC by Airline")
        ax.set_ylim(0.5, max(vals) + 0.05)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                    f"{v:.3f}", ha="center", fontsize=11)
        fig.tight_layout()
        save_fig(fig, "fig_08c_weather_only_auc_by_airline")
        summary["weather_only_auc"] = aucs

    # ── 8d: Same-airport comparison during severe weather ─────────────
    print("[08] fig_08d -- Same-airport severe weather comparison ...")
    severe_mask = (
        (combined["origin_daily_precip_sum_mm"] > 10) |
        (combined["origin_dep_windgusts_kmh"] > 50)
    )
    severe = combined[severe_mask]

    # find airports served by >= 3 airlines
    apt_airline_ct = severe.groupby("Origin")["airline"].nunique()
    shared_apts = apt_airline_ct[apt_airline_ct >= 3].index.tolist()[:15]

    if shared_apts:
        sub = severe[severe["Origin"].isin(shared_apts)]
        ct = sub.groupby(["Origin", "airline"])["y_dep_ge15"].mean().unstack()
        ct = ct.reindex(columns=AIRLINES).astype(float)
        ct = ct.loc[ct.mean(axis=1).sort_values(ascending=False).index]

        fig, ax = plt.subplots(figsize=(14, 7))
        from utils import safe_imshow
        im = safe_imshow(ax, ct.values, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(AIRLINES)))
        ax.set_xticklabels(AIRLINES, fontsize=11)
        ax.set_yticks(range(len(ct)))
        ax.set_yticklabels(ct.index, fontsize=9)
        ax.set_xlabel("Airline")
        ax.set_ylabel("Airport")
        arr = ct.fillna(0).values
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                v = float(ct.values[i, j]) if pd.notna(ct.values[i, j]) else np.nan
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if v > 0.4 else "black")
        fig.colorbar(im, ax=ax, label="Delay Rate (>=15 min)", shrink=0.7)
        ax.set_title("Shared-Airport Delay Rate During Severe Weather")
        fig.tight_layout()
        save_fig(fig, "fig_08d_airline_airport_weather_heatmap")

    # ── summary ───────────────────────────────────────────────────────
    with open(FIGURES / "08_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("[08] Done.")
    return summary


if __name__ == "__main__":
    sample = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run(sample)
