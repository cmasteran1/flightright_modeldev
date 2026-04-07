"""13 -- Interaction effects: CatBoost native + manual cross-feature analysis."""

import json, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import (load_full_df, train_catboost_full, set_plot_style, save_fig,
                   short_name, safe_imshow, FIGURES,
                   DEP_FEATURE_ORDER, DEP_CAT_FEATURES, DEP_NUM_FEATURES)
from catboost import Pool


def run(airline="WN", sample_n=None):
    set_plot_style()
    print("[13] Loading dep data ...")
    df = load_full_df(airline, "dep", sample_n)
    summary = {}

    # -- Train model --
    print("[13] Training full model ...")
    model, X_test, y_test, y_prob, auc, cat_idx = train_catboost_full(
        df, DEP_FEATURE_ORDER, DEP_CAT_FEATURES, "y_dep_ge15",
        iterations=1000, depth=6, verbose=200)

    # -- CatBoost native interaction importance --
    print("[13] Computing interaction importance ...")
    interaction_imp = model.get_feature_importance(type="Interaction")

    # Convert to readable format: list of (feat_a, feat_b, strength)
    interactions = []
    for row in interaction_imp:
        idx_a, idx_b = int(row[0]), int(row[1])
        strength = float(row[2])
        interactions.append({
            "feat_a": DEP_FEATURE_ORDER[idx_a],
            "feat_b": DEP_FEATURE_ORDER[idx_b],
            "strength": round(strength, 4),
        })
    interactions.sort(key=lambda x: -x["strength"])
    top20 = interactions[:20]
    summary["top20_interactions"] = top20

    # -- Plot: Top-20 interactions --
    print("[13] fig_13a -- Top-20 feature interactions ...")
    fig, ax = plt.subplots(figsize=(12, 8))
    labels = [f"{short_name(x['feat_a'])} x {short_name(x['feat_b'])}" for x in top20]
    vals = [x["strength"] for x in top20]
    # color by type: weather-operational=red, operational-operational=blue, weather-weather=green
    def _is_weather(f):
        return any(f.startswith(p) for p in ["origin_temp", "origin_daily", "origin_dep_"])
    colors = []
    for x in top20:
        wa, wb = _is_weather(x["feat_a"]), _is_weather(x["feat_b"])
        if wa and wb:
            colors.append("#4CAF50")  # weather x weather
        elif wa or wb:
            colors.append("#D32F2F")  # weather x operational
        else:
            colors.append("#1976D2")  # operational x operational
    ax.barh(range(len(top20)), vals, color=colors, alpha=0.85)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("CatBoost Interaction Strength")
    ax.set_title(f"Top-20 Feature Interactions ({airline})")
    # legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#D32F2F", label="Weather x Operational"),
        Patch(color="#1976D2", label="Operational x Operational"),
        Patch(color="#4CAF50", label="Weather x Weather"),
    ], fontsize=9)
    fig.tight_layout()
    save_fig(fig, "fig_13a_top20_interactions")

    # -- 2D heatmaps for top-5 interactions --
    print("[13] fig_13b -- Top-5 interaction heatmaps ...")
    target_col = "y_dep_ge15"
    # merge target back for heatmap computation
    df_plot = df[[target_col] + DEP_NUM_FEATURES].dropna(subset=[target_col])

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()
    for idx, inter in enumerate(top20[:5]):
        ax = axes[idx]
        fa, fb = inter["feat_a"], inter["feat_b"]
        if fa not in df_plot.columns or fb not in df_plot.columns:
            ax.set_visible(False)
            continue
        # Skip categorical features for heatmap
        if fa in DEP_CAT_FEATURES or fb in DEP_CAT_FEATURES:
            ax.set_title(f"{short_name(fa)} x {short_name(fb)}\n(categorical, skipped)")
            continue
        try:
            df_plot["_ba"] = pd.qcut(df_plot[fa], 5, labels=False, duplicates="drop")
            df_plot["_bb"] = pd.qcut(df_plot[fb], 5, labels=False, duplicates="drop")
        except ValueError:
            ax.set_visible(False)
            continue
        ct = df_plot.groupby(["_bb", "_ba"])[target_col].mean().unstack()
        safe_imshow(ax, ct.values, cmap="YlOrRd", aspect="auto", origin="lower")
        ax.set_xlabel(short_name(fa), fontsize=9)
        ax.set_ylabel(short_name(fb), fontsize=9)
        ax.set_title(f"{short_name(fa)} x {short_name(fb)}\nstrength={inter['strength']:.1f}", fontsize=9)
        # annotate
        arr = ct.fillna(0).values.astype(float)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                v = arr[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if v > 0.35 else "black")
    for j in range(5, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"Top-5 Feature Interaction Heatmaps ({airline})", fontsize=13, y=1.01)
    fig.tight_layout()
    save_fig(fig, "fig_13b_top5_interaction_heatmaps")

    # -- Cross-category analysis: weather x time, congestion x delay history --
    print("[13] fig_13c -- Cross-category interactions ...")
    cross_pairs = [
        ("origin_dep_windgusts_kmh", "carrier_depdelay_mean_last1", "Wind x CarrierDelay"),
        ("origin_daily_precip_sum_mm", "origin_congestion_3h_total", "Precip x Congestion"),
        ("origin_dep_temp_K", "sched_dep_hour", "Temp x DepartureHour"),
        ("origin_dep_cape_jkg", "origin_depdelay_mean_last1", "CAPE x OriginDelay"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()
    for idx, (fa, fb, title) in enumerate(cross_pairs):
        ax = axes[idx]
        if fa not in df_plot.columns or fb not in df_plot.columns:
            ax.set_visible(False)
            continue
        try:
            df_plot["_xa"] = pd.qcut(df_plot[fa], 5, labels=False, duplicates="drop")
            df_plot["_xb"] = pd.qcut(df_plot[fb], 5, labels=False, duplicates="drop")
        except ValueError:
            ax.set_visible(False)
            continue
        ct = df_plot.groupby(["_xb", "_xa"])[target_col].mean().unstack()
        safe_imshow(ax, ct.values, cmap="YlOrRd", aspect="auto", origin="lower")
        ax.set_xlabel(short_name(fa), fontsize=9)
        ax.set_ylabel(short_name(fb), fontsize=9)
        ax.set_title(title, fontsize=10)
    fig.suptitle(f"Cross-Category Interaction Heatmaps ({airline})", fontsize=13, y=1.01)
    fig.tight_layout()
    save_fig(fig, "fig_13c_cross_category_interactions")

    with open(FIGURES / "13_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("[13] Done.")
    return summary


if __name__ == "__main__":
    airline = sys.argv[1] if len(sys.argv) > 1 else "WN"
    sample = int(sys.argv[2]) if len(sys.argv) > 2 else None
    run(airline, sample)
