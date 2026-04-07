"""12 -- Drop candidates: features with low importance or high redundancy."""

import json, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import (load_full_df, train_catboost_full, set_plot_style, save_fig,
                   short_name, FIGURES,
                   DEP_FEATURE_ORDER, DEP_CAT_FEATURES, DEP_NUM_FEATURES)
from catboost import Pool


def run(airline="WN", sample_n=None):
    set_plot_style()
    print("[12] Loading dep data ...")
    df = load_full_df(airline, "dep", sample_n)
    summary = {}

    # -- Train baseline model --
    print("[12] Training baseline full model ...")
    model, X_test, y_test, y_prob, baseline_auc, cat_idx = train_catboost_full(
        df, DEP_FEATURE_ORDER, DEP_CAT_FEATURES, "y_dep_ge15",
        iterations=1000, depth=6, verbose=200)
    print(f"[12] Baseline AUC = {baseline_auc:.4f}")
    summary["baseline_auc"] = round(baseline_auc, 4)

    # -- Get feature importance --
    imp = model.get_feature_importance()
    imp_series = pd.Series(imp, index=DEP_FEATURE_ORDER).sort_values(ascending=True)

    # -- SHAP for comparison --
    shap_sample = X_test.sample(n=min(10_000, len(X_test)), random_state=42)
    shap_pool = Pool(shap_sample, cat_features=cat_idx)
    shap_vals = model.get_feature_importance(shap_pool, type="ShapValues")[:, :-1]
    mean_shap = pd.Series(np.abs(shap_vals).mean(axis=0), index=DEP_FEATURE_ORDER)

    # -- Null rate analysis --
    null_rates = df[DEP_FEATURE_ORDER].isnull().mean()

    # -- Correlation-based redundancy (from Spearman) --
    num_in_model = [c for c in DEP_NUM_FEATURES if c in df.columns]
    corr = df[num_in_model].corr(method="spearman")

    # -- Build drop recommendation table --
    print("[12] Building drop recommendation table ...")
    drop_candidates = []
    for feat in DEP_FEATURE_ORDER:
        imp_val = imp_series.get(feat, 0)
        shap_val = mean_shap.get(feat, 0)
        null_rate = null_rates.get(feat, 0)

        # find most correlated partner
        best_partner = None
        best_r = 0
        if feat in corr.index:
            row = corr.loc[feat].drop(feat, errors="ignore")
            if len(row) > 0:
                best_idx = row.abs().idxmax()
                best_r = row[best_idx]
                best_partner = best_idx

        reasons = []
        if imp_val < 0.5:
            reasons.append("low_importance")
        if abs(best_r) > 0.90 and best_partner:
            partner_imp = imp_series.get(best_partner, 0)
            if imp_val < partner_imp:
                reasons.append(f"redundant_with_{short_name(best_partner)}(r={best_r:.2f})")
        if null_rate > 0.30:
            reasons.append(f"high_null({null_rate:.1%})")

        drop_candidates.append({
            "feature": feat,
            "importance": round(float(imp_val), 2),
            "shap": round(float(shap_val), 5),
            "null_rate": round(float(null_rate), 3),
            "best_corr_partner": best_partner,
            "best_corr_r": round(float(best_r), 3) if best_partner else None,
            "reasons": reasons,
            "recommend_drop": len(reasons) > 0,
        })

    drop_df = pd.DataFrame(drop_candidates).sort_values("shap", ascending=True)
    recommended = drop_df[drop_df["recommend_drop"]]
    summary["n_recommended_drops"] = len(recommended)
    summary["recommended_drops"] = recommended[["feature", "importance", "shap",
                                                 "reasons"]].to_dict("records")
    summary["total_features"] = len(DEP_FEATURE_ORDER)

    # -- Plot: Drop candidates --
    print("[12] fig_12a -- Drop candidate ranking ...")
    plot_df = drop_df.head(30)  # bottom 30 by SHAP
    fig, ax = plt.subplots(figsize=(12, 10))
    colors = ["#D32F2F" if r else "#4CAF50" for r in plot_df["recommend_drop"]]
    ax.barh(range(len(plot_df)), plot_df["shap"].values, color=colors, alpha=0.8)
    ax.set_yticks(range(len(plot_df)))
    labels = []
    for _, row in plot_df.iterrows():
        lbl = short_name(row["feature"])
        if row["reasons"]:
            lbl += f"  [{', '.join(row['reasons'][:2])}]"
        labels.append(lbl)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Mean |SHAP| (lower = less important)")
    ax.set_title(f"Bottom-30 Features: Drop Candidates ({airline})\n"
                 f"Red = recommended drop, Green = keep")
    fig.tight_layout()
    save_fig(fig, "fig_12a_drop_candidates")

    # -- Plot: Importance distribution --
    print("[12] fig_12b -- Importance distribution ...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.hist(imp_series.values, bins=30, color="#1976D2", alpha=0.7, edgecolor="white")
    ax1.axvline(0.5, color="red", ls="--", label="Drop threshold (0.5)")
    ax1.set_xlabel("CatBoost Feature Importance")
    ax1.set_ylabel("Count")
    ax1.set_title("Feature Importance Distribution")
    ax1.legend()

    ax2.hist(mean_shap.values, bins=30, color="#FF7043", alpha=0.7, edgecolor="white")
    ax2.set_xlabel("Mean |SHAP|")
    ax2.set_ylabel("Count")
    ax2.set_title("SHAP Importance Distribution")
    fig.suptitle(f"Dep Model Feature Importance Distributions ({airline})")
    fig.tight_layout()
    save_fig(fig, "fig_12b_importance_distribution")

    with open(FIGURES / "12_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("[12] Done.")
    return summary


if __name__ == "__main__":
    airline = sys.argv[1] if len(sys.argv) > 1 else "WN"
    sample = int(sys.argv[2]) if len(sys.argv) > 2 else None
    run(airline, sample)
