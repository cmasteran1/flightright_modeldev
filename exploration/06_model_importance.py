"""06 -- Weather-only predictive model + feature importance."""

import json, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

from utils import (load_weather_df, NUMERIC_WEATHER, CAT_WEATHER, ALL_WEATHER,
                   set_plot_style, save_fig, short_name, FIGURES)

try:
    from catboost import CatBoostClassifier, Pool
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("[06] WARNING: catboost not installed, skipping model training")


def run(airline="WN", sample_n=None):
    if not HAS_CATBOOST:
        print("[06] catboost not available -- skipping.")
        return {}

    set_plot_style()
    print("[06] Loading data ...")
    df = load_weather_df(airline, sample_n)
    summary = {}

    # ── prepare features ──────────────────────────────────────────────
    feature_cols = NUMERIC_WEATHER + CAT_WEATHER
    target = "y_dep_ge15"

    # drop rows with NaN in features or target
    sub = df[feature_cols + [target, "FlightDate"]].dropna()
    for cc in CAT_WEATHER:
        sub[cc] = sub[cc].astype(int).astype(str)

    # date-based split: last 20% of dates as test
    dates_sorted = sorted(sub["FlightDate"].unique())
    split_idx = int(len(dates_sorted) * 0.8)
    train_dates = set(dates_sorted[:split_idx])
    test_dates  = set(dates_sorted[split_idx:])

    train = sub[sub["FlightDate"].isin(train_dates)]
    test  = sub[sub["FlightDate"].isin(test_dates)]
    print(f"[06] Train: {len(train):,} rows, Test: {len(test):,} rows")

    X_train, y_train = train[feature_cols], train[target]
    X_test,  y_test  = test[feature_cols],  test[target]

    cat_indices = [feature_cols.index(c) for c in CAT_WEATHER]

    # ── train weather-only model ──────────────────────────────────────
    print("[06] Training weather-only CatBoost ...")
    model = CatBoostClassifier(
        iterations=1000, depth=6, learning_rate=0.05,
        loss_function="Logloss", eval_metric="AUC",
        cat_features=cat_indices,
        verbose=200, random_seed=42,
        early_stopping_rounds=50,
        allow_writing_files=False,
    )
    model.fit(X_train, y_train,
              eval_set=Pool(X_test, y_test, cat_features=cat_indices),
              verbose=200)

    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print(f"[06] Weather-only AUC = {auc:.4f}")
    summary["weather_only_auc"] = round(auc, 4)

    # ── 6a: ROC curve ────────────────────────────────────────────────
    print("[06] fig_06a -- ROC curve ...")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, color="#D32F2F", linewidth=2, label=f"Weather-only (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random (AUC=0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Weather-Only Model ROC ({airline})")
    ax.legend(fontsize=11)
    ax.set_aspect("equal")
    fig.tight_layout()
    save_fig(fig, "fig_06a_weather_only_roc")

    # ── 6b: Feature importance bars ───────────────────────────────────
    print("[06] fig_06b -- Feature importance ...")
    importances = model.get_feature_importance()
    feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(feat_imp)), feat_imp.values, color="#FF7043", alpha=0.85)
    ax.set_yticks(range(len(feat_imp)))
    ax.set_yticklabels([short_name(c) for c in feat_imp.index], fontsize=9)
    ax.set_xlabel("CatBoost Feature Importance (PredictionValuesChange)")
    ax.set_title(f"Weather Feature Importance ({airline})")
    fig.tight_layout()
    save_fig(fig, "fig_06b_feature_importance_bars")

    summary["feature_importance"] = {short_name(c): round(v, 2)
                                     for c, v in feat_imp.items()}

    # ── 6c: SHAP summary (native CatBoost) ───────────────────────────
    print("[06] fig_06c -- SHAP values ...")
    shap_sample = X_test.sample(n=min(10_000, len(X_test)), random_state=42)
    shap_pool = Pool(shap_sample, cat_features=cat_indices)
    shap_vals = model.get_feature_importance(shap_pool, type="ShapValues")
    shap_vals = shap_vals[:, :-1]  # drop bias column

    # beeswarm approximation: mean |SHAP| per feature
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    shap_series = pd.Series(mean_abs_shap, index=feature_cols).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(shap_series)), shap_series.values, color="#7E57C2", alpha=0.85)
    ax.set_yticks(range(len(shap_series)))
    ax.set_yticklabels([short_name(c) for c in shap_series.index], fontsize=9)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"SHAP Feature Importance ({airline})")
    fig.tight_layout()
    save_fig(fig, "fig_06c_shap_summary")

    summary["shap_importance"] = {short_name(c): round(v, 5)
                                  for c, v in shap_series.items()}

    # ── 6d: SHAP direction scatter for top-6 features ─────────────────
    print("[06] fig_06d -- SHAP direction scatter ...")
    top6_idx = shap_series.sort_values(ascending=False).head(6).index.tolist()
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    for i, col in enumerate(top6_idx):
        ax = axes[i]
        cidx = feature_cols.index(col)
        vals = shap_sample[col].values
        if col in CAT_WEATHER:
            vals = vals.astype(float)
        shap_col = shap_vals[:, cidx]
        # subsample for scatter
        n_plot = min(3000, len(vals))
        idx = np.random.choice(len(vals), n_plot, replace=False)
        sc = ax.scatter(vals[idx], shap_col[idx], c=vals[idx], cmap="coolwarm",
                        alpha=0.3, s=5, edgecolors="none")
        ax.set_xlabel(short_name(col), fontsize=9)
        ax.set_ylabel("SHAP value", fontsize=9)
        ax.set_title(short_name(col), fontsize=10)
        ax.axhline(0, color="gray", lw=0.5)
    fig.suptitle(f"SHAP Direction: Top-6 Weather Features ({airline})", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "fig_06d_shap_direction_scatter")

    # ── 6e: Partial dependence plots ──────────────────────────────────
    print("[06] fig_06e -- Partial dependence ...")
    ncols = 4
    nrows = int(np.ceil(len(NUMERIC_WEATHER) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.5 * nrows))
    axes = axes.flatten()

    pd_sample = X_test.sample(n=min(5000, len(X_test)), random_state=42).copy()
    for i, col in enumerate(NUMERIC_WEATHER):
        ax = axes[i]
        vals_range = np.linspace(pd_sample[col].quantile(0.02),
                                 pd_sample[col].quantile(0.98), 30)
        preds = []
        for v in vals_range:
            tmp = pd_sample.copy()
            tmp[col] = v
            pool = Pool(tmp, cat_features=cat_indices)
            preds.append(model.predict_proba(pool)[:, 1].mean())
        ax.plot(vals_range, preds, color="#D32F2F", linewidth=2)
        ax.set_xlabel(short_name(col), fontsize=8)
        ax.set_ylabel("P(delay >=15)", fontsize=8)
        ax.set_title(short_name(col), fontsize=9)
        ax.tick_params(labelsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"Partial Dependence: Weather Features ({airline})", fontsize=13, y=1.01)
    fig.tight_layout()
    save_fig(fig, "fig_06e_partial_dependence")

    # ── summary ───────────────────────────────────────────────────────
    with open(FIGURES / "06_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("[06] Done.")
    return summary


if __name__ == "__main__":
    airline = sys.argv[1] if len(sys.argv) > 1 else "WN"
    sample  = int(sys.argv[2]) if len(sys.argv) > 2 else None
    run(airline, sample)
