"""
Test whether certain numeric features perform better as categorical in CatBoost.

Candidates:
  - dep_dow (0-6)
  - sched_dep_hour (0-23)
  - is_peak_hour (0/1)

Reference baseline: DepTimeBlk (already categorical in v10).

Methodology:
  - 50k random sample of WN v10 smoke features (June 2025).
  - Quick CatBoost (200 iter, depth 6) trained at y_dep_ge15 and y_dep_ge60.
  - Four variants per label:
      * all-numeric (baseline)
      * dep_dow categorical
      * sched_dep_hour categorical
      * both categorical
  - Compare held-out log-loss and AUC.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
from catboost import CatBoostClassifier, Pool

DATA_ROOT = Path("/Users/connermasteran/software/mpqc/flightrightdata")
REPO_ROOT = Path("/Users/connermasteran/software/mpqc/flightright_modeldev")
FEATS_PATH = DATA_ROOT / "data/processed/features_dep_WN_v10_smoke_unbalanced.parquet"

print("Loading features...")
df = pd.read_parquet(FEATS_PATH)
print(f"  Rows: {len(df)}")

# Build labels
df["y_dep_ge15"] = (df["DepDelayMinutes"] >= 15).astype(int)
df["y_dep_ge60"] = (df["DepDelayMinutes"] >= 60).astype(int)

# Feature set: a compact subset representing the full v10 feature groups plus the
# three candidate encoding features.
FEATURES = [
    # Schedule / route
    "CRSDepTime", "CRSElapsedTime", "Distance", "dep_dow", "sched_dep_hour", "is_peak_hour",
    # Weather (hourly + daily, a few representative cols)
    "origin_dep_windgusts_kmh", "origin_dep_precip_mm", "origin_dep_visibility_m",
    "origin_daily_windgusts_max_kmh",
    # Delay baselines
    "carrier_depdelay_median_last1", "carrier_depdelay_mean_last7",
    "flightnum_od_depdelay_median_last14", "origin_depdelay_mean_last14",
    # Late-aircraft rates
    "origin_lateaircraft_rate_last1", "carrier_lateaircraft_rate_last1",
    # Existing categoricals
    "Origin", "Dest", "DepTimeBlk", "origin_dep_hour_weathercode",
]

CAT_BASE = ["Origin", "Dest", "DepTimeBlk", "origin_dep_hour_weathercode"]

# Keep only rows with all features present
sub = df[FEATURES + ["y_dep_ge15", "y_dep_ge60"]].dropna().reset_index(drop=True)
print(f"  Rows after dropna: {len(sub)}")

if len(sub) > 50000:
    sub = sub.sample(50000, random_state=42).reset_index(drop=True)
    print(f"  Sampled to: {len(sub)}")

# String-cast the candidate features once (CatBoost requires strings for cat_features)
for cand in ["dep_dow", "sched_dep_hour", "is_peak_hour"]:
    sub[cand + "_str"] = sub[cand].astype(int).astype(str)

# Existing categoricals: ensure string
for c in CAT_BASE:
    sub[c] = sub[c].astype(str)


def prepare_x(variant: str) -> tuple[pd.DataFrame, list[str]]:
    """Return (X, categorical_feature_names) for a given variant."""
    feats = list(FEATURES)
    cat_feats = list(CAT_BASE)

    if variant == "baseline":
        pass
    elif variant == "dow_cat":
        feats[feats.index("dep_dow")] = "dep_dow_str"
        cat_feats.append("dep_dow_str")
    elif variant == "hour_cat":
        feats[feats.index("sched_dep_hour")] = "sched_dep_hour_str"
        cat_feats.append("sched_dep_hour_str")
    elif variant == "peak_cat":
        feats[feats.index("is_peak_hour")] = "is_peak_hour_str"
        cat_feats.append("is_peak_hour_str")
    elif variant == "all_three_cat":
        feats[feats.index("dep_dow")] = "dep_dow_str"
        feats[feats.index("sched_dep_hour")] = "sched_dep_hour_str"
        feats[feats.index("is_peak_hour")] = "is_peak_hour_str"
        cat_feats.extend(["dep_dow_str", "sched_dep_hour_str", "is_peak_hour_str"])
    else:
        raise ValueError(variant)

    return sub[feats], cat_feats


def train_eval(variant: str, label: str, seed: int = 0) -> dict:
    X, cat_feats = prepare_x(variant)
    y = sub[label].values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)

    train_pool = Pool(X_tr, y_tr, cat_features=cat_feats)
    test_pool = Pool(X_te, y_te, cat_features=cat_feats)

    clf = CatBoostClassifier(
        iterations=200,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        loss_function="Logloss",
        eval_metric="Logloss",
        od_type="Iter", od_wait=30,
        verbose=False,
        random_seed=seed,
    )
    clf.fit(train_pool, eval_set=test_pool, use_best_model=True)

    probs = clf.predict_proba(X_te)[:, 1]
    ll = log_loss(y_te, probs)
    auc = roc_auc_score(y_te, probs)

    return {"variant": variant, "label": label, "log_loss": ll, "auc": auc, "n_test": len(y_te),
            "best_iter": clf.get_best_iteration()}


variants = ["baseline", "dow_cat", "hour_cat", "peak_cat", "all_three_cat"]
labels = ["y_dep_ge15", "y_dep_ge60"]

print()
print(f"Training {len(variants) * len(labels)} models (CatBoost 200 iter, depth 6)...")
print()

results = []
for label in labels:
    for variant in variants:
        r = train_eval(variant, label)
        results.append(r)
        print(f"  {label}  {variant:15s}  logloss={r['log_loss']:.5f}  auc={r['auc']:.5f}  "
              f"best_iter={r['best_iter']}")
    print()

out_dir = REPO_ROOT / "exploration/reports"
out_dir.mkdir(parents=True, exist_ok=True)
pd.DataFrame(results).to_csv(out_dir / "2026-04-16-encoding-results.csv", index=False)

# Also: grouped delay rate plots by dep_dow and sched_dep_hour to illustrate non-monotonicity
print()
print("Delay rate by dep_dow:")
for dow in range(7):
    mask = sub["dep_dow"] == dow
    rate_15 = sub.loc[mask, "y_dep_ge15"].mean()
    rate_60 = sub.loc[mask, "y_dep_ge60"].mean()
    print(f"  dow={dow}  n={mask.sum():5d}  rate_ge15={rate_15:.4f}  rate_ge60={rate_60:.4f}")

print()
print("Delay rate by sched_dep_hour:")
for hr in sorted(sub["sched_dep_hour"].unique()):
    mask = sub["sched_dep_hour"] == hr
    if mask.sum() < 50:
        continue
    rate_15 = sub.loc[mask, "y_dep_ge15"].mean()
    rate_60 = sub.loc[mask, "y_dep_ge60"].mean()
    print(f"  hr={hr:2.0f}  n={mask.sum():5d}  rate_ge15={rate_15:.4f}  rate_ge60={rate_60:.4f}")

print()
print("Done.")
