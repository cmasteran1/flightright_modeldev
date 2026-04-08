#!/usr/bin/env python3
# src/training/train_cancellation_catboost.py
#
# Binary cancellation probability model via calibrated CatBoost.
#
# Pattern B (production-safe):
#   - Save a single deployable JOBLIB that is a plain dict:
#       {
#         "artifact_type": "cancellation_bundle_v1",
#         "categorical_features": [...],
#         "numeric_features": [...],
#         "feature_order": [...],
#         "calibrator": CalibratedClassifierCV,
#         "optimal_threshold": float,
#         "registry": {"model_path": "...cbm", "cal_path": "...joblib"},
#         "versions": {...},
#         "created_utc": "...",
#         "config_used": {...},
#       }
#
# Run:
#   python src/training/train_cancellation_catboost.py data/cancel_train_v9.json
#
import os, sys, json, time, signal, faulthandler
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone as dt_timezone

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, log_loss, accuracy_score,
    average_precision_score, brier_score_loss,
    precision_recall_curve,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier, Pool

import mlflow
from _mlflow_setup import init_mlflow, loggable_params, git_commit


REPO_ROOT = Path.cwd()
DATA_ROOT = (REPO_ROOT.parent / "flightrightdata").resolve()

def _as_data_path(p: Path) -> Path:
    return p if p.is_absolute() else (DATA_ROOT / p).resolve()

FAST = os.getenv("FAST_TRAIN", "0") == "1"
NO_PLOTS = os.getenv("NO_PLOTS", "0") == "1"
TRACE_EVERY_SEC = int(os.getenv("TRACE_EVERY_SEC", "0"))

_t0 = time.time()
_last_step_t = time.time()

def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def log(msg, *, step=False):
    global _last_step_t
    now = time.time()
    total = now - _t0
    delta = now - _last_step_t
    if step:
        _last_step_t = now
    print(f"[{_now()} +{total:7.2f}s D{delta:6.2f}s] {msg}", flush=True)

def enable_periodic_traces():
    if TRACE_EVERY_SEC > 0:
        faulthandler.dump_traceback_later(TRACE_EVERY_SEC, repeat=True)
        log(f"[trace] periodic faulthandler enabled every {TRACE_EVERY_SEC}s")

def enable_usr1_trace():
    try:
        faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
        log("[trace] SIGUSR1 handler registered (kill -USR1 <pid> to dump stacks)")
    except Exception as e:
        log(f"[trace] failed to register SIGUSR1 handler: {e}")


# ------------------------------ utils ------------------------------

def _load_json(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _safe_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    try:
        import sklearn
        versions["sklearn"] = getattr(sklearn, "__version__", "unknown")
    except Exception:
        versions["sklearn"] = "unknown"
    try:
        import catboost
        versions["catboost"] = getattr(catboost, "__version__", "unknown")
    except Exception:
        versions["catboost"] = "unknown"
    try:
        versions["numpy"] = getattr(np, "__version__", "unknown")
    except Exception:
        versions["numpy"] = "unknown"
    try:
        versions["pandas"] = getattr(pd, "__version__", "unknown")
    except Exception:
        versions["pandas"] = "unknown"
    try:
        versions["joblib"] = getattr(joblib, "__version__", "unknown")
    except Exception:
        versions["joblib"] = "unknown"
    return versions


def best_threshold_youden(y_true, y_proba):
    fpr, tpr, thr = roc_curve(y_true, y_proba)
    j = tpr - fpr
    i = int(np.argmax(j))
    return float(thr[i]), float(tpr[i]), float(fpr[i])


def as_str(series: pd.Series) -> pd.Series:
    s = series.astype("object")
    s = s.where(~pd.isna(s), "Unknown")
    s = s.map(lambda v: "Unknown" if str(v).strip().lower() in ("", "nan", "none") else str(v))
    return s.astype("object")


def reliability_table(y_true, p_pred, n_bins=10):
    y_true = pd.Series(y_true).reset_index(drop=True)
    p_pred = pd.Series(p_pred).reset_index(drop=True).clip(0, 1)
    bins = pd.interval_range(start=0.0, end=1.0, periods=n_bins, closed="right")
    cuts = pd.cut(p_pred, bins, include_lowest=True)
    out = (
        pd.DataFrame({"y": y_true, "p": p_pred, "bin": cuts})
        .groupby("bin", observed=True)
        .agg(count=("y", "size"), mean_pred=("p", "mean"), empirical_pos_rate=("y", "mean"))
        .reset_index()
    )
    out["bin_left"]  = out["bin"].apply(lambda iv: iv.left if pd.notna(iv) else np.nan)
    out["bin_right"] = out["bin"].apply(lambda iv: iv.right if pd.notna(iv) else np.nan)
    out["bin_mid"]   = (out["bin_left"].astype(float) + out["bin_right"].astype(float)) / 2.0
    return out


def plot_reliability(tag, y_true, p_pred, outdir: Path, n_bins=10):
    tab = reliability_table(y_true, p_pred, n_bins=n_bins)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="perfect")
    ax.plot(tab["mean_pred"], tab["empirical_pos_rate"], marker="o", linewidth=1.5, label="model")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical positive rate")
    ax.set_title(f"Reliability curve ({tag})")
    ax.legend(loc="best")
    fig.tight_layout()
    png_path = outdir / f"calibration_{tag}.png"
    csv_path = outdir / f"calibration_{tag}_table.csv"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    tab.drop(columns=["bin"], errors="ignore").to_csv(csv_path, index=False)
    log(f"[SAVE] reliability plot -> {png_path}")
    log(f"[SAVE] reliability table -> {csv_path}")


def build_X(df, feature_order, cat_feats):
    X = df[feature_order].copy()
    for c in cat_feats:
        if c in X.columns:
            X[c] = as_str(X[c])
    for c in X.columns:
        if c not in cat_feats:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


# ------------------------------ config resolution ------------------------------

def _resolve_run_args(argv) -> Tuple[Path, Path, Path]:
    if len(argv) != 2:
        raise SystemExit(
            "Usage: python train_cancellation_catboost.py CONFIG.json"
        )
    cfg_path = Path(argv[1])
    cfg = _load_json(cfg_path)
    in_path = cfg.get("input_features_path")
    outdir = cfg.get("outdir")
    if not in_path or not outdir:
        raise SystemExit("Config must have 'input_features_path' and 'outdir'.")
    return Path(in_path), Path(outdir), cfg_path


# ------------------------------ main ------------------------------

def main():
    enable_periodic_traces()
    enable_usr1_trace()

    INPUT, OUTDIR, CFG_PATH = _resolve_run_args(sys.argv)
    INPUT = _as_data_path(INPUT)
    OUTDIR = _as_data_path(OUTDIR)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    cfg = _load_json(CFG_PATH)

    # --- MLflow tracking setup -------------------------------------------------
    init_mlflow("cancellation")
    mlflow.start_run(run_name=CFG_PATH.stem)
    mlflow.set_tag("model_type", "cancellation")
    mlflow.set_tag("config_path", str(CFG_PATH))
    mlflow.set_tag("script", "train_cancellation_catboost.py")
    mlflow.set_tag("fast_mode", "1" if FAST else "0")
    _commit = git_commit()
    if _commit:
        mlflow.set_tag("git_commit", _commit)
    mlflow.log_params(loggable_params(cfg))
    # --------------------------------------------------------------------------

    if FAST:
        cfg["iterations"] = min(int(cfg.get("iterations", 4000)), 1200)
        cfg["depth"] = min(int(cfg.get("depth", 8)), 6)
        cfg["od_wait"] = min(int(cfg.get("od_wait", 200)), 120)
        cfg["learning_rate"] = max(float(cfg.get("learning_rate", 0.03)), 0.05)
        log("[FAST] using lighter CatBoost settings")

    SEED = int(cfg.get("random_seed", 42))
    TARGET_COL = str(cfg.get("target_col", "Cancelled"))

    log(f"Loading data: {INPUT}")
    df_all = pd.read_parquet(INPUT)
    if cfg.get("max_rows"):
        df_all = df_all.sample(n=int(cfg["max_rows"]), random_state=SEED).reset_index(drop=True)
    log(f"Rows={len(df_all)} Cols={len(df_all.columns)}", step=True)

    # Ensure target column exists and is binary
    if TARGET_COL not in df_all.columns:
        raise SystemExit(f"Missing target column: {TARGET_COL}")
    df_all[TARGET_COL] = pd.to_numeric(df_all[TARGET_COL], errors="coerce").fillna(0).astype(int)
    pos_rate = df_all[TARGET_COL].mean()
    log(f"Target '{TARGET_COL}': {df_all[TARGET_COL].sum():,} positives out of {len(df_all):,} ({pos_rate:.4f})")

    # Resolve features
    cat_feats = [c for c in cfg.get("categorical_features", []) if c in df_all.columns]
    num_feats = [c for c in cfg.get("numeric_features", []) if c in df_all.columns]
    feature_order = cat_feats + num_feats
    log(f"Features: {len(cat_feats)} cat + {len(num_feats)} num = {len(feature_order)} total")

    # ---- Split: eval set (unbalanced) ----
    eval_features_path = cfg.get("eval_features_path")
    if eval_features_path:
        eval_path = _as_data_path(Path(eval_features_path))
        df_eval_full = pd.read_parquet(eval_path)
        df_eval_full[TARGET_COL] = pd.to_numeric(df_eval_full[TARGET_COL], errors="coerce").fillna(0).astype(int)
        # Split 50/50 for calibration and eval
        cal_size = float(cfg.get("unbalanced_cal_size", 0.50))
        n_cal = int(len(df_eval_full) * cal_size)
        df_cal_unbal = df_eval_full.iloc[:n_cal].reset_index(drop=True)
        df_eval_unbal = df_eval_full.iloc[n_cal:].reset_index(drop=True)
        log(f"[unbalanced] cal={len(df_cal_unbal):,} eval={len(df_eval_unbal):,}")
        unbal_source = str(eval_path)
    else:
        # Fall back: time-based split using last N days
        eval_last_days = int(cfg.get("eval_last_days", 90))
        df_all["FlightDate"] = pd.to_datetime(df_all["FlightDate"], errors="coerce")
        cutoff = df_all["FlightDate"].max() - pd.Timedelta(days=eval_last_days)
        df_train_period = df_all[df_all["FlightDate"] <= cutoff].copy()
        df_eval_period = df_all[df_all["FlightDate"] > cutoff].copy()
        cal_size = 0.50
        n_cal = int(len(df_eval_period) * cal_size)
        df_cal_unbal = df_eval_period.iloc[:n_cal].reset_index(drop=True)
        df_eval_unbal = df_eval_period.iloc[n_cal:].reset_index(drop=True)
        log(f"[time split] train_period={len(df_train_period):,} cal={len(df_cal_unbal):,} eval={len(df_eval_unbal):,}")
        unbal_source = "time-based split"

    # ---- Load balanced training data ----
    balanced_path = cfg.get("balanced_train_path")
    if balanced_path:
        df_train = pd.read_parquet(_as_data_path(Path(balanced_path)))
        df_train[TARGET_COL] = pd.to_numeric(df_train[TARGET_COL], errors="coerce").fillna(0).astype(int)
        log(f"[balanced] loaded {len(df_train):,} rows from {balanced_path} (pos_rate={df_train[TARGET_COL].mean():.3f})")
    else:
        # Balance from the full dataset (undersample negatives)
        pos_frac = float(cfg.get("pos_frac", 0.20))
        df_pos = df_all[df_all[TARGET_COL] == 1]
        df_neg = df_all[df_all[TARGET_COL] == 0]
        n_pos = len(df_pos)
        n_neg_target = int(n_pos * (1 - pos_frac) / pos_frac)
        if n_neg_target < len(df_neg):
            df_neg = df_neg.sample(n=n_neg_target, random_state=SEED)
        df_train = pd.concat([df_pos, df_neg], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
        log(f"[balanced] auto-balanced to {len(df_train):,} rows (pos_rate={df_train[TARGET_COL].mean():.3f})")

    # ---- Build feature matrices ----
    y_train = df_train[TARGET_COL].astype(int).values
    X_train = build_X(df_train, feature_order, cat_feats)

    y_cal = df_cal_unbal[TARGET_COL].astype(int).values
    X_cal = build_X(df_cal_unbal, feature_order, cat_feats)

    y_eval = df_eval_unbal[TARGET_COL].astype(int).values
    X_eval = build_X(df_eval_unbal, feature_order, cat_feats)

    # ---- Train/validation split for early stopping ----
    split_cfg = cfg.get("split", {})
    val_size = float(split_cfg.get("val_size", 0.20))
    X_fit, X_es, y_fit, y_es = train_test_split(
        X_train, y_train, test_size=val_size, random_state=SEED, stratify=y_train
    )
    log(f"[split] fit={len(X_fit):,} earlystop={len(X_es):,}")

    cat_idx = [feature_order.index(c) for c in cat_feats]
    pool_fit = Pool(X_fit, label=y_fit, cat_features=cat_idx)
    pool_es = Pool(X_es, label=y_es, cat_features=cat_idx)

    # ---- CatBoost training ----
    iterations = int(cfg.get("iterations", 4000))
    depth = int(cfg.get("depth", 8))
    lr = float(cfg.get("learning_rate", 0.03))
    l2 = float(cfg.get("l2_leaf_reg", 5.0))
    od_wait = int(cfg.get("od_wait", 200))

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=iterations,
        depth=depth,
        learning_rate=lr,
        l2_leaf_reg=l2,
        od_type="Iter",
        od_wait=od_wait,
        random_seed=SEED,
        verbose=200,
        thread_count=-1,
        auto_class_weights="Balanced",
    )

    log(f"[cb] fit start (iter={iterations} depth={depth} lr={lr} l2={l2} od_wait={od_wait})", step=True)
    model.fit(pool_fit, eval_set=pool_es, use_best_model=True)
    log("[cb] fit done", step=True)

    model_path = OUTDIR / "cancel_model.cbm"
    model.save_model(str(model_path))
    log(f"[SAVE] model -> {model_path}")

    # ---- Calibration ----
    log(f"[cal] isotonic calibration on unbalanced cal set ({len(df_cal_unbal):,} rows)", step=True)
    cal = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
    cal.fit(X_cal, y_cal)
    log("[cal] done", step=True)

    cal_path = OUTDIR / "cancel_calibrator.joblib"
    joblib.dump(cal, cal_path)
    log(f"[SAVE] calibrator -> {cal_path}")

    # ---- Evaluation ----
    p_eval = cal.predict_proba(X_eval)[:, 1].astype(float)

    auc = float(roc_auc_score(y_eval, p_eval)) if len(np.unique(y_eval)) >= 2 else float("nan")
    ap = float(average_precision_score(y_eval, p_eval)) if len(np.unique(y_eval)) >= 2 else float("nan")
    ll = float(log_loss(y_eval, np.clip(p_eval, 1e-6, 1 - 1e-6)))
    brier = float(brier_score_loss(y_eval, p_eval))

    youden_thr, youden_tpr, youden_fpr = best_threshold_youden(y_eval, p_eval)
    log(f"[eval] AUC={auc:.4f} AP={ap:.4f} LogLoss={ll:.4f} Brier={brier:.4f}")
    log(f"[eval] Youden threshold={youden_thr:.3f} TPR={youden_tpr:.3f} FPR={youden_fpr:.3f}")

    mlflow.log_metrics({
        "auc": auc,
        "average_precision": ap,
        "log_loss": ll,
        "brier": brier,
        "youden_threshold": float(youden_thr),
        "youden_tpr": float(youden_tpr),
        "youden_fpr": float(youden_fpr),
        "train_rows": float(len(df_train)),
        "train_pos_rate": float(df_train[TARGET_COL].mean()),
        "cal_rows": float(len(df_cal_unbal)),
        "eval_rows": float(len(df_eval_unbal)),
        "eval_pos_rate": float(np.mean(y_eval)),
    })

    y_pred = (p_eval >= youden_thr).astype(int)
    log("[eval] Classification report (unbalanced eval):")
    print(classification_report(y_eval, y_pred, target_names=["Not Cancelled", "Cancelled"]))
    log("[eval] Confusion matrix:")
    print(confusion_matrix(y_eval, y_pred))

    # Calibration plot
    if not NO_PLOTS:
        plot_reliability("cancel_unbal_eval", y_eval, p_eval, OUTDIR)

    # ---- Feature importance ----
    importances = model.get_feature_importance()
    imp_df = pd.DataFrame({
        "feature": feature_order,
        "importance": importances
    }).sort_values("importance", ascending=False)
    imp_path = OUTDIR / "feature_importance.csv"
    imp_df.to_csv(imp_path, index=False)
    log(f"[SAVE] feature importance -> {imp_path}")
    log(f"[importance] top 10:")
    for _, row in imp_df.head(10).iterrows():
        log(f"  {row['feature']}: {row['importance']:.2f}")

    # ---- Save metrics ----
    metrics = {
        "artifact_type": "cancellation_train_metrics_v1",
        "created_utc": datetime.now(dt_timezone.utc).isoformat(),
        "target_col": TARGET_COL,
        "train_rows": int(len(df_train)),
        "train_pos_rate": float(df_train[TARGET_COL].mean()),
        "cal_rows": int(len(df_cal_unbal)),
        "eval_rows": int(len(df_eval_unbal)),
        "eval_pos_rate": float(np.mean(y_eval)),
        "auc": auc,
        "average_precision": ap,
        "log_loss": ll,
        "brier": brier,
        "youden_threshold": youden_thr,
        "youden_tpr": youden_tpr,
        "youden_fpr": youden_fpr,
        "unbalanced_source": unbal_source,
        "versions": _safe_versions(),
    }
    metrics_path = OUTDIR / "cancel_train_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    log(f"[SAVE] metrics -> {metrics_path}")

    # ---- Prediction samples ----
    df_eval_unbal = df_eval_unbal.copy()
    df_eval_unbal["p_cancelled"] = p_eval
    df_eval_unbal["pred_cancelled"] = y_pred

    sample_n = int(cfg.get("prediction_samples_n", 500))
    sample = df_eval_unbal.sample(n=min(sample_n, len(df_eval_unbal)), random_state=SEED).reset_index(drop=True)
    sample_path = OUTDIR / "prediction_samples.parquet"
    sample.to_parquet(sample_path, index=False)
    log(f"[SAVE] prediction samples -> {sample_path}")

    # ---- Deploy bundle (Pattern B) ----
    deploy_name = str(cfg.get("deploy_bundle_joblib_name", "cancellation_bundle.joblib")).strip() or "cancellation_bundle.joblib"
    deploy_path = OUTDIR / deploy_name

    bundle: Dict[str, Any] = {
        "artifact_type": "cancellation_bundle_v1",
        "created_utc": datetime.now(dt_timezone.utc).isoformat(),
        "target_col": TARGET_COL,
        "categorical_features": cat_feats,
        "numeric_features": num_feats,
        "feature_order": feature_order,
        "calibrator": cal,
        "optimal_threshold": youden_thr,
        "registry": {
            "model_path": str(model_path),
            "cal_path": str(cal_path),
        },
        "versions": _safe_versions(),
        "config_used": cfg,
    }

    joblib.dump(bundle, deploy_path)
    log(f"[SAVE] deployable bundle -> {deploy_path}")

    # --- MLflow artifact logging ----------------------------------------------
    for _artifact in (metrics_path, imp_path, sample_path, deploy_path):
        try:
            mlflow.log_artifact(str(_artifact))
        except Exception as _e:
            log(f"[mlflow][WARN] failed to log artifact {_artifact}: {_e}")
    mlflow.end_run()
    # --------------------------------------------------------------------------

    log("[OK] finished training cancellation model")


if __name__ == "__main__":
    main()
