#!/usr/bin/env python3
# src/training/train_arr_bins_ordinal_catboost.py
#
# Ordinal/binned ARRIVAL-delay probability model via calibrated P(delay >= thr).
#
# Pattern B (production-safe, mirrors train_dep_bins_ordinal_catboost.py):
# - DO NOT joblib/pickle any custom Python classes.
# - Save a single deployable JOBLIB that is a plain dict:
#   {
#     "artifact_type": "arr_delay_bins_bundle_v1",
#     "thresholds": [...],
#     "bin_labels": [...],
#     "categorical_features": [...],
#     "numeric_features": [...],
#     "feature_order": [...],
#     "calibrators": {thr: CalibratedClassifierCV, ...},  # OK to pickle sklearn objects
#     "registry": {thr: {"model_path": "...cbm", "cal_path": "...joblib"}, ...},
#     "versions": {...},
#     "created_utc": "...",
#   }
#
# Runtime/inference code loads this dict and implements:
# - enforce_monotone_ge_probs
# - ge_to_bins
#
# Run (config-only):
#   python src/training/train_arr_bins_ordinal_catboost.py data/models/arr_bins_WN_config.json
#
import os
import sys
import json
import time
import signal
import faulthandler
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone as dt_timezone

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    log_loss,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

import joblib
from catboost import CatBoostClassifier, Pool


REPO_ROOT = Path.cwd()
DATA_ROOT = (REPO_ROOT.parent / "flightrightdata").resolve()

FAST = os.getenv("FAST_TRAIN", "0") == "1"
NO_PLOTS = os.getenv("NO_PLOTS", "0") == "1"
TRACE_EVERY_SEC = int(os.getenv("TRACE_EVERY_SEC", "0"))

_t0 = time.time()
_last_step_t = time.time()


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str, *, step: bool = False) -> None:
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


def _as_data_path(p: Path) -> Path:
    return p if p.is_absolute() else (DATA_ROOT / p).resolve()


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _looks_like_json(p: Path) -> bool:
    return p.suffix.lower() == ".json"


def _resolve_run_args(argv) -> Tuple[Path, Path, Optional[Path]]:
    """
    Supported invocations:
      1) script CONFIG.json
      2) script INPUT.parquet OUTDIR
      3) script INPUT.parquet OUTDIR CONFIG.json
    Returns: (INPUT, OUTDIR, cfg_path_or_None)
    """
    if len(argv) == 2:
        p = Path(argv[1])
        if not _looks_like_json(p):
            raise SystemExit("Config-only mode requires a .json file: python ... train_arr_bins_...py path/to/config.json")
        cfg = _load_json(p)
        in_path = cfg.get("input_features_path") or cfg.get("input_features_unbalanced_path")
        outdir = cfg.get("outdir") or cfg.get("output_dir")
        if not in_path or not outdir:
            raise SystemExit("Config-only mode requires cfg['input_features_path'] (or input_features_unbalanced_path) and cfg['outdir'].")
        return Path(in_path), Path(outdir), p

    if len(argv) == 3:
        return Path(argv[1]), Path(argv[2]), None

    if len(argv) == 4:
        return Path(argv[1]), Path(argv[2]), Path(argv[3])

    raise SystemExit(
        "Usage:\n"
        "  python train_arr_bins_ordinal_catboost.py CONFIG.json\n"
        "  python train_arr_bins_ordinal_catboost.py features_unbalanced.parquet outdir\n"
        "  python train_arr_bins_ordinal_catboost.py features_unbalanced.parquet outdir config.json"
    )


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


# ------------------------------ ordinal helpers ------------------------------

def enforce_monotone_ge_probs(p_ge: np.ndarray) -> np.ndarray:
    p = p_ge.copy()
    for j in range(1, p.shape[1]):
        p[:, j] = np.minimum(p[:, j], p[:, j - 1])
    return p


def make_bin_labels(thresholds: List[int]) -> List[str]:
    labels = [f"< {thresholds[0]} min"]
    for i in range(len(thresholds) - 1):
        labels.append(f"{thresholds[i]}-{thresholds[i+1]} min")
    labels.append(f">= {thresholds[-1]} min")
    return labels


def ge_to_bins(p_ge_dict: Dict[int, np.ndarray], thresholds: List[int]) -> np.ndarray:
    p_ge = enforce_monotone_ge_probs(
        np.vstack([p_ge_dict[t] for t in thresholds]).T
    )
    bins = [1.0 - p_ge[:, 0]]
    for j in range(len(thresholds) - 1):
        bins.append(np.maximum(0.0, p_ge[:, j] - p_ge[:, j + 1]))
    bins.append(np.maximum(0.0, p_ge[:, -1]))
    P = np.vstack(bins).T
    Z = P.sum(axis=1, keepdims=True)
    Z[Z == 0] = 1.0
    return P / Z


# ------------------------------ calibration diagnostics ------------------------------

def reliability_table(y_true, p_pred, n_bins=10) -> pd.DataFrame:
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
    out["bin_left"] = out["bin"].apply(lambda iv: iv.left if pd.notna(iv) else np.nan)
    out["bin_right"] = out["bin"].apply(lambda iv: iv.right if pd.notna(iv) else np.nan)
    out["bin_mid"] = (out["bin_left"].astype(float) + out["bin_right"].astype(float)) / 2.0
    return out


def plot_reliability(tag, y_true, p_pred, outdir: Path, n_bins=10):
    tab = reliability_table(y_true, p_pred, n_bins=n_bins)
    csv_path = outdir / f"calibration_{tag}_table.csv"
    tab.drop(columns=["bin"], errors="ignore").to_csv(csv_path, index=False)
    log(f"[SAVE] reliability table -> {csv_path}")

    if NO_PLOTS:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        log(f"[WARN] matplotlib not available for plots: {e}")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="perfect")
    ax.plot(tab["mean_pred"], tab["empirical_pos_rate"], marker="o", linewidth=1.5, label="model")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical positive rate")
    ax.set_title(f"Reliability curve ({tag})")
    ax.legend(loc="best")
    fig.tight_layout()
    png_path = outdir / f"calibration_{tag}.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    log(f"[SAVE] reliability plot -> {png_path}")


def best_threshold_youden(y_true, y_proba) -> Tuple[float, float, float]:
    fpr, tpr, thr = roc_curve(y_true, y_proba)
    j = tpr - fpr
    i = int(np.argmax(j))
    return float(thr[i]), float(tpr[i]), float(fpr[i])


# ------------------------------ data splits (mirrors dep pattern) ------------------------------

def _time_eval_split(df: pd.DataFrame, *, date_col: str, last_days: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if date_col not in df.columns:
        raise RuntimeError(f"Missing {date_col} for time-based eval split.")
    fd = pd.to_datetime(df[date_col], errors="coerce")
    cutoff = fd.max() - pd.Timedelta(days=int(last_days))
    is_eval = fd >= cutoff
    return df.loc[~is_eval].reset_index(drop=True), df.loc[is_eval].reset_index(drop=True)


def _split_train_for_calibration(df_train: pd.DataFrame, y_col: str, *, cal_size: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calibration comes ONLY from the training period (so eval stays "future" clean).
    Stratify on the target label for that threshold.
    """
    cal_size = float(cal_size)
    cal_size = min(max(cal_size, 0.05), 0.95)

    y = pd.to_numeric(df_train[y_col], errors="coerce").fillna(0).astype(int).values
    idx = np.arange(len(df_train))

    if len(df_train) < 50 or len(np.unique(y)) < 2:
        # too small / degenerate
        return df_train.copy(), df_train.copy()

    idx_fit, idx_cal, _, _ = train_test_split(
        idx, y, test_size=cal_size, random_state=int(seed), stratify=y
    )
    df_fit = df_train.iloc[idx_fit].reset_index(drop=True)
    df_cal = df_train.iloc[idx_cal].reset_index(drop=True)
    return df_fit, df_cal


# ------------------------------ feature handling ------------------------------

def as_str(series: pd.Series) -> pd.Series:
    s = series.astype("object")
    s = s.where(~pd.isna(s), "Unknown")
    s = s.map(lambda v: "Unknown" if str(v).strip().lower() in ("", "nan", "none") else str(v))
    return s.astype("object")


def _resolve_feature_lists(cfg: dict, df_all: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    feat_cfg = cfg.get("features") or {}
    cat_feats = [str(x) for x in (feat_cfg.get("categorical") or [])]
    num_feats = [str(x) for x in (feat_cfg.get("numeric") or [])]
    requested = cat_feats + num_feats

    present = [c for c in requested if c in df_all.columns]

    # Exclusions: keep in parquet for validation, but never train on them
    train_cfg = cfg.get("training") or {}
    exclude_prefixes = tuple(train_cfg.get("exclude_feature_prefixes") or ["y_dep_ge"])
    exclude_cols = set(train_cfg.get("exclude_feature_columns") or [])

    present = [
        c for c in present
        if (c not in exclude_cols) and (not any(c.startswith(p) for p in exclude_prefixes))
    ]

    cat_feats = [c for c in cat_feats if c in present]
    num_feats = [c for c in num_feats if c in present]

    if not present:
        raise RuntimeError("No requested features exist after exclusions. Check cfg['features'].")

    return present, cat_feats, num_feats


def _make_pool(df: pd.DataFrame, feature_order: List[str], cat_feats: List[str], y: Optional[pd.Series] = None) -> Pool:
    X = df[feature_order].copy()
    for c in cat_feats:
        X[c] = as_str(X[c])
    if y is None:
        return Pool(X, cat_features=cat_feats)
    return Pool(X, y.astype(int), cat_features=cat_feats)


def _load_balanced_or_sample(cfg: dict, thr: int, df_train_period: pd.DataFrame) -> pd.DataFrame:
    """
    Mirrors dep trainer intent: use balanced dataset per threshold if present,
    else sample from training period to max_rows with stratification.
    """
    ycol = f"y_arr_ge{thr}"
    if ycol not in df_train_period.columns:
        raise RuntimeError(f"Missing label column in training data: {ycol}")

    train_cfg = cfg.get("training") or {}
    max_rows = int(train_cfg.get("max_train_rows", 0))
    seed = int(train_cfg.get("seed", 1337))

    # Candidate balanced templates
    fb = cfg.get("feature_balance") or {}
    tpl = fb.get("output_template") or cfg.get("output_features_balanced_path_template")

    cand = None
    if tpl:
        # support both {thr} and {target} styles
        try:
            rel = tpl.format(thr=thr, target="arr")
        except Exception:
            rel = tpl
        cand = _as_data_path(Path(rel))

    used_balanced = False
    if cand is not None and cand.exists():
        log(f"[DATA] Using balanced training file thr={thr}: {cand}")
        df = pd.read_parquet(cand)
        used_balanced = True
    else:
        log(f"[DATA] No balanced file for thr={thr}; sampling from unbalanced training period.")
        df = df_train_period.copy()

    df = df.dropna(subset=[ycol]).reset_index(drop=True)

    # Only downsample when we fell back to unbalanced data (balanced files are
    # already sized intentionally by the feature engineering step).
    if not used_balanced and max_rows > 0 and len(df) > max_rows:
        y = df[ycol].astype(int).values
        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        rng = np.random.default_rng(seed + thr)

        # preserve class balance in the source df (balanced files likely ~50/50 already)
        pos_keep = int(round(max_rows * (len(pos) / max(len(df), 1))))
        pos_keep = max(1, min(pos_keep, len(pos)))
        neg_keep = max(1, min(max_rows - pos_keep, len(neg)))

        pick = np.concatenate([
            rng.choice(pos, size=pos_keep, replace=False),
            rng.choice(neg, size=neg_keep, replace=False),
        ])
        df = df.iloc[pick].sample(frac=1.0, random_state=seed + thr).reset_index(drop=True)

    log(f"[DATA] thr={thr} train_rows={len(df):,} pos_frac={df[ycol].mean():.3f}")
    return df


# ------------------------------ main ------------------------------

def main():
    enable_periodic_traces()
    enable_usr1_trace()

    INPUT, OUTDIR, CFG_PATH = _resolve_run_args(sys.argv)
    cfg = _load_json(CFG_PATH) if CFG_PATH is not None else {}

    in_path = _as_data_path(INPUT)
    outdir = _as_data_path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)

    thresholds = cfg.get("thresholds", [15, 30, 60, 120])
    thresholds = [int(x) for x in thresholds]
    if len(thresholds) != 4:
        raise SystemExit("This trainer expects exactly 4 thresholds.")

    eval_cfg = cfg.get("eval") or {}
    eval_last_days = int(eval_cfg.get("last_days", 90))
    seed = int((cfg.get("training") or {}).get("seed", 1337))

    # CatBoost params (FAST_TRAIN supported like dep trainer)
    cb_cfg = cfg.get("catboost") or {}
    iterations = int(cb_cfg.get("iterations", 1200))
    depth = int(cb_cfg.get("depth", 8))
    lr = float(cb_cfg.get("learning_rate", 0.04))
    od_wait = int(cb_cfg.get("od_wait", 50))
    if FAST:
        iterations = min(iterations, 350)
        depth = min(depth, 8)
        lr = max(lr, 0.12)
        od_wait = min(od_wait, 30)
        log("[FAST_TRAIN] enabled: reducing iterations / boosting speed")

    params = dict(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=iterations,
        depth=depth,
        learning_rate=lr,
        l2_leaf_reg=float(cb_cfg.get("l2_leaf_reg", 3.0)),
        od_type="Iter",
        od_wait=od_wait,
        random_seed=int(cb_cfg.get("seed", seed)),
        verbose=int(cb_cfg.get("verbose", 200)),
        thread_count=int(cb_cfg.get("thread_count", -1)),
        allow_writing_files=False,
    )

    log(f"[LOAD] reading features: {in_path}", step=True)
    df_all = pd.read_parquet(in_path)

    # Resolve features (exclude y_dep_ge* etc.)
    feature_order, cat_feats, num_feats = _resolve_feature_lists(cfg, df_all)
    log(f"[FEATURES] cats={len(cat_feats)} nums={len(num_feats)} total={len(feature_order)}", step=True)

    # Save the resolved feature list so inference code can inspect it.
    resolved_feat_path = outdir / "resolved_features.json"
    resolved_feat_path.write_text(
        json.dumps({"categorical": cat_feats, "numeric": num_feats, "feature_order": feature_order}, indent=2),
        encoding="utf-8",
    )
    log(f"[SAVE] resolved features -> {resolved_feat_path}")

    # Ensure labels exist
    for thr in thresholds:
        ycol = f"y_arr_ge{thr}"
        if ycol not in df_all.columns:
            raise SystemExit(f"Missing label column: {ycol}")

    # --- Eval set: prefer pre-built random holdout, fall back to time-based split ---
    eval_feat_path_raw = cfg.get("eval_features_path")
    if eval_feat_path_raw:
        eval_feat_path = _as_data_path(Path(eval_feat_path_raw))
        log(f"[unbalanced] loaded eval_features_path={eval_feat_path} rows=?", step=True)
        df_eval_full = pd.read_parquet(eval_feat_path)
        log(f"[unbalanced] loaded rows={len(df_eval_full):,}", step=True)
        # Split 50/50 into calibration set and final eval set (mirrors dep trainer)
        cal_size_eval = float(eval_cfg.get("cal_size", 0.50))
        n_cal = int(len(df_eval_full) * cal_size_eval)
        df_cal_unbal = df_eval_full.iloc[:n_cal].reset_index(drop=True)
        df_eval = df_eval_full.iloc[n_cal:].reset_index(drop=True)
        log(f"[unbalanced] split -> cal={len(df_cal_unbal):,} eval={len(df_eval):,} (cal_size={cal_size_eval})", step=True)
        df_train_period = df_all   # full unbalanced parquet; balanced files handle actual training rows
    else:
        # Fall back: time-based split from the unbalanced features parquet
        df_train_period, df_eval = _time_eval_split(df_all, date_col="FlightDate", last_days=eval_last_days)
        df_cal_unbal = df_eval  # use the same eval set for calibration (no separate cal)
        log(f"[SPLIT] train_period={len(df_train_period):,} eval_future={len(df_eval):,} last_days={eval_last_days}", step=True)

    # Eval feature matrix shared across thresholds (labels differ, X is same)
    X_eval_df = df_eval[feature_order].copy()
    for c in cat_feats:
        X_eval_df[c] = as_str(X_eval_df[c])
    X_cal_unbal_df = df_cal_unbal[feature_order].copy()
    for c in cat_feats:
        X_cal_unbal_df[c] = as_str(X_cal_unbal_df[c])

    models: Dict[int, CatBoostClassifier] = {}
    calibrators: Dict[int, Any] = {}
    registry: Dict[int, Dict[str, str]] = {}
    metrics: Dict[str, Any] = {
        "artifact_type": "arr_train_metrics_v1",
        "created_utc": datetime.now(dt_timezone.utc).isoformat(),
        "thresholds": thresholds,
        "eval_last_days": eval_last_days,
        "n_eval": int(len(df_eval)),
        "per_threshold": {},
        "versions": _safe_versions(),
    }

    for thr in thresholds:
        log(f"================== Train arr >= {thr} ==================")
        ycol = f"y_arr_ge{thr}"

        # Train df: balanced file if exists; else sample to max_train_rows
        df_thr = _load_balanced_or_sample(cfg, thr, df_train_period)

        # 3-way split for CatBoost: fit / early-stop / balanced-test-sanity
        train_cfg = cfg.get("training") or {}
        es_frac = float(train_cfg.get("earlystop_frac", 0.25))
        test_frac = float(train_cfg.get("test_frac", 0.25))
        n = len(df_thr)
        n_es = max(1, int(n * es_frac))
        n_test = max(1, int(n * test_frac))
        n_fit = n - n_es - n_test
        rng = np.random.default_rng(seed + thr)
        idx = rng.permutation(n)
        df_fit = df_thr.iloc[idx[:n_fit]].reset_index(drop=True)
        df_es = df_thr.iloc[idx[n_fit:n_fit + n_es]].reset_index(drop=True)
        df_test_bal = df_thr.iloc[idx[n_fit + n_es:]].reset_index(drop=True)
        log(f"[data] per-threshold sizes train={len(df_fit):,} earlystop={len(df_es):,} test={len(df_test_bal):,}", step=True)

        pool_fit = _make_pool(df_fit, feature_order, cat_feats, y=df_fit[ycol])
        pool_es = _make_pool(df_es, feature_order, cat_feats, y=df_es[ycol])

        # Per-threshold overrides from hyperparam optimization (optional)
        thr_override = (cfg.get("per_threshold_catboost") or {}).get(str(thr), {})
        thr_params = {**params}
        if thr_override:
            for ovr_key in ("iterations", "depth", "learning_rate", "l2_leaf_reg", "od_wait",
                            "bagging_temperature", "random_strength", "border_count",
                            "min_data_in_leaf", "bootstrap_type", "subsample"):
                if ovr_key in thr_override:
                    thr_params[ovr_key] = thr_override[ovr_key]
            log(f"[cb] using per-threshold overrides for thr={thr}: {thr_override}")
        model = CatBoostClassifier(**thr_params)
        log(f"[cb] fit start (iter={thr_params['iterations']} depth={thr_params['depth']} lr={thr_params['learning_rate']} l2={thr_params['l2_leaf_reg']} od_wait={thr_params['od_wait']})", step=True)
        model.fit(pool_fit, eval_set=pool_es, use_best_model=True)
        log(f"[cb] fit done", step=True)

        # Save model to .cbm
        model_path = outdir / f"arr_thr{thr}.cbm"
        model.save_model(str(model_path))

        # Calibrate on the unbalanced calibration portion of the pre-built eval parquet
        log(f"[cal] isotonic calibration on UNBALANCED cal split size={len(df_cal_unbal):,}", step=True)
        cal = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
        y_cal_ub = df_cal_unbal[ycol].astype(int).values
        cal.fit(X_cal_unbal_df, y_cal_ub)
        log(f"[cal] done", step=True)

        cal_path = outdir / f"arr_thr{thr}_calibrator.joblib"
        joblib.dump(cal, cal_path)

        # AUC on all 3 sets — mirrors dep trainer output format
        p_cal_unbal = cal.predict_proba(X_cal_unbal_df)[:, 1].astype(float)
        auc_cal_unbal = float(roc_auc_score(y_cal_ub, p_cal_unbal)) if len(np.unique(y_cal_ub)) >= 2 else float("nan")

        X_test_bal = df_test_bal[feature_order].copy()
        for c in cat_feats:
            X_test_bal[c] = as_str(X_test_bal[c])
        y_test_bal = df_test_bal[ycol].astype(int).values
        p_test_bal = cal.predict_proba(X_test_bal)[:, 1].astype(float)
        auc_test_bal = float(roc_auc_score(y_test_bal, p_test_bal)) if len(np.unique(y_test_bal)) >= 2 else float("nan")

        y_eval = df_eval[ycol].astype(int).values
        p_eval = cal.predict_proba(X_eval_df)[:, 1].astype(float)
        auc_eval_unbal = float(roc_auc_score(y_eval, p_eval)) if len(np.unique(y_eval)) >= 2 else float("nan")

        youden_thr, youden_tpr, youden_fpr = best_threshold_youden(y_eval, p_eval)
        log(f"[thr>={thr}] Youden threshold (from UNBAL cal)={youden_thr:.3f} | TPR={youden_tpr:.3f} FPR={youden_fpr:.3f}")
        log(f"[thr>={thr}] AUC cal_unbal={auc_cal_unbal:.3f} | AUC test_bal_sanity={auc_test_bal:.3f} | AUC eval_unbal={auc_eval_unbal:.3f}")

        # Classification reports
        from sklearn.metrics import classification_report as _cr
        log(f"[thr>={thr}] Classification report on BALANCED sanity test @ Youden(from unbal cal):")
        y_pred_y_test = (p_test_bal >= youden_thr).astype(int)
        print(_cr(y_test_bal, y_pred_y_test))
        log(f"[thr>={thr}] Confusion matrix (balanced sanity):")
        print(confusion_matrix(y_test_bal, y_pred_y_test))
        log(f"[thr>={thr}] Classification report on UNBALANCED eval @ Youden(from unbal cal):")
        y_pred_y_eval = (p_eval >= youden_thr).astype(int)
        print(_cr(y_eval, y_pred_y_eval))
        log(f"[thr>={thr}] Confusion matrix (unbalanced eval):")
        print(confusion_matrix(y_eval, y_pred_y_eval))

        # Reliability outputs
        thr_outdir = outdir / f"thr_{thr}"
        thr_outdir.mkdir(parents=True, exist_ok=True)
        plot_reliability(f"arr_ge{thr}_unbal_cal", y_cal_ub, p_cal_unbal, thr_outdir)
        plot_reliability(f"arr_ge{thr}_unbal_eval", y_eval, p_eval, thr_outdir)

        ap = float(average_precision_score(y_eval, p_eval)) if len(np.unique(y_eval)) >= 2 else float("nan")
        ll = float(log_loss(y_eval, np.clip(p_eval, 1e-6, 1 - 1e-6)))
        brier = float(brier_score_loss(y_eval, p_eval))

        metrics["per_threshold"][str(thr)] = {
            "train_rows": int(len(df_thr)),
            "fit_rows": int(len(df_fit)),
            "es_rows": int(len(df_es)),
            "test_bal_rows": int(len(df_test_bal)),
            "cal_unbal_rows": int(len(df_cal_unbal)),
            "eval_rows": int(len(df_eval)),
            "pos_frac_train": float(df_thr[ycol].mean()),
            "pos_frac_eval": float(np.mean(y_eval)),
            "auc_cal_unbal": auc_cal_unbal,
            "auc_test_bal_sanity": auc_test_bal,
            "auc_eval_unbal": auc_eval_unbal,
            "ap": ap,
            "logloss": ll,
            "brier": brier,
            "youden_threshold": float(youden_thr),
            "youden_tpr": float(youden_tpr),
            "youden_fpr": float(youden_fpr),
        }

        # Keep in-memory refs for bundling
        models[thr] = model
        calibrators[thr] = cal
        registry[thr] = {
            "model_path": str(model_path),
            "cal_path": str(cal_path),
            "meta": {
                "threshold": thr,
                "auc_cal_unbalanced": auc_cal_unbal,
                "auc_test_balanced_sanity": auc_test_bal,
                "auc_eval_unbalanced": auc_eval_unbal,
                "threshold_youden": float(youden_thr),
                "tpr_at_threshold": float(youden_tpr),
                "fpr_at_threshold": float(youden_fpr),
            },
        }

        # Free training objects between thresholds to avoid OOM
        import gc as _gc
        _gc.collect()
        log(f"[gc] released threshold {thr} training objects")

    # Bin metadata — config-driven with defaults that mirror the dep trainer.
    bin_labels = make_bin_labels(thresholds)
    n_bins = len(thresholds) + 1
    bin_weights_minutes = cfg.get("bin_weights_minutes", [7.5, 22.5, 45.0, 90.0, 150.0])
    severity_weights    = cfg.get("severity_weights",    [0.0,  1.0,  2.0,  3.0,  4.0])
    w_min = np.array(bin_weights_minutes, dtype=float)
    w_sev = np.array(severity_weights,    dtype=float)
    if len(w_min) != n_bins or len(w_sev) != n_bins:
        raise ValueError(f"bin_weights_minutes and severity_weights must each have length {n_bins}.")

    # Write combined eval predictions (bins)
    p_ge_dict: Dict[int, np.ndarray] = {}
    for thr in thresholds:
        p_ge_dict[thr] = calibrators[thr].predict_proba(X_eval_df)[:, 1].astype(float)
    Pbins = ge_to_bins(p_ge_dict, thresholds)

    df_eval_out = df_eval.copy()
    for i, thr in enumerate(thresholds):
        df_eval_out[f"p_ge{thr}"] = p_ge_dict[thr]
    for i, lbl in enumerate(bin_labels):
        col = f"p_bin{i}_{lbl.replace(' ', '').replace('<','lt').replace('>=','ge')}"
        df_eval_out[col] = Pbins[:, i]
    # Derived scalar summaries (mirrors dep trainer output)
    df_eval_out["expected_delay_min"] = Pbins @ w_min
    df_eval_out["severity_score"]     = Pbins @ w_sev
    pred_idx = np.argmax(Pbins, axis=1)
    df_eval_out["pred_bin"] = pd.Categorical(
        [bin_labels[i] for i in pred_idx], categories=bin_labels, ordered=True
    )

    # Compute bin-level logloss and accuracy for bins_meta
    # Derive true_bin from binary threshold labels
    y_ge = np.column_stack([
        df_eval[f"y_arr_ge{thr}"].astype(int).values for thr in thresholds
    ])  # shape (N, 4)
    # true_bin: 0 = <15, 1 = 15-30, 2 = 30-45, 3 = 45-60, 4 = >=60
    y_true_mc = y_ge.sum(axis=1)  # number of thresholds exceeded = bin index
    try:
        bins_logloss = float(log_loss(y_true_mc, Pbins, labels=list(range(len(bin_labels)))))
    except Exception:
        bins_logloss = float("nan")
    bins_acc = float(accuracy_score(y_true_mc, pred_idx))
    log(f"[BINS] eval logloss={bins_logloss:.4f}  acc={bins_acc:.4f}")

    eval_out_name = str(cfg.get("eval_output_parquet_name", "arr_eval_with_probs.parquet"))
    eval_out_path = outdir / eval_out_name
    df_eval_out.to_parquet(eval_out_path, index=False)
    log(f"[SAVE] eval predictions -> {eval_out_path}")

    # Save metrics JSON
    metrics_name = str(cfg.get("metrics_output_json_name", "arr_train_metrics.json"))
    (outdir / metrics_name).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log(f"[SAVE] metrics -> {outdir / metrics_name}")

    # Deploy bundle dict (production-safe — no custom Python classes).
    # Inference code should call:
    #   enforce_monotone_ge_probs / ge_to_bins  (copy these two functions)
    #   expected_delay_min = Pbins @ bundle["bin_weights_minutes"]
    #   severity_score     = Pbins @ bundle["severity_weights"]
    bundle_name = str(cfg.get("deploy_bundle_joblib_name", "arr_delay_bins_bundle.joblib")).strip() or "arr_delay_bins_bundle.joblib"
    deploy_path = outdir / bundle_name

    bundle: Dict[str, Any] = {
        "artifact_type":        "arr_delay_bins_bundle_v1",
        "created_utc":          datetime.now(dt_timezone.utc).isoformat(),
        "thresholds":           thresholds,
        "bin_labels":           bin_labels,
        "bin_weights_minutes":  list(bin_weights_minutes),
        "severity_weights":     list(severity_weights),
        "categorical_features": cat_feats,
        "numeric_features":     num_feats,
        "feature_order":        feature_order,
        "calibrators":          calibrators,   # dict {thr: CalibratedClassifierCV}
        "registry":             registry,      # dict {thr: {"model_path": ..., "cal_path": ...}}
        "versions":             _safe_versions(),
        "preprocess":           {"categorical_na_value": "Unknown"},
        "config_used":          cfg,
        "bins_meta": {
            "bin_labels": bin_labels,
            "thresholds": thresholds,
            "eval_logloss": bins_logloss,
            "eval_acc": bins_acc,
            "bin_weights_minutes": list(bin_weights_minutes),
            "severity_weights": list(severity_weights),
            "feature_order": feature_order,
        },
    }

    joblib.dump(bundle, deploy_path)
    log(f"[SAVE] deploy bundle -> {deploy_path}")
    log("[DONE] arr training complete.", step=True)


if __name__ == "__main__":
    main()