#!/usr/bin/env python3
# src/training/hyperparam_optimize.py
#
# Bayesian hyperparameter optimization for CatBoost ordinal delay models.
# Works for both departure and arrival configs (auto-detected).
#
# Usage:
#   python src/training/hyperparam_optimize.py data/dep_train_WN_100_v8.json --n-trials 100
#   python src/training/hyperparam_optimize.py data/arr_train_WN_100_v8.json --n-trials 100 --mode arr
#   python src/training/hyperparam_optimize.py data/dep_train_WN_100_v8.json --thresholds 15,30 --n-trials 50
#
import os
import sys
import json
import time
import argparse
import copy
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from catboost import CatBoostClassifier, Pool

import optuna
from optuna.samplers import TPESampler

import mlflow
from _mlflow_setup import init_mlflow, loggable_params, git_commit


REPO_ROOT = Path.cwd()
DATA_ROOT = (REPO_ROOT.parent / "flightrightdata").resolve()

_t0 = time.time()


def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    elapsed = time.time() - _t0
    print(f"[{_now()} +{elapsed:7.1f}s] {msg}", flush=True)


# ------------------------------ path helpers ------------------------------

def _as_data_path(p: Path) -> Path:
    return p if p.is_absolute() else (DATA_ROOT / p).resolve()


def _load_json(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------ feature/data helpers ------------------------------

def as_str(series: pd.Series) -> pd.Series:
    s = series.astype("object")
    s = s.where(~pd.isna(s), "Unknown")
    s = s.map(lambda v: "Unknown" if str(v).strip().lower() in ("", "nan", "none") else str(v))
    return s.astype("object")


def build_X(df: pd.DataFrame, feature_order: List[str],
            cat_feats: List[str], num_feats: List[str]) -> pd.DataFrame:
    X = df[feature_order].copy()
    for c in cat_feats:
        X[c] = as_str(X[c])
    for c in num_feats:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        if X[c].isna().any():
            med = X[c].median()
            X[c] = X[c].fillna(med if pd.notna(med) else 0.0)
    return X


def enforce_monotone_ge_probs(p_ge: np.ndarray) -> np.ndarray:
    p = p_ge.copy()
    for j in range(1, p.shape[1]):
        p[:, j] = np.minimum(p[:, j], p[:, j - 1])
    return p


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


def true_bin_from_delay(delay_min: pd.Series, thresholds: List[int]) -> np.ndarray:
    d = pd.to_numeric(delay_min, errors="coerce").fillna(0.0).values
    codes = np.zeros(len(d), dtype=int)
    for i, thr in enumerate(thresholds):
        codes[d >= thr] = i + 1
    return codes


# ------------------------------ dep mode splits (mirrors dep trainer) ------

def _split_unbalanced_for_cal_and_eval(df_unbal, *, cal_size, seed):
    if len(df_unbal) < 20:
        return df_unbal.copy(), df_unbal.copy()
    d = pd.to_numeric(df_unbal.get("DepDelayMinutes"), errors="coerce").fillna(0.0)
    y15 = (d >= 15).astype(int).values
    idx = np.arange(len(df_unbal))
    cal_size = min(max(float(cal_size), 0.05), 0.95)
    idx_cal, idx_eval, _, _ = train_test_split(
        idx, y15, test_size=(1.0 - cal_size),
        random_state=int(seed), stratify=y15
    )
    return (df_unbal.iloc[idx_cal].reset_index(drop=True),
            df_unbal.iloc[idx_eval].reset_index(drop=True))


# ------------------------------ arr mode feature resolution ------

def _resolve_feature_lists_arr(cfg, df_all):
    feat_cfg = cfg.get("features") or {}
    cat_feats = [str(x) for x in (feat_cfg.get("categorical") or [])]
    num_feats = [str(x) for x in (feat_cfg.get("numeric") or [])]
    requested = cat_feats + num_feats
    present = [c for c in requested if c in df_all.columns]

    train_cfg = cfg.get("training") or {}
    exclude_prefixes = tuple(train_cfg.get("exclude_feature_prefixes") or ["y_dep_ge"])
    exclude_cols = set(train_cfg.get("exclude_feature_columns") or [])
    present = [
        c for c in present
        if (c not in exclude_cols) and (not any(c.startswith(p) for p in exclude_prefixes))
    ]
    cat_feats = [c for c in cat_feats if c in present]
    num_feats = [c for c in num_feats if c in present]
    return present, cat_feats, num_feats


def _load_balanced_or_sample_arr(cfg, thr, df_train_period):
    ycol = f"y_arr_ge{thr}"
    if ycol not in df_train_period.columns:
        raise RuntimeError(f"Missing label column: {ycol}")
    train_cfg = cfg.get("training") or {}
    max_rows = int(train_cfg.get("max_train_rows", 0))
    seed = int(train_cfg.get("seed", 1337))

    fb = cfg.get("feature_balance") or {}
    tpl = fb.get("output_template") or cfg.get("output_features_balanced_path_template")

    cand = None
    if tpl:
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

    if not used_balanced and max_rows > 0 and len(df) > max_rows:
        y = df[ycol].astype(int).values
        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        rng = np.random.default_rng(seed + thr)
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


# ------------------------------ data bundle loading ------------------------------

def load_data_dep(cfg: dict, thresholds: List[int]) -> Dict[int, dict]:
    """Load and split data for departure mode, returning a bundle per threshold."""
    seed = int(cfg.get("random_seed", 42))

    # Resolve features
    cat_feats = [c for c in cfg["categorical_features"]]
    num_feats = [c for c in cfg["numeric_features"]]
    feature_order = cat_feats + num_feats

    # Load unbalanced eval data
    eval_path = str(cfg.get("eval_features_path") or "").strip()
    if not eval_path:
        raise ValueError("cfg['eval_features_path'] must be set.")
    eval_path = str(_as_data_path(Path(eval_path)))
    df_unbal = pd.read_parquet(eval_path)
    log(f"[dep] loaded eval_features_path rows={len(df_unbal):,}")

    cal_frac = float(cfg.get("unbalanced_cal_size", 0.50))
    df_cal_unbal, df_eval_unbal = _split_unbalanced_for_cal_and_eval(
        df_unbal, cal_size=cal_frac, seed=seed
    )
    log(f"[dep] cal={len(df_cal_unbal):,} eval={len(df_eval_unbal):,}")

    # Filter features to those present in eval data
    cat_feats = [c for c in cat_feats if c in df_eval_unbal.columns]
    num_feats = [c for c in num_feats if c in df_eval_unbal.columns]
    feature_order = cat_feats + num_feats

    per_thr_paths = {str(k): str(v) for k, v in (cfg.get("per_threshold_train_paths") or {}).items()}

    bundles = {}
    for thr in thresholds:
        log(f"[dep] loading threshold {thr}...")
        if per_thr_paths:
            p = per_thr_paths.get(str(thr))
            if not p:
                raise ValueError(f"per_threshold_train_paths missing threshold {thr}")
            p = str(_as_data_path(Path(p)))
            df_tr = pd.read_parquet(p)
            base_y_col = f"y_dep_ge{thr}"
            if base_y_col in df_tr.columns:
                base_y = df_tr[base_y_col].astype(int).values
            else:
                d = pd.to_numeric(df_tr.get("DepDelayMinutes"), errors="coerce").fillna(0.0)
                base_y = (d >= thr).astype(int).values
            idx = np.arange(len(df_tr))
            idx_train, idx_temp, _, _ = train_test_split(
                idx, base_y, test_size=0.40, random_state=seed, stratify=base_y
            )
            idx_es, idx_test, _, _ = train_test_split(
                idx_temp, base_y[idx_temp], test_size=0.50,
                random_state=seed, stratify=base_y[idx_temp]
            )
            df_train = df_tr.iloc[idx_train].reset_index(drop=True)
            df_es = df_tr.iloc[idx_es].reset_index(drop=True)
        else:
            raise ValueError("dep mode requires per_threshold_train_paths in config")

        def make_y_dep(df, t):
            col = f"y_dep_ge{t}"
            if col in df.columns:
                return df[col].astype(int).values
            d = pd.to_numeric(df.get("DepDelayMinutes"), errors="coerce").fillna(0.0)
            return (d >= t).astype(int).values

        X_train = build_X(df_train, feature_order, cat_feats, num_feats)
        y_train = make_y_dep(df_train, thr)
        X_es = build_X(df_es, feature_order, cat_feats, num_feats)
        y_es = make_y_dep(df_es, thr)
        X_cal = build_X(df_cal_unbal, feature_order, cat_feats, num_feats)
        y_cal = make_y_dep(df_cal_unbal, thr)
        X_eval = build_X(df_eval_unbal, feature_order, cat_feats, num_feats)
        y_eval = make_y_dep(df_eval_unbal, thr)

        cat_idx = list(range(len(cat_feats)))
        bundles[thr] = {
            "X_train": X_train, "y_train": y_train,
            "X_es": X_es, "y_es": y_es,
            "X_cal": X_cal, "y_cal": y_cal,
            "X_eval": X_eval, "y_eval": y_eval,
            "cat_idx": cat_idx,
            "feature_order": feature_order,
            "cat_feats": cat_feats,
            "num_feats": num_feats,
        }
        log(f"[dep] thr={thr}: train={len(X_train):,} es={len(X_es):,} cal={len(X_cal):,} eval={len(X_eval):,}")

    # For multibin eval: delay column from eval set
    delay_col = pd.to_numeric(df_eval_unbal.get("DepDelayMinutes"), errors="coerce").fillna(0.0)
    y_true_mc = true_bin_from_delay(delay_col, thresholds)
    for thr in thresholds:
        bundles[thr]["y_true_mc"] = y_true_mc

    return bundles


def load_data_arr(cfg: dict, thresholds: List[int]) -> Dict[int, dict]:
    """Load and split data for arrival mode, returning a bundle per threshold."""
    seed = int((cfg.get("training") or {}).get("seed", 1337))

    in_path = cfg.get("input_features_path") or cfg.get("input_features_unbalanced_path")
    in_path = _as_data_path(Path(in_path))
    df_all = pd.read_parquet(in_path)
    log(f"[arr] loaded input_features_path rows={len(df_all):,}")

    feature_order, cat_feats, num_feats = _resolve_feature_lists_arr(cfg, df_all)
    log(f"[arr] features: cats={len(cat_feats)} nums={len(num_feats)} total={len(feature_order)}")

    # Eval split
    eval_cfg = cfg.get("eval") or {}
    eval_feat_path_raw = cfg.get("eval_features_path")
    if eval_feat_path_raw:
        eval_feat_path = _as_data_path(Path(eval_feat_path_raw))
        df_eval_full = pd.read_parquet(eval_feat_path)
        cal_size_eval = float(eval_cfg.get("cal_size", 0.50))
        n_cal = int(len(df_eval_full) * cal_size_eval)
        df_cal_unbal = df_eval_full.iloc[:n_cal].reset_index(drop=True)
        df_eval_unbal = df_eval_full.iloc[n_cal:].reset_index(drop=True)
        df_train_period = df_all
        log(f"[arr] eval from file: cal={len(df_cal_unbal):,} eval={len(df_eval_unbal):,}")
    else:
        eval_last_days = int(eval_cfg.get("last_days", 90))
        fd = pd.to_datetime(df_all["FlightDate"], errors="coerce")
        cutoff = fd.max() - pd.Timedelta(days=eval_last_days)
        is_eval = fd >= cutoff
        df_train_period = df_all.loc[~is_eval].reset_index(drop=True)
        df_eval_unbal = df_all.loc[is_eval].reset_index(drop=True)
        df_cal_unbal = df_eval_unbal
        log(f"[arr] time-based eval split: train={len(df_train_period):,} eval={len(df_eval_unbal):,}")

    X_cal = build_X(df_cal_unbal, feature_order, cat_feats, num_feats)
    X_eval = build_X(df_eval_unbal, feature_order, cat_feats, num_feats)

    train_cfg = cfg.get("training") or {}
    es_frac = float(train_cfg.get("earlystop_frac", 0.25))
    test_frac = float(train_cfg.get("test_frac", 0.25))

    bundles = {}
    for thr in thresholds:
        log(f"[arr] loading threshold {thr}...")
        ycol = f"y_arr_ge{thr}"
        df_thr = _load_balanced_or_sample_arr(cfg, thr, df_train_period)

        n = len(df_thr)
        n_es = max(1, int(n * es_frac))
        n_test = max(1, int(n * test_frac))
        rng = np.random.default_rng(seed + thr)
        idx = rng.permutation(n)
        df_fit = df_thr.iloc[idx[: n - n_es - n_test]].reset_index(drop=True)
        df_es = df_thr.iloc[idx[n - n_es - n_test: n - n_test]].reset_index(drop=True)

        X_train = build_X(df_fit, feature_order, cat_feats, num_feats)
        y_train = df_fit[ycol].astype(int).values
        X_es_thr = build_X(df_es, feature_order, cat_feats, num_feats)
        y_es = df_es[ycol].astype(int).values

        y_cal = df_cal_unbal[ycol].astype(int).values if ycol in df_cal_unbal.columns else np.zeros(len(df_cal_unbal), dtype=int)
        y_eval = df_eval_unbal[ycol].astype(int).values if ycol in df_eval_unbal.columns else np.zeros(len(df_eval_unbal), dtype=int)

        cat_idx = [feature_order.index(c) for c in cat_feats]
        bundles[thr] = {
            "X_train": X_train, "y_train": y_train,
            "X_es": X_es_thr, "y_es": y_es,
            "X_cal": X_cal.copy(), "y_cal": y_cal,
            "X_eval": X_eval.copy(), "y_eval": y_eval,
            "cat_idx": cat_idx,
            "feature_order": feature_order,
            "cat_feats": cat_feats,
            "num_feats": num_feats,
        }
        log(f"[arr] thr={thr}: train={len(X_train):,} es={len(X_es_thr):,} cal={len(X_cal):,} eval={len(X_eval):,}")

    # Multibin eval labels
    delay_col_name = "ArrDelayMinutes" if "ArrDelayMinutes" in df_eval_unbal.columns else "DepDelayMinutes"
    delay_col = pd.to_numeric(df_eval_unbal.get(delay_col_name), errors="coerce").fillna(0.0)
    y_true_mc = true_bin_from_delay(delay_col, thresholds)
    for thr in thresholds:
        bundles[thr]["y_true_mc"] = y_true_mc

    return bundles


# ------------------------------ search space ------------------------------

def suggest_catboost_params(trial: optuna.Trial, seed: int) -> dict:
    params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "od_type": "Iter",
        "thread_count": -1,
        "verbose": 0,
        "random_seed": seed,
        "allow_writing_files": False,
    }
    params["depth"] = trial.suggest_int("depth", 4, 12)
    params["learning_rate"] = trial.suggest_float("learning_rate", 0.005, 0.15, log=True)
    params["l2_leaf_reg"] = trial.suggest_float("l2_leaf_reg", 0.5, 20.0, log=True)
    params["iterations"] = trial.suggest_int("iterations", 2000, 10000, step=500)
    params["od_wait"] = trial.suggest_int("od_wait", 50, 500, step=50)
    params["random_strength"] = trial.suggest_float("random_strength", 0.0, 5.0)
    params["border_count"] = trial.suggest_int("border_count", 32, 255)
    params["min_data_in_leaf"] = trial.suggest_int("min_data_in_leaf", 1, 100)

    bootstrap_type = trial.suggest_categorical("bootstrap_type",
                                                ["Bayesian", "Bernoulli", "MVS"])
    params["bootstrap_type"] = bootstrap_type

    # bagging_temperature is ONLY valid for Bayesian bootstrap
    if bootstrap_type == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 5.0)
    # subsample is ONLY valid for Bernoulli and MVS
    if bootstrap_type in ("Bernoulli", "MVS"):
        params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)

    return params


# ------------------------------ objective functions ------------------------------

def _train_and_evaluate(params: dict, bundle: dict) -> float:
    """Train CatBoost, calibrate, return AUC on unbalanced eval set."""
    cat_idx = bundle["cat_idx"]
    train_pool = Pool(bundle["X_train"], bundle["y_train"], cat_features=cat_idx)
    es_pool = Pool(bundle["X_es"], bundle["y_es"], cat_features=cat_idx)

    clf = CatBoostClassifier(**params)
    clf.fit(train_pool, eval_set=es_pool, use_best_model=True)

    # Isotonic calibration on unbalanced cal set
    calibrator = CalibratedClassifierCV(FrozenEstimator(clf), method="isotonic")
    calibrator.fit(bundle["X_cal"], bundle["y_cal"])

    # Evaluate
    p_eval = calibrator.predict_proba(bundle["X_eval"])[:, 1]
    if len(np.unique(bundle["y_eval"])) < 2:
        return 0.5
    return float(roc_auc_score(bundle["y_eval"], p_eval))


def _log_trial_params_to_mlflow(params: dict) -> None:
    """Log scalar CatBoost params as a flat dict on the active MLflow run."""
    flat = {
        k: str(v) for k, v in params.items()
        if isinstance(v, (int, float, str, bool))
    }
    if flat:
        mlflow.log_params(flat)


def make_per_threshold_objective(thr: int, bundle: dict, seed: int):
    """Objective for per-threshold optimization: maximize AUC for one threshold."""
    def objective(trial):
        params = suggest_catboost_params(trial, seed)
        with mlflow.start_run(nested=True, run_name=f"per_thr{thr}_trial{trial.number}"):
            mlflow.set_tag("optuna_mode", "per-threshold")
            mlflow.set_tag("threshold", str(thr))
            mlflow.set_tag("trial_number", str(trial.number))
            _log_trial_params_to_mlflow(params)
            try:
                auc = _train_and_evaluate(params, bundle)
            except Exception as e:
                log(f"[trial {trial.number}] thr={thr} FAILED: {e}")
                mlflow.set_tag("status", "failed")
                mlflow.log_metric("auc", 0.5)
                return 0.5
            mlflow.log_metric("auc", auc)
        log(f"[trial {trial.number}] thr>={thr} AUC={auc:.5f}")
        return auc
    return objective


def make_shared_objective(bundles: Dict[int, dict], thresholds: List[int], seed: int):
    """Objective for shared-params optimization: maximize mean AUC across all thresholds."""
    def objective(trial):
        params = suggest_catboost_params(trial, seed)
        with mlflow.start_run(nested=True, run_name=f"shared_trial{trial.number}"):
            mlflow.set_tag("optuna_mode", "shared")
            mlflow.set_tag("trial_number", str(trial.number))
            _log_trial_params_to_mlflow(params)
            aucs = []
            for thr in thresholds:
                try:
                    auc = _train_and_evaluate(params, bundles[thr])
                    aucs.append(auc)
                    mlflow.log_metric(f"auc_ge{thr}", auc)
                except Exception as e:
                    log(f"[trial {trial.number}] thr={thr} FAILED: {e}")
                    aucs.append(0.5)
                    mlflow.log_metric(f"auc_ge{thr}", 0.5)
            mean_auc = float(np.mean(aucs))
            mlflow.log_metric("mean_auc", mean_auc)
        auc_str = " ".join(f"thr>={t}:{a:.4f}" for t, a in zip(thresholds, aucs))
        log(f"[trial {trial.number}] shared mean_AUC={mean_auc:.5f} ({auc_str})")
        return mean_auc
    return objective


# ------------------------------ final evaluation ------------------------------

def evaluate_params_all_thresholds(params_per_thr: Dict[int, dict],
                                   bundles: Dict[int, dict],
                                   thresholds: List[int],
                                   seed: int) -> dict:
    """Train all 4 thresholds with given params and compute per-threshold + multibin metrics."""
    results = {"per_threshold": {}, "multibin": {}}
    p_ge_eval = {}

    for thr in thresholds:
        p = params_per_thr.get(thr, params_per_thr.get(thresholds[0]))
        full_params = {**p, "random_seed": seed, "allow_writing_files": False, "verbose": 0}
        bundle = bundles[thr]
        cat_idx = bundle["cat_idx"]

        train_pool = Pool(bundle["X_train"], bundle["y_train"], cat_features=cat_idx)
        es_pool = Pool(bundle["X_es"], bundle["y_es"], cat_features=cat_idx)
        clf = CatBoostClassifier(**full_params)
        clf.fit(train_pool, eval_set=es_pool, use_best_model=True)

        calibrator = CalibratedClassifierCV(FrozenEstimator(clf), method="isotonic")
        calibrator.fit(bundle["X_cal"], bundle["y_cal"])

        p_eval = calibrator.predict_proba(bundle["X_eval"])[:, 1]
        auc = float(roc_auc_score(bundle["y_eval"], p_eval)) if len(np.unique(bundle["y_eval"])) > 1 else float("nan")
        p_ge_eval[thr] = p_eval
        results["per_threshold"][thr] = {"auc_eval": auc}

    # Multibin log loss
    if len(thresholds) == 4:
        P = ge_to_bins(p_ge_eval, thresholds)
        y_true_mc = bundles[thresholds[0]]["y_true_mc"]
        try:
            ll = float(log_loss(y_true_mc, P, labels=list(range(len(thresholds) + 1))))
        except Exception:
            ll = float("nan")
        results["multibin"]["log_loss"] = ll
        results["multibin"]["accuracy"] = float(np.mean(np.argmax(P, axis=1) == y_true_mc))

    return results


# ------------------------------ output generation ------------------------------

def generate_output_config_dep(original_cfg: dict, best_params: dict,
                               per_thr_params: Optional[Dict[int, dict]] = None) -> dict:
    """Generate a dep-mode config with optimized params (top-level keys)."""
    out = copy.deepcopy(original_cfg)
    # Write shared params at top level
    for key in ("iterations", "depth", "learning_rate", "l2_leaf_reg", "od_wait"):
        if key in best_params:
            out[key] = best_params[key]
    # Store per-threshold overrides
    if per_thr_params:
        out["per_threshold_catboost"] = {}
        for thr, params in per_thr_params.items():
            out["per_threshold_catboost"][str(thr)] = {
                k: params[k] for k in (
                    "iterations", "depth", "learning_rate", "l2_leaf_reg", "od_wait",
                    "bagging_temperature", "random_strength", "border_count",
                    "min_data_in_leaf", "bootstrap_type"
                ) if k in params
            }
            if "subsample" in params:
                out["per_threshold_catboost"][str(thr)]["subsample"] = params["subsample"]
    return out


def generate_output_config_arr(original_cfg: dict, best_params: dict,
                               per_thr_params: Optional[Dict[int, dict]] = None) -> dict:
    """Generate an arr-mode config with optimized params (nested catboost key)."""
    out = copy.deepcopy(original_cfg)
    cb = out.setdefault("catboost", {})
    for key in ("iterations", "depth", "learning_rate", "l2_leaf_reg", "od_wait"):
        if key in best_params:
            cb[key] = best_params[key]
    if per_thr_params:
        out["per_threshold_catboost"] = {}
        for thr, params in per_thr_params.items():
            out["per_threshold_catboost"][str(thr)] = {
                k: params[k] for k in (
                    "iterations", "depth", "learning_rate", "l2_leaf_reg", "od_wait",
                    "bagging_temperature", "random_strength", "border_count",
                    "min_data_in_leaf", "bootstrap_type"
                ) if k in params
            }
            if "subsample" in params:
                out["per_threshold_catboost"][str(thr)]["subsample"] = params["subsample"]
    return out


def _extract_catboost_params(trial_params: dict) -> dict:
    """Extract just the CatBoost-relevant params from an Optuna trial's params dict."""
    out = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "od_type": "Iter",
    }
    keys = [
        "depth", "learning_rate", "l2_leaf_reg", "iterations", "od_wait",
        "bagging_temperature", "random_strength", "border_count",
        "min_data_in_leaf", "bootstrap_type", "subsample",
    ]
    for k in keys:
        if k in trial_params:
            out[k] = trial_params[k]
    return out


# ------------------------------ main ------------------------------

def main():
    parser = argparse.ArgumentParser(description="CatBoost hyperparameter optimization for delay models")
    parser.add_argument("config", help="Path to training config JSON")
    parser.add_argument("--mode", choices=["dep", "arr"], default=None,
                        help="Model mode (auto-detected if omitted)")
    parser.add_argument("--thresholds", default="15,30,60,120",
                        help="Comma-separated thresholds to optimize (default: 15,30,60,120)")
    parser.add_argument("--n-trials", type=int, default=100,
                        help="Number of Optuna trials per threshold (default: 100)")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Max seconds per threshold optimization")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: {outdir}/hyperparam_search/)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed override")
    parser.add_argument("--max-train-rows", type=int, default=0,
                        help="Max training rows per threshold (0=no limit). Subsamples balanced data for faster trials.")
    parser.add_argument("--max-eval-rows", type=int, default=0,
                        help="Max eval/cal rows (0=no limit). Subsamples unbalanced data for faster eval.")
    args = parser.parse_args()

    cfg = _load_json(args.config)
    thresholds = [int(x) for x in args.thresholds.split(",")]

    # Auto-detect mode
    mode = args.mode
    if mode is None:
        mode = "arr" if "catboost" in cfg or "features" in cfg else "dep"
    log(f"Mode: {mode} | Thresholds: {thresholds} | Trials: {args.n_trials}")

    # --- MLflow tracking setup -------------------------------------------------
    init_mlflow("hyperparam-optimization")
    _parent_name = f"hpo_{mode}_{Path(args.config).stem}"
    mlflow.start_run(run_name=_parent_name)
    mlflow.set_tag("mode", mode)
    mlflow.set_tag("script", "hyperparam_optimize.py")
    mlflow.set_tag("config_path", str(args.config))
    _commit = git_commit()
    if _commit:
        mlflow.set_tag("git_commit", _commit)
    mlflow.log_params({
        "hpo.mode": mode,
        "hpo.n_trials": str(args.n_trials),
        "hpo.thresholds": ",".join(str(t) for t in thresholds),
        "hpo.timeout": str(args.timeout) if args.timeout is not None else "",
        "hpo.max_train_rows": str(args.max_train_rows),
        "hpo.max_eval_rows": str(args.max_eval_rows),
        "hpo.config": Path(args.config).name,
    })
    # --------------------------------------------------------------------------

    # Resolve seed
    if args.seed is not None:
        seed = args.seed
    elif mode == "dep":
        seed = int(cfg.get("random_seed", 42))
    else:
        seed = int((cfg.get("training") or {}).get("seed", 1337))

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        outdir_raw = cfg.get("outdir") or cfg.get("output_dir") or "data/models/hyperparam/"
        output_dir = _as_data_path(Path(outdir_raw)) / "hyperparam_search"
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Output directory: {output_dir}")

    # Load data
    log("=" * 60)
    log("Loading data...")
    if mode == "dep":
        bundles = load_data_dep(cfg, thresholds)
    else:
        bundles = load_data_arr(cfg, thresholds)
    log("Data loaded.")

    # Subsample for faster iteration
    if args.max_train_rows > 0 or args.max_eval_rows > 0:
        rng = np.random.default_rng(seed)
        for thr in thresholds:
            b = bundles[thr]
            if args.max_train_rows > 0 and len(b["X_train"]) > args.max_train_rows:
                idx = rng.choice(len(b["X_train"]), size=args.max_train_rows, replace=False)
                b["X_train"] = b["X_train"].iloc[idx].reset_index(drop=True)
                b["y_train"] = b["y_train"][idx]
                log(f"[subsample] thr={thr} train -> {args.max_train_rows:,} rows")
            if args.max_train_rows > 0 and len(b["X_es"]) > args.max_train_rows // 3:
                n_es = args.max_train_rows // 3
                idx = rng.choice(len(b["X_es"]), size=n_es, replace=False)
                b["X_es"] = b["X_es"].iloc[idx].reset_index(drop=True)
                b["y_es"] = b["y_es"][idx]
                log(f"[subsample] thr={thr} earlystop -> {n_es:,} rows")
            if args.max_eval_rows > 0:
                for key_x, key_y in [("X_cal", "y_cal"), ("X_eval", "y_eval")]:
                    if len(b[key_x]) > args.max_eval_rows:
                        idx = rng.choice(len(b[key_x]), size=args.max_eval_rows, replace=False)
                        b[key_x] = b[key_x].iloc[idx].reset_index(drop=True)
                        b[key_y] = b[key_y][idx]
                        log(f"[subsample] thr={thr} {key_x} -> {args.max_eval_rows:,} rows")
                if "y_true_mc" in b and len(b["y_true_mc"]) > args.max_eval_rows:
                    b["y_true_mc"] = b["y_true_mc"][:args.max_eval_rows]

    # ========== MODE A: Per-threshold optimization ==========
    log("=" * 60)
    log("MODE A: Per-threshold optimization")
    log("=" * 60)

    per_thr_studies = {}
    per_thr_best_params = {}

    for thr in thresholds:
        log(f"\n{'='*50}")
        log(f"Optimizing threshold >= {thr} min")
        log(f"{'='*50}")

        sampler = TPESampler(seed=seed + thr, n_startup_trials=20, multivariate=True)
        study = optuna.create_study(
            study_name=f"per_thr_{thr}",
            direction="maximize",
            sampler=sampler,
        )
        study.optimize(
            make_per_threshold_objective(thr, bundles[thr], seed),
            n_trials=args.n_trials,
            timeout=args.timeout,
        )
        per_thr_studies[thr] = study
        per_thr_best_params[thr] = _extract_catboost_params(study.best_params)
        log(f"[thr>={thr}] BEST AUC={study.best_value:.5f}")
        log(f"[thr>={thr}] BEST params: {json.dumps(study.best_params, indent=2)}")

        # Save per-threshold results
        thr_dir = output_dir / f"thr_{thr}"
        thr_dir.mkdir(parents=True, exist_ok=True)
        with open(thr_dir / "best_params.json", "w") as f:
            json.dump({"best_auc": study.best_value, "params": study.best_params}, f, indent=2)

        # Save all trials
        trials_data = []
        for t in study.trials:
            trials_data.append({
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "state": str(t.state),
            })
        with open(thr_dir / "all_trials.json", "w") as f:
            json.dump(trials_data, f, indent=2)

    # ========== MODE B: Shared params optimization ==========
    log("\n" + "=" * 60)
    log("MODE B: Shared params optimization")
    log("=" * 60)

    sampler_shared = TPESampler(seed=seed, n_startup_trials=20, multivariate=True)
    study_shared = optuna.create_study(
        study_name="shared_params",
        direction="maximize",
        sampler=sampler_shared,
    )
    study_shared.optimize(
        make_shared_objective(bundles, thresholds, seed),
        n_trials=args.n_trials,
        timeout=args.timeout,
    )
    shared_best_params = _extract_catboost_params(study_shared.best_params)
    log(f"[shared] BEST mean_AUC={study_shared.best_value:.5f}")
    log(f"[shared] BEST params: {json.dumps(study_shared.best_params, indent=2)}")

    shared_dir = output_dir / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)
    with open(shared_dir / "best_params.json", "w") as f:
        json.dump({"best_mean_auc": study_shared.best_value, "params": study_shared.best_params}, f, indent=2)

    trials_data = []
    for t in study_shared.trials:
        trials_data.append({
            "number": t.number,
            "value": t.value,
            "params": t.params,
            "state": str(t.state),
        })
    with open(shared_dir / "all_trials.json", "w") as f:
        json.dump(trials_data, f, indent=2)

    # ========== Final evaluation: baseline vs shared vs per-threshold ==========
    log("\n" + "=" * 60)
    log("FINAL COMPARISON")
    log("=" * 60)

    # Baseline params from original config
    if mode == "dep":
        baseline_params = {
            "loss_function": "Logloss", "eval_metric": "AUC", "od_type": "Iter",
            "iterations": int(cfg.get("iterations", 4000)),
            "depth": int(cfg.get("depth", 8)),
            "learning_rate": float(cfg.get("learning_rate", 0.03)),
            "l2_leaf_reg": float(cfg.get("l2_leaf_reg", 5.0)),
            "od_wait": int(cfg.get("od_wait", 200)),
        }
    else:
        cb_cfg = cfg.get("catboost") or {}
        baseline_params = {
            "loss_function": "Logloss", "eval_metric": "AUC", "od_type": "Iter",
            "iterations": int(cb_cfg.get("iterations", 1200)),
            "depth": int(cb_cfg.get("depth", 8)),
            "learning_rate": float(cb_cfg.get("learning_rate", 0.04)),
            "l2_leaf_reg": float(cb_cfg.get("l2_leaf_reg", 3.0)),
            "od_wait": int(cb_cfg.get("od_wait", 200)),
        }

    log("Evaluating BASELINE params...")
    baseline_per_thr = {thr: baseline_params for thr in thresholds}
    baseline_results = evaluate_params_all_thresholds(baseline_per_thr, bundles, thresholds, seed)

    log("Evaluating SHARED optimized params...")
    shared_per_thr = {thr: shared_best_params for thr in thresholds}
    shared_results = evaluate_params_all_thresholds(shared_per_thr, bundles, thresholds, seed)

    log("Evaluating PER-THRESHOLD optimized params...")
    per_thr_results = evaluate_params_all_thresholds(per_thr_best_params, bundles, thresholds, seed)

    # Print comparison table
    log("\n" + "=" * 80)
    log(f"{'':>12} | {'BASELINE':>12} | {'SHARED OPT':>12} | {'PER-THR OPT':>12}")
    log("-" * 80)
    for thr in thresholds:
        b_auc = baseline_results["per_threshold"][thr]["auc_eval"]
        s_auc = shared_results["per_threshold"][thr]["auc_eval"]
        p_auc = per_thr_results["per_threshold"][thr]["auc_eval"]
        log(f"  thr>={thr:>3} | AUC {b_auc:.5f} | AUC {s_auc:.5f} | AUC {p_auc:.5f}")
    log("-" * 80)

    if "log_loss" in baseline_results.get("multibin", {}):
        b_ll = baseline_results["multibin"]["log_loss"]
        s_ll = shared_results["multibin"]["log_loss"]
        p_ll = per_thr_results["multibin"]["log_loss"]
        b_acc = baseline_results["multibin"]["accuracy"]
        s_acc = shared_results["multibin"]["accuracy"]
        p_acc = per_thr_results["multibin"]["accuracy"]
        log(f"  {'multibin':>10} | LL {b_ll:.5f}  | LL {s_ll:.5f}  | LL {p_ll:.5f}")
        log(f"  {'':>10} | acc {b_acc:.4f}  | acc {s_acc:.4f}  | acc {p_acc:.4f}")
    log("=" * 80)

    # Save comparison summary
    comparison = {
        "baseline": {
            "params": baseline_params,
            "results": _jsonify(baseline_results),
        },
        "shared_optimized": {
            "params": _jsonify(shared_best_params),
            "results": _jsonify(shared_results),
        },
        "per_threshold_optimized": {
            "params": {str(k): _jsonify(v) for k, v in per_thr_best_params.items()},
            "results": _jsonify(per_thr_results),
        },
    }
    with open(output_dir / "optimization_summary.json", "w") as f:
        json.dump(comparison, f, indent=2)
    log(f"[SAVE] optimization_summary.json -> {output_dir / 'optimization_summary.json'}")

    # Generate output configs
    if mode == "dep":
        out_cfg_shared = generate_output_config_dep(cfg, shared_best_params)
        out_cfg_per_thr = generate_output_config_dep(cfg, shared_best_params, per_thr_best_params)
    else:
        out_cfg_shared = generate_output_config_arr(cfg, shared_best_params)
        out_cfg_per_thr = generate_output_config_arr(cfg, shared_best_params, per_thr_best_params)

    shared_cfg_path = output_dir / f"optimized_{mode}_train_config_shared.json"
    per_thr_cfg_path = output_dir / f"optimized_{mode}_train_config_per_thr.json"
    with open(shared_cfg_path, "w") as f:
        json.dump(out_cfg_shared, f, indent=2)
    with open(per_thr_cfg_path, "w") as f:
        json.dump(out_cfg_per_thr, f, indent=2)
    log(f"[SAVE] shared config -> {shared_cfg_path}")
    log(f"[SAVE] per-threshold config -> {per_thr_cfg_path}")

    # --- MLflow final summary on the parent run --------------------------------
    _final_metrics: Dict[str, float] = {}
    for thr in thresholds:
        _final_metrics[f"baseline_auc_ge{thr}"] = float(
            baseline_results["per_threshold"][thr]["auc_eval"]
        )
        _final_metrics[f"shared_opt_auc_ge{thr}"] = float(
            shared_results["per_threshold"][thr]["auc_eval"]
        )
        _final_metrics[f"per_thr_opt_auc_ge{thr}"] = float(
            per_thr_results["per_threshold"][thr]["auc_eval"]
        )
    if "log_loss" in baseline_results.get("multibin", {}):
        _final_metrics["baseline_multibin_logloss"] = float(baseline_results["multibin"]["log_loss"])
        _final_metrics["shared_opt_multibin_logloss"] = float(shared_results["multibin"]["log_loss"])
        _final_metrics["per_thr_opt_multibin_logloss"] = float(per_thr_results["multibin"]["log_loss"])
        _final_metrics["baseline_multibin_acc"] = float(baseline_results["multibin"]["accuracy"])
        _final_metrics["shared_opt_multibin_acc"] = float(shared_results["multibin"]["accuracy"])
        _final_metrics["per_thr_opt_multibin_acc"] = float(per_thr_results["multibin"]["accuracy"])
    _final_metrics["shared_best_mean_auc"] = float(study_shared.best_value)
    _clean_final = {k: (0.0 if (isinstance(v, float) and np.isnan(v)) else v) for k, v in _final_metrics.items()}
    mlflow.log_metrics(_clean_final)

    for _artifact in (
        output_dir / "optimization_summary.json",
        shared_cfg_path,
        per_thr_cfg_path,
    ):
        try:
            mlflow.log_artifact(str(_artifact))
        except Exception as _e:
            log(f"[mlflow][WARN] failed to log artifact {_artifact}: {_e}")
    mlflow.end_run()
    # --------------------------------------------------------------------------

    log("\n[OK] Hyperparameter optimization complete.")


def _jsonify(obj):
    """Make an object JSON-serializable."""
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


if __name__ == "__main__":
    main()
