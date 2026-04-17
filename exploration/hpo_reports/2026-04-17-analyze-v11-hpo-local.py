"""
Analyze the v11 local HPO sweep results after `hyperparam_optimize.py` finishes.

Reads two sources:
  1. The sweep's own hpo_summary.json / best_params files under
     {outdir}/hyperparam_search/
  2. MLflow runs under experiment "hyperparam-optimization"
     (../flightrightdata/mlruns/)

Prints a concise summary:
  - Mode A (per-threshold) best AUC and params per threshold
  - Mode B (shared-params) best mean AUC and params
  - Convergence diagnostics (how many trials before best found)
  - Comparison vs v10/v11 baseline defaults (depth=8, lr=0.03, l2=3.0)
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

DATA = Path("/Users/connermasteran/software/mpqc/flightrightdata")
HPO_OUT = DATA / "data/models/dep_WN_100_v11_hpo/hyperparam_search"
MLRUNS = DATA / "mlruns"
BASELINE = {"depth": 8, "learning_rate": 0.03, "l2_leaf_reg": 3.0,
            "iterations": 6000, "od_wait": 200}


def _try_read_json(p: Path) -> Optional[Dict[str, Any]]:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def print_sweep_artifacts() -> None:
    print("=" * 72)
    print("SWEEP ARTIFACTS (from hyperparam_search/ outputs)")
    print("=" * 72)
    if not HPO_OUT.exists():
        print(f"[WARN] {HPO_OUT} not found — sweep may not have completed")
        return

    for candidate in ["hpo_summary.json", "best_params_per_threshold.json",
                      "best_params_shared.json"]:
        payload = _try_read_json(HPO_OUT / candidate)
        print()
        print(f"--- {candidate} ---")
        if payload is None:
            print("  (missing)")
            continue
        print(json.dumps(payload, indent=2)[:2000])


def fetch_mlflow_summary() -> None:
    print()
    print("=" * 72)
    print("MLFLOW SUMMARY (experiment='hyperparam-optimization')")
    print("=" * 72)
    uri = MLRUNS.resolve().as_uri()
    mlflow.set_tracking_uri(uri)
    client = MlflowClient()
    exp = client.get_experiment_by_name("hyperparam-optimization")
    if exp is None:
        print("[WARN] no such experiment")
        return
    # Fetch nested child runs (each trial is a nested run) for this parent
    # via tag.config_path filter.
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string="tags.config_path LIKE '%dep_train_WN_100_v11_hpo.json%'",
        order_by=["attribute.start_time DESC"],
        max_results=500,
    )
    if not runs:
        print("  (no runs yet)")
        return

    # Separate parent vs child by run_name pattern
    parents = [r for r in runs if r.info.run_name and r.info.run_name.startswith("hpo_")]
    children = [r for r in runs if r.info.run_name and r.info.run_name.startswith(("per_thr", "shared_trial"))]

    print(f"  Total runs: {len(runs)}   parents: {len(parents)}   child trials: {len(children)}")
    print()

    # Per-threshold analysis
    print("--- Mode A: Per-threshold best trials ---")
    for thr in [15, 30, 60, 120]:
        thr_children = [r for r in children
                        if r.info.run_name and r.info.run_name.startswith(f"per_thr{thr}_")]
        if not thr_children:
            continue
        # AUC metric is called "auc"
        with_auc = [(r, r.data.metrics.get("auc")) for r in thr_children]
        with_auc = [(r, a) for r, a in with_auc if a is not None]
        if not with_auc:
            continue
        with_auc.sort(key=lambda x: x[1], reverse=True)
        best, best_auc = with_auc[0]
        print(f"  thr={thr:>3}:  best AUC = {best_auc:.5f}   ({len(with_auc)} trials)")
        # Show best-trial params
        best_params = {k: best.data.params.get(k) for k in
                       ["depth", "learning_rate", "l2_leaf_reg", "iterations",
                        "od_wait", "bootstrap_type", "random_strength", "min_data_in_leaf"]}
        print(f"          params: {best_params}")
        # AUC distribution
        aucs = [a for _, a in with_auc]
        print(f"          AUC range across trials: "
              f"min={min(aucs):.5f}  median={pd.Series(aucs).median():.5f}  max={max(aucs):.5f}")

    # Shared-params analysis
    print()
    print("--- Mode B: Shared-params best trials ---")
    shared_children = [r for r in children
                       if r.info.run_name and r.info.run_name.startswith("shared_trial")]
    if shared_children:
        with_m = [(r, r.data.metrics.get("mean_auc")) for r in shared_children]
        with_m = [(r, m) for r, m in with_m if m is not None]
        with_m.sort(key=lambda x: x[1], reverse=True)
        if with_m:
            best, best_m = with_m[0]
            print(f"  best mean_auc = {best_m:.5f}   ({len(with_m)} trials)")
            best_params = {k: best.data.params.get(k) for k in
                           ["depth", "learning_rate", "l2_leaf_reg", "iterations",
                            "od_wait", "bootstrap_type", "random_strength",
                            "min_data_in_leaf"]}
            print(f"  best params: {best_params}")
            # Per-threshold AUCs at best trial
            for thr in [15, 30, 60, 120]:
                a = best.data.metrics.get(f"auc_ge{thr}")
                if a is not None:
                    print(f"    auc_ge{thr}: {a:.5f}")
            aucs = [m for _, m in with_m]
            print(f"  mean_auc range: min={min(aucs):.5f}  "
                  f"median={pd.Series(aucs).median():.5f}  max={max(aucs):.5f}")
    else:
        print("  (no shared_trial runs yet)")


def print_vs_baseline() -> None:
    print()
    print("=" * 72)
    print("BASELINE REFERENCE (v10/v11 default hyperparameters)")
    print("=" * 72)
    print(json.dumps(BASELINE, indent=2))
    print()
    print("  The best trial params above should be compared against these defaults.")
    print("  A promising sweep produces best params materially different from baseline")
    print("  AND a best AUC > baseline AUC (which would require a separate baseline run).")


if __name__ == "__main__":
    print_sweep_artifacts()
    fetch_mlflow_summary()
    print_vs_baseline()
