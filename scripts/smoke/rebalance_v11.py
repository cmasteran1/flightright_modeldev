"""
Rebalance a v11 blueprint at a new pos_frac WITHOUT re-running feature engineering.

Reads the existing unbalanced features parquet (whatever is at
`output_features_unbalanced_path`), redoes the eval split per `feature_balance.eval`,
and rewrites the balanced per-threshold parquets at the pos_frac specified in the
blueprint.

Use when you've already run `prepare_dataset.py + features_{dep,arr}.py` once at one
pos_frac and want to produce additional balanced parquets at other pos_fracs for the
same airline / direction / year slice. Saves ~70% of feature-gen wall-clock on a
3-pos_frac pilot.

Usage:
  python scripts/smoke/rebalance_v11.py data/blueprint_dep_WN_v11_2025_pf030.json

Determines direction (dep vs arr) from the blueprint's `target` field. Supports both.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = (REPO_ROOT.parent / "flightrightdata").resolve()

# Import the project's balancer so the semantics are identical to
# features_dep.py / features_arr.py.
sys.path.insert(0, str(REPO_ROOT / "src" / "fetch_prune"))
from features_dep import balance_to_pos_frac  # noqa: E402


def _abspath(p: str, *, base: str) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    if base == "repo":
        return (REPO_ROOT / pp).resolve()
    if base == "data":
        return (DATA_ROOT / pp).resolve()
    raise ValueError(f"base must be 'repo' or 'data', got {base!r}")


def _load_unbalanced(cfg: dict, target: str) -> pd.DataFrame:
    """features_dep.py resolves `output_features_unbalanced_path` against DATA_ROOT,
    features_arr.py resolves against REPO_ROOT. Handle both."""
    raw = cfg.get("output_features_unbalanced_path")
    if not raw:
        raise KeyError("blueprint missing output_features_unbalanced_path")
    base = "repo" if target == "arr" else "data"
    path = _abspath(raw, base=base)
    if not path.exists():
        raise FileNotFoundError(
            f"Unbalanced features parquet not found at {path}\n"
            f"Run `features_{target}.py {cfg.get('_source_config', '<blueprint>')}` first "
            f"to produce it, then rebalance."
        )
    print(f"[INFO] Loading unbalanced features: {path}")
    df = pd.read_parquet(path)
    print(f"[INFO] Loaded rows={len(df):,} cols={len(df.columns)}")
    return df


def _eval_split(df: pd.DataFrame, fb: dict) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Split `df` into train_pool and eval_df according to fb['eval'].
    If eval disabled, returns (df, None) and no eval parquet is written."""
    eval_cfg = fb.get("eval") or {}
    eval_enabled = bool(eval_cfg.get("enabled", True))
    if not eval_enabled:
        return df, None

    by = (eval_cfg.get("by") or "last_days").lower()
    if by == "random":
        eval_frac = float(eval_cfg.get("eval_frac", 0.15))
        eval_seed = int(eval_cfg.get("seed", 42))
        eval_idx = df.sample(frac=eval_frac, random_state=eval_seed).index
        eval_mask = df.index.isin(eval_idx)
        eval_df = df[eval_mask].copy()
        train_pool = df[~eval_mask].copy()
        print(
            f"[INFO] eval random frac={eval_frac} seed={eval_seed} "
            f"eval_rows={len(eval_df):,} train_pool_rows={len(train_pool):,}"
        )
        return train_pool, eval_df
    if by == "last_days":
        last_days = int(eval_cfg.get("last_days", 90))
        fd = pd.to_datetime(df["FlightDate"], errors="coerce").dt.normalize()
        cutoff = fd.max() - pd.Timedelta(days=last_days)
        eval_mask = fd > cutoff
        eval_df = df[eval_mask].copy()
        train_pool = df[~eval_mask].copy()
        print(
            f"[INFO] eval last_days={last_days} cutoff>{cutoff.date()} "
            f"eval_rows={len(eval_df):,} train_pool_rows={len(train_pool):,}"
        )
        return train_pool, eval_df
    raise ValueError(f"feature_balance.eval.by must be 'random' or 'last_days', got {by!r}")


def main(cfg_path: str) -> None:
    blueprint = Path(cfg_path).resolve()
    cfg = json.loads(blueprint.read_text())
    target = str(cfg.get("target", "dep")).strip().lower()
    if target not in ("dep", "arr"):
        raise ValueError(f"blueprint target must be 'dep' or 'arr', got {target!r}")

    fb = cfg.get("feature_balance") or {}
    if not fb or not fb.get("enabled", True):
        raise ValueError("feature_balance is missing or disabled in the blueprint")
    out_tmpl = fb.get("output_template")
    if not out_tmpl:
        raise ValueError("feature_balance.output_template is required")

    pos_frac = float(fb.get("pos_frac", 0.50))
    seed = int(fb.get("seed", 42))
    max_rows = fb.get("max_rows")
    max_rows = int(max_rows) if max_rows is not None else None

    balance_thresholds = [int(x) for x in (fb.get("thresholds") or [15, 30, 60, 120])]

    df = _load_unbalanced(cfg, target)
    train_pool, eval_df = _eval_split(df, fb)

    # Write eval parquet if configured.
    eval_cfg = fb.get("eval") or {}
    eval_out = eval_cfg.get("output_path")
    if eval_df is not None and eval_out:
        base = "repo" if target == "arr" else "data"
        eval_path = _abspath(eval_out, base=base)
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        eval_df.to_parquet(eval_path, index=False)
        print(f"[OK] wrote eval unbalanced -> {eval_path}")

    base = "repo" if target == "arr" else "data"
    label_prefix = "y_arr_ge" if target == "arr" else "y_dep_ge"

    for thr in balance_thresholds:
        label_col = f"{label_prefix}{thr}"
        if label_col not in train_pool.columns:
            print(f"[WARN] {label_col} not in features; skipping thr={thr}")
            continue
        bal_df = balance_to_pos_frac(
            train_pool,
            label_col=label_col,
            pos_frac=pos_frac,
            seed=seed,
            max_rows=max_rows,
        )
        out_rel = out_tmpl.format(thr=thr)
        out_path = _abspath(out_rel, base=base)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        bal_df.to_parquet(out_path, index=False)
        print(f"[OK] wrote balanced {target} train thr={thr} pos_frac={pos_frac} -> {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/smoke/rebalance_v11.py <blueprint.json>")
        sys.exit(1)
    main(sys.argv[1])
