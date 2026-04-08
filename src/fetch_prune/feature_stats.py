#!/usr/bin/env python3
"""
src/fetch_prune/feature_stats.py

Compute and persist basic descriptive statistics for an unbalanced features
parquet. Called at the end of features_dep.py / features_arr.py main() so
that every training run has a companion stats file to sanity-check feature
distributions without having to reload the parquet.

The output JSON sits next to the parquet with an ``_stats.json`` suffix:

    data/processed/features_dep_WN_v10_unbalanced.parquet
    data/processed/features_dep_WN_v10_unbalanced_stats.json
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


# Percentiles to compute for every numeric column. Anchored on the ones the
# user asked for (P01, P50, P99) plus the standard quantiles around them so
# the stats file is useful for tail inspection without being huge.
_PERCENTILES: tuple = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)

# Numeric column types accepted by the stats path. Everything else (object,
# string, category, datetime, bool) goes through the categorical path.
_NUMERIC_DTYPES = ("int", "float", "uint")


def _json_safe(value: Any) -> Any:
    """Convert numpy / pandas scalars to JSON-safe Python primitives.
    NaN and +/-Inf become None (JSON has no native representation)."""
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        f = float(value)
        if np.isnan(f) or np.isinf(f):
            return None
        return round(f, 6)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return str(value)
    return value


def _numeric_stats(s: pd.Series) -> Dict[str, Any]:
    non_null = s.dropna()
    info: Dict[str, Any] = {
        "dtype": str(s.dtype),
        "count": int(len(s)),
        "non_null_count": int(len(non_null)),
        "null_count": int(len(s) - len(non_null)),
        "null_pct": round(float(s.isna().mean()) * 100, 4),
    }
    if len(non_null) == 0:
        for k in ("min", "max", "mean", "median", "std"):
            info[k] = None
        info["percentiles"] = {f"p{int(p*100):02d}": None for p in _PERCENTILES}
        return info

    info["min"] = _json_safe(non_null.min())
    info["max"] = _json_safe(non_null.max())
    info["mean"] = _json_safe(non_null.mean())
    info["median"] = _json_safe(non_null.median())
    info["std"] = _json_safe(non_null.std(ddof=0))

    quantiles = non_null.quantile(list(_PERCENTILES))
    info["percentiles"] = {
        f"p{int(p*100):02d}": _json_safe(quantiles.loc[p]) for p in _PERCENTILES
    }
    return info


def _categorical_stats(s: pd.Series, top_k: int = 5) -> Dict[str, Any]:
    non_null = s.dropna()
    info: Dict[str, Any] = {
        "dtype": str(s.dtype),
        "count": int(len(s)),
        "non_null_count": int(len(non_null)),
        "null_count": int(len(s) - len(non_null)),
        "null_pct": round(float(s.isna().mean()) * 100, 4),
        "unique_count": int(non_null.nunique()),
    }
    if len(non_null) == 0:
        info["top_values"] = {}
        return info

    top = non_null.value_counts().head(top_k)
    info["top_values"] = {str(k): int(v) for k, v in top.items()}
    return info


def compute_feature_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Return a dict of per-column descriptive stats for `df`.
    Numeric columns get count/null/min/max/mean/median/std + P01/P05/P25/P50/P75/P95/P99.
    Non-numeric columns get count/null/unique + top-5 value counts.
    """
    stats: Dict[str, Dict[str, Any]] = {}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            stats[col] = _numeric_stats(s)
        else:
            stats[col] = _categorical_stats(s)

    global_info: Dict[str, Any] = {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    if "FlightDate" in df.columns:
        dates = pd.to_datetime(df["FlightDate"], errors="coerce").dropna()
        if len(dates) > 0:
            global_info["date_min"] = str(dates.min().date())
            global_info["date_max"] = str(dates.max().date())
    if "Reporting_Airline" in df.columns:
        airlines = sorted(df["Reporting_Airline"].dropna().astype(str).unique().tolist())
        global_info["airlines"] = airlines

    return {"global": global_info, "features": stats}


def write_feature_stats_json(df: pd.DataFrame, parquet_path: Path) -> Path:
    """Compute stats from `df` and write them to `{parquet_stem}_stats.json`
    alongside the parquet. Returns the output path."""
    parquet_path = Path(parquet_path)
    out_path = parquet_path.with_name(parquet_path.stem + "_stats.json")
    report = compute_feature_stats(df)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return out_path
