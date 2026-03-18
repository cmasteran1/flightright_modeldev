#!/usr/bin/env python3
"""
src/health/feature_report.py

Standalone feature health report generator.

Reads a features parquet and writes a JSON report with per-feature
statistics, global metadata, and PASS/WARN/FAIL flags.

Usage:
    python src/health/feature_report.py <features.parquet> [--output report.json]
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Support running both as `python src/health/feature_report.py` and as an import
try:
    from src.health.health_checks import validate_features, CheckResult
except ModuleNotFoundError:
    from health_checks import validate_features, CheckResult


def _safe_float(v) -> Optional[float]:
    """Convert to JSON-safe float (handle NaN/Inf)."""
    if v is None:
        return None
    f = float(v)
    if np.isnan(f) or np.isinf(f):
        return None
    return round(f, 6)


def generate_feature_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate a comprehensive feature health report as a dict."""
    report: Dict[str, Any] = {}

    # --- global metadata ---
    global_info: Dict[str, Any] = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }

    if "FlightDate" in df.columns:
        dates = pd.to_datetime(df["FlightDate"], errors="coerce").dropna()
        if len(dates) > 0:
            global_info["date_min"] = str(dates.min().date())
            global_info["date_max"] = str(dates.max().date())

    if "Reporting_Airline" in df.columns:
        global_info["airlines"] = sorted(df["Reporting_Airline"].dropna().unique().tolist())

    for col in ["Origin", "Dest"]:
        if col in df.columns:
            vals = sorted(df[col].dropna().unique().tolist())
            global_info[f"{col.lower()}_count"] = len(vals)
            global_info[f"{col.lower()}_sample"] = vals[:10]

    report["global"] = global_info

    # --- per-feature stats ---
    features: Dict[str, Dict[str, Any]] = {}
    for col in df.columns:
        s = df[col]
        info: Dict[str, Any] = {
            "dtype": str(s.dtype),
            "null_count": int(s.isnull().sum()),
            "null_pct": round(float(s.isnull().mean()) * 100, 2),
        }

        if pd.api.types.is_numeric_dtype(s):
            desc = s.describe()
            info["min"] = _safe_float(desc.get("min"))
            info["max"] = _safe_float(desc.get("max"))
            info["mean"] = _safe_float(desc.get("mean"))
            info["std"] = _safe_float(desc.get("std"))
            info["median"] = _safe_float(s.median())
        else:
            nunique = s.nunique()
            info["unique_count"] = int(nunique)
            top_vals = s.value_counts().head(5)
            info["top_values"] = {str(k): int(v) for k, v in top_vals.items()}

        features[col] = info

    report["features"] = features

    # --- validation flags ---
    check_result = validate_features(df)
    report["validation"] = check_result.to_dict()

    # --- overall status ---
    has_fail = any(i["status"] == "FAIL" for i in check_result.items)
    has_warn = any(i["status"] == "WARN" for i in check_result.items)
    if has_fail:
        report["status"] = "FAIL"
    elif has_warn:
        report["status"] = "WARN"
    else:
        report["status"] = "PASS"

    return report


def main():
    parser = argparse.ArgumentParser(description="Generate feature health report")
    parser.add_argument("features_path", help="Path to features parquet file")
    parser.add_argument("--output", "-o", help="Output JSON path (default: stdout)")
    args = parser.parse_args()

    path = Path(args.features_path)
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(path)
    report = generate_feature_report(df)

    out = json.dumps(report, indent=2, default=str)

    if args.output:
        Path(args.output).write_text(out, encoding="utf-8")
        print(f"Report written to {args.output}")
    else:
        print(out)


if __name__ == "__main__":
    main()
