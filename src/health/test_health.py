#!/usr/bin/env python3
"""
src/health/test_health.py

Test suite for the health-check and feature-report modules.

Run:
    python -m pytest src/health/test_health.py -v
    python -m pytest src/health/test_health.py -v -m "not network"   # skip API tests
    python -m pytest src/health/test_health.py -v -m "not network and not data"  # skip API + data tests
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

try:
    from src.health.health_checks import (
        CheckResult,
        _FEATURE_REGISTRY,
        check_bts_api,
        check_open_meteo_api,
        validate_features,
    )
    from src.health.feature_report import generate_feature_report
except ModuleNotFoundError:
    from health_checks import (
        CheckResult,
        _FEATURE_REGISTRY,
        check_bts_api,
        check_open_meteo_api,
        validate_features,
    )
    from feature_report import generate_feature_report


# ---------------------------------------------------------------------------
# Real data paths — relative to repo root
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parents[2] / ".." / "flightrightdata" / "data"
_PROCESSED = _DATA_DIR / "processed"
_MODELS = _DATA_DIR / "models"

# All 8 feature parquets from the current v7g training run
_REAL_FEATURE_PARQUETS = {
    "dep_WN": _PROCESSED / "features_dep_WN_100_v7g_unbalanced.parquet",
    "dep_AA": _PROCESSED / "features_dep_AA_100_v7g_unbalanced.parquet",
    "dep_DL": _PROCESSED / "features_dep_DL_100_v7g_unbalanced.parquet",
    "dep_UA": _PROCESSED / "features_dep_UA_100_v7g_unbalanced.parquet",
    "arr_WN": _PROCESSED / "features_arr_WN_100_v7g_unbalanced.parquet",
    "arr_AA": _PROCESSED / "features_arr_AA_100_v7g_unbalanced.parquet",
    "arr_DL": _PROCESSED / "features_arr_DL_100_v7g_unbalanced.parquet",
    "arr_UA": _PROCESSED / "features_arr_UA_100_v7g_unbalanced.parquet",
}

_REAL_BUNDLES = {
    "dep_WN": _MODELS / "dep_WN_100_v7g" / "dep_delay_bins_bundle_WN_100_v7g.joblib",
    "dep_AA": _MODELS / "dep_AA_100_v7g" / "dep_delay_bins_bundle_AA_100_v7g.joblib",
    "dep_DL": _MODELS / "dep_DL_100_v7g" / "dep_delay_bins_bundle_DL_100_v7g.joblib",
    "dep_UA": _MODELS / "dep_UA_100_v7g" / "dep_delay_bins_bundle_UA_100_v7g.joblib",
    "arr_WN": _MODELS / "arr_WN_100_v7g" / "arr_delay_bins_bundle_WN_100_v7g.joblib",
    "arr_AA": _MODELS / "arr_AA_100_v7g" / "arr_delay_bins_bundle_AA_100_v7g.joblib",
    "arr_DL": _MODELS / "arr_DL_100_v7g" / "arr_delay_bins_bundle_DL_100_v7g.joblib",
    "arr_UA": _MODELS / "arr_UA_100_v7g" / "arr_delay_bins_bundle_UA_100_v7g.joblib",
}

_HAS_REAL_DATA = any(p.exists() for p in _REAL_FEATURE_PARQUETS.values())


def _available_feature_parquets():
    """Yield (name, path) for all v7g feature parquets that exist on disk."""
    for name, path in _REAL_FEATURE_PARQUETS.items():
        if path.exists():
            yield name, path


def _load_sample(path: Path, n: int = 50_000) -> pd.DataFrame:
    """Load a random sample from a parquet to keep tests fast."""
    df = pd.read_parquet(path)
    if len(df) > n:
        df = df.sample(n=n, random_state=42)
    return df


# ---------------------------------------------------------------------------
# API tests (require network)
# ---------------------------------------------------------------------------

@pytest.mark.network
def test_bts_api_reachable():
    result = check_bts_api(timeout=20)
    result.print_report()
    assert result.passed, result.summary


@pytest.mark.network
def test_open_meteo_api_reachable():
    result = check_open_meteo_api(timeout=20)
    result.print_report()
    assert result.passed, result.summary


# ---------------------------------------------------------------------------
# Unit-level validation logic tests (no data files needed)
# ---------------------------------------------------------------------------

def test_empty_df_fails():
    df = pd.DataFrame()
    result = validate_features(df)
    assert not result.passed
    assert any("empty" in i["detail"].lower() for i in result.items)


def test_expected_columns_check():
    """validate_features should FAIL when expected columns are missing."""
    df = pd.DataFrame({"Origin": ["DEN", "PHX"], "Dest": ["BWI", "MDW"]})
    result = validate_features(df, expected_columns=["Origin", "Dest", "MISSING_COL"])
    fails = [i for i in result.items if i["status"] == "FAIL" and "expected_columns" in i["name"]]
    assert len(fails) == 1


# ---------------------------------------------------------------------------
# Real-data feature validation — runs on every v7g parquet on disk
# ---------------------------------------------------------------------------

@pytest.mark.data
@pytest.mark.skipif(not _HAS_REAL_DATA, reason="No v7g feature parquets on disk")
@pytest.mark.parametrize("name,path", list(_available_feature_parquets()), ids=lambda x: x if isinstance(x, str) else x.stem)
def test_real_features_pass_validation(name, path):
    """Real feature parquets should pass full validation (no FAILs)."""
    df = _load_sample(path)
    result = validate_features(df)
    result.print_report()
    fails = [i for i in result.items if i["status"] == "FAIL"]
    assert len(fails) == 0, f"{name}: {[f['name'] + ': ' + f['detail'] for f in fails]}"


@pytest.mark.data
@pytest.mark.skipif(not _HAS_REAL_DATA, reason="No v7g feature parquets on disk")
@pytest.mark.parametrize("name,path", list(_available_feature_parquets()), ids=lambda x: x if isinstance(x, str) else x.stem)
def test_real_multiday_magnitude(name, path):
    """Multi-day mean/median P95 should be within sane limits on real data."""
    df = _load_sample(path)
    result = validate_features(df)

    magnitude_items = [i for i in result.items if "multiday_magnitude" in i["name"]]
    assert len(magnitude_items) >= 1, f"{name}: multiday_magnitude check did not run"

    for item in magnitude_items:
        assert item["status"] != "WARN", (
            f"{name}: multi-day magnitude warning — likely computation bug: {item['detail']}"
        )


@pytest.mark.data
@pytest.mark.skipif(not _HAS_REAL_DATA, reason="No v7g feature parquets on disk")
@pytest.mark.parametrize("name,path", list(_available_feature_parquets()), ids=lambda x: x if isinstance(x, str) else x.stem)
def test_real_mean_median_p99_spot_check(name, path):
    """Spot-check P99 of every delay mean/median column on real data.

    Multi-day delay means should have P99 < 200, medians P99 < 150.
    These are looser than the P95 health-check limits to catch only gross errors.
    Non-delay features (airtime, wo_slip) are excluded — different scale.
    """
    df = _load_sample(path)
    violations = []
    _EXEMPT = ("airtime_", "wo_slip_")

    for col in df.columns:
        if any(col.startswith(pfx) for pfx in _EXEMPT):
            continue
        is_mean = "mean" in col and "last1" not in col and "mean_last1" not in col
        is_median = "median" in col and "last1" not in col
        is_multiday = any(k in col for k in ("last7", "last14", "_7d_", "_14d_"))
        if not ((is_mean or is_median) and is_multiday):
            continue

        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) < 50:
            continue

        p99 = float(s.quantile(0.99))
        limit = 150.0 if is_median else 200.0

        if p99 > limit:
            violations.append(f"{col}: P99={p99:.1f} > {limit:.0f}")

    assert len(violations) == 0, f"{name} has {len(violations)} P99 violations:\n" + "\n".join(violations)


# ---------------------------------------------------------------------------
# Holiday logic tests
# ---------------------------------------------------------------------------

def test_thanksgiving_2024():
    """Thanksgiving 2024 = Nov 28 (4th Thursday of November)."""
    from src.fetch_prune.features_dep import thanksgiving_day
    assert thanksgiving_day(2024).month == 11
    assert thanksgiving_day(2024).day == 28


# ---------------------------------------------------------------------------
# Feature report tests — use real data when available, otherwise minimal df
# ---------------------------------------------------------------------------

def _get_report_df():
    """Return a real sample if available, else a minimal synthetic df."""
    for path in _REAL_FEATURE_PARQUETS.values():
        if path.exists():
            return _load_sample(path, n=5000)
    # Fallback: minimal df with a few registered features
    return pd.DataFrame({
        "Origin": ["DEN"] * 50,
        "Dest": ["PHX"] * 50,
        "origin_temp_max_K": np.random.uniform(200, 320, 50),
        "Distance": np.random.uniform(100, 3000, 50),
    })


def test_feature_report_structure():
    df = _get_report_df()
    report = generate_feature_report(df)

    assert "global" in report
    assert "features" in report
    assert "validation" in report
    assert "status" in report

    assert report["global"]["row_count"] == len(df)
    assert report["global"]["column_count"] == len(df.columns)
    # Real data may have warnings (e.g. null columns), so just check it ran
    assert report["status"] in ("PASS", "WARN")


def test_feature_report_per_feature_stats():
    df = _get_report_df()
    report = generate_feature_report(df)

    # Find a numeric feature present in both the df and registry
    numeric_col = None
    for col in df.columns:
        if col in _FEATURE_REGISTRY and _FEATURE_REGISTRY[col]["type"] == "num":
            if col in report["features"]:
                numeric_col = col
                break
    assert numeric_col is not None, "No registered numeric feature found in report"

    info = report["features"][numeric_col]
    assert "min" in info
    assert "max" in info
    assert "mean" in info

    # Find a categorical feature
    cat_col = None
    for col in df.columns:
        if col in _FEATURE_REGISTRY and _FEATURE_REGISTRY[col]["type"] == "cat":
            if col in report["features"]:
                cat_col = col
                break
    if cat_col:
        cat_info = report["features"][cat_col]
        assert "unique_count" in cat_info
