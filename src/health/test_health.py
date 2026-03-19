#!/usr/bin/env python3
"""
src/health/test_health.py

Small test suite for the health-check and feature-report modules.

Run:
    python -m pytest src/health/test_health.py -v
    python -m pytest src/health/test_health.py -v -m "not network"   # skip API tests
"""

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
# Helpers
# ---------------------------------------------------------------------------

def _make_good_df(n: int = 100) -> pd.DataFrame:
    """Create a synthetic features DataFrame covering ALL registered features."""
    rng = np.random.RandomState(42)
    airports = ["DEN", "PHX", "BWI", "MDW", "BNA", "LAS", "OAK", "DAL"]
    data = {}

    for name, spec in _FEATURE_REGISTRY.items():
        if spec["type"] == "num":
            lo = max(spec["min"], -100)  # clamp to reasonable synthetic range
            hi = min(spec["max"], 1000)
            # keep values well within bounds
            margin = (hi - lo) * 0.1
            data[name] = rng.uniform(lo + margin, hi - margin, n)
        elif spec["type"] == "cat":
            if name in ("Origin", "Dest"):
                data[name] = rng.choice(airports, n)
            elif name == "Reporting_Airline":
                data[name] = "WN"
            elif name == "FlightDate":
                data[name] = pd.date_range("2024-01-01", periods=n, freq="h")
            elif name in ("Flight_Number_Reporting_Airline",):
                data[name] = rng.choice(["1234", "5678", "9012"], n)
            elif name == "Tail_Number":
                data[name] = rng.choice(["N12345", "N67890", "N11111"], n)
            elif name == "od_pair":
                data[name] = rng.choice(["DEN_PHX", "BWI_MDW", "BNA_LAS"], n)
            elif name == "DepTimeBlk":
                data[name] = rng.choice(["0600-0659", "0700-0759", "0800-0859"], n)
            elif name == "aircraft_type":
                data[name] = rng.choice(["B737", "B738", "7M8"], n)
            elif name == "Aircraft_Age_Bucket":
                data[name] = rng.choice(["0-5", "5-10", "10-15", "15+"], n)
            elif name in ("dep_dow", "sched_dep_hour", "sched_arr_hour", "flight_month"):
                data[name] = rng.choice(["1", "2", "3", "4", "5"], n)
            elif name in ("is_holiday", "is_spring_break", "has_recent_arrival_turn_5h", "is_first_leg_of_day"):
                data[name] = rng.choice([0, 1], n)
            elif "weathercode" in name:
                data[name] = rng.choice(["0", "1", "2", "3", "51", "61", "71"], n)
            else:
                data[name] = rng.choice(["A", "B", "C"], n)

    return pd.DataFrame(data)


def _make_bad_df(n: int = 100) -> pd.DataFrame:
    """Create a DataFrame with deliberate problems: out-of-range, high nulls."""
    df = _make_good_df(n)
    # temperature way too high (out of [180, 340] range)
    df["origin_temp_max_K"] = 999.0
    # negative delay (out of [0, 2000] range)
    df["DepDelayMinutes"] = -5.0
    # all nulls in a column
    df["origin_dep_precip_mm"] = np.nan
    # bad IATA code
    df.loc[0, "Origin"] = "ZZZZ"
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
# Feature validation tests (no network needed)
# ---------------------------------------------------------------------------

def test_good_features_pass():
    df = _make_good_df()
    result = validate_features(df)
    result.print_report()
    assert result.passed, result.summary


def test_bad_features_caught():
    df = _make_bad_df()
    result = validate_features(df)
    result.print_report()
    # Should have failures for out-of-range temp, negative delay, bad IATA
    fails = [i for i in result.items if i["status"] == "FAIL"]
    assert len(fails) >= 2, f"Expected at least 2 failures, got {len(fails)}: {fails}"


def test_empty_df_fails():
    df = pd.DataFrame()
    result = validate_features(df)
    assert not result.passed
    assert any("empty" in i["detail"].lower() for i in result.items)


def test_expected_columns_check():
    df = _make_good_df()
    result = validate_features(df, expected_columns=["Origin", "Dest", "MISSING_COL"])
    fails = [i for i in result.items if i["status"] == "FAIL" and "expected_columns" in i["name"]]
    assert len(fails) == 1


def test_null_threshold_configurable():
    df = _make_good_df()
    # Make one column ~60% null
    mask = np.random.RandomState(0).random(len(df)) < 0.6
    df.loc[mask, "Distance"] = np.nan

    # Default threshold is 0.50 => should warn
    result_default = validate_features(df, null_threshold=0.50)
    warns = [i for i in result_default.items if "null_fraction" in i["name"] and i["status"] == "WARN"]
    assert len(warns) == 1

    # Raise threshold to 0.80 => should pass
    result_lax = validate_features(df, null_threshold=0.80)
    warns_lax = [i for i in result_lax.items if "null_fraction" in i["name"] and i["status"] == "WARN"]
    assert len(warns_lax) == 0


# ---------------------------------------------------------------------------
# Holiday logic tests
# ---------------------------------------------------------------------------

def test_thanksgiving_2024():
    """Thanksgiving 2024 = Nov 28 (4th Thursday of November)."""
    # Import the function from features_dep
    from src.fetch_prune.features_dep import thanksgiving_day
    assert thanksgiving_day(2024).month == 11
    assert thanksgiving_day(2024).day == 28


# ---------------------------------------------------------------------------
# Feature report tests
# ---------------------------------------------------------------------------

def test_feature_report_structure():
    df = _make_good_df()
    report = generate_feature_report(df)

    assert "global" in report
    assert "features" in report
    assert "validation" in report
    assert "status" in report

    assert report["global"]["row_count"] == len(df)
    assert report["global"]["column_count"] == len(df.columns)
    assert report["status"] == "PASS"


def test_feature_report_bad_data():
    df = _make_bad_df()
    report = generate_feature_report(df)
    assert report["status"] == "FAIL"


def test_feature_report_per_feature_stats():
    df = _make_good_df(50)
    report = generate_feature_report(df)

    # Numeric feature should have min/max/mean
    temp_info = report["features"]["origin_temp_max_K"]
    assert "min" in temp_info
    assert "max" in temp_info
    assert "mean" in temp_info
    assert temp_info["null_pct"] == 0.0

    # Categorical feature should have unique_count
    origin_info = report["features"]["Origin"]
    assert "unique_count" in origin_info
    assert "top_values" in origin_info
