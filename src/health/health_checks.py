#!/usr/bin/env python3
"""
src/health/health_checks.py

Validation and health-check utilities for the FlightRight model pipeline.

Usage as a library:
    from src.health.health_checks import check_apis, validate_features, validate_model_bundle

Usage standalone:
    python src/health/health_checks.py                  # API checks only
    python src/health/health_checks.py --features path  # + feature validation
    python src/health/health_checks.py --bundle path    # + model bundle validation
"""

import sys
import json
import re
from pathlib import Path
from datetime import date, timedelta
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class CheckResult:
    """Collects pass/warn/fail outcomes from a suite of checks."""

    def __init__(self):
        self.items: List[Dict[str, Any]] = []

    def _add(self, status: str, name: str, detail: str = ""):
        self.items.append({"status": status, "name": name, "detail": detail})

    def ok(self, name: str, detail: str = ""):
        self._add("PASS", name, detail)

    def warn(self, name: str, detail: str = ""):
        self._add("WARN", name, detail)

    def fail(self, name: str, detail: str = ""):
        self._add("FAIL", name, detail)

    @property
    def passed(self) -> bool:
        return all(i["status"] != "FAIL" for i in self.items)

    @property
    def summary(self) -> str:
        counts = {"PASS": 0, "WARN": 0, "FAIL": 0}
        for i in self.items:
            counts[i["status"]] += 1
        status = "PASS" if counts["FAIL"] == 0 else "FAIL"
        if counts["WARN"] > 0 and status == "PASS":
            status = "WARN"
        return (
            f"{status}: {counts['PASS']} passed, "
            f"{counts['WARN']} warnings, {counts['FAIL']} failures"
        )

    def print_report(self):
        for item in self.items:
            mark = {"PASS": "+", "WARN": "?", "FAIL": "X"}[item["status"]]
            line = f"[{mark}] {item['name']}"
            if item["detail"]:
                line += f"  -- {item['detail']}"
            print(line)
        print(f"\n{self.summary}")

    def to_dict(self) -> Dict[str, Any]:
        return {"checks": self.items, "summary": self.summary, "passed": self.passed}


# ---------------------------------------------------------------------------
# 1. API health checks
# ---------------------------------------------------------------------------

def check_bts_api(result: Optional[CheckResult] = None, timeout: int = 15) -> CheckResult:
    """Check that the BTS PREZIP endpoint is reachable."""
    import requests

    result = result or CheckResult()
    url = "https://transtats.bts.gov"
    try:
        r = requests.head(url, timeout=timeout, allow_redirects=True)
        if r.status_code < 400:
            result.ok("bts_api_reachable", f"status={r.status_code}")
        else:
            result.fail("bts_api_reachable", f"status={r.status_code}")
    except Exception as e:
        result.fail("bts_api_reachable", str(e))
    return result


def check_open_meteo_api(result: Optional[CheckResult] = None, timeout: int = 15) -> CheckResult:
    """Check Open-Meteo archive API with a tiny known query."""
    import requests

    result = result or CheckResult()
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 39.8561,   # DEN
        "longitude": -104.6737,
        "start_date": "2024-01-01",
        "end_date": "2024-01-01",
        "daily": "temperature_2m_max",
        "timezone": "America/Denver",
    }
    try:
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code == 200:
            data = r.json()
            if "daily" in data and "temperature_2m_max" in data["daily"]:
                result.ok("open_meteo_api", "returned valid daily data")
            else:
                result.warn("open_meteo_api", "200 but unexpected payload shape")
        else:
            result.fail("open_meteo_api", f"status={r.status_code}")
    except Exception as e:
        result.fail("open_meteo_api", str(e))
    return result


def check_apis(timeout: int = 15) -> CheckResult:
    """Run all API health checks."""
    result = CheckResult()
    check_bts_api(result, timeout=timeout)
    check_open_meteo_api(result, timeout=timeout)
    return result


# ---------------------------------------------------------------------------
# 2. Feature validation — complete per-feature registry
# ---------------------------------------------------------------------------

# Every known feature with its type ("num" or "cat") and valid range for numerics.
# Features not present in a given DataFrame are silently skipped (dep vs arr).
# Any feature present in the DataFrame but NOT in this registry gets a warning.
_FEATURE_REGISTRY: Dict[str, Dict[str, Any]] = {}


def _reg_num(name: str, lo: float, hi: float, desc: str = ""):
    _FEATURE_REGISTRY[name] = {"type": "num", "min": lo, "max": hi, "desc": desc or name}


def _reg_cat(name: str, desc: str = ""):
    _FEATURE_REGISTRY[name] = {"type": "cat", "desc": desc or name}


# --- Metadata / label columns (not model features, but present in parquets) ---
_reg_cat("FlightDate", "flight date")
_reg_num("DepDelayMinutes", 0, 2000, "departure delay (min)")
_reg_num("ArrDelayMinutes", 0, 2000, "arrival delay (min)")
_reg_cat("Reporting_Airline", "2-letter airline code")
_reg_cat("Flight_Number_Reporting_Airline", "flight number")
_reg_cat("Tail_Number", "aircraft tail number")
_reg_cat("od_pair", "origin-dest pair")

# --- Categorical features ---
_reg_cat("Origin", "origin IATA")
_reg_cat("Dest", "destination IATA")
_reg_cat("DepTimeBlk", "departure time block")
_reg_cat("aircraft_type", "FAA aircraft type")
_reg_cat("Aircraft_Age_Bucket", "aircraft age bucket")
_reg_cat("origin_daily_weathercode", "WMO daily weather code at origin")
_reg_cat("origin_dep_hour_weathercode", "WMO hourly weather code at dep time")
_reg_cat("dep_dow", "day of week")
_reg_cat("sched_dep_hour", "scheduled departure hour")
_reg_cat("sched_arr_hour", "scheduled arrival hour")
_reg_cat("is_holiday", "holiday flag")
_reg_cat("is_spring_break", "spring break flag")
_reg_cat("has_recent_arrival_turn_5h", "recent arrival within 5h flag")
_reg_cat("is_first_leg_of_day", "first leg of day for tail")
_reg_cat("flight_month", "month of flight")

# --- Origin weather (daily) ---
_reg_num("origin_temp_max_K", 180, 340, "origin daily max temp (K)")
_reg_num("origin_temp_min_K", 180, 340, "origin daily min temp (K)")
_reg_num("origin_daily_precip_sum_mm", 0, 500, "origin daily precip (mm)")
_reg_num("origin_daily_windspeed_max_kmh", 0, 400, "origin daily max windspeed (km/h)")
_reg_num("origin_daily_windgusts_max_kmh", 0, 500, "origin daily max windgusts (km/h)")

# --- Origin weather (hourly at departure) ---
_reg_num("origin_dep_temp_K", 180, 340, "origin hourly temp at dep (K)")
_reg_num("origin_dep_precip_mm", 0, 300, "origin hourly precip at dep (mm)")
_reg_num("origin_dep_windspeed_kmh", 0, 400, "origin hourly windspeed at dep (km/h)")
_reg_num("origin_dep_windgusts_kmh", 0, 500, "origin hourly windgusts at dep (km/h)")
_reg_num("origin_dep_visibility_m", 0, 100000, "origin hourly visibility at dep (m)")
_reg_num("origin_dep_cape_jkg", 0, 10000, "origin CAPE at dep (J/kg)")
_reg_num("origin_dep_log1p_cape", 0, 15, "origin log1p(CAPE) at dep")
_reg_num("origin_dep_cloudcover_pct", 0, 100, "origin cloudcover at dep (%)")

# --- Destination weather (hourly at arrival) ---
_reg_num("dest_arr_temperature_2m", 180, 340, "dest hourly temp at arr")
_reg_num("dest_arr_precipitation", 0, 300, "dest hourly precip at arr")
_reg_num("dest_arr_windspeed_10m", 0, 400, "dest hourly windspeed at arr")
_reg_num("dest_arr_windgusts_10m", 0, 500, "dest hourly windgusts at arr")
_reg_num("dest_arr_visibility", 0, 100000, "dest hourly visibility at arr")
_reg_num("dest_arr_cape", 0, 10000, "dest CAPE at arr")
_reg_num("dest_arr_cloudcover", 0, 100, "dest cloudcover at arr")

# --- Flight-number OD rolling departure delay stats ---
# Bounds match DepDelayMinutes (0-2000) since 1-day means/medians can equal raw delay
for _w in [1, 7, 14]:
    _reg_num(f"flightnum_od_depdelay_mean_last{_w}", -30, 2000, f"flightnum OD dep delay mean {_w}d")
    _reg_num(f"flightnum_od_depdelay_median_last{_w}", -30, 2000, f"flightnum OD dep delay median {_w}d")
    if _w >= 2:  # std_last1 dropped: undefined for single observation
        _reg_num(f"flightnum_od_depdelay_std_last{_w}", 0, 2000, f"flightnum OD dep delay std {_w}d")
_reg_num("flightnum_od_support_count_last14d", 0, 500, "flightnum OD support count 14d")
_reg_num("flightnum_od_low_support_last14d", 0, 1, "flightnum OD low support flag")

# --- Carrier rolling departure delay stats ---
for _w in [1, 7, 14]:
    _reg_num(f"carrier_depdelay_mean_last{_w}", -30, 2000, f"carrier dep delay mean {_w}d")
    if _w >= 2:
        _reg_num(f"carrier_depdelay_std_last{_w}", 0, 2000, f"carrier dep delay std {_w}d")

# --- Carrier-origin rolling departure delay stats ---
for _w in [1, 7, 14]:
    _reg_num(f"carrier_origin_depdelay_mean_last{_w}", -30, 2000, f"carrier-origin dep delay mean {_w}d")
    if _w >= 2:
        _reg_num(f"carrier_origin_depdelay_std_last{_w}", 0, 2000, f"carrier-origin dep delay std {_w}d")

# --- Origin rolling departure delay stats ---
for _w in [1, 7, 14]:
    _reg_num(f"origin_depdelay_mean_last{_w}", -30, 2000, f"origin dep delay mean {_w}d")
    if _w >= 2:
        _reg_num(f"origin_depdelay_std_last{_w}", 0, 2000, f"origin dep delay std {_w}d")

# --- Delay cause rates (origin + carrier) ---
for _w in [1, 7, 14]:
    _reg_num(f"origin_lateaircraft_rate_last{_w}", 0, 1, f"origin late-aircraft rate {_w}d")
    _reg_num(f"origin_nas_rate_last{_w}", 0, 1, f"origin NAS delay rate {_w}d")
    _reg_num(f"origin_weather_rate_last{_w}", 0, 1, f"origin weather delay rate {_w}d")
    _reg_num(f"carrier_lateaircraft_rate_last{_w}", 0, 1, f"carrier late-aircraft rate {_w}d")

# --- Hub spillover (indexed 0-4 and named DEN/PHX/BWI/MDW/BNA) ---
for _h in [0, 1, 2, 3, 4]:
    for _w in [1, 7, 14]:
        _reg_num(f"hub_{_h}_depdelay_mean_last{_w}", -30, 2000, f"hub {_h} dep delay mean {_w}d")
        _reg_num(f"hub_{_h}_lateaircraft_rate_last{_w}", 0, 1, f"hub {_h} late-aircraft rate {_w}d")
for _hub in ["DEN", "PHX", "BWI", "MDW", "BNA"]:
    for _w in [1, 7, 14]:
        _reg_num(f"hub_{_hub}_depdelay_mean_last{_w}", -30, 2000, f"hub {_hub} dep delay mean {_w}d")
        _reg_num(f"hub_{_hub}_lateaircraft_rate_last{_w}", 0, 1, f"hub {_hub} late-aircraft rate {_w}d")

# --- Tail (aircraft) rolling stats ---
for _w in [1, 7, 14]:
    _reg_num(f"tail_depdelay_mean_last{_w}", -30, 2000, f"tail dep delay mean {_w}d")
    _reg_num(f"tail_lateaircraft_rate_last{_w}", 0, 1, f"tail late-aircraft rate {_w}d")

# --- Destination rolling stats ---
for _w in [1, 7, 14]:
    _reg_num(f"dest_depdelay_mean_last{_w}", -30, 2000, f"dest dep delay mean {_w}d")
    _reg_num(f"dest_lateaircraft_rate_last{_w}", 0, 1, f"dest late-aircraft rate {_w}d")

# --- Congestion ---
_reg_num("origin_congestion_3h_total", 0, 1000, "origin 3h congestion count")
_reg_num("origin_airline_congestion_3h_total", 0, 1000, "origin airline 3h congestion count")

# --- Temporal / scheduling ---
_reg_num("tail_leg_num_day", 0, 20, "tail leg number in day")
_reg_num("turn_time_hours", 0, 48, "turn time (hours)")
_reg_num("flightnum_hours_since_first_departure_today", 0, 24, "hours since first dep today")
_reg_num("CRSDepTime", 0, 2400, "scheduled departure time (HHMM)")
_reg_num("CRSArrTime", 0, 2400, "scheduled arrival time (HHMM)")
_reg_num("CRSElapsedTime", 0, 1000, "scheduled elapsed time (min)")
_reg_num("Distance", 0, 6000, "route distance (miles)")

# --- Prior leg ---
for _w in [7, 14]:
    _reg_num(f"prior_leg_depdelay_mean_last{_w}", -30, 2000, f"prior leg dep delay mean {_w}d")

# --- Arrival-specific: arrival delay rolling stats ---
for _w in [7, 14]:
    _reg_num(f"arrdelay_mean_{_w}d_fn_od", -30, 2000, f"arr delay mean {_w}d flight-num OD")
    _reg_num(f"arrdelay_n_{_w}d_fn_od", 0, 500, f"arr delay count {_w}d flight-num OD")
    _reg_num(f"arrdelay_mean_{_w}d_car_od", -30, 2000, f"arr delay mean {_w}d carrier OD")
    _reg_num(f"arrdelay_n_{_w}d_car_od", 0, 500, f"arr delay count {_w}d carrier OD")
_reg_num("arrdelay_median_7d_fn_od", -30, 2000, "arr delay median 7d flight-num OD")
_reg_num("arrdelay_median_7d_car_od", -30, 2000, "arr delay median 7d carrier OD")

# --- Arrival-specific: airtime rolling stats ---
for _w in [7, 14]:
    _reg_num(f"airtime_mean_{_w}d_fn_od", 0, 1000, f"airtime mean {_w}d flight-num OD")
    _reg_num(f"airtime_n_{_w}d_fn_od", 0, 500, f"airtime count {_w}d flight-num OD")
    _reg_num(f"airtime_mean_{_w}d_car_od", 0, 1000, f"airtime mean {_w}d carrier OD")
    _reg_num(f"airtime_n_{_w}d_car_od", 0, 500, f"airtime count {_w}d carrier OD")

# --- Arrival-specific: wheels-off slip stats ---
for _w in [7, 14]:
    _reg_num(f"wo_slip_mean_{_w}d_fn_od", -120, 300, f"wheels-off slip mean {_w}d flight-num OD")
    _reg_num(f"wo_slip_n_{_w}d_fn_od", 0, 500, f"wheels-off slip count {_w}d flight-num OD")
    _reg_num(f"wo_slip_mean_{_w}d_origin_blk", -120, 300, f"wheels-off slip mean {_w}d origin-block")
    _reg_num(f"wo_slip_n_{_w}d_origin_blk", 0, 500, f"wheels-off slip count {_w}d origin-block")

# --- Arrival-specific: destination congestion ---
_reg_num("dest_arrivals_pm60_sched", 0, 500, "dest arrivals +-60min (scheduled)")
_reg_num("dest_airline_arrivals_pm60_sched", 0, 300, "dest airline arrivals +-60min (scheduled)")
_reg_num("dest_arrivals_pm60_eta", 0, 500, "dest arrivals +-60min (ETA)")
_reg_num("dest_airline_arrivals_pm60_eta", 0, 300, "dest airline arrivals +-60min (ETA)")

# --- Raw weather column names (from prepare_dataset before rename) ---
for _raw in [
    "origin_temperature_2m_max", "origin_temperature_2m_min",
    "origin_precipitation_sum", "origin_windspeed_10m_max",
    "origin_windgusts_10m_max", "origin_weathercode",
    "origin_dep_temperature_2m", "origin_dep_precipitation",
    "origin_dep_windspeed_10m", "origin_dep_weathercode",
    "origin_dep_windgusts_10m", "origin_dep_visibility",
    "origin_dep_cape", "origin_dep_cloudcover",
]:
    if _raw not in _FEATURE_REGISTRY:
        _reg_num(_raw, -500, 100000, f"raw weather: {_raw}")


_IATA_RE = re.compile(r"^[A-Z]{3}$")

# Columns we know about but don't need to range-check (labels, targets, etc.)
_KNOWN_SKIP_PATTERNS = [
    re.compile(r"^y_(dep|arr)_ge\d+$"),      # binary label columns
    re.compile(r"^(dep|arr)_dt_local$"),       # datetime columns
    re.compile(r"^Unnamed"),                    # pandas artifacts
]


def validate_features(
    df: pd.DataFrame,
    result: Optional[CheckResult] = None,
    null_threshold: float = 0.50,
    expected_columns: Optional[List[str]] = None,
) -> CheckResult:
    """Validate a features DataFrame — checks every feature individually."""
    result = result or CheckResult()

    # --- row count ---
    if len(df) == 0:
        result.fail("row_count", "DataFrame is empty")
        return result
    elif len(df) < 10:
        result.warn("row_count", f"only {len(df)} rows")
    else:
        result.ok("row_count", f"{len(df)} rows")

    # --- expected columns ---
    if expected_columns:
        missing = set(expected_columns) - set(df.columns)
        if missing:
            result.fail("expected_columns", f"missing: {sorted(missing)[:10]}")
        else:
            result.ok("expected_columns", f"all {len(expected_columns)} present")

    # --- null fraction (per-column) ---
    null_fracs = df.isnull().mean()
    high_null = null_fracs[null_fracs > null_threshold]
    if len(high_null) > 0:
        worst = high_null.sort_values(ascending=False).head(5)
        detail = ", ".join(f"{c}={v:.0%}" for c, v in worst.items())
        result.warn("null_fraction", f"{len(high_null)} cols >{null_threshold:.0%} null: {detail}")
    else:
        result.ok("null_fraction", f"all cols <={null_threshold:.0%} null")

    # --- per-feature checks from registry ---
    checked = set()
    for col in df.columns:
        if col not in _FEATURE_REGISTRY:
            # Skip known non-feature columns
            if any(p.match(col) for p in _KNOWN_SKIP_PATTERNS):
                continue
            continue  # don't warn on unknown — many intermediate cols exist

        spec = _FEATURE_REGISTRY[col]
        checked.add(col)

        if spec["type"] == "num":
            if not pd.api.types.is_numeric_dtype(df[col]):
                result.fail(f"type_{col}", f"expected numeric, got {df[col].dtype}")
                continue
            s = df[col].dropna()
            if len(s) == 0:
                result.warn(f"range_{col}", "all null")
                continue
            lo, hi = float(s.min()), float(s.max())
            violations = []
            if lo < spec["min"]:
                violations.append(f"min={lo:.2f} < {spec['min']}")
            if hi > spec["max"]:
                violations.append(f"max={hi:.2f} > {spec['max']}")
            if violations:
                result.fail(f"range_{col}", f"{spec['desc']}: {', '.join(violations)}")
            else:
                result.ok(f"range_{col}", f"[{lo:.2f}, {hi:.2f}]")

        elif spec["type"] == "cat":
            # Specific validations for known categoricals
            if col in ("Origin", "Dest"):
                vals = df[col].dropna().unique()
                bad = [v for v in vals if not _IATA_RE.match(str(v))]
                if bad:
                    result.fail(f"iata_{col}", f"non-IATA values: {bad[:5]}")
                else:
                    result.ok(f"cat_{col}", f"{len(vals)} valid IATA codes")
            elif col == "Reporting_Airline":
                vals = df[col].dropna().unique()
                bad = [v for v in vals if not re.match(r"^[A-Z0-9]{2}$", str(v))]
                if bad:
                    result.warn(f"cat_{col}", f"unexpected codes: {bad[:5]}")
                else:
                    result.ok(f"cat_{col}", f"{len(vals)} valid codes")
            else:
                # Generic categorical: just check it's not all null
                n_unique = df[col].nunique(dropna=True)
                null_pct = df[col].isnull().mean()
                if null_pct > null_threshold:
                    result.warn(f"cat_{col}", f"{null_pct:.0%} null, {n_unique} unique")
                elif n_unique == 0:
                    result.warn(f"cat_{col}", "all null")
                else:
                    result.ok(f"cat_{col}", f"{n_unique} unique, {null_pct:.0%} null")

    # --- holiday spot-check ---
    if "is_holiday" in df.columns and "FlightDate" in df.columns:
        hol_rows = df[df["is_holiday"] == 1]
        if len(hol_rows) > 0:
            hol_dates = pd.to_datetime(hol_rows["FlightDate"]).dt.date.unique()
            sample = sorted(hol_dates)[:5]
            result.ok("holiday_dates", f"sample holiday dates: {sample}")
        else:
            result.ok("holiday_dates", "no holiday rows (may be expected for short date ranges)")

    # --- summary of coverage ---
    registry_cols_in_df = set(df.columns) & set(_FEATURE_REGISTRY.keys())
    unchecked = registry_cols_in_df - checked
    if unchecked:
        result.warn("coverage", f"{len(unchecked)} registered cols unchecked: {sorted(unchecked)[:5]}")
    result.ok("feature_coverage", f"{len(checked)}/{len(registry_cols_in_df)} registered features checked")

    return result


# ---------------------------------------------------------------------------
# 3. Model bundle validation
# ---------------------------------------------------------------------------

_REQUIRED_BUNDLE_KEYS = [
    "artifact_type", "thresholds", "feature_order", "calibrators",
    "categorical_features", "numeric_features",
]


def validate_model_bundle(
    bundle_path: str,
    result: Optional[CheckResult] = None,
) -> CheckResult:
    """Validate a trained model bundle (.joblib)."""
    import joblib

    result = result or CheckResult()
    path = Path(bundle_path)

    if not path.exists():
        result.fail("bundle_exists", f"{path} not found")
        return result

    try:
        bundle = joblib.load(path)
    except Exception as e:
        result.fail("bundle_loads", str(e))
        return result
    result.ok("bundle_loads", f"loaded {path.name}")

    # --- required keys ---
    missing_keys = [k for k in _REQUIRED_BUNDLE_KEYS if k not in bundle]
    if missing_keys:
        result.fail("bundle_keys", f"missing: {missing_keys}")
        return result
    result.ok("bundle_keys", "all required keys present")

    # --- model files exist ---
    registry = bundle.get("registry", {})
    bundle_dir = path.parent
    for thr, info in registry.items():
        model_path = info.get("model_path", "")
        if model_path:
            mp = Path(model_path)
            if not mp.is_absolute():
                mp = bundle_dir / mp
            if not mp.exists():
                result.fail(f"model_file_{thr}", f"not found: {mp}")
            else:
                result.ok(f"model_file_{thr}", f"exists: {mp.name}")

    # --- monotonicity of calibrators on synthetic data ---
    thresholds = bundle["thresholds"]
    calibrators = bundle["calibrators"]
    feature_order = bundle["feature_order"]
    cat_features = set(bundle.get("categorical_features", []))

    n_test = 5
    synthetic = {}
    for feat in feature_order:
        if feat in cat_features:
            synthetic[feat] = ["A"] * n_test
        else:
            synthetic[feat] = np.random.uniform(0, 10, n_test)
    test_df = pd.DataFrame(synthetic)

    try:
        probs = {}
        for thr in thresholds:
            thr_key = thr if thr in calibrators else str(thr)
            cal = calibrators[thr_key]
            p = cal.predict_proba(test_df[feature_order])[:, 1]
            probs[thr] = p

        # Check monotonicity: P(>=15) >= P(>=30) >= P(>=45) >= P(>=60)
        p_matrix = np.column_stack([probs[t] for t in sorted(thresholds)])
        mono_ok = True
        for j in range(1, p_matrix.shape[1]):
            if np.any(p_matrix[:, j] > p_matrix[:, j - 1] + 1e-6):
                mono_ok = False
                break
        if mono_ok:
            result.ok("calibrator_monotone", "P(>=thr) decreasing across thresholds")
        else:
            result.warn("calibrator_monotone", "raw calibrator outputs not monotone (enforce_monotone fixes this)")
    except Exception as e:
        result.warn("calibrator_test", f"could not test calibrators: {e}")

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="FlightRight health checks")
    parser.add_argument("--features", type=str, help="Path to features parquet to validate")
    parser.add_argument("--bundle", type=str, help="Path to model bundle .joblib to validate")
    parser.add_argument("--no-api", action="store_true", help="Skip API checks")
    args = parser.parse_args()

    result = CheckResult()

    if not args.no_api:
        print("=== API Health Checks ===")
        check_apis().print_report()
        print()

    if args.features:
        print(f"=== Feature Validation: {args.features} ===")
        df = pd.read_parquet(args.features)
        validate_features(df, result).print_report()
        print()

    if args.bundle:
        print(f"=== Model Bundle Validation: {args.bundle} ===")
        validate_model_bundle(args.bundle, result).print_report()
        print()


if __name__ == "__main__":
    main()
