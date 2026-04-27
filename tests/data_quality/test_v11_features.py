"""
v11 feature validation tests.

These tests check that the v11 smoke feature parquet has the expected schema,
value ranges, and invariants. Run with:

    .venv/bin/pytest tests/data_quality/test_v11_features.py -v

Assumes the smoke pipeline has already been run:

    python src/fetch_prune/prepare_dataset.py data/blueprint_dep_WN_v11_smoke.json
    python src/fetch_prune/features_dep.py data/blueprint_dep_WN_v11_smoke.json

Skips cleanly when the parquet file is missing.
"""
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import pytest

DATA = Path(__file__).parent.parent.parent.parent / "flightrightdata"
SMOKE_PARQUET = DATA / "data/processed/features_dep_WN_v11_smoke_unbalanced.parquet"


def _resolve_features_parquet() -> Path:
    """Honor FLIGHTRIGHT_FEATURES_PARQUET to retarget at any features_dep parquet."""
    env = os.environ.get("FLIGHTRIGHT_FEATURES_PARQUET")
    return Path(env) if env else SMOKE_PARQUET


@pytest.fixture(scope="module")
def smoke_df() -> pd.DataFrame:
    p = _resolve_features_parquet()
    if not p.exists():
        pytest.skip(f"features parquet not found at {p}")
    return pd.read_parquet(p)


# ---------------- SCHEMA ----------------

REQUIRED_V11_LABELS = ["y_dep_ge15", "y_dep_ge30", "y_dep_ge60", "y_dep_ge120"]
REQUIRED_V11_FEATURES = ["flightnum_od_otp_rate_last14d"]
FORBIDDEN_COLS = [
    "y_dep_ge45",                 # old v10 bucket
    "origin_nasdelay_rate_last1d", # dropped in v11
]


@pytest.mark.parametrize("col", REQUIRED_V11_LABELS + REQUIRED_V11_FEATURES)
def test_v11_required_column_present(col, smoke_df):
    assert col in smoke_df.columns, f"Required v11 column {col!r} missing from smoke parquet"


@pytest.mark.parametrize("col", FORBIDDEN_COLS)
def test_v11_forbidden_column_absent(col, smoke_df):
    assert col not in smoke_df.columns, (
        f"Forbidden column {col!r} present in v11 smoke parquet "
        f"(should be dropped per v11 manifest)"
    )


# ---------------- RANGE ----------------

RATE_COLUMNS_0_1 = {
    "flightnum_od_otp_rate_last14d": (0.0, 1.0),
    "origin_lateaircraft_rate_last1": (0.0, 1.0),
    "origin_lateaircraft_rate_last7": (0.0, 1.0),
    "carrier_lateaircraft_rate_last1": (0.0, 1.0),
    "carrier_lateaircraft_rate_last7": (0.0, 1.0),
    "hub_max_lateaircraft_last1": (0.0, 1.0),
}


@pytest.mark.parametrize("col,bounds", RATE_COLUMNS_0_1.items())
def test_rate_column_in_bounds(col, bounds, smoke_df):
    if col not in smoke_df.columns:
        pytest.skip(f"{col} not present in smoke parquet")
    lo, hi = bounds
    s = smoke_df[col]
    bad = smoke_df[(s < lo) | (s > hi)]
    assert bad.empty, (
        f"{col} has {len(bad)} rows outside [{lo}, {hi}]. "
        f"First offender: {bad.iloc[0][[col]].to_dict()}"
    )


def test_label_base_rates_monotone(smoke_df):
    """Stricter delay thresholds should have lower base rates."""
    rates = [smoke_df[col].mean() for col in REQUIRED_V11_LABELS]
    for i in range(len(rates) - 1):
        assert rates[i] >= rates[i + 1], (
            f"Non-monotone label base rates: "
            f"{REQUIRED_V11_LABELS[i]}={rates[i]:.4f} < "
            f"{REQUIRED_V11_LABELS[i+1]}={rates[i+1]:.4f}"
        )


def test_label_nesting_ge120_implies_ge60(smoke_df):
    """If a flight is delayed >=120, it must also be delayed >=60."""
    violate = smoke_df[(smoke_df["y_dep_ge120"] == 1) & (smoke_df["y_dep_ge60"] == 0)]
    assert violate.empty, (
        f"{len(violate)} rows violate y_dep_ge120 => y_dep_ge60. "
        f"First: {violate.iloc[0][REQUIRED_V11_LABELS].to_dict()}"
    )


# ---------------- DISTRIBUTION ----------------

def test_otp_rate_coverage_reasonable(smoke_df):
    """OTP rate should be non-null for most rows (some new flight identities
    lack 14d history)."""
    col = "flightnum_od_otp_rate_last14d"
    if col not in smoke_df.columns:
        pytest.skip(f"{col} not present")
    coverage = smoke_df[col].notna().mean()
    assert 0.85 <= coverage <= 0.99, (
        f"{col} coverage={coverage:.1%} outside expected [85%, 99%]"
    )


def test_otp_rate_distribution_plausible(smoke_df):
    """OTP rate median should reflect 'most flights mostly on-time' — typically
    0.5–0.9 over a 14d window for scheduled US carriers."""
    col = "flightnum_od_otp_rate_last14d"
    if col not in smoke_df.columns:
        pytest.skip(f"{col} not present")
    median = smoke_df[col].median()
    assert 0.4 <= median <= 0.95, (
        f"{col} median={median:.4f} outside plausible [0.4, 0.95]"
    )


# ---------------- LOCAL TIMEZONE INVARIANTS ----------------
# Regression guard for the mixed-tz parquet-coercion bug in
# src/fetch_prune/prepare_dataset.py::add_timezone_local_times. Building
# dep_dt_local as a Python list of per-row tz-aware datetimes with different
# ZoneInfos forces pyarrow to pick one tz for the whole column (observed:
# America/Chicago), so .dt.hour on the round-tripped column returns Chicago
# wall-clock instead of airport-local. Pre-fix, this silently corrupted
# sched_dep_hour, is_peak_hour, and dep_dow for every non-Central airport
# (PT +2, MT +1, ET -1 vs true local). BTS's raw CRSDepTime is already
# airport-local HHMM, so it is the ground-truth anchor for these invariants.


def _crs_hour(df: pd.DataFrame) -> pd.Series:
    return (pd.to_numeric(df["CRSDepTime"], errors="coerce") // 100).astype("Int64")


def test_sched_dep_hour_is_airport_local(smoke_df):
    """sched_dep_hour must equal CRSDepTime // 100 (BTS HHMM is airport-local)."""
    for c in ("sched_dep_hour", "CRSDepTime"):
        if c not in smoke_df.columns:
            pytest.skip(f"{c} missing")
    sdh = pd.to_numeric(smoke_df["sched_dep_hour"], errors="coerce").astype("Int64")
    crsh = _crs_hour(smoke_df)
    mask = sdh.notna() & crsh.notna()
    bad = smoke_df[mask & (sdh != crsh)]
    assert bad.empty, (
        f"sched_dep_hour != CRSDepTime//100 in {len(bad)}/{int(mask.sum())} rows — "
        f"mixed-tz parquet-coercion bug regressed. First offender: "
        f"Origin={bad.iloc[0]['Origin']} CRSDepTime={bad.iloc[0]['CRSDepTime']} "
        f"sched_dep_hour={bad.iloc[0]['sched_dep_hour']}"
    )


def test_sched_dep_hour_in_range(smoke_df):
    if "sched_dep_hour" not in smoke_df.columns:
        pytest.skip("sched_dep_hour missing")
    sdh = pd.to_numeric(smoke_df["sched_dep_hour"], errors="coerce")
    bad = smoke_df[sdh.notna() & ((sdh < 0) | (sdh > 23))]
    assert bad.empty, (
        f"sched_dep_hour outside [0, 23] in {len(bad)} rows. "
        f"First offender: {bad.iloc[0][['Origin', 'CRSDepTime', 'sched_dep_hour']].to_dict()}"
    )


def test_is_peak_hour_consistent_with_sched_dep_hour(smoke_df):
    """is_peak_hour := 1 iff 16 <= sched_dep_hour <= 20 (v10+ feature spec)."""
    for c in ("is_peak_hour", "sched_dep_hour"):
        if c not in smoke_df.columns:
            pytest.skip(f"{c} missing")
    sdh = pd.to_numeric(smoke_df["sched_dep_hour"], errors="coerce").astype("Int64")
    ip = pd.to_numeric(smoke_df["is_peak_hour"], errors="coerce").astype("Int64")
    expected = ((sdh >= 16) & (sdh <= 20)).astype("Int64")
    mask = sdh.notna() & ip.notna()
    bad = smoke_df[mask & (ip != expected)]
    assert bad.empty, (
        f"is_peak_hour inconsistent with sched_dep_hour in {len(bad)} rows. "
        f"First offender: Origin={bad.iloc[0]['Origin']} "
        f"sched_dep_hour={bad.iloc[0]['sched_dep_hour']} "
        f"is_peak_hour={bad.iloc[0]['is_peak_hour']}"
    )


def test_dep_dow_matches_flightdate_weekday(smoke_df):
    """dep_dow is the airport-local day-of-week of FlightDate. A Chicago-coerced
    column silently shifts ET late-night flights back one day; CT 0; MT/PT typically
    unaffected because the CT→local delta is small at their wall-clock evening."""
    for c in ("dep_dow", "FlightDate"):
        if c not in smoke_df.columns:
            pytest.skip(f"{c} missing")
    dow = pd.to_numeric(smoke_df["dep_dow"], errors="coerce").astype("Int64")
    assert dow.dropna().between(0, 6).all(), "dep_dow has values outside [0, 6]"
    fd_dow = pd.to_datetime(smoke_df["FlightDate"], errors="coerce").dt.dayofweek.astype("Int64")
    crs = pd.to_numeric(smoke_df.get("CRSDepTime"), errors="coerce")
    # BTS FlightDate is the local departure date; a CRSDepTime < 2400 cannot cross
    # midnight in local time, so dep_dow must equal FlightDate's weekday.
    mask = dow.notna() & fd_dow.notna() & (crs < 2400)
    bad = smoke_df[mask & (dow != fd_dow)]
    assert bad.empty, (
        f"dep_dow != FlightDate weekday in {len(bad)}/{int(mask.sum())} rows — "
        f"tz-coercion bug suspected. First offender: "
        f"Origin={bad.iloc[0]['Origin']} FlightDate={bad.iloc[0]['FlightDate']} "
        f"CRSDepTime={bad.iloc[0]['CRSDepTime']} dep_dow={bad.iloc[0]['dep_dow']}"
    )


def test_no_per_origin_systematic_hour_shift(smoke_df):
    """Per-origin, sched_dep_hour must not be offset from CRSDepTime//100 by any
    constant. Under the Chicago-coercion bug, every PT airport shows delta=+2,
    every MT +1, every ET -1 — test_sched_dep_hour_is_airport_local already
    catches this row-by-row, but this version produces a cleaner failure message
    grouped by origin so a reviewer can see the pattern at a glance."""
    for c in ("sched_dep_hour", "Origin", "CRSDepTime"):
        if c not in smoke_df.columns:
            pytest.skip(f"{c} missing")
    sdh = pd.to_numeric(smoke_df["sched_dep_hour"], errors="coerce").astype("Int64")
    crsh = _crs_hour(smoke_df)
    df = smoke_df.assign(_delta=(sdh - crsh))
    df = df[df["_delta"].notna()]
    bad = []
    for origin, g in df.groupby("Origin"):
        if len(g) < 20:
            continue
        nonzero_frac = float((g["_delta"] != 0).mean())
        if nonzero_frac > 0.02:
            modal = int(g["_delta"].mode().iloc[0])
            bad.append((origin, round(nonzero_frac, 3), modal, len(g)))
    assert not bad, (
        f"{len(bad)} origins show a systematic sched_dep_hour shift (>2% of rows). "
        f"Chicago-coercion bug signature: PT +2, MT +1, ET -1. "
        f"Offenders (origin, nonzero_frac, modal_delta, n_rows): {bad[:10]}"
    )


# ---------------- TRAINING BUNDLE ----------------

TRAINED_BUNDLE = DATA / "data/models/dep_WN_100_v11_smoke/dep_delay_bins_bundle_WN_100_v11_smoke.joblib"


@pytest.fixture(scope="module")
def bundle():
    if not TRAINED_BUNDLE.exists():
        pytest.skip(f"v11 smoke bundle not found at {TRAINED_BUNDLE}")
    import joblib
    return joblib.load(TRAINED_BUNDLE)


def test_bundle_thresholds_are_v11(bundle):
    assert bundle["thresholds"] == [15, 30, 60, 120], (
        f"v11 bundle thresholds={bundle['thresholds']}, expected [15, 30, 60, 120]"
    )


def test_bundle_bin_weights_are_v11(bundle):
    assert bundle["bin_weights_minutes"] == [7.5, 22.5, 45.0, 90.0, 180.0], (
        f"v11 bundle bin_weights_minutes={bundle['bin_weights_minutes']}, "
        f"expected [7.5, 22.5, 45.0, 90.0, 180.0]"
    )


def test_bundle_feature_order_has_otp(bundle):
    assert "flightnum_od_otp_rate_last14d" in bundle["feature_order"], (
        "v11 bundle missing flightnum_od_otp_rate_last14d from feature_order"
    )


def test_bundle_feature_order_excludes_nas(bundle):
    assert "origin_nasdelay_rate_last1d" not in bundle["feature_order"], (
        "v11 bundle should NOT include origin_nasdelay_rate_last1d (dropped)"
    )
