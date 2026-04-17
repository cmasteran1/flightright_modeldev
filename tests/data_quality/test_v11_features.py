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
from pathlib import Path
import pandas as pd
import pytest

DATA = Path(__file__).parent.parent.parent.parent / "flightrightdata"
SMOKE_PARQUET = DATA / "data/processed/features_dep_WN_v11_smoke_unbalanced.parquet"


@pytest.fixture(scope="module")
def smoke_df() -> pd.DataFrame:
    if not SMOKE_PARQUET.exists():
        pytest.skip(f"v11 smoke parquet not found at {SMOKE_PARQUET}")
    return pd.read_parquet(SMOKE_PARQUET)


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
