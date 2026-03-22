#!/usr/bin/env python3
"""
Standalone labor-action data collector for FlightRight.

Reads the curated CSV of US aviation labor actions and writes a validated
parquet file for use by prepare_dataset.py during enrichment.

Usage:
    python src/fetch_prune/collect_strike_cache.py
    python src/fetch_prune/collect_strike_cache.py \
        --csv ../flightrightdata/data/meta/us_aviation_labor_actions.csv \
        --output-dir ../flightrightdata/strike_cache
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = (REPO_ROOT.parent / "flightrightdata").resolve()

DEFAULT_CSV = DATA_ROOT / "data" / "meta" / "us_aviation_labor_actions.csv"
DEFAULT_OUTPUT_DIR = DATA_ROOT / "strike_cache"

REQUIRED_COLUMNS = {
    "event_id", "airline_iata", "worker_type", "scope", "affected_airports",
    "announced_date", "cooling_off_start", "action_start_date", "action_end_date",
    "workers_involved", "action_type", "severity", "source", "notes",
}

VALID_WORKER_TYPES = {"pilot", "flight_attendant", "mechanic", "ground_crew", "atc", "dispatcher"}
VALID_ACTION_TYPES = {
    "strike", "sickout", "work_slowdown", "lockout",
    "strike_authorization_vote", "informational_picketing", "picketing",
}
VALID_SCOPES = {"airline_wide", "airport_specific", "system_wide"}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_and_load(csv_path: Path) -> pd.DataFrame:
    """Load the curated CSV, validate schema & values, return DataFrame."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Strike CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, dtype=str)
    df.columns = df.columns.str.strip()

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in strike CSV: {sorted(missing)}")

    # Parse dates
    for col in ["announced_date", "cooling_off_start", "action_start_date", "action_end_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Parse numeric
    df["workers_involved"] = pd.to_numeric(df["workers_involved"], errors="coerce").astype("Int32")
    df["severity"] = pd.to_numeric(df["severity"], errors="coerce").astype("Int8")

    # Normalize strings
    df["airline_iata"] = df["airline_iata"].str.strip().str.upper()
    df["worker_type"] = df["worker_type"].str.strip().str.lower()
    df["scope"] = df["scope"].str.strip().str.lower()
    df["action_type"] = df["action_type"].str.strip().str.lower()
    df["affected_airports"] = df["affected_airports"].str.strip().str.upper()

    # Validate enums
    bad_worker = df[~df["worker_type"].isin(VALID_WORKER_TYPES)]
    if len(bad_worker):
        print(f"[WARN] Unknown worker_type values: {bad_worker['worker_type'].unique().tolist()}")

    bad_action = df[~df["action_type"].isin(VALID_ACTION_TYPES)]
    if len(bad_action):
        print(f"[WARN] Unknown action_type values: {bad_action['action_type'].unique().tolist()}")

    bad_scope = df[~df["scope"].isin(VALID_SCOPES)]
    if len(bad_scope):
        print(f"[WARN] Unknown scope values: {bad_scope['scope'].unique().tolist()}")

    # Validate severity range
    sev = df["severity"].dropna()
    if (sev < 1).any() or (sev > 5).any():
        print(f"[WARN] severity values outside 1-5 range: {sev[(sev < 1) | (sev > 5)].tolist()}")

    # Validate date ordering
    has_dates = df.dropna(subset=["action_start_date"])
    both_dates = has_dates.dropna(subset=["action_end_date"])
    bad_order = both_dates[both_dates["action_end_date"] < both_dates["action_start_date"]]
    if len(bad_order):
        print(f"[WARN] action_end_date < action_start_date for: {bad_order['event_id'].tolist()}")

    print(f"[OK] Loaded {len(df)} labor action events from {csv_path}")
    print(f"     Date range: {df['action_start_date'].min()} to {df['action_end_date'].max()}")
    print(f"     Airlines: {sorted(df['airline_iata'].unique())}")
    print(f"     Action types: {sorted(df['action_type'].unique())}")
    print(f"     Severity distribution: {df['severity'].value_counts().sort_index().to_dict()}")

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build strike/labor-action cache for FlightRight")
    parser.add_argument("--csv", type=str, default=str(DEFAULT_CSV),
                        help=f"Path to curated labor actions CSV (default: {DEFAULT_CSV})")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help=f"Output directory for parquet cache (default: {DEFAULT_OUTPUT_DIR})")
    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()
    output_dir = Path(args.output_dir).resolve()

    df = validate_and_load(csv_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "us_aviation_labor_actions.parquet"
    df.to_parquet(out_path, index=False)
    print(f"[OK] Wrote strike cache -> {out_path} ({len(df)} events)")


if __name__ == "__main__":
    main()
