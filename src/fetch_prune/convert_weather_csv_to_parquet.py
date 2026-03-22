#!/usr/bin/env python3
"""
Convert weather_cache_historical CSV files to parquet format.

Reads all CSV files from the historical cache directory and writes
corresponding parquet files in the same directory.

Usage:
    python src/fetch_prune/convert_weather_csv_to_parquet.py
    python src/fetch_prune/convert_weather_csv_to_parquet.py --cache-dir <path>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = (REPO_ROOT.parent / "flightrightdata").resolve()
DEFAULT_CACHE_DIR = DATA_ROOT / "weather_cache_historical"


def convert_all(cache_dir: Path, overwrite: bool = False):
    csvs = sorted(cache_dir.glob("*.csv"))
    if not csvs:
        print(f"[WARN] No CSV files found in {cache_dir}")
        return

    converted = 0
    skipped = 0
    for csv_path in csvs:
        pq_path = csv_path.with_suffix(".parquet")
        if pq_path.exists() and not overwrite:
            skipped += 1
            continue

        try:
            df = pd.read_csv(csv_path)
            # Normalize date columns
            if "FlightDate" in df.columns:
                df["FlightDate"] = pd.to_datetime(df["FlightDate"], errors="coerce")
            if "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"], errors="coerce")
            df.to_parquet(pq_path, index=False)
            converted += 1
        except Exception as e:
            print(f"[ERROR] {csv_path.name}: {e}")

    print(f"[OK] Converted {converted} CSVs to parquet, skipped {skipped} "
          f"(already exist) in {cache_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert historical weather CSVs to parquet")
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing parquets")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).resolve()
    if not cache_dir.exists():
        print(f"[ERROR] Directory not found: {cache_dir}")
        sys.exit(1)

    convert_all(cache_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
