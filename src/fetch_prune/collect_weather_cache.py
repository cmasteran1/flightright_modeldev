#!/usr/bin/env python3
"""
Standalone historical weather data collector for FlightRight.

Pre-populates a weather cache (daily + hourly parquet files) for all unique
airports across the WN/AA/DL/UA top-100 lists.  Designed to be run slowly
over hours/days, resuming from a checkpoint file if interrupted.

Usage:
    python src/fetch_prune/collect_weather_cache.py \
        --start-date 2023-01-01 \
        --end-date 2026-02-28 \
        --cache-dir ../flightrightdata/weather_cache_historical

See --help for all options.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Imports from prepare_dataset.py  (reuse existing helpers)
# ---------------------------------------------------------------------------
# We import private helpers by name; they are stable within this project.
from prepare_dataset import (
    DAILY_WEATHER_VARS,
    HOURLY_WEATHER_VARS,
    _HOURLY_ARCHIVE_VARS,
    _HOURLY_FORECAST_ONLY_VARS,
    _chunks,
    _daily_df_from_payload,
    _normalize_weather_columns,
    _note_open_meteo_request_attempt,
    _open_meteo_historical_forecast_url,
    _open_meteo_url,
    _sleep_before_open_meteo_request,
    load_airport_meta,
    _validate_tz_strings_for_needed,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = (REPO_ROOT.parent / "flightrightdata").resolve()

AIRLINE_TOP100_DIR = DATA_ROOT / "data" / "meta" / "airport_rankings"
AIRPORTS_CSV = DATA_ROOT / "data" / "meta" / "airports.csv"
AIRLINES = ["WN", "AA", "DL", "UA"]

_DEFAULT_PACING = "2" if os.getenv("OPEN_METEO_API_KEY") else "15"
PACING = float(os.getenv("OPEN_METEO_MIN_SECONDS_BETWEEN_CALLS", _DEFAULT_PACING))

# ---------------------------------------------------------------------------
# Airport loading
# ---------------------------------------------------------------------------

def load_all_top100_airports() -> List[str]:
    """Union of all airports across the 4 airline top-100 lists."""
    all_airports: set = set()
    for airline in AIRLINES:
        csv_path = AIRLINE_TOP100_DIR / f"{airline}_top_100.csv"
        if not csv_path.exists():
            print(f"[WARN] Missing {csv_path}, skipping")
            continue
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                for code in row:
                    code = code.strip().strip('"').upper()
                    if len(code) == 3 and code.isalpha():
                        all_airports.add(code)
    return sorted(all_airports)


def load_airports_meta_dict(
    airports: List[str],
) -> Dict[str, Tuple[float, float, str]]:
    """Load lat/lon/tz for each airport using the existing metadata CSV."""
    meta_df = load_airport_meta(str(AIRPORTS_CSV))
    return _validate_tz_strings_for_needed(meta_df, airports, airports_csv_for_msg=AIRPORTS_CSV)


# ---------------------------------------------------------------------------
# Date chunking
# ---------------------------------------------------------------------------

def date_chunks(start: str, end: str, chunk_days: int) -> List[Tuple[str, str]]:
    """Split a date range into (start, end) chunks of at most chunk_days."""
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    chunks = []
    while s <= e:
        chunk_end = min(s + timedelta(days=chunk_days - 1), e)
        chunks.append((s.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        s = chunk_end + timedelta(days=1)
    return chunks


# ---------------------------------------------------------------------------
# Cache path helpers
# ---------------------------------------------------------------------------

def cache_path(cache_dir: Path, airport: str, start: str, end: str, tz: str, granularity: str) -> Path:
    safe_tz = tz.replace("/", "__")
    return cache_dir / f"{airport}_{start}_{end}_{safe_tz}_{granularity}_historical.csv"


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

class Checkpoint:
    """Manages resume state across runs."""

    def __init__(self, path: Path, config: dict):
        self.path = path
        self.config = config
        self.completed: set = set()  # frozenset keys: (airport, start, end, granularity)
        self.failed: List[dict] = []
        self.stats = {
            "total_items": 0,
            "completed_count": 0,
            "failed_count": 0,
            "api_calls": 0,
            "started_utc": datetime.now().astimezone().isoformat(),
        }
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        try:
            with open(self.path) as f:
                data = json.load(f)
            for item in data.get("completed", []):
                key = (item["airport"], item["start"], item["end"], item.get("granularity", "both"))
                self.completed.add(key)
            prev_failed = data.get("failed", [])
            self.failed = []  # clear so they get retried
            saved_stats = data.get("stats", {})
            self.stats["api_calls"] = saved_stats.get("api_calls", 0)
            self.stats["completed_count"] = len(self.completed)
            self.stats["failed_count"] = 0
            if prev_failed:
                print(f"[CHECKPOINT] Clearing {len(prev_failed)} previously failed items for retry")
            if "started_utc" in saved_stats:
                self.stats["started_utc"] = saved_stats["started_utc"]
            print(f"[CHECKPOINT] Loaded {len(self.completed)} completed items from {self.path}")
        except Exception as e:
            print(f"[WARN] Could not load checkpoint {self.path}: {e}")

    def is_done(self, airport: str, start: str, end: str, granularity: str = "both") -> bool:
        return (airport, start, end, granularity) in self.completed

    def mark_done(self, airport: str, start: str, end: str, granularity: str = "both"):
        self.completed.add((airport, start, end, granularity))
        self.stats["completed_count"] = len(self.completed)

    def mark_failed(self, airport: str, start: str, end: str, error: str):
        self.failed.append({
            "airport": airport,
            "start": start,
            "end": end,
            "error": str(error)[:500],
            "timestamp": datetime.now().astimezone().isoformat(),
        })
        self.stats["failed_count"] = len(self.failed)

    def add_api_calls(self, n: int = 1):
        self.stats["api_calls"] += n

    def save(self):
        data = {
            "version": 1,
            "config": self.config,
            "stats": {**self.stats, "last_updated_utc": datetime.now().astimezone().isoformat()},
            "completed": [
                {"airport": a, "start": s, "end": e, "granularity": g}
                for a, s, e, g in sorted(self.completed)
            ],
            "failed": self.failed,
        }
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        tmp.replace(self.path)


# ---------------------------------------------------------------------------
# HTTP helper (standalone, does not rely on prepare_dataset's global state)
# ---------------------------------------------------------------------------

_last_request_mono: Optional[float] = None
_total_api_calls = 0


def _pace():
    """Enforce minimum gap between Open-Meteo requests."""
    global _last_request_mono
    if _last_request_mono is not None:
        elapsed = time.monotonic() - _last_request_mono
        if elapsed < PACING:
            sleep_s = (PACING - elapsed) + random.uniform(0.25, 1.0)
            time.sleep(sleep_s)


def _mark_request():
    global _last_request_mono, _total_api_calls
    _last_request_mono = time.monotonic()
    _total_api_calls += 1


class RateLimitHit(Exception):
    """Raised when we get a 429 and need to wait a long time."""
    def __init__(self, retry_after: Optional[float] = None):
        self.retry_after = retry_after
        super().__init__(f"Rate limited (Retry-After={retry_after})")


def _request(url: str, params: dict, max_tries: int = 5) -> dict:
    """
    Make a GET request with backoff.  Raises RateLimitHit on persistent 429s
    so the caller can decide whether to wait or checkpoint-and-exit.
    """
    import requests

    backoff = 4.0
    consecutive_429 = 0

    for attempt in range(1, max_tries + 1):
        try:
            _pace()
            r = requests.get(url, params=params, timeout=120)
            _mark_request()

            if r.status_code == 200:
                return r.json()

            if r.status_code == 429:
                consecutive_429 += 1
                retry_after = None
                try:
                    ra = r.headers.get("Retry-After")
                    if ra is not None:
                        retry_after = float(ra)
                except Exception:
                    pass

                if consecutive_429 >= 3:
                    raise RateLimitHit(retry_after)

                sleep_s = retry_after if retry_after is not None else max(60.0, backoff)
                print(f"  [429] attempt {attempt}/{max_tries}, sleeping {sleep_s:.0f}s ...")
                time.sleep(sleep_s + random.uniform(1.0, 5.0))
                backoff = min(900, backoff * 2)
                continue

            if r.status_code in (500, 502, 503, 504):
                sleep_s = backoff
                print(f"  [HTTP {r.status_code}] attempt {attempt}/{max_tries}, sleeping {sleep_s:.0f}s ...")
                if attempt == max_tries:
                    raise RuntimeError(f"HTTP {r.status_code} after {max_tries} tries: {r.text[:300]}")
                time.sleep(sleep_s + random.uniform(0.5, 1.5))
                backoff = min(300, backoff * 2)
                continue

            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")

        except (RateLimitHit, RuntimeError):
            raise
        except Exception as e:
            if attempt == max_tries:
                raise RuntimeError(f"Request failed after {max_tries} tries: {e}")
            print(f"  [ERR] attempt {attempt}/{max_tries}: {e}, sleeping {backoff:.0f}s ...")
            time.sleep(backoff)
            backoff = min(300, backoff * 2)

    raise RuntimeError("Should not reach here")


# ---------------------------------------------------------------------------
# Fetch + save helpers
# ---------------------------------------------------------------------------

def fetch_and_save_daily(
    airport: str,
    meta: Tuple[float, float, str],
    start: str,
    end: str,
    cache_dir: Path,
) -> Path:
    """Fetch daily weather from archive API and save as parquet. Returns path."""
    lat, lon, tz = meta
    out_path = cache_path(cache_dir, airport, start, end, tz, "daily")

    if out_path.exists():
        return out_path

    url = _open_meteo_url()
    params = {
        "latitude": str(lat),
        "longitude": str(lon),
        "start_date": start,
        "end_date": end,
        "daily": ",".join(DAILY_WEATHER_VARS),
        "timezone": tz,
    }

    payload = _request(url, params)
    df = _daily_df_from_payload(payload)
    df.insert(0, "airport", airport)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


def fetch_and_save_hourly(
    airport: str,
    meta: Tuple[float, float, str],
    start: str,
    end: str,
    cache_dir: Path,
) -> Path:
    """Fetch hourly weather (archive + forecast-only) and save as parquet."""
    lat, lon, tz = meta
    out_path = cache_path(cache_dir, airport, start, end, tz, "hourly")

    if out_path.exists():
        return out_path

    # 1) Archive hourly vars
    archive_url = _open_meteo_url()
    archive_params = {
        "latitude": str(lat),
        "longitude": str(lon),
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(_HOURLY_ARCHIVE_VARS),
        "timezone": tz,
    }
    archive_payload = _request(archive_url, archive_params)

    hourly_data = archive_payload.get("hourly", {})
    if not hourly_data or "time" not in hourly_data:
        # Empty result — save empty frame
        df = pd.DataFrame(columns=["time"] + _HOURLY_ARCHIVE_VARS + _HOURLY_FORECAST_ONLY_VARS)
        df.insert(0, "airport", airport)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        return out_path

    df = pd.DataFrame(hourly_data)
    df = _normalize_weather_columns(df)

    # 2) Forecast-only hourly vars (visibility, cape)
    try:
        fc_url = _open_meteo_historical_forecast_url()
        fc_params = {
            "latitude": str(lat),
            "longitude": str(lon),
            "start_date": start,
            "end_date": end,
            "hourly": ",".join(_HOURLY_FORECAST_ONLY_VARS),
            "timezone": tz,
        }
        fc_payload = _request(fc_url, fc_params)
        fc_hourly = fc_payload.get("hourly", {})
        if fc_hourly and "time" in fc_hourly:
            fc_df = pd.DataFrame(fc_hourly)
            fc_df = _normalize_weather_columns(fc_df)
            # Merge on time column
            for v in _HOURLY_FORECAST_ONLY_VARS:
                if v in fc_df.columns:
                    time_to_val = dict(zip(fc_df["time"], fc_df[v]))
                    df[v] = df["time"].map(time_to_val)
    except RateLimitHit:
        raise  # propagate rate limits
    except Exception as e:
        print(f"  [WARN] forecast-only vars failed for {airport} {start}..{end}: {e}")
        # Continue without visibility/cape — they'll be NaN

    # Ensure all expected columns exist
    for v in HOURLY_WEATHER_VARS:
        if v not in df.columns:
            df[v] = np.nan

    df.insert(0, "airport", airport)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


# ---------------------------------------------------------------------------
# Rate-limit wait with countdown
# ---------------------------------------------------------------------------

def wait_for_rate_limit_reset(retry_after: Optional[float], checkpoint: Checkpoint):
    """Wait for rate limit reset, logging countdown every 60s."""
    wait_s = retry_after if retry_after is not None else 3600.0
    wait_s = max(wait_s, 60.0)  # at least 60s

    checkpoint.save()
    print(f"\n[RATE LIMITED] Waiting {wait_s:.0f}s for rate limit reset ...")
    print(f"  Checkpoint saved. You can safely Ctrl+C and resume later.\n")

    start = time.monotonic()
    while True:
        remaining = wait_s - (time.monotonic() - start)
        if remaining <= 0:
            break
        next_log = min(remaining, 60.0)
        print(f"  ... {remaining:.0f}s remaining")
        time.sleep(next_log)

    print("[RATE LIMITED] Wait complete, resuming.\n")


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

def collect(
    airports: List[str],
    airports_meta: Dict[str, Tuple[float, float, str]],
    start_date: str,
    end_date: str,
    cache_dir: Path,
    checkpoint: Checkpoint,
    *,
    chunk_days: int = 90,
    granularity: str = "both",
):
    """Run the full collection loop with checkpointing."""
    chunks = date_chunks(start_date, end_date, chunk_days)
    total = len(airports) * len(chunks)
    checkpoint.stats["total_items"] = total

    print(f"\n{'='*60}")
    print(f"  Airports:    {len(airports)}")
    print(f"  Date range:  {start_date} .. {end_date}")
    print(f"  Chunks:      {len(chunks)} x {chunk_days}d")
    print(f"  Total items: {total}")
    print(f"  Granularity: {granularity}")
    print(f"  Cache dir:   {cache_dir}")
    print(f"  Already done:{checkpoint.stats['completed_count']}")
    print(f"{'='*60}\n")

    cache_dir.mkdir(parents=True, exist_ok=True)
    done_count = checkpoint.stats["completed_count"]
    consecutive_rate_limits = 0
    t_start = time.monotonic()

    for chunk_start, chunk_end in chunks:
        for airport in airports:
            item_key = f"{airport} {chunk_start}..{chunk_end}"

            if checkpoint.is_done(airport, chunk_start, chunk_end, granularity):
                continue

            # Also skip if cache files already exist on disk
            lat, lon, tz = airports_meta[airport]
            daily_exists = cache_path(cache_dir, airport, chunk_start, chunk_end, tz, "daily").exists()
            hourly_exists = cache_path(cache_dir, airport, chunk_start, chunk_end, tz, "hourly").exists()

            if granularity == "daily" and daily_exists:
                checkpoint.mark_done(airport, chunk_start, chunk_end, granularity)
                done_count += 1
                continue
            if granularity == "hourly" and hourly_exists:
                checkpoint.mark_done(airport, chunk_start, chunk_end, granularity)
                done_count += 1
                continue
            if granularity == "both" and daily_exists and hourly_exists:
                checkpoint.mark_done(airport, chunk_start, chunk_end, granularity)
                done_count += 1
                continue

            t_item = time.monotonic()
            try:
                api_calls = 0

                if granularity in ("both", "daily") and not daily_exists:
                    fetch_and_save_daily(airport, airports_meta[airport], chunk_start, chunk_end, cache_dir)
                    api_calls += 1

                if granularity in ("both", "hourly") and not hourly_exists:
                    fetch_and_save_hourly(airport, airports_meta[airport], chunk_start, chunk_end, cache_dir)
                    api_calls += 2  # archive + forecast-only

                checkpoint.add_api_calls(api_calls)
                checkpoint.mark_done(airport, chunk_start, chunk_end, granularity)
                done_count += 1
                consecutive_rate_limits = 0

                elapsed_item = time.monotonic() - t_item
                elapsed_total = time.monotonic() - t_start

                remaining = total - done_count
                if done_count > 0:
                    rate = elapsed_total / (done_count - checkpoint.stats["completed_count"] + len(checkpoint.completed))
                    eta_s = remaining * (elapsed_item + PACING)  # rough ETA
                else:
                    eta_s = 0

                print(
                    f"[{done_count}/{total}] {item_key} {granularity} OK "
                    f"({elapsed_item:.1f}s, ~{eta_s/3600:.1f}h remaining)"
                )

                # Save checkpoint after every item (cheap JSON write)
                checkpoint.save()

            except RateLimitHit as e:
                consecutive_rate_limits += 1
                print(f"\n[RATE LIMIT] Hit on {item_key} (consecutive={consecutive_rate_limits})")

                if consecutive_rate_limits >= 3:
                    checkpoint.save()
                    print(f"\n{'='*60}")
                    print(f"  STOPPED: 3 consecutive rate limits.")
                    print(f"  Completed: {done_count}/{total}")
                    print(f"  Checkpoint saved to: {checkpoint.path}")
                    print(f"  Resume with the same command.")
                    print(f"{'='*60}")
                    return False

                wait_for_rate_limit_reset(e.retry_after, checkpoint)
                # Don't mark as done — will retry on next iteration

            except KeyboardInterrupt:
                checkpoint.save()
                print(f"\n[INTERRUPTED] Checkpoint saved at {done_count}/{total}. Resume with same command.")
                return False

            except Exception as e:
                checkpoint.mark_failed(airport, chunk_start, chunk_end, str(e))
                print(f"  [FAIL] {item_key}: {e}")
                # Continue to next item

    checkpoint.save()
    elapsed = time.monotonic() - t_start

    print(f"\n{'='*60}")
    print(f"  COLLECTION COMPLETE")
    print(f"  Completed:  {done_count}/{total}")
    print(f"  Failed:     {checkpoint.stats['failed_count']}")
    print(f"  API calls:  {checkpoint.stats['api_calls']}")
    print(f"  Elapsed:    {elapsed/3600:.1f}h")
    print(f"  Cache dir:  {cache_dir}")
    print(f"{'='*60}")

    if checkpoint.failed:
        print(f"\nFailed items:")
        for f in checkpoint.failed[-20:]:
            print(f"  {f['airport']} {f['start']}..{f['end']}: {f['error'][:100]}")

    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pre-fetch historical weather data for all top-100 airports.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (1 week):
  python src/fetch_prune/collect_weather_cache.py \\
    --start-date 2025-01-01 --end-date 2025-01-07 --chunk-days 7

  # Full collection (2023-2026):
  python src/fetch_prune/collect_weather_cache.py \\
    --start-date 2023-01-01 --end-date 2026-02-28
        """,
    )
    parser.add_argument("--start-date", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--cache-dir",
        default=str(DATA_ROOT / "weather_cache_historical"),
        help="Directory for cached parquet files (default: ../flightrightdata/weather_cache_historical)",
    )
    parser.add_argument(
        "--checkpoint-file",
        default=None,
        help="Checkpoint JSON path (default: {cache-dir}/_checkpoint.json)",
    )
    parser.add_argument("--chunk-days", type=int, default=90, help="Days per API chunk (default: 90)")
    parser.add_argument(
        "--granularity",
        choices=["both", "daily", "hourly"],
        default="both",
        help="What to fetch: both, daily, or hourly (default: both)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show work plan without fetching",
    )

    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).resolve()
    checkpoint_path = Path(args.checkpoint_file) if args.checkpoint_file else cache_dir / "_checkpoint.json"

    # Load airports
    print("[1/3] Loading airport lists ...")
    airports = load_all_top100_airports()
    print(f"  {len(airports)} unique airports across {len(AIRLINES)} airlines")

    # Load coordinates
    print("[2/3] Loading airport metadata ...")
    airports_meta = load_airports_meta_dict(airports)
    print(f"  {len(airports_meta)} airports with valid coordinates")

    # Only keep airports we have metadata for
    airports = [a for a in airports if a in airports_meta]

    # Build work plan
    chunks = date_chunks(args.start_date, args.end_date, args.chunk_days)
    total = len(airports) * len(chunks)
    print(f"\n  Work plan: {len(airports)} airports x {len(chunks)} chunks = {total} items")

    if args.dry_run:
        print("\n[DRY RUN] Would fetch:")
        for i, (cs, ce) in enumerate(chunks[:5]):
            print(f"  chunk {i+1}: {cs} .. {ce}")
        if len(chunks) > 5:
            print(f"  ... and {len(chunks) - 5} more chunks")
        print(f"\nFirst 20 airports: {', '.join(airports[:20])}")
        if len(airports) > 20:
            print(f"  ... and {len(airports) - 20} more")
        return

    # Load checkpoint
    config = {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "chunk_days": args.chunk_days,
        "granularity": args.granularity,
    }
    checkpoint = Checkpoint(checkpoint_path.resolve(), config)

    # Run collection
    print("[3/3] Starting collection ...")
    try:
        success = collect(
            airports=airports,
            airports_meta=airports_meta,
            start_date=args.start_date,
            end_date=args.end_date,
            cache_dir=cache_dir,
            checkpoint=checkpoint,
            chunk_days=args.chunk_days,
            granularity=args.granularity,
        )
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        checkpoint.save()
        print(f"\n[INTERRUPTED] Checkpoint saved. Resume with same command.")
        sys.exit(1)


if __name__ == "__main__":
    main()
