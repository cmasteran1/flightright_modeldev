"""
collect_isd_visibility.py

Download at-airport METAR visibility observations from the Iowa Environmental
Mesonet (IEM) ASOS archive and cache them locally for training-time feature
extraction.

Why this exists: v11's Open-Meteo visibility feature has univariate AUC ~0.50
(near-random) across all 4 airlines because Open-Meteo visibility comes from a
~10 km-grid NWP model that smooths over airport-scale fog/low-vis events. IEM
ASOS provides 1/4-statute-mile-precision at-airport readings derived from the
same METAR observations pilots and dispatchers actually use.

Source: https://mesonet.agron.iastate.edu/request/download.phtml (free, no auth)

Output layout (mirrors the existing Open-Meteo caches under
  ../flightrightdata/weather_cache_hourly/):

  ../flightrightdata/weather_cache_isd/
      {IATA}_{start}_{end}_UTC_hourly_vis.parquet

Each per-airport parquet has columns:
  - valid_utc: pd.Timestamp UTC (observation time)
  - visibility_m: float (statute miles × 1609.344), clipped to [0, 25000]
  - visibility_sm: float (original statute miles)

Usage:
  python src/fetch_prune/collect_isd_visibility.py \\
      --start-date 2024-10-01 --end-date 2025-12-31 \\
      --cache-dir ../flightrightdata/weather_cache_isd

  python src/fetch_prune/collect_isd_visibility.py \\
      --iatas DFW,CLT,ATL \\
      --start-date 2024-10-01 --end-date 2025-12-31

By default, airports are the UNION of top-100 lists across WN/AA/DL/UA
(~141 unique airports).

IMPORTANT: TRAINING-TIME ONLY. This cache is pre-computed once for a date
range and re-used for feature extraction. Serving a production model that
uses these features requires a serve-time METAR source (see
feature-transferability skill). Do not deploy a model trained on this
feature until that audit is complete.
"""
from __future__ import annotations

import argparse
import io
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import urllib.request
import urllib.error

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = (REPO_ROOT.parent / "flightrightdata").resolve()
DEFAULT_CACHE_DIR = DATA_ROOT / "weather_cache_isd"
AIRLINE_TOP100_DIR = DATA_ROOT / "data" / "meta" / "airport_rankings"

SM_TO_M = 1609.344  # statute mile → meter
MAX_VIS_M = 25000.0  # same clip used for Open-Meteo visibility in features_dep.py

IEM_URL_TEMPLATE = (
    "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
    "?station={iata}&data=vsby"
    "&year1={y1}&month1={m1}&day1={d1}"
    "&year2={y2}&month2={m2}&day2={d2}"
    "&tz=Etc/UTC&format=onlycomma&latlon=no"
    "&missing=M&trace=T&direct=no&report_type=3"
)

# NOAA NCEI global-hourly backend: one CSV per year per station, HTTPS/S3-backed,
# no rate limiting. Requires USAF-WBAN station lookup from isd-history.csv.
NOAA_BASE = "https://www.ncei.noaa.gov/data/global-hourly/access"
NOAA_STATIONS_URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
_STATION_LOOKUP_CACHE: dict[str, tuple[str, str]] | None = None


def default_airport_iatas() -> list[str]:
    """Union of top-100 IATA lists across WN, AA, DL, UA."""
    airports: set[str] = set()
    for al in ("WN", "AA", "DL", "UA"):
        p = AIRLINE_TOP100_DIR / f"{al}_top_100.csv"
        if not p.exists():
            continue
        # These CSVs use the IATAs AS the column headers (single row of data).
        df = pd.read_csv(p)
        airports.update(str(c).strip() for c in df.columns)
    return sorted(airports)


def _fetch_one(
    iata: str,
    start: datetime,
    end: datetime,
    retries: int = 5,
    timeout_s: int = 120,
) -> pd.DataFrame:
    """Fetch IEM ASOS visibility for `iata` in [start, end] (UTC). Returns a
    DataFrame with columns valid_utc, visibility_sm, visibility_m. Empty frame
    on permanent failure (caller should skip writing the parquet).

    IEM rate-limits aggressively; we back off starting at 10s on 429/5xx."""
    url = IEM_URL_TEMPLATE.format(
        iata=iata,
        y1=start.year, m1=start.month, d1=start.day,
        y2=end.year,   m2=end.month,   d2=end.day,
    )
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout_s) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            break
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code == 429 or 500 <= e.code < 600:
                # Honor Retry-After if present, otherwise aggressive backoff.
                ra = e.headers.get("Retry-After")
                if ra and ra.isdigit():
                    delay = int(ra) + random.random()
                else:
                    delay = 10.0 * (2 ** attempt) + random.random()
                print(f"  [retry] {iata} attempt {attempt+1}/{retries} after HTTP {e.code}: sleeping {delay:.1f}s", flush=True)
                time.sleep(delay)
                continue
            # other HTTP errors: don't retry
            print(f"  [skip] {iata}: HTTP {e.code} non-retriable: {e.reason}", flush=True)
            return pd.DataFrame(columns=["valid_utc", "visibility_sm", "visibility_m"])
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = e
            delay = 5.0 * (2 ** attempt) + random.random()
            print(f"  [retry] {iata} attempt {attempt+1}/{retries} after {e}: sleeping {delay:.1f}s", flush=True)
            time.sleep(delay)
    else:
        print(f"  [skip] {iata}: failed after {retries} retries: {last_err}", flush=True)
        return pd.DataFrame(columns=["valid_utc", "visibility_sm", "visibility_m"])

    if not raw or raw.startswith("ERROR"):
        print(f"  [skip] {iata}: IEM returned empty or error response", flush=True)
        return pd.DataFrame(columns=["valid_utc", "visibility_sm", "visibility_m"])

    df = pd.read_csv(io.StringIO(raw))
    if "vsby" not in df.columns or "valid" not in df.columns:
        print(f"  [skip] {iata}: missing expected columns in IEM response: {list(df.columns)}", flush=True)
        return pd.DataFrame(columns=["valid_utc", "visibility_sm", "visibility_m"])

    # Parse. vsby is statute miles; 'M' or 'T' (trace) become NaN.
    df["valid_utc"] = pd.to_datetime(df["valid"], errors="coerce", utc=True)
    df["visibility_sm"] = pd.to_numeric(df["vsby"], errors="coerce")
    df["visibility_m"] = (df["visibility_sm"] * SM_TO_M).clip(lower=0.0, upper=MAX_VIS_M)
    df = df.dropna(subset=["valid_utc"])
    # Deduplicate on timestamp (some stations report multiple obs per hour).
    # Keep the earliest report in the hour; features_*.py joins on hour-floor.
    return df[["valid_utc", "visibility_sm", "visibility_m"]].sort_values("valid_utc").reset_index(drop=True)


def _load_station_lookup() -> dict[str, tuple[str, str]]:
    """Download NOAA isd-history.csv and build IATA → (USAF, WBAN) lookup for US airports.

    Station CALL codes for US airports are ICAO (K+IATA). Returned dict keys are IATA.
    Cached in-memory for the life of the process."""
    global _STATION_LOOKUP_CACHE
    if _STATION_LOOKUP_CACHE is not None:
        return _STATION_LOOKUP_CACHE
    print("  [noaa] fetching isd-history.csv station index...", flush=True)
    with urllib.request.urlopen(NOAA_STATIONS_URL, timeout=120) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    df = pd.read_csv(io.StringIO(raw), dtype=str)
    # Column names in current isd-history.csv: USAF, WBAN, STATION NAME, CTRY, STATE, ICAO, LAT, LON, ELEV(M), BEGIN, END
    df = df[df["CTRY"] == "US"].copy()
    df = df[df["ICAO"].str.match(r"^K[A-Z]{3}$", na=False)]
    df["IATA"] = df["ICAO"].str.slice(1)  # K-prefix stripped
    # Prefer stations that are still active at BEGIN <= 2024 and END > 20240000
    df["END_INT"] = pd.to_numeric(df["END"], errors="coerce").fillna(0)
    df = df.sort_values(["IATA", "END_INT"], ascending=[True, False])
    df = df.drop_duplicates("IATA", keep="first")
    lookup = {r["IATA"]: (str(r["USAF"]).zfill(6), str(r["WBAN"]).zfill(5)) for _, r in df.iterrows()}
    print(f"  [noaa] loaded {len(lookup)} US IATA → (USAF, WBAN) mappings", flush=True)
    _STATION_LOOKUP_CACHE = lookup
    return lookup


def _parse_noaa_vis(df: pd.DataFrame) -> pd.DataFrame:
    """Parse NOAA global-hourly CSV into (valid_utc, visibility_sm, visibility_m).

    The VIS field in NCEI CSV is a compound string like '016000,1,N,5,N' where the
    first field is visibility in meters, followed by 4 quality / variability codes.
    See the ISD format document; we just take the first integer."""
    if "VIS" not in df.columns or "DATE" not in df.columns:
        return pd.DataFrame(columns=["valid_utc", "visibility_sm", "visibility_m"])
    vis_parts = df["VIS"].astype(str).str.split(",", n=1).str[0]
    # Missing/invalid visibility codes: 999999 (missing)
    vis_m = pd.to_numeric(vis_parts, errors="coerce")
    vis_m = vis_m.where(vis_m < 900000, other=pd.NA)
    out = pd.DataFrame({
        "valid_utc": pd.to_datetime(df["DATE"], errors="coerce", utc=True),
        "visibility_m": vis_m.astype("float64"),
    })
    out["visibility_sm"] = out["visibility_m"] / SM_TO_M
    out["visibility_m"] = out["visibility_m"].clip(lower=0.0, upper=MAX_VIS_M)
    out = out.dropna(subset=["valid_utc", "visibility_m"])
    return out[["valid_utc", "visibility_sm", "visibility_m"]].sort_values("valid_utc").reset_index(drop=True)


def _fetch_one_noaa(
    iata: str,
    start: datetime,
    end: datetime,
    retries: int = 3,
    timeout_s: int = 300,
) -> pd.DataFrame:
    """Fetch NOAA global-hourly for `iata` by downloading per-year CSVs and concatenating.
    NOAA serves directly from NCEI (HTTPS); no rate limit.
    """
    lookup = _load_station_lookup()
    if iata not in lookup:
        print(f"  [skip] {iata}: no NOAA station mapping (not US/ICAO K-prefix?)", flush=True)
        return pd.DataFrame(columns=["valid_utc", "visibility_sm", "visibility_m"])
    usaf, wban = lookup[iata]
    station_id = f"{usaf}{wban}"

    years = range(start.year, end.year + 1)
    parts = []
    for yr in years:
        url = f"{NOAA_BASE}/{yr}/{station_id}.csv"
        last_err: Exception | None = None
        raw = None
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(url, timeout=timeout_s) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
                break
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    print(f"  [noaa] {iata} {yr}: no data (404)", flush=True)
                    break
                last_err = e
                delay = 5.0 * (2 ** attempt) + random.random()
                print(f"  [retry] {iata} {yr} after HTTP {e.code}: sleeping {delay:.1f}s", flush=True)
                time.sleep(delay)
            except (urllib.error.URLError, TimeoutError) as e:
                last_err = e
                delay = 5.0 * (2 ** attempt) + random.random()
                print(f"  [retry] {iata} {yr} after {e}: sleeping {delay:.1f}s", flush=True)
                time.sleep(delay)
        if raw is None or not raw:
            continue
        year_df = pd.read_csv(io.StringIO(raw), dtype=str, low_memory=False)
        parsed = _parse_noaa_vis(year_df)
        # Clip to requested date range
        parsed = parsed[
            (parsed["valid_utc"] >= pd.Timestamp(start, tz="UTC")) &
            (parsed["valid_utc"] <= pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1))
        ]
        parts.append(parsed)
    if not parts:
        return pd.DataFrame(columns=["valid_utc", "visibility_sm", "visibility_m"])
    return pd.concat(parts, ignore_index=True).sort_values("valid_utc").reset_index(drop=True)


def collect(
    iatas: Iterable[str],
    start: datetime,
    end: datetime,
    cache_dir: Path,
    force: bool = False,
    source: str = "iem",
) -> dict[str, int]:
    """Fetch visibility for each IATA and write a per-airport parquet. Returns
    dict {iata: rows_written}."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"{start:%Y-%m-%d}_{end:%Y-%m-%d}_UTC_hourly_vis"
    report: dict[str, int] = {}
    for iata in iatas:
        iata = iata.strip().upper()
        out = cache_dir / f"{iata}_{stamp}.parquet"
        if out.exists() and not force:
            try:
                n = len(pd.read_parquet(out, columns=["valid_utc"]))
            except Exception:
                n = -1
            report[iata] = n
            print(f"  [cached] {iata}: {n} rows -> {out.name}", flush=True)
            continue
        print(f"  [fetch {source}]  {iata} {start:%Y-%m-%d}..{end:%Y-%m-%d}", flush=True)
        if source == "noaa":
            df = _fetch_one_noaa(iata, start, end)
            sleep_after = 0.2  # NOAA is static files; short courtesy delay
        else:
            df = _fetch_one(iata, start, end)
            sleep_after = 3.0  # IEM rate limits aggressively
        if len(df) == 0:
            report[iata] = 0
            continue
        df.to_parquet(out, index=False)
        report[iata] = len(df)
        print(f"  [ok]     {iata}: {len(df)} rows -> {out.name}", flush=True)
        time.sleep(sleep_after)
    return report


def _parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iatas", default=None,
                    help="Comma-separated airport IATAs. Default: union of top-100 lists for WN/AA/DL/UA.")
    ap.add_argument("--start-date", required=True, help="Inclusive YYYY-MM-DD")
    ap.add_argument("--end-date", required=True, help="Inclusive YYYY-MM-DD")
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR),
                    help=f"Output directory (default: {DEFAULT_CACHE_DIR})")
    ap.add_argument("--force", action="store_true", help="Re-download even if parquet exists.")
    ap.add_argument("--source", choices=["iem", "noaa"], default="iem",
                    help="Data source. iem=IEM ASOS (rate-limited, SM units). "
                         "noaa=NOAA NCEI global-hourly (no rate limit, meter units, recommended for bulk).")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv or sys.argv[1:])
    start = datetime.strptime(args.start_date, "%Y-%m-%d")
    end = datetime.strptime(args.end_date, "%Y-%m-%d")
    if args.iatas:
        iatas = [x.strip().upper() for x in args.iatas.split(",") if x.strip()]
    else:
        iatas = default_airport_iatas()

    cache_dir = Path(args.cache_dir).resolve()
    print(f"Airports: {len(iatas)}")
    print(f"Date range: {start:%Y-%m-%d} .. {end:%Y-%m-%d}")
    print(f"Cache dir: {cache_dir}")
    print(f"Source: {args.source}")
    report = collect(iatas, start, end, cache_dir, force=args.force, source=args.source)
    ok = sum(1 for v in report.values() if v > 0)
    empty = sum(1 for v in report.values() if v == 0)
    print()
    print(f"[SUMMARY] ok={ok}/{len(iatas)} empty_or_failed={empty}")


if __name__ == "__main__":
    main()
