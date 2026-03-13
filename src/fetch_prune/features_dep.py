#!/usr/bin/env python3
# src/fetch_prune/features_dep.py
#
# Run from REPO_ROOT:
#   python src/fetch_prune/features_dep.py data/dep_arr_config.json
#
import sys
import json
from pathlib import Path
from datetime import date, timedelta
from typing import Optional, List, Tuple, Dict
from collections import deque

import numpy as np
import pandas as pd


REPO_ROOT = Path.cwd()
DATA_ROOT = (REPO_ROOT.parent / "flightrightdata").resolve()


def _abspath(p: str, *, base: str = "repo") -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    if base == "repo":
        return (REPO_ROOT / pp).resolve()
    if base == "data":
        return (DATA_ROOT / pp).resolve()
    raise ValueError("base must be 'repo' or 'data'")


def _format_path_template(p: str, *, target: str, thr: Optional[int] = None) -> str:
    if thr is None:
        return p.format(target=target)
    return p.format(target=target, thr=thr)


# ------------------------------ parquet read helpers (column projection) ------------------------------

def _parquet_columns(path: Path) -> Optional[List[str]]:
    try:
        import pyarrow.parquet as pq  # type: ignore
        pf = pq.ParquetFile(str(path))
        return list(pf.schema.names)
    except Exception:
        return None


def _read_parquet_projected(path: Path, want_cols: List[str]) -> pd.DataFrame:
    cols = _parquet_columns(path)
    if cols is None:
        return pd.read_parquet(path)

    keep = [c for c in want_cols if c in cols]
    if not keep:
        return pd.read_parquet(path)

    return pd.read_parquet(path, columns=keep)


# ---------- holidays ----------
def _nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date:
    d = date(year, month, 1)
    shift = (weekday - d.weekday()) % 7
    d = d + timedelta(days=shift)
    d = d + timedelta(weeks=n - 1)
    return d


def thanksgiving_day(year: int) -> date:
    return _nth_weekday_of_month(year, 11, weekday=3, n=4)


def _last_weekday_of_month(year: int, month: int, weekday: int) -> date:
    """Last occurrence of `weekday` (0=Mon…6=Sun) in the given month."""
    import calendar as _cal
    last_day_num = _cal.monthrange(year, month)[1]
    d = date(year, month, last_day_num)
    # weekday() returns 0=Mon…6=Sun; shift backward to reach desired weekday
    shift = (d.weekday() - weekday) % 7
    return d - timedelta(days=shift)


def _us_travel_holidays(year: int) -> List[Tuple[date, int]]:
    """
    Returns list of (holiday_date, window_days) for travel-impactful US holidays.
    Window = how many days ± around the anchor date to flag as 'holiday'.
    """
    mlk_day       = _nth_weekday_of_month(year, 1, weekday=0, n=3)  # 3rd Mon Jan
    presidents_day = _nth_weekday_of_month(year, 2, weekday=0, n=3) # 3rd Mon Feb
    memorial_day  = _last_weekday_of_month(year, 5, weekday=0)       # Last Mon May
    labor_day     = _nth_weekday_of_month(year, 9, weekday=0, n=1)   # 1st Mon Sep

    return [
        (date(year, 1, 1),        1),   # New Year's Day
        (mlk_day,                 1),   # MLK Day (Mon holiday, light travel)
        (presidents_day,          1),   # Presidents' Day
        (memorial_day,            2),   # Memorial Day weekend (big travel)
        (date(year, 7, 4),        2),   # Independence Day
        (labor_day,               2),   # Labor Day weekend (big travel)
        (thanksgiving_day(year),  2),   # Thanksgiving
        (date(year, 12, 25),      3),   # Christmas
    ]


def add_holiday_flag(
    df: pd.DataFrame,
    thanksgiving_window: int = 2,
    christmas_window: int = 3,
    spring_break_window: Tuple[Tuple[int,int], Tuple[int,int]] = ((3, 15), (4, 15)),
) -> pd.DataFrame:
    """
    Sets is_holiday=1 for major US travel holidays + spring break proxy.
    thanksgiving_window / christmas_window kept for backward compat but
    the full holiday list now uses built-in windows per holiday.
    Spring break is flagged as is_spring_break separately.
    """
    df = df.copy()
    base = pd.to_datetime(df["FlightDate"], errors="coerce")
    years = sorted(pd.unique(base.dt.year.dropna()))

    # Build per-year set of (date, window) pairs
    holiday_map: Dict[int, List[Tuple[date, int]]] = {}
    for y in years:
        iy = int(y)
        # Override thanksgiving/christmas windows from config
        hols = _us_travel_holidays(iy)
        # Apply config overrides for thanksgiving/christmas
        hols_out = []
        for h_date, h_win in hols:
            if h_date.month == 11:  # thanksgiving
                hols_out.append((h_date, thanksgiving_window))
            elif h_date.month == 12 and h_date.day == 25:  # christmas
                hols_out.append((h_date, christmas_window))
            else:
                hols_out.append((h_date, h_win))
        holiday_map[iy] = hols_out

    def is_near(d0: date, center: date, win: int) -> bool:
        return abs((d0 - center).days) <= win

    # Spring break: March 15 - April 15 (proxy for US spring break travel peak)
    sb_start = spring_break_window[0]  # (month, day)
    sb_end   = spring_break_window[1]

    flags = []
    sb_flags = []
    for ts in base:
        if pd.isna(ts):
            flags.append(0)
            sb_flags.append(0)
            continue
        d0 = ts.date()
        y = d0.year
        f = 0
        hols = holiday_map.get(y, [])
        for h_date, h_win in hols:
            if is_near(d0, h_date, h_win):
                f = 1
                break
        flags.append(f)
        # Spring break flag
        try:
            sb_s = date(y, sb_start[0], sb_start[1])
            sb_e = date(y, sb_end[0],   sb_end[1])
            sb_flags.append(1 if sb_s <= d0 <= sb_e else 0)
        except Exception:
            sb_flags.append(0)

    df["is_holiday"] = pd.Series(flags, index=df.index).astype("Int8")
    df["is_spring_break"] = pd.Series(sb_flags, index=df.index).astype("Int8")
    return df


# ---------- weather unit conversions ----------
def add_weather_kelvin(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Daily origin weather ---
    if "origin_temperature_2m_max" in df.columns:
        df["origin_temp_max_K"] = pd.to_numeric(df["origin_temperature_2m_max"], errors="coerce") + 273.15
    if "origin_temperature_2m_min" in df.columns:
        df["origin_temp_min_K"] = pd.to_numeric(df["origin_temperature_2m_min"], errors="coerce") + 273.15
    if "origin_precipitation_sum" in df.columns:
        df["origin_daily_precip_sum_mm"] = pd.to_numeric(df["origin_precipitation_sum"], errors="coerce")
    if "origin_windspeed_10m_max" in df.columns:
        df["origin_daily_windspeed_max_kmh"] = pd.to_numeric(df["origin_windspeed_10m_max"], errors="coerce")
    # Daily windgusts (already km/h from Open-Meteo)
    if "origin_windgusts_10m_max" in df.columns:
        df["origin_daily_windgusts_max_kmh"] = pd.to_numeric(df["origin_windgusts_10m_max"], errors="coerce")

    # --- Hourly origin weather at departure time ---
    if "origin_dep_temperature_2m" in df.columns:
        df["origin_dep_temp_K"] = pd.to_numeric(df["origin_dep_temperature_2m"], errors="coerce") + 273.15
    if "origin_dep_precipitation" in df.columns:
        df["origin_dep_precip_mm"] = pd.to_numeric(df["origin_dep_precipitation"], errors="coerce")
    if "origin_dep_windspeed_10m" in df.columns:
        df["origin_dep_windspeed_kmh"] = pd.to_numeric(df["origin_dep_windspeed_10m"], errors="coerce")
    # Extended hourly: windgusts, visibility, CAPE, cloudcover
    if "origin_dep_windgusts_10m" in df.columns:
        df["origin_dep_windgusts_kmh"] = pd.to_numeric(df["origin_dep_windgusts_10m"], errors="coerce")
    if "origin_dep_visibility" in df.columns:
        # visibility in meters from Open-Meteo; cap at 25000m (25 km = clear VFR ceiling)
        vis = pd.to_numeric(df["origin_dep_visibility"], errors="coerce")
        df["origin_dep_visibility_m"] = vis.clip(upper=25000.0)
    if "origin_dep_cape" in df.columns:
        # CAPE in J/kg; log1p transform to compress the heavy right tail
        cape = pd.to_numeric(df["origin_dep_cape"], errors="coerce").clip(lower=0.0)
        df["origin_dep_cape_jkg"] = cape
        df["origin_dep_log1p_cape"] = np.log1p(cape)
    if "origin_dep_cloudcover" in df.columns:
        df["origin_dep_cloudcover_pct"] = pd.to_numeric(df["origin_dep_cloudcover"], errors="coerce")

    return df


# ---------- helper: ensure we have a usable scheduled-departure datetime ----------
def _ensure_dep_dt(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "dep_dt_local" in df.columns:
        df["dep_dt_local"] = pd.to_datetime(df["dep_dt_local"], errors="coerce")
    else:
        fd = pd.to_datetime(df.get("FlightDate"), errors="coerce")
        hhmm = pd.to_numeric(df.get("CRSDepTime"), errors="coerce")

        def _mk(fd0, t0):
            if pd.isna(fd0) or pd.isna(t0):
                return pd.NaT
            try:
                s = str(int(t0)).zfill(4)
                hh = int(s[:2])
                mm = int(s[2:])
                return pd.Timestamp(fd0.date()) + pd.Timedelta(hours=hh, minutes=mm)
            except Exception:
                return pd.NaT

        df["dep_dt_local"] = [_mk(a, b) for a, b in zip(fd.tolist(), hhmm.tolist())]

    dts = pd.to_datetime(df["dep_dt_local"], errors="coerce")
    df["dep_local_date"] = dts.dt.date
    return df


# ---------- congestion features (±3 hours around scheduled dep), from FULL HISTORY ----------

def _counts_within_window_for_queries(sorted_pool_ns: np.ndarray, sorted_query_ns: np.ndarray, window_ns: int) -> np.ndarray:
    n_pool = len(sorted_pool_ns)
    out = np.zeros(len(sorted_query_ns), dtype=np.int32)
    left = 0
    right = 0
    for i, t in enumerate(sorted_query_ns):
        lo = t - window_ns
        hi = t + window_ns
        while left < n_pool and sorted_pool_ns[left] < lo:
            left += 1
        if right < left:
            right = left
        while right < n_pool and sorted_pool_ns[right] <= hi:
            right += 1
        out[i] = max(0, right - left)
    return out


def _build_congestion_time_dicts_from_history(
    hist: pd.DataFrame,
) -> Tuple[Dict[Tuple[str, object], np.ndarray], Dict[Tuple[str, str, object], np.ndarray]]:
    h = hist.copy()
    h["Origin"] = h.get("Origin", "").astype(str).str.upper().str.strip()
    h["Reporting_Airline"] = h.get("Reporting_Airline", "").astype(str).str.upper().str.strip()

    if "dep_dt_local" not in h.columns:
        raise KeyError("History pool must include dep_dt_local for congestion-from-history. Re-run prepare_dataset.py to rebuild history pool with dep_dt_local kept.")

    dep_local = pd.to_datetime(h["dep_dt_local"], errors="coerce")
    dep_utc = pd.to_datetime(dep_local, errors="coerce", utc=True)
    dep_ns = dep_utc.astype("int64")
    valid = dep_ns != np.iinfo("int64").min

    if "dep_local_date" in h.columns:
        dep_local_date = pd.Series(h["dep_local_date"])
    else:
        dep_local_date = dep_local.dt.date

    h2 = pd.DataFrame(
        {
            "Origin": h.loc[valid, "Origin"].values,
            "Reporting_Airline": h.loc[valid, "Reporting_Airline"].values,
            "dep_local_date": pd.Series(dep_local_date.loc[valid].values),
            "dep_ns": dep_ns[valid],
        }
    )

    h2 = h2.sort_values(["Origin", "dep_local_date", "dep_ns"], kind="mergesort")

    times_by_origin_date: Dict[Tuple[str, object], np.ndarray] = {}
    for (o, d0), g in h2.groupby(["Origin", "dep_local_date"], sort=False):
        times_by_origin_date[(o, d0)] = g["dep_ns"].to_numpy(dtype=np.int64, copy=False)

    h3 = h2.sort_values(["Origin", "Reporting_Airline", "dep_local_date", "dep_ns"], kind="mergesort")
    times_by_origin_airline_date: Dict[Tuple[str, str, object], np.ndarray] = {}
    for (o, carr, d0), g in h3.groupby(["Origin", "Reporting_Airline", "dep_local_date"], sort=False):
        times_by_origin_airline_date[(o, carr, d0)] = g["dep_ns"].to_numpy(dtype=np.int64, copy=False)

    return times_by_origin_date, times_by_origin_airline_date


def add_congestion_3h_features_from_history(df: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _ensure_dep_dt(df)

    df["Origin"] = df.get("Origin", "").astype(str).str.upper().str.strip()
    df["Reporting_Airline"] = df.get("Reporting_Airline", "").astype(str).str.upper().str.strip()

    dep_local = pd.to_datetime(df["dep_dt_local"], errors="coerce")
    dep_utc = pd.to_datetime(dep_local, errors="coerce", utc=True)
    dep_ns = dep_utc.astype("int64")
    valid = dep_ns != np.iinfo("int64").min
    if "dep_local_date" not in df.columns:
        df["dep_local_date"] = dep_local.dt.date

    df["origin_congestion_3h_total"] = pd.Series([0] * len(df), index=df.index, dtype="Int16")
    df["origin_airline_congestion_3h_total"] = pd.Series([0] * len(df), index=df.index, dtype="Int16")
    if not valid.any():
        return df

    WINDOW_NS = int(pd.Timedelta(hours=3).value)
    times_by_origin_date, times_by_origin_airline_date = _build_congestion_time_dicts_from_history(hist)

    work = pd.DataFrame(
        {
            "idx": df.index[valid],
            "Origin": df.loc[valid, "Origin"].values,
            "Reporting_Airline": df.loc[valid, "Reporting_Airline"].values,
            "dep_local_date": pd.Series(df.loc[valid, "dep_local_date"].values),
            "dep_ns": dep_ns[valid],
        }
    )

    for (o, d0), g in work.groupby(["Origin", "dep_local_date"], sort=False):
        pool = times_by_origin_date.get((o, d0))
        if pool is None or len(pool) == 0:
            continue
        g2 = g.sort_values("dep_ns", kind="mergesort")
        q = g2["dep_ns"].to_numpy(dtype=np.int64, copy=False)
        counts = _counts_within_window_for_queries(pool, q, WINDOW_NS)
        counts = np.maximum(0, counts - 1)
        counts = np.clip(counts, 0, np.iinfo(np.int16).max).astype(np.int16)
        df.loc[g2["idx"].to_numpy(), "origin_congestion_3h_total"] = pd.Series(counts, index=g2["idx"].to_numpy()).astype("Int16")

    for (o, carr, d0), g in work.groupby(["Origin", "Reporting_Airline", "dep_local_date"], sort=False):
        pool = times_by_origin_airline_date.get((o, carr, d0))
        if pool is None or len(pool) == 0:
            continue
        g2 = g.sort_values("dep_ns", kind="mergesort")
        q = g2["dep_ns"].to_numpy(dtype=np.int64, copy=False)
        counts = _counts_within_window_for_queries(pool, q, WINDOW_NS)
        counts = np.maximum(0, counts - 1)
        counts = np.clip(counts, 0, np.iinfo(np.int16).max).astype(np.int16)
        df.loc[g2["idx"].to_numpy(), "origin_airline_congestion_3h_total"] = pd.Series(counts, index=g2["idx"].to_numpy()).astype("Int16")

    return df


# ---------- tail + flight-number temporal features ----------
def add_tail_and_flightnum_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _ensure_dep_dt(df)

    df["Origin"] = df.get("Origin", "").astype(str).str.upper().str.strip()
    df["Dest"] = df.get("Dest", "").astype(str).str.upper().str.strip()
    df["Reporting_Airline"] = df.get("Reporting_Airline", "").astype(str).str.upper().str.strip()

    tail = df.get("Tail_Number", "").astype(str).str.upper().str.strip()
    bad_tail = tail.isin(["", "NAN", "NONE", "NULL", "UNKNOWN", "UNKN", "NA"])
    tail = tail.where(~bad_tail, "")
    df["Tail_Number"] = tail

    df["Flight_Number_Reporting_Airline"] = pd.to_numeric(df.get("Flight_Number_Reporting_Airline"), errors="coerce")

    dep_utc = pd.to_datetime(df.get("dep_dt_local"), errors="coerce", utc=True)
    arr_utc = pd.to_datetime(df.get("arr_dt_local"), errors="coerce", utc=True) if "arr_dt_local" in df.columns else pd.to_datetime(pd.Series([pd.NaT] * len(df)), utc=True)

    df["_dep_utc"] = dep_utc
    df["_arr_utc"] = arr_utc

    dep_local_date = df.get("dep_local_date")
    if dep_local_date is None:
        dts = pd.to_datetime(df.get("dep_dt_local"), errors="coerce")
        dep_local_date = dts.dt.date
        df["dep_local_date"] = dep_local_date

    valid_tail = df["Tail_Number"].astype(str).str.len() > 0
    valid_dep = df["_dep_utc"].notna()
    valid_day = pd.Series(dep_local_date).notna()

    df["tail_leg_num_day"] = pd.Series([pd.NA] * len(df), index=df.index, dtype="Int16")
    mask_leg = valid_tail & valid_dep & valid_day
    if mask_leg.any():
        key = ["Tail_Number", "dep_local_date"]
        order = df.loc[mask_leg].sort_values(key + ["_dep_utc"]).copy()
        order["tail_leg_num_day"] = order.groupby(key, sort=False).cumcount() + 1
        df.loc[order.index, "tail_leg_num_day"] = pd.to_numeric(order["tail_leg_num_day"], errors="coerce").astype("Int16")

    df["flightnum_hours_since_first_departure_today"] = np.nan
    valid_fn = df["Flight_Number_Reporting_Airline"].notna() & valid_dep & valid_day
    if valid_fn.any():
        key = ["Reporting_Airline", "Flight_Number_Reporting_Airline", "dep_local_date"]
        first_dep = df.loc[valid_fn].groupby(key, sort=False)["_dep_utc"].transform("min")
        hours = (df.loc[valid_fn, "_dep_utc"] - first_dep).dt.total_seconds() / 3600.0
        df.loc[valid_fn, "flightnum_hours_since_first_departure_today"] = hours.astype(float)

    df["has_recent_arrival_turn_5h"] = pd.Series(np.zeros(len(df), dtype=np.int8), index=df.index).astype("Int8")

    ok_rows = valid_tail & valid_dep
    if ok_rows.any():
        w = df.loc[ok_rows, ["Tail_Number", "Origin", "Dest", "_dep_utc", "_arr_utc"]].copy()
        w = w.sort_values(["Tail_Number", "_dep_utc"])

        w["prev_arr_utc"] = w.groupby("Tail_Number", sort=False)["_arr_utc"].shift(1)
        w["prev_dest"] = w.groupby("Tail_Number", sort=False)["Dest"].shift(1)

        gap_h = (w["_dep_utc"] - w["prev_arr_utc"]).dt.total_seconds() / 3600.0
        cond = (
            w["prev_arr_utc"].notna()
            & w["prev_dest"].notna()
            & (w["prev_dest"].astype(str).str.upper() == w["Origin"].astype(str).str.upper())
            & (gap_h >= 0.0)
            & (gap_h <= 5.0)
        )

        df.loc[w.index, "has_recent_arrival_turn_5h"] = pd.Series(cond.astype(np.int8).values, index=w.index).astype("Int8")

    df = df.drop(columns=["_dep_utc", "_arr_utc"], errors="ignore")
    return df


# ---------- NEW: flight-number OD rolling means over last {7,14,21,28} days ----------
def add_flightnum_od_depdelay_means_from_history(
    df: pd.DataFrame,
    hist: pd.DataFrame,
    windows_days: List[int],
    *,
    n_recent: int = 20,
    low_support_leq: int = 2,
) -> pd.DataFrame:
    """
    Produces features:
      - flightnum_od_depdelay_mean_last{W}  for W in windows_days (calendar-day windows)
      - flightnum_od_support_count_last{Wmax}d
      - flightnum_od_low_support_last{Wmax}d

    IMPORTANT:
      - uses strictly PRIOR observations (no leakage) by processing in time order and only
        adding current-row delay AFTER computing features for that row.
      - windowed means are over the last W calendar days (not last W flights).
    """
    df = df.copy()
    windows = sorted({int(w) for w in windows_days if int(w) > 0})
    if not windows:
        return df
    wmax = int(max(windows))

    key = ["Reporting_Airline", "Flight_Number_Reporting_Airline", "Origin", "Dest"]

    for c in ["Reporting_Airline", "Origin", "Dest"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.upper().str.strip()
    df["Flight_Number_Reporting_Airline"] = pd.to_numeric(df.get("Flight_Number_Reporting_Airline"), errors="coerce")
    df["FlightDate"] = pd.to_datetime(df.get("FlightDate"), errors="coerce").dt.normalize()

    hist = hist.copy()
    for c in ["Reporting_Airline", "Origin", "Dest"]:
        if c in hist.columns:
            hist[c] = hist[c].astype(str).str.upper().str.strip()
    hist["Flight_Number_Reporting_Airline"] = pd.to_numeric(hist.get("Flight_Number_Reporting_Airline"), errors="coerce")
    hist["FlightDate"] = pd.to_datetime(hist.get("FlightDate"), errors="coerce").dt.normalize()
    hist["DepDelayMinutes"] = pd.to_numeric(hist.get("DepDelayMinutes"), errors="coerce")

    need_df = ["FlightDate"] + key + ["DepDelayMinutes", "dep_dt_local"]
    need_hist = ["FlightDate"] + key + ["DepDelayMinutes", "dep_dt_local"]
    need_df = [c for c in need_df if c in df.columns]
    need_hist = [c for c in need_hist if c in hist.columns]

    df_small = df[need_df].copy()
    hist_small = hist[need_hist].copy()

    df_small = df_small.dropna(subset=["FlightDate"] + key).copy()
    hist_small = hist_small.dropna(subset=["FlightDate"] + key).copy()

    def _time_i8(frame: pd.DataFrame) -> np.ndarray:
        if "dep_dt_local" in frame.columns:
            t = pd.to_datetime(frame["dep_dt_local"], errors="coerce", utc=True)
            out = t.astype("int64")
            out = np.where(out == np.iinfo("int64").min, 0, out)
            return out.astype(np.int64, copy=False)
        return np.zeros(len(frame), dtype=np.int64)

    df_small["_t"] = _time_i8(df_small)
    hist_small["_t"] = _time_i8(hist_small)

    sort_cols = key + ["FlightDate", "_t"]
    df_small = df_small.sort_values(sort_cols, kind="mergesort")
    hist_small = hist_small.sort_values(sort_cols, kind="mergesort")

    # output columns
    out_means = {w: pd.Series(np.nan, index=df.index, dtype="float64") for w in windows}
    out_medians = {w: pd.Series(np.nan, index=df.index, dtype="float64") for w in windows}
    out_stds = {w: pd.Series(np.nan, index=df.index, dtype="float64") for w in windows}
    out_supp = pd.Series(np.zeros(len(df), dtype=np.int16), index=df.index, dtype="Int16")
    out_low = pd.Series(np.ones(len(df), dtype=np.int8), index=df.index, dtype="Int8")

    # build grouped history arrays
    hist_groups: Dict[Tuple, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    if not hist_small.empty:
        h_dates = hist_small["FlightDate"].to_numpy()
        h_date_int = h_dates.astype("datetime64[D]").astype(np.int32)
        h_t = hist_small["_t"].to_numpy(dtype=np.int64, copy=False)
        h_delay = pd.to_numeric(hist_small["DepDelayMinutes"], errors="coerce").to_numpy(dtype=np.float32, copy=False)

        h_key_df = hist_small[key]
        h_key_vals = [h_key_df[c].to_numpy() for c in key]
        n = len(hist_small)

        if n > 0:
            starts = [0]
            for i in range(1, n):
                changed = False
                for arr in h_key_vals:
                    if arr[i] != arr[i - 1]:
                        changed = True
                        break
                if changed:
                    starts.append(i)
            starts.append(n)

            def _key_at(i: int) -> Tuple:
                return tuple(arr[i] for arr in h_key_vals)

            for a, b in zip(starts[:-1], starts[1:]):
                k = _key_at(a)
                hist_groups[k] = (h_date_int[a:b], h_t[a:b], h_delay[a:b])

    if df_small.empty:
        for w in windows:
            df[f"flightnum_od_depdelay_mean_last{w}"] = out_means[w]
        df[f"flightnum_od_support_count_last{wmax}d"] = out_supp
        df[f"flightnum_od_low_support_last{wmax}d"] = out_low
        return df

    d_dates = df_small["FlightDate"].to_numpy()
    d_date_int = d_dates.astype("datetime64[D]").astype(np.int32)
    d_t = df_small["_t"].to_numpy(dtype=np.int64, copy=False)
    d_delay = pd.to_numeric(df_small.get("DepDelayMinutes"), errors="coerce").to_numpy(dtype=np.float32, copy=False)
    d_idx = df_small.index.to_numpy()

    d_key_df = df_small[key]
    d_key_vals = [d_key_df[c].to_numpy() for c in key]
    ndf = len(df_small)

    starts = [0]
    for i in range(1, ndf):
        changed = False
        for arr in d_key_vals:
            if arr[i] != arr[i - 1]:
                changed = True
                break
        if changed:
            starts.append(i)
    starts.append(ndf)

    def _df_key_at(i: int) -> Tuple:
        return tuple(arr[i] for arr in d_key_vals)

    for a, b in zip(starts[:-1], starts[1:]):
        k = _df_key_at(a)

        df_dates_i = d_date_int[a:b]
        df_t_i = d_t[a:b]
        df_delay_i = d_delay[a:b]
        df_idx_i = d_idx[a:b]

        h = hist_groups.get(k)
        if h is None:
            h_dates_i = np.empty(0, dtype=np.int32)
            h_t_i = np.empty(0, dtype=np.int64)
            h_delay_i = np.empty(0, dtype=np.float32)
        else:
            h_dates_i, h_t_i, h_delay_i = h

        i_hist = 0
        i_df = 0
        n_hist = len(h_dates_i)
        n_df_g = len(df_dates_i)

        # Maintain:
        # - lastN flights deque for "recent flight history" (not requested as feature but useful if you later want it)
        # - window deques for each W in days: store (date_int, delay) for last W calendar days
        lastN = deque(maxlen=int(n_recent))
        win_deques = {w: deque() for w in windows}
        win_sums = {w: 0.0 for w in windows}
        win_counts = {w: 0 for w in windows}

        def _evict(win: int, current_date_int: int):
            cutoff = current_date_int - int(win)
            dq = win_deques[win]
            while dq and dq[0][0] < cutoff:
                d0, v0 = dq.popleft()
                win_sums[win] -= float(v0)
                win_counts[win] -= 1

        def _evict_all(current_date_int: int):
            for w in windows:
                _evict(w, current_date_int)

        def _push_to_all(date_int: int, delayv: float):
            for w in windows:
                # store the observation at its FlightDate
                win_deques[w].append((date_int, float(delayv)))
                win_sums[w] += float(delayv)
                win_counts[w] += 1

        while i_df < n_df_g:
            next_df_date = int(df_dates_i[i_df])
            next_df_t = int(df_t_i[i_df])

            # add all HISTORY events strictly before this df row (date, then time)
            while i_hist < n_hist:
                hd = int(h_dates_i[i_hist])
                ht = int(h_t_i[i_hist])
                if (hd < next_df_date) or (hd == next_df_date and ht < next_df_t):
                    delayv = float(h_delay_i[i_hist])
                    if np.isfinite(delayv):
                        lastN.append(delayv)
                        _push_to_all(hd, delayv)
                    i_hist += 1
                else:
                    break

            # evict old observations per window
            _evict_all(next_df_date)

            # compute features for this df row (based on prior obs only)
            for w in windows:
                if win_counts[w] > 0:
                    vals_w = [x[1] for x in win_deques[w]]
                    out_means[w].loc[df_idx_i[i_df]] = float(win_sums[w] / max(1, win_counts[w]))
                    out_medians[w].loc[df_idx_i[i_df]] = float(np.median(vals_w))
                    if len(vals_w) > 1:
                        out_stds[w].loc[df_idx_i[i_df]] = float(np.std(vals_w, ddof=0))
                else:
                    out_means[w].loc[df_idx_i[i_df]] = np.nan
                    out_medians[w].loc[df_idx_i[i_df]] = np.nan
                    out_stds[w].loc[df_idx_i[i_df]] = np.nan

            # support + low-support based on longest window
            supp = int(win_counts[wmax])
            out_supp.loc[df_idx_i[i_df]] = np.int16(min(supp, np.iinfo(np.int16).max))
            out_low.loc[df_idx_i[i_df]] = np.int8(1 if supp <= int(low_support_leq) else 0)

            # now add THIS row’s delay into the windows for future rows (still no leakage)
            dv = float(df_delay_i[i_df]) if np.isfinite(df_delay_i[i_df]) else np.nan
            if np.isfinite(dv):
                lastN.append(dv)
                _push_to_all(next_df_date, dv)

            i_df += 1

    for w in windows:
        df[f"flightnum_od_depdelay_mean_last{w}"] = pd.to_numeric(out_means[w], errors="coerce")
        df[f"flightnum_od_depdelay_median_last{w}"] = pd.to_numeric(out_medians[w], errors="coerce")
        df[f"flightnum_od_depdelay_std_last{w}"] = pd.to_numeric(out_stds[w], errors="coerce")

    df[f"flightnum_od_support_count_last{wmax}d"] = pd.to_numeric(out_supp, errors="coerce").fillna(0).astype("Int16")
    df[f"flightnum_od_low_support_last{wmax}d"] = pd.to_numeric(out_low, errors="coerce").fillna(1).astype("Int8")
    return df


# ---------- NEW: carrier + carrier-origin rolling means/stds over multiple windows ----------
def add_carrier_dep_delay_baselines_from_history_multi(
    df: pd.DataFrame,
    hist: pd.DataFrame,
    windows_days: List[int],
) -> pd.DataFrame:
    """
    Adds:
      carrier_depdelay_mean_last{W}
      carrier_depdelay_std_last{W}   ← volatility signal
      carrier_origin_depdelay_mean_last{W}
    where W is in windows_days.

    Uses daily carrier (and carrier-origin) mean dep delay, then rolling over W days, shifted by 1 day (no leakage).
    std captures carrier operational volatility: high std = chaotic ops (weather/ATC events).
    """
    df = df.copy()
    windows = sorted({int(w) for w in windows_days if int(w) > 0})
    if not windows:
        return df

    df["FlightDate"] = pd.to_datetime(df["FlightDate"], errors="coerce").dt.normalize()
    df["Reporting_Airline"] = df["Reporting_Airline"].astype(str).str.upper().str.strip()
    df["Origin"] = df["Origin"].astype(str).str.upper().str.strip()

    hist = hist.copy()
    hist["FlightDate"] = pd.to_datetime(hist.get("FlightDate"), errors="coerce").dt.normalize()
    hist["Reporting_Airline"] = hist.get("Reporting_Airline", "").astype(str).str.upper().str.strip()
    hist["Origin"] = hist.get("Origin", "").astype(str).str.upper().str.strip()
    hist["DepDelayMinutes"] = pd.to_numeric(hist.get("DepDelayMinutes"), errors="coerce")

    carr = hist.dropna(subset=["Reporting_Airline", "FlightDate", "DepDelayMinutes"]).copy()
    carr_daily = (
        carr.groupby(["Reporting_Airline", "FlightDate"], as_index=False)["DepDelayMinutes"]
        .mean()
        .sort_values(["Reporting_Airline", "FlightDate"])
    )

    for w in windows:
        grp = carr_daily.groupby("Reporting_Airline")["DepDelayMinutes"]
        carr_daily[f"carrier_depdelay_mean_last{w}"] = (
            grp.apply(lambda s: s.shift(1).rolling(window=int(w), min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )
        carr_daily[f"carrier_depdelay_std_last{w}"] = (
            grp.apply(lambda s: s.shift(1).rolling(window=int(w), min_periods=min(2, int(w))).std())
            .reset_index(level=0, drop=True)
        )

    mean_cols = [f"carrier_depdelay_mean_last{w}" for w in windows]
    std_cols  = [f"carrier_depdelay_std_last{w}"  for w in windows]
    keep_cols = ["Reporting_Airline", "FlightDate"] + mean_cols + std_cols
    df = df.merge(carr_daily[keep_cols], on=["Reporting_Airline", "FlightDate"], how="left")

    co = carr.dropna(subset=["Origin"]).copy()
    co_daily = (
        co.groupby(["Reporting_Airline", "Origin", "FlightDate"], as_index=False)["DepDelayMinutes"]
        .mean()
        .sort_values(["Reporting_Airline", "Origin", "FlightDate"])
    )

    for w in windows:
        grp_co = co_daily.groupby(["Reporting_Airline", "Origin"])["DepDelayMinutes"]
        co_daily[f"carrier_origin_depdelay_mean_last{w}"] = (
            grp_co
            .apply(lambda s: s.shift(1).rolling(window=int(w), min_periods=1).mean())
            .reset_index(level=[0, 1], drop=True)
        )
        co_daily[f"carrier_origin_depdelay_std_last{w}"] = (
            grp_co
            .apply(lambda s: s.shift(1).rolling(window=int(w), min_periods=min(2, int(w))).std())
            .reset_index(level=[0, 1], drop=True)
        )

    mean_cols_co = [f"carrier_origin_depdelay_mean_last{w}" for w in windows]
    std_cols_co  = [f"carrier_origin_depdelay_std_last{w}"  for w in windows]
    keep_cols = ["Reporting_Airline", "Origin", "FlightDate"] + mean_cols_co + std_cols_co
    df = df.merge(co_daily[keep_cols], on=["Reporting_Airline", "Origin", "FlightDate"], how="left")

    return df


# ---------- NEW: origin-airport rolling means over multiple windows ----------
def add_origin_dep_delay_baselines_from_history_multi(
    df: pd.DataFrame,
    hist: pd.DataFrame,
    windows_days: List[int],
) -> pd.DataFrame:
    """
    Adds:
      origin_depdelay_mean_last{W}
      origin_depdelay_std_last{W}   -- airport-level operational volatility
    where W is in windows_days.

    Uses daily origin mean dep delay (all carriers), rolling over W days, shifted by 1 day (no leakage).
    """
    df = df.copy()
    windows = sorted({int(w) for w in windows_days if int(w) > 0})
    if not windows:
        return df

    df["FlightDate"] = pd.to_datetime(df["FlightDate"], errors="coerce").dt.normalize()
    df["Origin"] = df["Origin"].astype(str).str.upper().str.strip()

    hist = hist.copy()
    hist["FlightDate"] = pd.to_datetime(hist.get("FlightDate"), errors="coerce").dt.normalize()
    hist["Origin"] = hist.get("Origin", "").astype(str).str.upper().str.strip()
    hist["DepDelayMinutes"] = pd.to_numeric(hist.get("DepDelayMinutes"), errors="coerce")

    o = hist.dropna(subset=["Origin", "FlightDate", "DepDelayMinutes"]).copy()
    o_daily = (
        o.groupby(["Origin", "FlightDate"], as_index=False)["DepDelayMinutes"]
        .mean()
        .sort_values(["Origin", "FlightDate"])
    )

    for w in windows:
        grp_o = o_daily.groupby("Origin")["DepDelayMinutes"]
        o_daily[f"origin_depdelay_mean_last{w}"] = (
            grp_o
            .apply(lambda s: s.shift(1).rolling(window=int(w), min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )
        o_daily[f"origin_depdelay_std_last{w}"] = (
            grp_o
            .apply(lambda s: s.shift(1).rolling(window=int(w), min_periods=min(2, int(w))).std())
            .reset_index(level=0, drop=True)
        )

    mean_cols_o = [f"origin_depdelay_mean_last{w}" for w in windows]
    std_cols_o  = [f"origin_depdelay_std_last{w}"  for w in windows]
    keep_cols = ["Origin", "FlightDate"] + mean_cols_o + std_cols_o
    df = df.merge(o_daily[keep_cols], on=["Origin", "FlightDate"], how="left")
    return df


def add_tail_rolling_history(
    df: pd.DataFrame,
    hist: pd.DataFrame,
    windows_days: List[int],
) -> pd.DataFrame:
    """
    Rolling departure delay stats per individual aircraft (tail number).

    Captures aircraft-specific reliability — maintenance issues, chronic
    problems with a specific airframe show up as persistent delays on that tail.
    Distinct from flight-number history: same flight can fly on different aircraft.

    New columns (per window W in windows_days):
      tail_depdelay_mean_last{W}
      tail_lateaircraft_rate_last{W}

    All computed with 1-day lag (no leakage).
    """
    df = df.copy()
    windows = sorted({int(w) for w in windows_days if int(w) > 0})
    if not windows:
        return df

    if "Tail_Number" not in df.columns or "Tail_Number" not in hist.columns:
        for w in windows:
            df[f"tail_depdelay_mean_last{w}"] = np.nan
            df[f"tail_lateaircraft_rate_last{w}"] = np.nan
        return df

    h = hist.copy()
    h["FlightDate"] = pd.to_datetime(h.get("FlightDate"), errors="coerce").dt.normalize()
    h["Tail_Number"] = h["Tail_Number"].astype(str).str.upper().str.strip()
    h["DepDelayMinutes"] = pd.to_numeric(h.get("DepDelayMinutes"), errors="coerce")

    bad = h["Tail_Number"].isin(["", "NAN", "NONE", "NULL", "UNKNOWN", "UNKN", "NA"])
    h = h[~bad].dropna(subset=["FlightDate", "Tail_Number", "DepDelayMinutes"]).copy()

    has_la = "LateAircraftDelay" in h.columns
    if has_la:
        h["LateAircraftDelay"] = pd.to_numeric(h["LateAircraftDelay"], errors="coerce").fillna(0.0)
        h["_has_la"] = (h["LateAircraftDelay"] > 0).astype(float)

    if has_la:
        tail_daily = (
            h.groupby(["Tail_Number", "FlightDate"], as_index=False)
            .agg(_delay_mean=("DepDelayMinutes", "mean"), _la_rate=("_has_la", "mean"))
            .sort_values(["Tail_Number", "FlightDate"])
        )
    else:
        tail_daily = (
            h.groupby(["Tail_Number", "FlightDate"], as_index=False)
            .agg(_delay_mean=("DepDelayMinutes", "mean"))
            .sort_values(["Tail_Number", "FlightDate"])
        )

    for w in windows:
        tail_daily[f"tail_depdelay_mean_last{w}"] = (
            tail_daily.groupby("Tail_Number")["_delay_mean"]
            .apply(lambda s: s.shift(1).rolling(window=int(w), min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )
        if has_la:
            tail_daily[f"tail_lateaircraft_rate_last{w}"] = (
                tail_daily.groupby("Tail_Number")["_la_rate"]
                .apply(lambda s: s.shift(1).rolling(window=int(w), min_periods=1).mean())
                .reset_index(level=0, drop=True)
            )
        else:
            tail_daily[f"tail_lateaircraft_rate_last{w}"] = np.nan

    feat_cols = (
        [f"tail_depdelay_mean_last{w}" for w in windows] +
        [f"tail_lateaircraft_rate_last{w}" for w in windows]
    )
    df["Tail_Number"] = df["Tail_Number"].astype(str).str.upper().str.strip()
    df["FlightDate"] = pd.to_datetime(df["FlightDate"], errors="coerce").dt.normalize()
    df = df.merge(tail_daily[["Tail_Number", "FlightDate"] + feat_cols],
                  on=["Tail_Number", "FlightDate"], how="left")
    return df


def add_dest_rolling_stats(
    df: pd.DataFrame,
    hist: pd.DataFrame,
    windows_days: List[int],
) -> pd.DataFrame:
    """
    Rolling departure delay stats for the DESTINATION airport.

    Captures destination operational health independently of origin.
    A congested/delayed destination creates ATC ground delay programs that
    ripple back as departure holds at origin airports.  Covers all routes —
    complementing hub_spillover which only covers named hub airports.

    New columns (per window W in windows_days):
      dest_depdelay_mean_last{W}
      dest_lateaircraft_rate_last{W}

    All computed with 1-day lag (no leakage).
    """
    df = df.copy()
    windows = sorted({int(w) for w in windows_days if int(w) > 0})
    if not windows:
        return df

    h = hist.copy()
    h["FlightDate"] = pd.to_datetime(h.get("FlightDate"), errors="coerce").dt.normalize()
    h["Origin"] = h.get("Origin", "").astype(str).str.upper().str.strip()
    h["DepDelayMinutes"] = pd.to_numeric(h.get("DepDelayMinutes"), errors="coerce")

    has_la = "LateAircraftDelay" in h.columns
    if has_la:
        h["LateAircraftDelay"] = pd.to_numeric(h["LateAircraftDelay"], errors="coerce").fillna(0.0)
        h["_has_la"] = (h["LateAircraftDelay"] > 0).astype(float)

    o = h.dropna(subset=["Origin", "FlightDate", "DepDelayMinutes"]).copy()

    if has_la:
        o_daily = (
            o.groupby(["Origin", "FlightDate"], as_index=False)
            .agg(_dep_mean=("DepDelayMinutes", "mean"), _la_rate=("_has_la", "mean"))
            .sort_values(["Origin", "FlightDate"])
        )
    else:
        o_daily = (
            o.groupby(["Origin", "FlightDate"], as_index=False)
            .agg(_dep_mean=("DepDelayMinutes", "mean"))
            .sort_values(["Origin", "FlightDate"])
        )

    for w in windows:
        o_daily[f"dest_depdelay_mean_last{w}"] = (
            o_daily.groupby("Origin")["_dep_mean"]
            .apply(lambda s: s.shift(1).rolling(window=int(w), min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )
        if has_la:
            o_daily[f"dest_lateaircraft_rate_last{w}"] = (
                o_daily.groupby("Origin")["_la_rate"]
                .apply(lambda s: s.shift(1).rolling(window=int(w), min_periods=1).mean())
                .reset_index(level=0, drop=True)
            )
        else:
            o_daily[f"dest_lateaircraft_rate_last{w}"] = np.nan

    feat_cols = (
        [f"dest_depdelay_mean_last{w}" for w in windows] +
        [f"dest_lateaircraft_rate_last{w}" for w in windows]
    )
    df["Dest"] = df.get("Dest", "").astype(str).str.upper().str.strip()
    df["FlightDate"] = pd.to_datetime(df["FlightDate"], errors="coerce").dt.normalize()
    df = df.merge(
        o_daily[["Origin", "FlightDate"] + feat_cols].rename(columns={"Origin": "Dest"}),
        on=["Dest", "FlightDate"],
        how="left",
    )
    return df


def add_wn_hub_spillover_from_history(
    df: pd.DataFrame,
    hist: pd.DataFrame,
    windows_days: List[int],
    hubs: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Captures network-level delay propagation from WN's key hub airports.

    For each hub airport, compute rolling historical:
      - mean departure delay (depdelay_mean)
      - late-aircraft rate (lateaircraft_rate, if LateAircraftDelay available)

    These are joined to ALL flights (not just hub departures) on FlightDate alone,
    capturing the network propagation effect: a bad day at DEN yesterday tends to
    ripple through the entire WN network tomorrow (validated r=0.38 for DEN last1).

    All rolling stats use a 1-day shift (shift(1)) to prevent leakage.

    New features (per hub H, per window W):
      hub_{H}_depdelay_mean_last{W}
      hub_{H}_lateaircraft_rate_last{W}   (only if LateAircraftDelay in history)

    Parameters
    ----------
    hubs : list of IATA codes.  Defaults to WN's top hubs by next-day spillover
           correlation: ['DEN', 'PHX', 'BWI', 'MDW', 'BNA']
    """
    df = df.copy()
    windows = sorted({int(w) for w in windows_days if int(w) > 0})
    if not windows:
        return df

    if hubs is None:
        hubs = ["DEN", "PHX", "BWI", "MDW", "BNA"]  # WN default; see AIRLINE_HUB_PRESETS
    hubs = [h.upper().strip() for h in hubs]

    h = hist.copy()
    h["FlightDate"] = pd.to_datetime(h.get("FlightDate"), errors="coerce").dt.normalize()
    h["Origin"] = h.get("Origin", "").astype(str).str.upper().str.strip()
    h["DepDelayMinutes"] = pd.to_numeric(h.get("DepDelayMinutes"), errors="coerce")

    has_late_aircraft = "LateAircraftDelay" in h.columns
    if has_late_aircraft:
        h["LateAircraftDelay"] = pd.to_numeric(h.get("LateAircraftDelay"), errors="coerce").fillna(0.0)
        h["_has_lateaircraft"] = (h["LateAircraftDelay"] > 0).astype(float)

    h_hub = h[h["Origin"].isin(hubs)].dropna(subset=["FlightDate"]).copy()

    # Daily aggregates per hub airport
    agg_dict = {"DepDelayMinutes": "mean"}
    if has_late_aircraft:
        agg_dict["_has_lateaircraft"] = "mean"

    hub_daily = (
        h_hub.groupby(["Origin", "FlightDate"], as_index=False)
        .agg(agg_dict)
        .rename(columns={"DepDelayMinutes": "_dep_mean", "_has_lateaircraft": "_la_rate"})
        .sort_values(["Origin", "FlightDate"])
    )

    # Rolling with 1-day shift per hub, then pivot to wide (one row per date)
    all_dates = hub_daily["FlightDate"].drop_duplicates().sort_values()
    date_df = pd.DataFrame({"FlightDate": all_dates})

    for hub in hubs:
        h_sub = hub_daily[hub_daily["Origin"] == hub].set_index("FlightDate")
        h_sub = h_sub.reindex(all_dates)  # fill missing dates with NaN

        for w in windows:
            _w = int(w)
            rolled_mean = (
                h_sub["_dep_mean"]
                .shift(1)
                .rolling(window=_w, min_periods=1)
                .mean()
            )
            date_df[f"hub_{hub}_depdelay_mean_last{w}"] = rolled_mean.values

            if has_late_aircraft and "_la_rate" in h_sub.columns:
                rolled_la = (
                    h_sub["_la_rate"]
                    .shift(1)
                    .rolling(window=_w, min_periods=1)
                    .mean()
                )
                date_df[f"hub_{hub}_lateaircraft_rate_last{w}"] = rolled_la.values
            else:
                date_df[f"hub_{hub}_lateaircraft_rate_last{w}"] = np.nan

    df["FlightDate"] = pd.to_datetime(df.get("FlightDate"), errors="coerce").dt.normalize()
    df = df.merge(date_df, on="FlightDate", how="left")
    return df


# Generic alias (works for any airline when hubs list is provided or looked up via AIRLINE_HUB_PRESETS)
add_hub_spillover_from_history = add_wn_hub_spillover_from_history


# ---------- Hub airport weather (Open-Meteo, daily) ----------

# Fixed coordinates for WN's key hubs.  Lat/lon taken from official airport data.
# tz is the IANA timezone used by Open-Meteo for daily aggregation.
WN_HUB_COORDS: Dict[str, Tuple[float, float, str]] = {
    "DEN": (39.8561, -104.6737, "America/Denver"),
    "PHX": (33.4373, -112.0078, "America/Phoenix"),
    "BWI": (39.1754, -76.6684, "America/New_York"),
    "MDW": (41.7868, -87.7522, "America/Chicago"),
    "BNA": (36.1245, -86.6782, "America/Chicago"),
}

# Hub lists for major US carriers, ordered by network-spillover importance.
# Override per-airline via cfg["features_dep"]["hub_spillover"]["hubs"] or
# use cfg["features_dep"]["hub_spillover"]["airline"] to select a preset.
AIRLINE_HUB_PRESETS: Dict[str, List[str]] = {
    "WN": ["DEN", "PHX", "BWI", "MDW", "BNA"],   # Southwest
    "AA": ["DFW", "CLT", "ORD", "MIA", "PHX"],   # American
    "DL": ["ATL", "DTW", "MSP", "LAX", "SLC"],   # Delta
    "UA": ["ORD", "EWR", "IAH", "DEN", "LAX"],   # United
    "B6": ["JFK", "BOS", "FLL", "MCO", "LGB"],   # JetBlue
    "AS": ["SEA", "LAX", "PDX", "ANC", "SFO"],   # Alaska
    "NK": ["FLL", "MCO", "DFW", "ORD", "DEN"],   # Spirit
    "F9": ["DEN", "MCO", "FLL", "LAX", "PHX"],   # Frontier
    "G4": ["PIE", "SFB", "AZA", "PGD", "LAL"],   # Allegiant
}

# Alias without the WN-specific name so callers for other airlines read naturally.

# Daily variables requested from Open-Meteo archive
_HUB_DAILY_VARS = [
    "temperature_2m_max",   # °C → convert to K
    "temperature_2m_min",   # °C → convert to K
    "precipitation_sum",    # mm
    "windgusts_10m_max",    # km/h
]

# Hourly variables for daily-aggregated hub features (cape, visibility).
# These aren't available as daily aggregates from Open-Meteo, so we fetch
# hourly and reduce: max(cape) per day, min(visibility) per day.
_HUB_HOURLY_VARS = ["cape", "visibility"]


def _hub_req_with_backoff(url: str, params: dict, max_tries: int = 8) -> "requests.Response":  # type: ignore[name-defined]
    """Minimal HTTP GET with exponential-backoff retry (mirrors prepare_dataset._req_with_backoff)."""
    import requests as _req
    import time as _time

    backoff, delay = 1.8, 1.0
    last_status = last_text = last_exc = None
    for i in range(1, max_tries + 1):
        try:
            r = _req.get(url, params=params, timeout=60)
            if r.status_code == 200:
                return r
            last_status = r.status_code
            try:
                last_text = r.text[:400]
            except Exception:
                last_text = ""
            if r.status_code in (429, 500, 502, 503, 504):
                ra = r.headers.get("Retry-After")
                sleep_s = max(delay, float(ra)) if ra else delay
                print(f"[WARN] HTTP {r.status_code} attempt {i}/{max_tries}. sleep {sleep_s:.1f}s")
                _time.sleep(sleep_s)
                delay = min(120, delay * backoff)
                continue
            raise RuntimeError(f"Request failed HTTP {r.status_code}. Response: {last_text}")
        except RuntimeError:
            raise
        except Exception as e:
            last_exc = e
            print(f"[WARN] Request exception attempt {i}/{max_tries}: {repr(e)}. sleep {delay:.1f}s")
            import time as _time2; _time2.sleep(delay)
            delay = min(120, delay * backoff)
    raise RuntimeError(
        f"Request failed after {max_tries} retries. status={last_status} exc={repr(last_exc)}"
    )


def _fetch_hub_daily_weather(hub: str, lat: float, lon: float, tz: str,
                              start: str, end: str, cache_dir: Path) -> pd.DataFrame:
    """
    Fetch daily weather for one hub airport from Open-Meteo archive, with local cache.
    Returns DataFrame with columns: FlightDate, temp_max_K, temp_min_K,
    precip_sum_mm, windgusts_max_kmh.
    """
    safe_tz = tz.replace("/", "__")
    cache_path = cache_dir / f"hub_{hub}_{start}_{end}_{safe_tz}_daily.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    url = "https://archive-api.open-meteo.com/v1/archive"
    r = _hub_req_with_backoff(url, {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": ",".join(_HUB_DAILY_VARS),
        "timezone": tz,
    })
    payload = r.json()
    daily = payload.get("daily", {})
    if not daily or "time" not in daily:
        wx = pd.DataFrame(columns=["FlightDate"] + _HUB_DAILY_VARS)
    else:
        wx = pd.DataFrame(daily)
        wx["FlightDate"] = pd.to_datetime(wx["time"])
        wx = wx.drop(columns=["time"])
        # Convert temperature °C → K
        for col in ["temperature_2m_max", "temperature_2m_min"]:
            if col in wx.columns:
                wx[col] = pd.to_numeric(wx[col], errors="coerce") + 273.15
        # Rename to model-friendly names
        wx = wx.rename(columns={
            "temperature_2m_max": "temp_max_K",
            "temperature_2m_min": "temp_min_K",
            "precipitation_sum": "precip_sum_mm",
            "windgusts_10m_max": "windgusts_max_kmh",
        })

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    wx.to_parquet(cache_path, index=False)
    return wx


def _fetch_hub_hourly_weather(hub: str, lat: float, lon: float, tz: str,
                               start: str, end: str, cache_dir: Path) -> pd.DataFrame:
    """
    Fetch hourly cape + visibility for one hub, aggregate to daily:
      cape_max    -- daily max CAPE (J/kg), indicates convective storm potential
      visibility_min_m -- daily minimum visibility (m), captures IFR / fog / smoke events

    Returns DataFrame with columns: FlightDate, cape_max, visibility_min_m.
    """
    safe_tz = tz.replace("/", "__")
    cache_path = cache_dir / f"hub_{hub}_{start}_{end}_{safe_tz}_hourly_cape_vis.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    url = "https://archive-api.open-meteo.com/v1/archive"
    r = _hub_req_with_backoff(url, {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(_HUB_HOURLY_VARS),
        "timezone": tz,
    })
    payload = r.json()
    hourly = payload.get("hourly", {})
    if not hourly or "time" not in hourly:
        out = pd.DataFrame(columns=["FlightDate", "cape_max", "visibility_min_m"])
    else:
        h = pd.DataFrame(hourly)
        h["FlightDate"] = pd.to_datetime(h["time"]).dt.normalize()
        h["cape"] = pd.to_numeric(h.get("cape"), errors="coerce").fillna(0.0)
        h["visibility"] = pd.to_numeric(h.get("visibility"), errors="coerce")
        daily = h.groupby("FlightDate", as_index=False).agg(
            cape_max=("cape", "max"),
            visibility_min_m=("visibility", "min"),
        )
        out = daily

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache_path, index=False)
    return out


def add_wn_hub_weather_from_openmeteo(
    df: pd.DataFrame,
    *,
    hubs: Optional[List[str]] = None,
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Adds daily weather features for WN's key hub airports, joined to ALL flights
    on FlightDate alone.  Provides both day-of ("_today") and previous-day
    ("_yesterday") variants as leading indicators of network-wide delay risk.

    Daily features per hub H (from Open-Meteo daily API):
      hub_{H}_precip_sum_mm_today / _yesterday
      hub_{H}_windgusts_max_kmh_today / _yesterday

    Hourly-aggregated features per hub H (max/min over each calendar day):
      hub_{H}_log1p_cape_max_today / _yesterday    -- log1p(daily max CAPE), thunderstorm risk
      hub_{H}_visibility_min_m_today / _yesterday  -- daily min visibility, IFR/flow reduction

    4 vars × 2 time offsets × N hubs numeric features total.

    Weather data is fetched from Open-Meteo archive API (free, no key required)
    and cached locally.  In production, swap archive URL for forecast endpoint.

    Parameters
    ----------
    hubs : list of IATA codes.  Defaults to WN_HUB_COORDS keys.
    cache_dir : path (relative to DATA_ROOT) for caching raw weather parquets.
                Defaults to "weather_cache".
    """
    df = df.copy()
    df["FlightDate"] = pd.to_datetime(df.get("FlightDate"), errors="coerce").dt.normalize()

    if hubs is None:
        hubs = list(WN_HUB_COORDS.keys())
    hubs = [h.upper().strip() for h in hubs if h.upper().strip() in WN_HUB_COORDS]

    _cache_root = DATA_ROOT / (cache_dir or "weather_cache")
    _cache_root.mkdir(parents=True, exist_ok=True)

    start_str = df["FlightDate"].min().strftime("%Y-%m-%d")
    end_str   = df["FlightDate"].max().strftime("%Y-%m-%d")

    for hub in hubs:
        lat, lon, tz = WN_HUB_COORDS[hub]
        print(f"[hub weather] fetching daily {hub} {start_str}..{end_str}")
        wx_d = _fetch_hub_daily_weather(hub, lat, lon, tz, start_str, end_str, _cache_root)
        wx_d["FlightDate"] = pd.to_datetime(wx_d["FlightDate"]).dt.normalize()
        wx_d = wx_d.set_index("FlightDate").sort_index()

        print(f"[hub weather] fetching hourly (cape/vis) {hub} {start_str}..{end_str}")
        wx_h = _fetch_hub_hourly_weather(hub, lat, lon, tz, start_str, end_str, _cache_root)
        wx_h["FlightDate"] = pd.to_datetime(wx_h["FlightDate"]).dt.normalize()
        wx_h = wx_h.set_index("FlightDate").sort_index()
        # log1p-transform CAPE (heavy right tail)
        if "cape_max" in wx_h.columns:
            wx_h["log1p_cape_max"] = np.log1p(wx_h["cape_max"].clip(lower=0))

        # Merge hourly-derived daily stats into the daily frame
        for col in ["log1p_cape_max", "visibility_min_m"]:
            if col in wx_h.columns:
                wx_d[col] = wx_h[col]

        # Reindex to cover all dates
        all_dates = pd.date_range(wx_d.index.min(), wx_d.index.max(), freq="D")
        wx_d = wx_d.reindex(all_dates)

        VARS = ["precip_sum_mm", "windgusts_max_kmh", "log1p_cape_max", "visibility_min_m"]
        for var in VARS:
            col = wx_d[var] if var in wx_d.columns else pd.Series(np.nan, index=wx_d.index)
            today_df = pd.DataFrame({
                "FlightDate": wx_d.index,
                f"hub_{hub}_{var}_today":     col.values,
                f"hub_{hub}_{var}_yesterday": col.shift(1).values,
            })
            df = df.merge(today_df, on="FlightDate", how="left")

    return df


def add_delay_cause_rates_from_history(
    df: pd.DataFrame,
    hist: pd.DataFrame,
    windows_days: List[int],
) -> pd.DataFrame:
    """
    Computes rolling historical delay-cause attribution rates from BTS cause columns.
    All computed with 1-day lag (no leakage).

    New numeric features per window W in windows_days:
      origin_nas_rate_last{W}           -- fraction of flights at Origin with NASDelay > 0
      origin_lateaircraft_rate_last{W}  -- fraction with LateAircraftDelay > 0 (propagation proxy)
      origin_weather_rate_last{W}       -- fraction with WeatherDelay > 0
      carrier_lateaircraft_rate_last{W} -- network-wide fraction of carrier flights with LateAircraftDelay > 0

    If the cause columns are not present in hist (old history pool), all features are NaN.
    """
    df = df.copy()
    windows = sorted({int(w) for w in windows_days if int(w) > 0})
    if not windows:
        return df

    cause_cols_needed = ["NASDelay", "LateAircraftDelay", "WeatherDelay"]
    if any(c not in hist.columns for c in cause_cols_needed):
        for w in windows:
            for feat in ["origin_nas_rate", "origin_lateaircraft_rate",
                         "origin_weather_rate", "carrier_lateaircraft_rate"]:
                df[f"{feat}_last{w}"] = np.nan
        return df

    h = hist.copy()
    h["FlightDate"] = pd.to_datetime(h.get("FlightDate"), errors="coerce").dt.normalize()
    h["Origin"] = h.get("Origin", "").astype(str).str.upper().str.strip()
    h["Reporting_Airline"] = h.get("Reporting_Airline", "").astype(str).str.upper().str.strip()

    for c in cause_cols_needed:
        h[c] = pd.to_numeric(h.get(c), errors="coerce").fillna(0.0)

    # Binary flags: delay cause was non-zero
    h["_has_nas"]          = (h["NASDelay"] > 0).astype(float)
    h["_has_lateaircraft"] = (h["LateAircraftDelay"] > 0).astype(float)
    h["_has_weather"]      = (h["WeatherDelay"] > 0).astype(float)

    # ---- Origin-level daily rates ----
    o = h.dropna(subset=["Origin", "FlightDate"]).copy()
    o_daily = (
        o.groupby(["Origin", "FlightDate"], as_index=False)
        [["_has_nas", "_has_lateaircraft", "_has_weather"]]
        .mean()
        .sort_values(["Origin", "FlightDate"])
    )

    for w in windows:
        grp = o_daily.groupby("Origin")
        o_daily[f"origin_nas_rate_last{w}"] = (
            grp["_has_nas"]
            .apply(lambda s: s.shift(1).rolling(window=int(w), min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )
        o_daily[f"origin_lateaircraft_rate_last{w}"] = (
            grp["_has_lateaircraft"]
            .apply(lambda s: s.shift(1).rolling(window=int(w), min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )
        o_daily[f"origin_weather_rate_last{w}"] = (
            grp["_has_weather"]
            .apply(lambda s: s.shift(1).rolling(window=int(w), min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )

    origin_rate_cols = (
        [f"origin_nas_rate_last{w}" for w in windows] +
        [f"origin_lateaircraft_rate_last{w}" for w in windows] +
        [f"origin_weather_rate_last{w}" for w in windows]
    )
    df["FlightDate"] = pd.to_datetime(df.get("FlightDate"), errors="coerce").dt.normalize()
    df["Origin"] = df.get("Origin", "").astype(str).str.upper().str.strip()
    df = df.merge(o_daily[["Origin", "FlightDate"] + origin_rate_cols],
                  on=["Origin", "FlightDate"], how="left")

    # ---- Carrier-level late-aircraft rate (captures network-wide propagation) ----
    c = h.dropna(subset=["Reporting_Airline", "FlightDate"]).copy()
    c_daily = (
        c.groupby(["Reporting_Airline", "FlightDate"], as_index=False)
        [["_has_lateaircraft"]]
        .mean()
        .sort_values(["Reporting_Airline", "FlightDate"])
    )

    for w in windows:
        c_daily[f"carrier_lateaircraft_rate_last{w}"] = (
            c_daily.groupby("Reporting_Airline")["_has_lateaircraft"]
            .apply(lambda s: s.shift(1).rolling(window=int(w), min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )

    carrier_rate_cols = [f"carrier_lateaircraft_rate_last{w}" for w in windows]
    df["Reporting_Airline"] = df.get("Reporting_Airline", "").astype(str).str.upper().str.strip()
    df = df.merge(c_daily[["Reporting_Airline", "FlightDate"] + carrier_rate_cols],
                  on=["Reporting_Airline", "FlightDate"], how="left")

    return df


def add_turn_time_hours(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Flight_Number_Reporting_Airline"] = pd.to_numeric(df.get("Flight_Number_Reporting_Airline"), errors="coerce")

    df["dep_dt_utc"] = pd.to_datetime(df.get("dep_dt_local"), errors="coerce", utc=True)
    df["arr_dt_utc"] = pd.to_datetime(df.get("arr_dt_local"), errors="coerce", utc=True)

    key = ["Reporting_Airline", "Flight_Number_Reporting_Airline"]
    df = df.sort_values(key + ["dep_dt_utc"])

    df["prev_arr_dt_utc"] = df.groupby(key, sort=False)["arr_dt_utc"].shift(1)
    df["prev_dest"] = df.groupby(key, sort=False)["Dest"].shift(1)

    ok = df["prev_dest"].astype(str).str.upper() == df["Origin"].astype(str).str.upper()
    delta_hours = (df["dep_dt_utc"] - df["prev_arr_dt_utc"]).dt.total_seconds() / 3600.0
    df["turn_time_hours"] = np.where(ok, delta_hours, np.nan)
    return df


def add_prior_leg_propagation_features(
    df: pd.DataFrame,
    hist: pd.DataFrame,
    windows_days: List[int],
) -> pd.DataFrame:
    """
    Propagation delay risk: for each flight, estimate the expected departure
    delay of the inbound leg (prior rotation) using historical tail-number data.

    Algorithm
    ---------
    1. Sort history by (Tail_Number, FlightDate, dep_dt_local).
    2. For each hist flight, shift(1) within (Tail, Date) to get the prior
       leg's DepDelayMinutes, but only when prev_dest == this flight's Origin
       (confirming it's a connected rotation, not a re-assignment).
    3. Aggregate these prior-leg delays daily by (Airline, FlightNum, Origin).
    4. Rolling mean over windows_days with a 1-day shift for no leakage.
    5. Merge back to df.

    Also adds:
      is_first_leg_of_day  — 1 when tail_leg_num_day == 1 (no inbound risk),
                             0 otherwise.

    New columns: prior_leg_depdelay_mean_last{W}  for W in windows_days.
    """
    df = df.copy()
    windows = sorted({int(w) for w in windows_days if int(w) > 0})

    # is_first_leg_of_day (derived from tail_leg_num_day already in df)
    if "tail_leg_num_day" in df.columns:
        leg = pd.to_numeric(df["tail_leg_num_day"], errors="coerce")
        df["is_first_leg_of_day"] = (leg == 1).astype("Int8")
    else:
        df["is_first_leg_of_day"] = pd.Series(pd.NA, index=df.index, dtype="Int8")

    if not windows:
        return df

    # Require tail number and timing in history
    if "Tail_Number" not in hist.columns or "dep_dt_local" not in hist.columns:
        for w in windows:
            df[f"prior_leg_depdelay_mean_last{w}"] = np.nan
        return df

    # --- Step 1: prior-leg delay for every history flight ---
    need = ["FlightDate", "Tail_Number", "Reporting_Airline",
            "Flight_Number_Reporting_Airline", "Origin", "Dest",
            "dep_dt_local", "DepDelayMinutes"]
    h = hist[[c for c in need if c in hist.columns]].copy()

    h["Tail_Number"] = h["Tail_Number"].astype(str).str.upper().str.strip()
    bad = h["Tail_Number"].isin(["", "NAN", "NONE", "NULL", "UNKNOWN", "UNKN", "NA"])
    h = h[~bad].dropna(subset=["FlightDate", "dep_dt_local", "Tail_Number"]).copy()

    h["FlightDate"] = pd.to_datetime(h["FlightDate"], errors="coerce").dt.normalize()
    h["dep_dt_utc"] = pd.to_datetime(h["dep_dt_local"], utc=True, errors="coerce")
    h["DepDelayMinutes"] = pd.to_numeric(h["DepDelayMinutes"], errors="coerce")
    h["Flight_Number_Reporting_Airline"] = pd.to_numeric(
        h.get("Flight_Number_Reporting_Airline"), errors="coerce"
    )
    h["Reporting_Airline"] = h.get("Reporting_Airline", "").astype(str).str.upper().str.strip()
    h = h.dropna(subset=["dep_dt_utc"]).sort_values(
        ["Tail_Number", "FlightDate", "dep_dt_utc"], kind="mergesort"
    )

    grp = h.groupby(["Tail_Number", "FlightDate"], sort=False)
    h["_prev_delay"] = grp["DepDelayMinutes"].shift(1)
    h["_prev_dest"] = grp["Dest"].shift(1)
    connected = (
        h["_prev_dest"].astype(str).str.upper() == h["Origin"].astype(str).str.upper()
    )
    h["prior_leg_depdelay"] = np.where(connected, h["_prev_delay"], np.nan)

    # --- Step 2: daily mean by (Airline, FlightNum, Origin) ---
    key = ["Reporting_Airline", "Flight_Number_Reporting_Airline", "Origin"]
    h_valid = h.dropna(subset=key + ["FlightDate", "prior_leg_depdelay"]).copy()
    h_valid = h_valid.dropna(subset=["Flight_Number_Reporting_Airline"])

    if h_valid.empty:
        for w in windows:
            df[f"prior_leg_depdelay_mean_last{w}"] = np.nan
        return df

    daily = (
        h_valid.groupby(key + ["FlightDate"], as_index=False)["prior_leg_depdelay"]
        .mean()
        .sort_values(key + ["FlightDate"])
    )

    # --- Step 3: rolling with 1-day shift (no leakage) ---
    for w in windows:
        _w = int(w)
        daily[f"prior_leg_depdelay_mean_last{w}"] = (
            daily.groupby(key)["prior_leg_depdelay"]
            .apply(lambda s: s.shift(1).rolling(window=_w, min_periods=1).mean())
            .reset_index(level=[0, 1, 2], drop=True)
        )

    # --- Step 4: merge into df ---
    feat_cols = [f"prior_leg_depdelay_mean_last{w}" for w in windows]
    keep = key + ["FlightDate"] + feat_cols

    df["Flight_Number_Reporting_Airline"] = pd.to_numeric(
        df.get("Flight_Number_Reporting_Airline"), errors="coerce"
    )
    df["FlightDate"] = pd.to_datetime(df["FlightDate"], errors="coerce").dt.normalize()
    df["Reporting_Airline"] = df.get("Reporting_Airline", "").astype(str).str.upper().str.strip()

    df = df.merge(daily[keep], on=key + ["FlightDate"], how="left")
    return df


def add_dep_labels_for_thresholds(df: pd.DataFrame, thresholds: List[int], delay_col: str = "DepDelayMinutes") -> pd.DataFrame:
    df = df.copy()
    d = pd.to_numeric(df.get(delay_col), errors="coerce").fillna(0.0)
    for thr in thresholds:
        df[f"y_dep_ge{int(thr)}"] = (d >= int(thr)).astype("Int8")
    if "y_dep15" not in df.columns and 15 in thresholds:
        df["y_dep15"] = df["y_dep_ge15"]
    return df


def balance_to_pos_frac(df: pd.DataFrame, label_col: str, pos_frac: float, seed: int, max_rows: Optional[int]) -> pd.DataFrame:
    df = df.copy()
    y = df[label_col].astype(int)
    pos = df[y == 1]
    neg = df[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        print(f"[WARN] Cannot balance {label_col}: one class empty.")
        return df

    target = float(pos_frac)
    if not (0.0 < target < 1.0):
        raise ValueError("pos_frac must be in (0,1)")

    a_max = len(pos)
    b_needed_for_a_max = int(round(a_max * (1 - target) / target))
    if b_needed_for_a_max <= len(neg):
        a = a_max
        b = b_needed_for_a_max
        limiting = "pos"
    else:
        b_max = len(neg)
        a_needed_for_b_max = int(round(b_max * target / (1 - target)))
        a = min(a_needed_for_b_max, a_max)
        b = b_max
        limiting = "neg"

    pos_s = pos.sample(n=a, random_state=seed)
    neg_s = neg.sample(n=b, random_state=seed)
    out = pd.concat([pos_s, neg_s], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    if max_rows is not None and len(out) > int(max_rows):
        out = out.sample(n=int(max_rows), random_state=seed).reset_index(drop=True)

    achieved = out[label_col].mean()
    print(f"[INFO] Balanced {label_col} to pos_frac~{achieved:.3f} (target={target:.3f}, limiting={limiting}) rows={len(out)}")
    return out


# ------------------------------ main ------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python features_dep.py data/dep_arr_config.json")
        sys.exit(1)

    cfg_path = _abspath(sys.argv[1], base="repo")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    target = (cfg.get("target") or "dep").lower()
    if target != "dep":
        raise ValueError("features_dep.py expects cfg.target == 'dep'")

    in_enriched = cfg.get("output_enriched_unbalanced_path")
    if not in_enriched:
        raise KeyError("cfg.output_enriched_unbalanced_path is required (written by prepare_dataset.py)")
    in_path = _abspath(_format_path_template(in_enriched, target=target), base="data")
    if not in_path.exists():
        raise FileNotFoundError(f"Enriched parquet not found: {in_path}")

    df_cols_want = [
        "FlightDate",
        "Origin",
        "Dest",
        "Reporting_Airline",
        "Flight_Number_Reporting_Airline",
        "CRSDepTime",
        "CRSElapsedTime",
        "Distance",
        "DepTimeBlk",
        "dep_dt_local",
        "arr_dt_local",
        "DepDelayMinutes",
        "Tail_Number",
        "aircraft_type",
        "Aircraft_Age_Bucket",
        # daily weather
        "origin_temperature_2m_max",
        "origin_temperature_2m_min",
        "origin_precipitation_sum",
        "origin_windspeed_10m_max",
        "origin_windgusts_10m_max",
        "origin_weathercode",
        # hourly weather at departure time
        "origin_dep_temperature_2m",
        "origin_dep_precipitation",
        "origin_dep_windspeed_10m",
        "origin_dep_weathercode",
        "origin_dep_windgusts_10m",
        "origin_dep_visibility",
        "origin_dep_cape",
        "origin_dep_cloudcover",
    ]

    print(f"[INFO] reading enriched -> {in_path}")
    df = _read_parquet_projected(in_path, df_cols_want)

    # Optional date range filter — enables fast 2023-only experiments without re-running prepare_dataset
    cfg_start = cfg.get("start_date")
    cfg_end   = cfg.get("end_date")
    if cfg_start or cfg_end:
        fd = pd.to_datetime(df["FlightDate"], errors="coerce").dt.normalize()
        mask = pd.Series(True, index=fd.index)
        if cfg_start:
            mask &= fd >= pd.Timestamp(cfg_start)
        if cfg_end:
            mask &= fd <= pd.Timestamp(cfg_end)
        df = df[mask].reset_index(drop=True)
        print(f"[INFO] date filter {cfg_start or '..'}..{cfg_end or '..'} -> {len(df):,} rows")

    hist_path_cfg = cfg.get("history_output_path", "intermediate/history_pool_dep.parquet")
    hist_path = _abspath(_format_path_template(hist_path_cfg, target=target), base="data")
    if not hist_path.exists():
        raise FileNotFoundError(
            f"History pool parquet not found: {hist_path}\n"
            "Run prepare_dataset.py with history_output_path enabled, or set cfg.history_output_path correctly."
        )

    hist_cols_want = [
        "FlightDate",
        "Origin",
        "Dest",
        "Reporting_Airline",
        "Flight_Number_Reporting_Airline",
        "DepDelayMinutes",
        "CRSDepTime",
        "dep_dt_local",
        "dep_local_date",
        "Tail_Number",
        # BTS delay cause breakdown — for rolling cause-rate features
        "NASDelay",
        "LateAircraftDelay",
        "WeatherDelay",
    ]
    hist = _read_parquet_projected(hist_path, hist_cols_want)

    df["FlightDate"] = pd.to_datetime(df.get("FlightDate"), errors="coerce")
    df["Origin"] = df.get("Origin", "").astype(str).str.upper().str.strip()
    df["Dest"] = df.get("Dest", "").astype(str).str.upper().str.strip()
    df["Reporting_Airline"] = df.get("Reporting_Airline", "").astype(str).str.upper().str.strip()

    hist["FlightDate"] = pd.to_datetime(hist.get("FlightDate"), errors="coerce")
    hist["Origin"] = hist.get("Origin", "").astype(str).str.upper().str.strip()
    hist["Dest"] = hist.get("Dest", "").astype(str).str.upper().str.strip()
    hist["Reporting_Airline"] = hist.get("Reporting_Airline", "").astype(str).str.upper().str.strip()

    df["od_pair"] = df["Origin"] + "_" + df["Dest"]

    if "dep_dt_local" in df.columns:
        dts = pd.to_datetime(df["dep_dt_local"], errors="coerce")
        df["dep_dow"] = dts.dt.dayofweek.astype("Int8")
        df["sched_dep_hour"] = dts.dt.hour.astype("Int8")
    else:
        fd = pd.to_datetime(df["FlightDate"], errors="coerce")
        df["dep_dow"] = fd.dt.dayofweek.astype("Int8")
        df["sched_dep_hour"] = pd.to_numeric(df.get("CRSDepTime"), errors="coerce").floordiv(100).astype("Int8")

    feat_cfg = cfg.get("features_dep") or {}
    hw = (feat_cfg.get("holiday_windows") or {})
    df = add_holiday_flag(
        df,
        thanksgiving_window=int(hw.get("thanksgiving", 2)),
        christmas_window=int(hw.get("christmas", 3)),
    )

    if "aircraft_type" not in df.columns:
        df["aircraft_type"] = "Unknown"

    if "Aircraft_Age_Bucket" in df.columns:
        df["Aircraft_Age_Bucket"] = df["Aircraft_Age_Bucket"].astype(str).str.strip()
    else:
        df["Aircraft_Age_Bucket"] = "Unknown"

    if "DepTimeBlk" in df.columns:
        df["DepTimeBlk"] = df["DepTimeBlk"].astype(str).str.upper().str.strip()
    else:
        df["DepTimeBlk"] = "Unknown"

    if "origin_weathercode" in df.columns:
        df["origin_daily_weathercode"] = pd.to_numeric(df["origin_weathercode"], errors="coerce").astype("Int64").astype("object")
    if "origin_dep_weathercode" in df.columns:
        df["origin_dep_hour_weathercode"] = pd.to_numeric(df["origin_dep_weathercode"], errors="coerce").astype("Int64").astype("object")

    df = add_weather_kelvin(df)

    # Numeric route features from BTS
    if "CRSElapsedTime" in df.columns:
        df["CRSElapsedTime"] = pd.to_numeric(df["CRSElapsedTime"], errors="coerce")
    if "Distance" in df.columns:
        df["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")

    # Month of year — captures seasonal demand/weather patterns over a full year
    fd_dt = pd.to_datetime(df["FlightDate"], errors="coerce")
    df["flight_month"] = fd_dt.dt.month.astype("Int8")

    df = add_congestion_3h_features_from_history(df, hist)
    df = add_tail_and_flightnum_time_features(df)

    windows_days = feat_cfg.get("rolling_windows_days") or [7, 14, 21, 28]
    windows_days = [int(x) for x in windows_days]

    n_recent = int(feat_cfg.get("n_recent_flightnum_od", 20))
    low_support_leq = int(feat_cfg.get("low_support_leq", 2))

    df = add_flightnum_od_depdelay_means_from_history(
        df,
        hist,
        windows_days=windows_days,
        n_recent=n_recent,
        low_support_leq=low_support_leq,
    )

    df = add_carrier_dep_delay_baselines_from_history_multi(df, hist, windows_days=windows_days)
    df = add_origin_dep_delay_baselines_from_history_multi(df, hist, windows_days=windows_days)
    df = add_delay_cause_rates_from_history(df, hist, windows_days=windows_days)

    tail_cfg = feat_cfg.get("tail_history") or {}
    if tail_cfg.get("enabled", True):
        # Tail_Number is not stored in the prepared history pool — read it directly
        # from the full enriched parquet (all dates, so Jan-2023 has Dec-2022 lookback)
        _tail_hist_cols = ["FlightDate", "Tail_Number", "DepDelayMinutes", "LateAircraftDelay"]
        _tail_hist = _read_parquet_projected(in_path, _tail_hist_cols)
        _tail_hist["FlightDate"] = pd.to_datetime(_tail_hist.get("FlightDate"), errors="coerce").dt.normalize()
        df = add_tail_rolling_history(df, _tail_hist, windows_days=windows_days)
        del _tail_hist

    dest_cfg = feat_cfg.get("dest_rolling") or {}
    if dest_cfg.get("enabled", True):
        df = add_dest_rolling_stats(df, hist, windows_days=windows_days)

    hub_cfg = feat_cfg.get("hub_spillover") or {}
    hub_hubs = hub_cfg.get("hubs") or None  # explicit list wins
    if hub_hubs is None:
        hub_airline = str(hub_cfg.get("airline") or "WN").upper()
        hub_hubs = AIRLINE_HUB_PRESETS.get(hub_airline)
        if hub_hubs:
            print(f"[INFO] hub_spillover: using preset hubs for airline={hub_airline}: {hub_hubs}")
        else:
            print(f"[WARN] hub_spillover: no preset for airline={hub_airline!r}; add to AIRLINE_HUB_PRESETS or set hub_spillover.hubs explicitly")
    df = add_hub_spillover_from_history(df, hist, windows_days=windows_days, hubs=hub_hubs)

    wx_cfg = feat_cfg.get("hub_weather") or {}
    if wx_cfg.get("enabled", True):
        wx_hubs = wx_cfg.get("hubs") or None  # None → use WN_HUB_COORDS keys
        wx_cache = wx_cfg.get("cache_dir") or "weather_cache"
        df = add_wn_hub_weather_from_openmeteo(df, hubs=wx_hubs, cache_dir=wx_cache)

    df = add_turn_time_hours(df)

    # prior_leg needs Tail_Number + dep_dt_local, which are not in the prepared history pool.
    # Read the columns we need from the full enriched parquet (no date filter, so Jan-2023
    # lookback can reach Dec-2022). Disable via features_dep.prior_leg.enabled=false.
    prior_leg_cfg = feat_cfg.get("prior_leg") or {}
    if prior_leg_cfg.get("enabled", True):
        _prior_hist_cols = [
            "FlightDate", "Tail_Number", "Reporting_Airline",
            "Flight_Number_Reporting_Airline", "Origin", "Dest",
            "dep_dt_local", "DepDelayMinutes",
        ]
        _prior_hist = _read_parquet_projected(in_path, _prior_hist_cols)
        _prior_hist["FlightDate"] = pd.to_datetime(_prior_hist.get("FlightDate"), errors="coerce").dt.normalize()
        df = add_prior_leg_propagation_features(df, _prior_hist, windows_days=windows_days)
        del _prior_hist
    else:
        print("[INFO] prior_leg disabled via config")

    thresholds = (cfg.get("balance", {}) or {}).get("thresholds", [15, 30, 45, 60])
    thresholds = [int(x) for x in thresholds]
    delay_col = (cfg.get("balance", {}) or {}).get("delay_col", "DepDelayMinutes")
    df = add_dep_labels_for_thresholds(df, thresholds, delay_col=delay_col)

    out_feat_cfg = cfg.get("output_features_unbalanced_path", "processed/features_dep_unbalanced.parquet")
    out_feat_cfg = _format_path_template(out_feat_cfg, target=target)
    out_feat_path = _abspath(out_feat_cfg, base="data")
    out_feat_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(out_feat_path, index=False)
    print(f"[OK] wrote features unbalanced rows={len(df)} -> {out_feat_path}")

    fb = cfg.get("feature_balance") or {}
    if not bool(fb.get("enabled", True)):
        return

    eval_cfg = fb.get("eval") or {}
    eval_enabled = bool(eval_cfg.get("enabled", True))
    train_pool = df

    if eval_enabled:
        by = (eval_cfg.get("by") or "last_days").lower()
        if by == "last_days":
            last_days = int(eval_cfg.get("last_days", 90))
            fd = pd.to_datetime(df["FlightDate"], errors="coerce").dt.normalize()
            cutoff = fd.max() - pd.Timedelta(days=last_days)
            eval_mask = fd > cutoff
            eval_df = df[eval_mask].copy()
            train_pool = df[~eval_mask].copy()
            print(f"[INFO] feature eval last_days={last_days} cutoff>{cutoff.date()} rows={len(eval_df)} train_pool rows={len(train_pool)}")
        elif by == "random":
            eval_frac = float(eval_cfg.get("eval_frac", 0.15))
            eval_seed = int(eval_cfg.get("seed", 42))
            eval_idx = df.sample(frac=eval_frac, random_state=eval_seed).index
            eval_mask = df.index.isin(eval_idx)
            eval_df = df[eval_mask].copy()
            train_pool = df[~eval_mask].copy()
            print(f"[INFO] feature eval random frac={eval_frac} seed={eval_seed} rows={len(eval_df)} train_pool rows={len(train_pool)}")
        else:
            raise ValueError(f"feature_balance.eval.by must be 'last_days' or 'random', got {by!r}")

        eval_out = eval_cfg.get("output_path", "processed/eval_dep_unbalanced.parquet")
        eval_out = _format_path_template(eval_out, target=target)
        eval_path = _abspath(eval_out, base="data")
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        eval_df.to_parquet(eval_path, index=False)
        print(f"[OK] wrote feature eval unbalanced -> {eval_path}")

    thr_list = [int(x) for x in (fb.get("thresholds") or thresholds)]
    pos_frac = float(fb.get("pos_frac", 0.50))
    seed = int(fb.get("seed", 42))
    max_rows = fb.get("max_rows")
    max_rows = int(max_rows) if max_rows is not None else None

    out_tmpl = fb.get("output_template", "processed/train_{target}_ge{thr}_balanced.parquet")

    for thr in thr_list:
        label_col = f"y_dep_ge{thr}"
        if label_col not in train_pool.columns:
            raise KeyError(f"Missing label column {label_col} in features dataframe")

        bal_df = balance_to_pos_frac(train_pool, label_col=label_col, pos_frac=pos_frac, seed=seed, max_rows=max_rows)

        out_p = _format_path_template(out_tmpl, target=target, thr=thr)
        out_p = _abspath(out_p, base="data")
        out_p.parent.mkdir(parents=True, exist_ok=True)
        bal_df.to_parquet(out_p, index=False)
        print(f"[OK] wrote balanced feature train thr={thr} -> {out_p}")


if __name__ == "__main__":
    main()
