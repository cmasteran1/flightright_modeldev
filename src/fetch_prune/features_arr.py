#!/usr/bin/env python3
# src/fetch_prune/features_arr.py
#
# (same header as before; only the ETA congestion computation was made bulletproof)

import sys
import json
import gc
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype


REPO_ROOT = Path.cwd()
DATA_ROOT = (REPO_ROOT.parent / "flightrightdata").resolve()

MEAN_WINDOWS = [7, 14]
MEDIAN_WINDOWS = [7]
# v10 hybrid mean/median rule: route arrival-delay history adds 14d medians alongside the
# 7d median that v8 already produced, so the model has a stable median signal at the longer
# window without giving up the v8 mean features. Toggle via cfg["arrdelay_median_14d"]["enabled"].
MEDIAN_WINDOWS_V10 = [7, 14]
THRESHOLDS_DEFAULT = [15, 30, 45, 60]


def _abspath(p: str, *, base: str = "repo") -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    if base == "repo":
        return (REPO_ROOT / pp).resolve()
    if base == "data":
        return (DATA_ROOT / pp).resolve()
    raise ValueError("base must be 'repo' or 'data'")


def _ensure_base_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "FlightDate" in df.columns:
        df["FlightDate"] = pd.to_datetime(df["FlightDate"], errors="coerce")

    for col in ["Origin", "Dest", "Reporting_Airline", "DepTimeBlk"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.upper().str.strip()

    if "Flight_Number_Reporting_Airline" in df.columns:
        df["Flight_Number_Reporting_Airline"] = pd.to_numeric(df["Flight_Number_Reporting_Airline"], errors="coerce")

    for c in ["CRSDepTime", "CRSArrTime", "WheelsOff", "AirTime", "ArrDelayMinutes"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["Origin", "Dest", "Reporting_Airline", "DepTimeBlk"]:
        if c in df.columns:
            df[c] = df[c].astype("category")

    return df


def _hhmm_series_to_minutes(s: pd.Series) -> np.ndarray:
    x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    m = ~np.isnan(x)
    if not np.any(m):
        return out
    xi = x[m].astype(np.int64)
    hh = xi // 100
    mm = xi % 100
    ok = (hh >= 0) & (hh <= 47) & (mm >= 0) & (mm <= 59)
    mins = hh * 60 + mm
    tmp = np.full(xi.shape, np.nan, dtype=float)
    tmp[ok] = mins[ok].astype(float)
    out[m] = tmp
    return out


def _minutes_diff_wrap_vec(actual_min: np.ndarray, sched_min: np.ndarray) -> np.ndarray:
    d = actual_min - sched_min
    m = ~np.isnan(d)
    d2 = d.copy()
    d2[m & (d2 < -720)] += 1440.0
    d2[m & (d2 > 720)] -= 1440.0
    return d2


def _prefilter_hist(hist: pd.DataFrame, keys_df: pd.DataFrame, key_cols: List[str]) -> pd.DataFrame:
    keys_df = keys_df[key_cols].drop_duplicates()
    return hist.merge(keys_df, on=key_cols, how="inner")


def _attach_asof(df: pd.DataFrame, stats_tbl: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    df2 = df.copy()
    st = stats_tbl.copy()

    df2["FlightDate"] = pd.to_datetime(df2["FlightDate"], errors="coerce")
    st["FlightDate"] = pd.to_datetime(st["FlightDate"], errors="coerce")

    for k in keys:
        if k not in df2.columns or k not in st.columns:
            continue

        dfd = df2[k].dtype
        std = st[k].dtype

        if isinstance(dfd, CategoricalDtype):
            st[k] = st[k].astype(str)
            st[k] = pd.Categorical(st[k], categories=df2[k].cat.categories)
        elif isinstance(std, CategoricalDtype):
            df2[k] = df2[k].astype(str)
            df2[k] = pd.Categorical(df2[k], categories=st[k].cat.categories)
        else:
            if dfd != std:
                df2[k] = df2[k].astype(str)
                st[k] = st[k].astype(str)

    mask_left = df2["FlightDate"].notna()
    left = df2.loc[mask_left].copy()
    left_nat = df2.loc[~mask_left].copy()

    mask_right = st["FlightDate"].notna()
    right = st.loc[mask_right].copy()

    sort_cols = ["FlightDate"] + keys
    left = left.sort_values(sort_cols, kind="mergesort")
    right = right.sort_values(sort_cols, kind="mergesort")

    merged = pd.merge_asof(
        left,
        right,
        on="FlightDate",
        by=keys,
        direction="backward",
        allow_exact_matches=True,
    )

    out = pd.concat([merged, left_nat], axis=0)
    out = out.loc[df.index]
    return out


def _rolling_tbl_mean_count(hist: pd.DataFrame, keys: List[str], value_col: str, window_days: int, prefix: str) -> pd.DataFrame:
    h = hist.dropna(subset=["FlightDate"]).copy()
    h = h.dropna(subset=keys, how="any").copy()
    h[value_col] = pd.to_numeric(h[value_col], errors="coerce")
    h = h.dropna(subset=[value_col]).copy()
    h = h.sort_values(["FlightDate"] + keys, kind="mergesort")

    g = h.set_index("FlightDate").groupby(keys, sort=False, observed=True)[value_col]
    rw = g.rolling(f"{window_days}D", closed="left")

    mean_s = rw.mean()
    cnt_s = rw.count()

    out = pd.concat([mean_s, cnt_s], axis=1).reset_index()
    out.columns = keys + ["FlightDate", f"{prefix}mean_{window_days}d", f"{prefix}count_{window_days}d"]
    out = out.sort_values(["FlightDate"] + keys, kind="mergesort")
    return out


def _rolling_tbl_median(hist: pd.DataFrame, keys: List[str], value_col: str, window_days: int, prefix: str) -> pd.DataFrame:
    h = hist.dropna(subset=["FlightDate"]).copy()
    h = h.dropna(subset=keys, how="any").copy()
    h[value_col] = pd.to_numeric(h[value_col], errors="coerce")
    h = h.dropna(subset=[value_col]).copy()
    h = h.sort_values(["FlightDate"] + keys, kind="mergesort")

    g = h.set_index("FlightDate").groupby(keys, sort=False, observed=True)[value_col]
    rw = g.rolling(f"{window_days}D", closed="left")
    med_s = rw.median()

    out = med_s.reset_index()
    out = out.rename(columns={value_col: f"{prefix}median_{window_days}d"})
    out = out.sort_values(["FlightDate"] + keys, kind="mergesort")
    return out


def add_route_arrival_delay_stats(df: pd.DataFrame, hist: pd.DataFrame, median_windows: Optional[List[int]] = None) -> pd.DataFrame:
    df = _ensure_base_types(df)
    hist = _ensure_base_types(hist)

    if median_windows is None:
        median_windows = MEDIAN_WINDOWS

    keys_fn = ["Reporting_Airline", "Flight_Number_Reporting_Airline", "Origin", "Dest"]
    hist_fn = _prefilter_hist(hist, df, keys_fn)

    for w in MEAN_WINDOWS:
        tbl = _rolling_tbl_mean_count(hist_fn, keys_fn, "ArrDelayMinutes", w, prefix="arrdelay_").rename(
            columns={f"arrdelay_mean_{w}d": f"arrdelay_mean_{w}d_fn_od", f"arrdelay_count_{w}d": f"arrdelay_n_{w}d_fn_od"}
        )
        df = _attach_asof(df, tbl, keys_fn)
        del tbl
        gc.collect()

    for w in median_windows:
        tbl = _rolling_tbl_median(hist_fn, keys_fn, "ArrDelayMinutes", w, prefix="arrdelay_").rename(
            columns={f"arrdelay_median_{w}d": f"arrdelay_median_{w}d_fn_od"}
        )
        df = _attach_asof(df, tbl, keys_fn)
        del tbl
        gc.collect()
    del hist_fn
    gc.collect()

    keys_car = ["Reporting_Airline", "Origin", "Dest"]
    hist_car = _prefilter_hist(hist, df, keys_car)

    for w in MEAN_WINDOWS:
        tbl = _rolling_tbl_mean_count(hist_car, keys_car, "ArrDelayMinutes", w, prefix="arrdelay_").rename(
            columns={f"arrdelay_mean_{w}d": f"arrdelay_mean_{w}d_car_od", f"arrdelay_count_{w}d": f"arrdelay_n_{w}d_car_od"}
        )
        df = _attach_asof(df, tbl, keys_car)
        del tbl
        gc.collect()

    for w in median_windows:
        tbl = _rolling_tbl_median(hist_car, keys_car, "ArrDelayMinutes", w, prefix="arrdelay_").rename(
            columns={f"arrdelay_median_{w}d": f"arrdelay_median_{w}d_car_od"}
        )
        df = _attach_asof(df, tbl, keys_car)
        del tbl
        gc.collect()
    del hist_car
    gc.collect()

    return df


def add_elapsed_time_ratio_stats(df: pd.DataFrame, hist: pd.DataFrame, window_days: int = 14) -> pd.DataFrame:
    """Rolling mean of `ActualElapsedTime / CRSElapsedTime` for (Airline, FlightNum, Origin, Dest)
    over the previous `window_days`. Both inputs are gate-based BTS fields, so this feature does
    NOT depend on WheelsOff/WheelsOn and is safe under the v10 AeroDataBox constraint."""
    df = _ensure_base_types(df)
    hist = _ensure_base_types(hist)

    if "ActualElapsedTime" not in hist.columns or "CRSElapsedTime" not in hist.columns:
        df = df.copy()
        df[f"elapsed_time_ratio_last{window_days}d"] = np.nan
        return df

    h = hist.copy()
    h["ActualElapsedTime"] = pd.to_numeric(h["ActualElapsedTime"], errors="coerce")
    h["CRSElapsedTime"] = pd.to_numeric(h["CRSElapsedTime"], errors="coerce").replace(0, np.nan)
    h["_elapsed_ratio"] = h["ActualElapsedTime"] / h["CRSElapsedTime"]
    h = h.dropna(subset=["_elapsed_ratio"]).copy()

    keys_fn = ["Reporting_Airline", "Flight_Number_Reporting_Airline", "Origin", "Dest"]
    hist_fn = _prefilter_hist(h, df, keys_fn)
    tbl = _rolling_tbl_mean_count(hist_fn, keys_fn, "_elapsed_ratio", window_days, prefix="elapsed_time_ratio_").rename(
        columns={
            f"elapsed_time_ratio_mean_{window_days}d": f"elapsed_time_ratio_last{window_days}d",
            f"elapsed_time_ratio_count_{window_days}d": f"elapsed_time_ratio_n_last{window_days}d",
        }
    )
    df = _attach_asof(df, tbl, keys_fn)
    del tbl, hist_fn, h
    gc.collect()
    return df


def add_recent_airtime_stats(df: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_base_types(df)
    hist = _ensure_base_types(hist)

    keys_fn = ["Reporting_Airline", "Flight_Number_Reporting_Airline", "Origin", "Dest"]
    hist_fn = _prefilter_hist(hist.dropna(subset=["AirTime"]), df, keys_fn)

    for w in MEAN_WINDOWS:
        tbl = _rolling_tbl_mean_count(hist_fn, keys_fn, "AirTime", w, prefix="airtime_").rename(
            columns={f"airtime_mean_{w}d": f"airtime_mean_{w}d_fn_od", f"airtime_count_{w}d": f"airtime_n_{w}d_fn_od"}
        )
        df = _attach_asof(df, tbl, keys_fn)
        del tbl
        gc.collect()

    del hist_fn
    gc.collect()

    keys_car = ["Reporting_Airline", "Origin", "Dest"]
    hist_car = _prefilter_hist(hist.dropna(subset=["AirTime"]), df, keys_car)

    for w in MEAN_WINDOWS:
        tbl = _rolling_tbl_mean_count(hist_car, keys_car, "AirTime", w, prefix="airtime_").rename(
            columns={f"airtime_mean_{w}d": f"airtime_mean_{w}d_car_od", f"airtime_count_{w}d": f"airtime_n_{w}d_car_od"}
        )
        df = _attach_asof(df, tbl, keys_car)
        del tbl
        gc.collect()

    del hist_car
    gc.collect()

    return df


def add_recent_wheels_off_slip_stats(df: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_base_types(df)
    hist = _ensure_base_types(hist)

    if "DepTimeBlk" not in df.columns:
        hh = pd.to_numeric(df.get("CRSDepTime"), errors="coerce").fillna(-1).astype(int)
        hh = (hh // 100).clip(0, 23)
        df["DepTimeBlk"] = hh.map(lambda x: f"{x:02d}00-{x:02d}59").astype("category")

    hist2 = hist.copy()
    if "DepTimeBlk" not in hist2.columns:
        hh = pd.to_numeric(hist2.get("CRSDepTime"), errors="coerce").fillna(-1).astype(int)
        hh = (hh // 100).clip(0, 23)
        hist2["DepTimeBlk"] = hh.map(lambda x: f"{x:02d}00-{x:02d}59").astype("category")

    crs_min = _hhmm_series_to_minutes(hist2["CRSDepTime"])
    wo_min = _hhmm_series_to_minutes(hist2["WheelsOff"])
    hist2["wo_slip_min"] = _minutes_diff_wrap_vec(wo_min, crs_min)
    hist2 = hist2.dropna(subset=["wo_slip_min"]).copy()

    keys_fn = ["Reporting_Airline", "Flight_Number_Reporting_Airline", "Origin", "Dest"]
    hist_fn = _prefilter_hist(hist2, df, keys_fn)

    for w in MEAN_WINDOWS:
        tbl = _rolling_tbl_mean_count(hist_fn, keys_fn, "wo_slip_min", w, prefix="wo_slip_").rename(
            columns={f"wo_slip_mean_{w}d": f"wo_slip_mean_{w}d_fn_od", f"wo_slip_count_{w}d": f"wo_slip_n_{w}d_fn_od"}
        )
        df = _attach_asof(df, tbl, keys_fn)
        del tbl
        gc.collect()

    del hist_fn
    gc.collect()

    keys_blk = ["Origin", "DepTimeBlk"]
    hist_blk = _prefilter_hist(hist2, df, keys_blk)

    for w in MEAN_WINDOWS:
        tbl = _rolling_tbl_mean_count(hist_blk, keys_blk, "wo_slip_min", w, prefix="wo_slip_").rename(
            columns={f"wo_slip_mean_{w}d": f"wo_slip_mean_{w}d_origin_blk", f"wo_slip_count_{w}d": f"wo_slip_n_{w}d_origin_blk"}
        )
        df = _attach_asof(df, tbl, keys_blk)
        del tbl
        gc.collect()

    del hist_blk, hist2
    gc.collect()
    return df


def add_projected_arrival_congestion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Congestion counts in ±60 minutes around:
      - scheduled arrival time (arr_dt_local)
      - ETA-shifted arrival time computed in ns:
            eta_ns = arr_ns + shift_minutes * 60e9
    This avoids tz-aware datetime casting pitfalls.
    """
    df = _ensure_base_types(df).copy()

    if "arr_dt_local" not in df.columns:
        raise KeyError("Projected arrival congestion requires arr_dt_local in enriched parquet.")

    arr_dt = pd.to_datetime(df["arr_dt_local"], errors="coerce")
    df["_arr_local_date"] = arr_dt.dt.date

    arr_ns_all = arr_dt.astype("int64").to_numpy()
    valid = arr_ns_all != np.iinfo("int64").min

    # shift source: 14d carrier OD mean, else 7d carrier OD mean, else 0
    if "arrdelay_mean_14d_car_od" in df.columns:
        shift_min_all = pd.to_numeric(df["arrdelay_mean_14d_car_od"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    elif "arrdelay_mean_7d_car_od" in df.columns:
        shift_min_all = pd.to_numeric(df["arrdelay_mean_7d_car_od"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    else:
        shift_min_all = np.zeros(len(df), dtype=float)

    # ns shift (int64) — round to nearest minute in ns
    shift_ns_valid = np.rint(shift_min_all[valid] * 60.0 * 1_000_000_000.0).astype(np.int64)

    df["dest_arrivals_pm60_sched"] = 0
    df["dest_airline_arrivals_pm60_sched"] = 0
    df["dest_arrivals_pm60_eta"] = 0
    df["dest_airline_arrivals_pm60_eta"] = 0

    window_ns = int(pd.Timedelta(minutes=60).value)

    work = df.loc[valid, ["Dest", "Reporting_Airline", "_arr_local_date"]].copy()
    work["idx"] = work.index
    work["arr_ns"] = arr_ns_all[valid]
    work["eta_ns"] = work["arr_ns"].to_numpy(dtype=np.int64, copy=False) + shift_ns_valid

    # (Dest, date)
    for (dest, d0), g in work.groupby(["Dest", "_arr_local_date"], sort=False):
        times = np.sort(g["arr_ns"].to_numpy(dtype=np.int64, copy=False))

        q = g["arr_ns"].to_numpy(dtype=np.int64, copy=False)
        left = np.searchsorted(times, q - window_ns, side="left")
        right = np.searchsorted(times, q + window_ns, side="right")
        counts = (right - left - 1).astype(np.int32)
        counts[counts < 0] = 0
        df.loc[g["idx"].to_numpy(), "dest_arrivals_pm60_sched"] = counts

        q2 = g["eta_ns"].to_numpy(dtype=np.int64, copy=False)
        left2 = np.searchsorted(times, q2 - window_ns, side="left")
        right2 = np.searchsorted(times, q2 + window_ns, side="right")
        counts2 = (right2 - left2 - 1).astype(np.int32)
        counts2[counts2 < 0] = 0
        df.loc[g["idx"].to_numpy(), "dest_arrivals_pm60_eta"] = counts2

    # (Dest, airline, date)
    for (dest, carr, d0), g in work.groupby(["Dest", "Reporting_Airline", "_arr_local_date"], sort=False):
        times = np.sort(g["arr_ns"].to_numpy(dtype=np.int64, copy=False))

        q = g["arr_ns"].to_numpy(dtype=np.int64, copy=False)
        left = np.searchsorted(times, q - window_ns, side="left")
        right = np.searchsorted(times, q + window_ns, side="right")
        counts = (right - left - 1).astype(np.int32)
        counts[counts < 0] = 0
        df.loc[g["idx"].to_numpy(), "dest_airline_arrivals_pm60_sched"] = counts

        q2 = g["eta_ns"].to_numpy(dtype=np.int64, copy=False)
        left2 = np.searchsorted(times, q2 - window_ns, side="left")
        right2 = np.searchsorted(times, q2 + window_ns, side="right")
        counts2 = (right2 - left2 - 1).astype(np.int32)
        counts2[counts2 < 0] = 0
        df.loc[g["idx"].to_numpy(), "dest_airline_arrivals_pm60_eta"] = counts2

    df = df.drop(columns=["_arr_local_date"], errors="ignore")

    for c in ["dest_arrivals_pm60_sched", "dest_airline_arrivals_pm60_sched", "dest_arrivals_pm60_eta", "dest_airline_arrivals_pm60_eta"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("Int32")

    return df


def _normalize_join_keys(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    if "FlightDate" in x.columns:
        fd = pd.to_datetime(x["FlightDate"], errors="coerce")
        x["FlightDate"] = fd.dt.normalize()
    for c in ["Reporting_Airline", "Origin", "Dest"]:
        if c in x.columns:
            x[c] = x[c].astype(str).str.upper().str.strip()
    for c in ["CRSDepTime", "CRSArrTime"]:
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce").astype("Int64")
    if "Flight_Number_Reporting_Airline" in x.columns:
        x["Flight_Number_Reporting_Airline"] = pd.to_numeric(x["Flight_Number_Reporting_Airline"], errors="coerce").astype("Int64")
    return x


def merge_dep_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    dep_feat_path_raw = cfg.get("dep_features_unbalanced_path")
    if not dep_feat_path_raw:
        print("[INFO] dep_features_unbalanced_path not set; skipping dep-feature merge.")
        return df

    dep_feat_path = _abspath(dep_feat_path_raw, base="repo")
    if not dep_feat_path.exists():
        print(f"[WARN] dep features file not found; skipping dep-feature merge: {dep_feat_path}")
        return df

    print(f"[INFO] Merging dep features from: {dep_feat_path}")

    dep = pd.read_parquet(dep_feat_path)
    dep = _normalize_join_keys(dep)
    df_norm = _normalize_join_keys(df)

    join_keys = ["FlightDate", "Reporting_Airline", "Flight_Number_Reporting_Airline", "Origin", "Dest", "CRSDepTime", "CRSArrTime"]
    join_keys = [k for k in join_keys if (k in df_norm.columns and k in dep.columns)]
    if not join_keys:
        raise RuntimeError("No common join keys found to merge dep features into arrival features.")

    dep_cols = [c for c in dep.columns if (c not in df_norm.columns) or (c in join_keys)]
    dep = dep[dep_cols].copy()

    merged = df_norm.merge(dep, on=join_keys, how="left", suffixes=("", "_depdup"))
    dup_cols = [c for c in merged.columns if c.endswith("_depdup")]
    if dup_cols:
        merged = merged.drop(columns=dup_cols)

    del dep, df_norm
    gc.collect()
    return merged


def add_arrival_labels(df: pd.DataFrame, thresholds: List[int]) -> pd.DataFrame:
    df = df.copy()
    y = pd.to_numeric(df.get("ArrDelayMinutes"), errors="coerce")
    for thr in thresholds:
        df[f"y_arr_ge{thr}"] = (y >= float(thr)).astype("Int8")
    return df


# ------------------------------ per-threshold balancing ------------------------------

def _balance_to_pos_frac(
    df: pd.DataFrame,
    label_col: str,
    pos_frac: float,
    seed: int,
    max_rows: Optional[int],
) -> pd.DataFrame:
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
    b_needed = int(round(a_max * (1 - target) / target))
    if b_needed <= len(neg):
        a, b, limiting = a_max, b_needed, "pos"
    else:
        b_max = len(neg)
        a = min(int(round(b_max * target / (1 - target))), a_max)
        b, limiting = b_max, "neg"
    pos_s = pos.sample(n=a, random_state=seed)
    neg_s = neg.sample(n=b, random_state=seed)
    out = pd.concat([pos_s, neg_s], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if max_rows is not None and len(out) > int(max_rows):
        out = out.sample(n=int(max_rows), random_state=seed).reset_index(drop=True)
    achieved = out[label_col].mean()
    print(f"[INFO] Balanced {label_col} to pos_frac~{achieved:.3f} (target={target:.3f}, limiting={limiting}) rows={len(out)}")
    return out


def _write_balanced_arr_splits(df: pd.DataFrame, cfg: dict) -> None:
    """Create per-threshold 50/50 balanced training parquets and (optionally) an
    eval holdout parquet, mirroring features_dep.py so that
    train_arr_bins_ordinal_catboost.py can pick them up via
    feature_balance.output_template and eval_features_path."""
    fb = cfg.get("feature_balance") or {}
    if not fb:
        return
    out_tmpl = fb.get("output_template")
    if not out_tmpl:
        print("[WARN] feature_balance.output_template not set; skipping balanced split write.")
        return

    thr_list = [int(x) for x in (fb.get("thresholds") or THRESHOLDS_DEFAULT)]
    pos_frac = float(fb.get("pos_frac", 0.50))
    seed = int(fb.get("seed", 42))
    max_rows = fb.get("max_rows")
    max_rows = int(max_rows) if max_rows is not None else None

    # Determine eval split strategy and separate out the eval/training pools.
    eval_cfg = fb.get("eval") or {}
    eval_enabled = bool(eval_cfg.get("enabled", True))
    by = str(eval_cfg.get("by") or "last_days").lower()

    if eval_enabled and by == "random":
        eval_frac = float(eval_cfg.get("eval_frac", 0.15))
        eval_seed = int(eval_cfg.get("seed", 42))
        eval_idx = df.sample(frac=eval_frac, random_state=eval_seed).index
        eval_mask = df.index.isin(eval_idx)
        eval_df = df[eval_mask].copy()
        train_pool = df[~eval_mask].copy()
        print(f"[INFO] arr eval random frac={eval_frac} seed={eval_seed} eval_rows={len(eval_df):,} train_pool_rows={len(train_pool):,}")

        eval_out = eval_cfg.get("output_path")
        if eval_out:
            eval_out_p = _abspath(eval_out, base="repo")
            eval_out_p.parent.mkdir(parents=True, exist_ok=True)
            eval_df.to_parquet(eval_out_p, index=False)
            print(f"[OK] wrote arr eval unbalanced -> {eval_out_p}")

    elif eval_enabled and by == "last_days":
        last_days = int(eval_cfg.get("last_days", 0))
        if last_days > 0:
            fd = pd.to_datetime(df["FlightDate"], errors="coerce").dt.normalize()
            cutoff = fd.max() - pd.Timedelta(days=last_days)
            eval_df = df[fd > cutoff].copy()
            train_pool = df[fd <= cutoff].copy()
            print(f"[INFO] arr eval last_days={last_days} cutoff>{cutoff.date()} eval_rows={len(eval_df):,} train_pool_rows={len(train_pool):,}")

            eval_out = eval_cfg.get("output_path")
            if eval_out:
                eval_out_p = _abspath(eval_out, base="repo")
                eval_out_p.parent.mkdir(parents=True, exist_ok=True)
                eval_df.to_parquet(eval_out_p, index=False)
                print(f"[OK] wrote arr eval unbalanced -> {eval_out_p}")
        else:
            train_pool = df.copy()
    else:
        train_pool = df.copy()

    for thr in thr_list:
        label_col = f"y_arr_ge{thr}"
        if label_col not in train_pool.columns:
            print(f"[WARN] {label_col} not in features; skipping thr={thr}")
            continue
        bal_df = _balance_to_pos_frac(train_pool, label_col=label_col, pos_frac=pos_frac, seed=seed, max_rows=max_rows)
        out_p = _abspath(out_tmpl.format(thr=thr), base="repo")
        out_p.parent.mkdir(parents=True, exist_ok=True)
        bal_df.to_parquet(out_p, index=False)
        print(f"[OK] wrote balanced arr train thr={thr} -> {out_p}")


def main(cfg_path: str) -> None:
    cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    target = str(cfg.get("target", "dep")).strip().lower()
    if target != "arr":
        raise ValueError("features_arr.py expects cfg['target'] == 'arr'")

    thresholds = [int(x) for x in cfg.get("thresholds", THRESHOLDS_DEFAULT)]

    enriched_path = _abspath(cfg["output_enriched_unbalanced_path"], base="repo")
    history_path = _abspath(cfg["history_output_path"], base="repo")
    out_path = _abspath(cfg["output_features_unbalanced_path"], base="repo")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Reading enriched: {enriched_path}")
    need_df_want = [
        "FlightDate", "Reporting_Airline", "Flight_Number_Reporting_Airline", "Origin", "Dest",
        "CRSDepTime", "CRSArrTime", "arr_dt_local", "dep_dt_local", "DepTimeBlk",
        "ArrDelayMinutes", "aircraft_type", "Aircraft_Age_Bucket",
    ]
    # Project only columns that actually exist in the parquet
    import pyarrow.parquet as _pq
    _avail = set(_pq.read_schema(str(enriched_path)).names)
    need_df = [c for c in need_df_want if c in _avail]
    df = pd.read_parquet(enriched_path, columns=need_df)
    df = _ensure_base_types(df)

    # Optional date range filter (mirrors features_dep.py behaviour)
    start_date = cfg.get("start_date")
    end_date = cfg.get("end_date")
    if start_date or end_date:
        fd = pd.to_datetime(df["FlightDate"], errors="coerce")
        mask = pd.Series(True, index=df.index)
        if start_date:
            mask &= fd >= pd.Timestamp(start_date)
        if end_date:
            mask &= fd <= pd.Timestamp(end_date)
        df = df[mask].reset_index(drop=True)  # reset index so _attach_asof works
        print(f"[INFO] date filter {start_date}..{end_date} -> {len(df):,} rows")

    # ── Optional: destination hourly weather at scheduled arrival time ─────────
    # Enabled via cfg["dest_weather"]["enabled"] = true.
    # Imports add_dest_weather_hourly_at_arr from sibling prepare_dataset.py.
    # Outputs 8 dest_arr_* columns (temp, precip, wind, weathercode, etc.)
    dest_weather_cfg = cfg.get("dest_weather") or {}
    if dest_weather_cfg.get("enabled", False):
        _here = Path(__file__).parent
        if str(_here) not in sys.path:
            sys.path.insert(0, str(_here))
        from prepare_dataset import add_dest_weather_hourly_at_arr  # noqa: E402

        airports_csv = dest_weather_cfg.get("airports_csv", "data/meta/airports.csv")
        airports_csv_p = _abspath(airports_csv, base="repo")
        if not airports_csv_p.exists():
            raise FileNotFoundError(f"[dest_weather] airports_csv not found: {airports_csv_p}")
        _meta = pd.read_csv(airports_csv_p, dtype=str)
        _meta["IATA"] = _meta["IATA"].astype(str).str.upper().str.strip()
        _meta["Latitude"] = pd.to_numeric(_meta["Latitude"], errors="coerce")
        _meta["Longitude"] = pd.to_numeric(_meta["Longitude"], errors="coerce")
        needed_apts = set(df["Dest"].dropna().astype(str).str.upper().unique())
        airports_meta = {}
        for _, _row in _meta.iterrows():
            ap = str(_row["IATA"])
            if ap in needed_apts:
                lat, lon, tz = _row.get("Latitude"), _row.get("Longitude"), _row.get("Timezone")
                if pd.notna(lat) and pd.notna(lon) and pd.notna(tz) and str(tz).strip() not in ("", "nan", "none"):
                    airports_meta[ap] = (float(lat), float(lon), str(tz).strip())
        dest_hourly_cache = dest_weather_cfg.get("hourly_cache_dir", "../flightrightdata/weather_cache_dest_hourly")
        dest_hourly_cache_p = str(_abspath(dest_hourly_cache, base="repo"))
        print(f"[INFO] dest_weather: fetching hourly weather for {len(airports_meta)} dest airports -> {dest_hourly_cache_p}")
        df = add_dest_weather_hourly_at_arr(df, airports_meta=airports_meta, cache_dir=dest_hourly_cache_p)
        _dest_cols = [c for c in df.columns if c.startswith("dest_arr_")]
        print(f"[INFO] dest_weather: added {len(_dest_cols)} columns: {_dest_cols}")

    # Departure-timing features: day-of-week and scheduled departure hour.
    # These are strong predictors for both departure and arrival delay.
    # Must read the pre-computed per-row ints from prepare_dataset; the round-tripped
    # dep_dt_local parquet column has a single coerced tz (Chicago) so .dt.hour on it
    # returns Chicago wall-clock, not origin-local.
    if "sched_dep_hour" in df.columns and "dep_dow" in df.columns:
        df["sched_dep_hour"] = pd.to_numeric(df["sched_dep_hour"], errors="coerce").astype("Int8")
        df["dep_dow"] = pd.to_numeric(df["dep_dow"], errors="coerce").astype("Int8")
    else:
        _fd = pd.to_datetime(df["FlightDate"], errors="coerce")
        df["dep_dow"] = _fd.dt.dayofweek.astype("Int8")
        df["sched_dep_hour"] = pd.to_numeric(df.get("CRSDepTime"), errors="coerce").floordiv(100).astype("Int8")

    # Scheduled arrival hour — later arrivals accumulate more network delay.
    if "sched_arr_hour" in df.columns:
        df["sched_arr_hour"] = pd.to_numeric(df["sched_arr_hour"], errors="coerce").astype("Int8")
    else:
        df["sched_arr_hour"] = pd.to_numeric(df.get("CRSArrTime"), errors="coerce").floordiv(100).astype("Int8")

    print(f"[INFO] Reading history pool: {history_path}")
    need_hist_want = [
        "FlightDate", "Reporting_Airline", "Flight_Number_Reporting_Airline", "Origin", "Dest",
        "CRSDepTime", "ArrDelayMinutes", "AirTime", "WheelsOff",
        "ActualElapsedTime", "CRSElapsedTime",
    ]
    import pyarrow.parquet as _pq_h
    _avail_hist = set(_pq_h.read_schema(str(history_path)).names)
    need_hist = [c for c in need_hist_want if c in _avail_hist]
    hist = pd.read_parquet(history_path, columns=need_hist)
    hist = _ensure_base_types(hist)
    if "ActualElapsedTime" in hist.columns:
        hist["ActualElapsedTime"] = pd.to_numeric(hist["ActualElapsedTime"], errors="coerce")
    if "CRSElapsedTime" in hist.columns:
        hist["CRSElapsedTime"] = pd.to_numeric(hist["CRSElapsedTime"], errors="coerce")

    df_start = df["FlightDate"].min()
    df_end = df["FlightDate"].max()
    if pd.notna(df_start) and pd.notna(df_end):
        hist = hist[(hist["FlightDate"] >= (df_start - pd.Timedelta(days=14))) & (hist["FlightDate"] <= df_end)].copy()
        gc.collect()

    # v10: 14d medians for arrdelay route stats. Gated by cfg["arrdelay_median_14d"]["enabled"];
    # default off so v8 pipelines remain byte-identical.
    arrmed14_cfg = cfg.get("arrdelay_median_14d") or {}
    arrmed_windows = MEDIAN_WINDOWS_V10 if arrmed14_cfg.get("enabled", False) else MEDIAN_WINDOWS
    df = add_route_arrival_delay_stats(df, hist, median_windows=arrmed_windows); gc.collect()

    # v10: airtime and wheels-off slip features depend on AirTime / WheelsOff, both of which
    # cannot be sourced from AeroDataBox at predict time. Gate them so v10 pipelines skip them.
    if (cfg.get("recent_airtime") or {}).get("enabled", True):
        df = add_recent_airtime_stats(df, hist); gc.collect()
    else:
        print("[INFO] v10: recent_airtime disabled (skipping airtime_*_d_* features)")
    if (cfg.get("wheels_off_slip") or {}).get("enabled", True):
        df = add_recent_wheels_off_slip_stats(df, hist); gc.collect()
    else:
        print("[INFO] v10: wheels_off_slip disabled (skipping wo_slip_*_d_* features)")

    # v10: elapsed_time_ratio_last14d uses ActualElapsedTime / CRSElapsedTime, both gate-based
    # BTS fields. Safe under the AeroDataBox constraint. Gated by cfg["elapsed_time_ratio"]["enabled"].
    if (cfg.get("elapsed_time_ratio") or {}).get("enabled", False):
        df = add_elapsed_time_ratio_stats(df, hist, window_days=14); gc.collect()

    df = add_projected_arrival_congestion(df); gc.collect()
    df = merge_dep_features(df, cfg); gc.collect()
    df = add_arrival_labels(df, thresholds)

    _write_balanced_arr_splits(df, cfg)

    df.to_parquet(out_path, index=False)
    print(f"[OK] wrote arrival features rows={len(df):,} -> {out_path}")

    # v10: write a small descriptive-stats JSON next to the unbalanced parquet so every
    # training run has a companion file with min/mean/median/std/P01/P50/P99/etc. per
    # feature for sanity-checking distributions before training starts.
    try:
        from feature_stats import write_feature_stats_json  # noqa: E402
    except ModuleNotFoundError:
        from src.fetch_prune.feature_stats import write_feature_stats_json  # type: ignore  # noqa: E402
    stats_path = write_feature_stats_json(df, out_path)
    print(f"[OK] wrote feature stats -> {stats_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python src/fetch_prune/features_arr.py data/arr_config.json")
    main(sys.argv[1])