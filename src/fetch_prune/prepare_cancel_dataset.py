#!/usr/bin/env python3
# src/fetch_prune/prepare_cancel_dataset.py
#
# Dedicated data preparation for the cancellation prediction model.
# Loads raw BTS parquets for all airlines, generates departure features,
# and produces balanced training data targeting a configurable cancelled:not-cancelled ratio.
#
# Run:
#   python src/fetch_prune/prepare_cancel_dataset.py data/blueprint_cancel_v9.json
#
import sys
import json
import gc
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd

# Reuse feature engineering from the departure pipeline
from prepare_dataset import (
    read_input_parquet,
    canonicalize_bts_columns,
    load_airport_meta,
)
from features_dep import (
    add_holiday_flag,
    add_tail_and_flightnum_time_features,
    add_turn_time_hours,
    add_is_peak_hour,
    add_weather_kelvin,
    add_flightnum_od_depdelay_means_from_history,
    add_carrier_dep_delay_baselines_from_history_multi,
    add_origin_dep_delay_baselines_from_history_multi,
    add_tail_rolling_history,
    add_dest_rolling_stats,
    add_congestion_3h_features_from_history,
    add_wn_hub_spillover_from_history,
    add_hub_max_lateaircraft_last1,
    add_delay_cause_rates_from_history,
    add_strike_proximity_features,
    add_carrier_delay_rate_anomaly,
    add_wind_x_precip,
    add_dep_labels_for_thresholds,
)


REPO_ROOT = Path.cwd()
DATA_ROOT = (REPO_ROOT.parent / "flightrightdata").resolve()

# Columns needed for feature engineering + target
_KEEP_COLS = {
    "FlightDate",
    "Reporting_Airline", "IATA_Code_Operating_Airline",
    "Flight_Number_Reporting_Airline", "Flight_Number_Operating_Airline",
    "Origin", "Dest",
    "CRSDepTime", "CRSArrTime", "CRSElapsedTime",
    "DepDelayMinutes", "ArrDelayMinutes",
    "DepDel15", "ArrDel15",
    "Tail_Number", "AirTime",
    "WheelsOff", "WheelsOn", "TaxiOut", "TaxiIn",
    "DepTimeBlk", "Distance",
    "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay",
    "Cancelled",
}


def _abspath(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (DATA_ROOT / pp).resolve()


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python prepare_cancel_dataset.py BLUEPRINT.json")

    bp = _load_json(sys.argv[1])

    # ---- Load raw flights ----
    input_paths = bp.get("input_flights_path", [])
    if isinstance(input_paths, str):
        input_paths = [input_paths]
    input_paths = [_abspath(p) for p in input_paths]

    airlines = bp.get("airlines", ["WN", "DL", "AA", "UA"])
    start_date = bp.get("start_date")
    end_date = bp.get("end_date")

    print(f"[INFO] Loading flights from {len(input_paths)} source(s) for airlines: {airlines}")
    print(f"[INFO] Date range: {start_date} to {end_date}")

    # Load with the shared reader (which now includes Cancelled column)
    parts: List[pd.DataFrame] = []
    for fp in input_paths:
        if fp.is_dir():
            parquets = sorted(fp.rglob("*.parquet"))
        else:
            parquets = [fp]
        for pq in parquets:
            print(f"[INFO] Reading: {pq}")
            # Only load columns we need to save memory
            pq_meta = pd.read_parquet(pq, columns=[])
            avail_cols = list(pq_meta.columns) if hasattr(pq_meta, 'columns') else []
            if not avail_cols:
                import pyarrow.parquet as _pq
                avail_cols = _pq.read_schema(pq).names
            use_cols = [c for c in avail_cols if c in _KEEP_COLS]
            df = pd.read_parquet(pq, columns=use_cols if use_cols else None)
            df = canonicalize_bts_columns(df)

            # Filter airlines
            if "Reporting_Airline" in df.columns:
                df["Reporting_Airline"] = df["Reporting_Airline"].astype(str).str.upper().str.strip()
                df = df[df["Reporting_Airline"].isin(airlines)]

            # Filter dates
            if start_date and "FlightDate" in df.columns:
                df["FlightDate"] = pd.to_datetime(df["FlightDate"], errors="coerce")
                df = df[df["FlightDate"] >= pd.Timestamp(start_date)]
            if end_date and "FlightDate" in df.columns:
                df = df[df["FlightDate"] <= pd.Timestamp(end_date)]

            if len(df) > 0:
                parts.append(df)
            del df
            gc.collect()

    if not parts:
        raise SystemExit("No data loaded!")

    df = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()
    print(f"[INFO] Total rows loaded: {len(df):,}")

    # Ensure Cancelled column
    if "Cancelled" not in df.columns:
        raise SystemExit("Cancelled column not found in data. Ensure prepare_dataset.py includes it in keep_cols.")
    df["Cancelled"] = pd.to_numeric(df["Cancelled"], errors="coerce").fillna(0).astype(int)

    n_cancelled = df["Cancelled"].sum()
    n_total = len(df)
    print(f"[INFO] Cancelled flights: {n_cancelled:,} / {n_total:,} ({n_cancelled/n_total:.4f})")

    # ---- Load history pool for feature engineering ----
    history_paths = bp.get("history_flights_path", [])
    if isinstance(history_paths, str):
        history_paths = [history_paths]
    history_paths = [_abspath(p) for p in history_paths]

    # ---- Compute dep_dt_local on full dataset (needed by history functions) ----
    print("[INFO] Computing dep_dt_local on full dataset for history pool...")
    fd = pd.to_datetime(df["FlightDate"], errors="coerce")
    hhmm = pd.to_numeric(df.get("CRSDepTime"), errors="coerce").fillna(0).astype(int)
    hours = hhmm // 100
    mins = hhmm % 100
    df["dep_dt_local"] = fd + pd.to_timedelta(hours, unit="h") + pd.to_timedelta(mins, unit="m")
    del fd, hhmm, hours, mins
    gc.collect()

    # ---- Pre-sample: keep all cancelled + subsample not-cancelled for feature engineering ----
    # This avoids OOM by reducing the feature-engineering target from 15M to ~1.5M rows.
    # The full dataset is still used as the history pool for rolling stats.
    balance_cfg = bp.get("balance", {})
    pos_frac = float(balance_cfg.get("pos_frac", 0.20))
    seed = int(balance_cfg.get("seed", 42))
    eval_frac = float(bp.get("feature_balance", {}).get("eval", {}).get("eval_frac", 0.15))

    df_pos_all = df[df["Cancelled"] == 1]
    df_neg_all = df[df["Cancelled"] == 0]
    n_pos = len(df_pos_all)
    # We need enough negatives for: training (80% of pos / pos_frac * (1-pos_frac)) + eval
    # Overshoot to account for eval split
    n_neg_target = int(n_pos * (1 - pos_frac) / pos_frac / (1 - eval_frac) * 1.2)
    n_neg_target = min(n_neg_target, len(df_neg_all))
    df_neg_sampled = df_neg_all.sample(n=n_neg_target, random_state=seed)
    # Trim history pool to only columns needed for rolling lookups (saves ~60% memory)
    _HIST_COLS = [
        "FlightDate", "Reporting_Airline", "Flight_Number_Reporting_Airline",
        "Origin", "Dest", "CRSDepTime", "CRSArrTime",
        "DepDelayMinutes", "ArrDelayMinutes", "DepDel15", "ArrDel15",
        "Tail_Number", "Distance", "DepTimeBlk",
        "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay",
        "Cancelled", "dep_dt_local",
    ]
    hist_cols_avail = [c for c in _HIST_COLS if c in df.columns]
    df_full = df  # keep reference before overwriting df
    df_hist = df_full[hist_cols_avail].copy()
    del df_full
    gc.collect()
    df = pd.concat([df_pos_all, df_neg_sampled], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    del df_pos_all, df_neg_all, df_neg_sampled
    gc.collect()
    print(f"[INFO] Pre-sampled for feature engineering: {len(df):,} rows (cancelled={df['Cancelled'].sum():,}, rate={df['Cancelled'].mean():.4f})")

    # If history paths match input paths, reuse already-loaded data
    if history_paths and set(str(p) for p in history_paths) == set(str(p) for p in input_paths):
        print(f"[INFO] History paths == input paths, reusing full dataset as history pool ({len(df_hist):,} rows)")
    elif history_paths:
        hist_parts = []
        for fp in history_paths:
            if fp.is_dir():
                parquets = sorted(fp.rglob("*.parquet"))
            else:
                parquets = [fp]
            for pq in parquets:
                print(f"[INFO] Reading history: {pq}")
                import pyarrow.parquet as _pq
                avail_cols = _pq.read_schema(pq).names
                use_cols = [c for c in avail_cols if c in _KEEP_COLS]
                hdf = pd.read_parquet(pq, columns=use_cols if use_cols else None)
                hdf = canonicalize_bts_columns(hdf)
                if "Reporting_Airline" in hdf.columns:
                    hdf["Reporting_Airline"] = hdf["Reporting_Airline"].astype(str).str.upper().str.strip()
                    hdf = hdf[hdf["Reporting_Airline"].isin(airlines)]
                hist_parts.append(hdf)
                del hdf
                gc.collect()
        df_hist = pd.concat(hist_parts, ignore_index=True) if hist_parts else df.copy()
        del hist_parts
        gc.collect()
        print(f"[INFO] History pool rows: {len(df_hist):,}")
    else:
        df_hist = df.copy()
        print("[INFO] Using main dataset as history pool")

    # ---- Feature engineering (reuse departure pipeline functions) ----
    feat_cfg = bp.get("features_dep", {})

    print("[INFO] Adding temporal features...")
    df = add_holiday_flag(df)
    df = add_is_peak_hour(df)

    if feat_cfg.get("tail_and_flightnum_time", True):
        print("[INFO] Adding tail/flightnum time features...")
        df = add_tail_and_flightnum_time_features(df)
        df = add_turn_time_hours(df)

    print("[INFO] Adding weather features...")
    weather_cfg = bp.get("weather", {})
    if weather_cfg.get("daily", {}).get("enabled", True):
        df = add_weather_kelvin(df)

    # Standard rolling windows for history features
    flightnum_od_cfg = feat_cfg.get("flightnum_od", {})
    fn_n_recent = int(flightnum_od_cfg.get("n_recent", 20))
    fn_lookback = int(flightnum_od_cfg.get("lookback_days", 14))

    print("[INFO] Adding flight-number OD history...")
    df = add_flightnum_od_depdelay_means_from_history(
        df, df_hist, windows_days=[1, 7, 14], n_recent=fn_n_recent
    )

    print("[INFO] Adding carrier baselines...")
    df = add_carrier_dep_delay_baselines_from_history_multi(df, df_hist, windows_days=[1, 7, 14])

    print("[INFO] Adding origin baselines...")
    df = add_origin_dep_delay_baselines_from_history_multi(df, df_hist, windows_days=[1, 7, 14])

    print("[INFO] Adding delay cause rates...")
    df = add_delay_cause_rates_from_history(df, df_hist, windows_days=[1, 7])

    print("[INFO] Adding congestion features...")
    df = add_congestion_3h_features_from_history(df, df_hist)

    print("[INFO] Adding destination rolling stats...")
    df = add_dest_rolling_stats(df, df_hist, windows_days=[1, 7, 14])

    hub_cfg = feat_cfg.get("hub_spillover", {})
    if hub_cfg.get("enabled", True):
        hubs = hub_cfg.get("hubs", ["ATL", "DEN", "DFW", "ORD", "EWR"])
        print(f"[INFO] Adding hub spillover for hubs={hubs}...")
        df = add_wn_hub_spillover_from_history(
            df, df_hist, windows_days=[1, 7, 14], hubs=hubs
        )
        df = add_hub_max_lateaircraft_last1(df)

    tail_hist_cfg = feat_cfg.get("tail_history", {})
    if tail_hist_cfg.get("enabled", False):
        tail_windows = tail_hist_cfg.get("windows", [1, 14])
        print(f"[INFO] Adding tail rolling history (windows={tail_windows})...")
        df = add_tail_rolling_history(df, df_hist, windows_days=tail_windows)

    strike_cfg = bp.get("strike", {})
    if strike_cfg.get("enabled", False):
        cache_path = str(_abspath(strike_cfg.get("cache_path", "data/strike_cache.json")))
        print("[INFO] Adding strike features...")
        df = add_strike_proximity_features(df, cache_path)
        df = add_carrier_delay_rate_anomaly(df, df_hist)

    df = add_wind_x_precip(df)

    del df_hist
    gc.collect()

    # ---- Save unbalanced features ----
    output_unbal = bp.get("output_features_unbalanced_path",
                          "data/processed/features_cancel_all_v9_unbalanced.parquet")
    output_unbal = _abspath(output_unbal)
    output_unbal.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_unbal, index=False)
    print(f"[SAVE] Unbalanced features ({len(df):,} rows) -> {output_unbal}")

    # ---- Save eval set ----
    eval_cfg = bp.get("feature_balance", {}).get("eval", {})
    if eval_cfg.get("enabled", True):
        eval_frac = float(eval_cfg.get("eval_frac", 0.15))
        # Time-based: use the last N% of data as eval
        df = df.sort_values("FlightDate").reset_index(drop=True)
        n_eval = int(len(df) * eval_frac)
        df_eval = df.iloc[-n_eval:].reset_index(drop=True)
        df_train_pool = df.iloc[:-n_eval].reset_index(drop=True)

        eval_path = eval_cfg.get("output_path",
                                 "data/processed/features_cancel_all_v9_eval_unbalanced.parquet")
        eval_path = _abspath(eval_path)
        df_eval.to_parquet(eval_path, index=False)
        print(f"[SAVE] Eval set ({len(df_eval):,} rows) -> {eval_path}")
    else:
        df_train_pool = df

    # ---- Balance and save training data ----
    balance_cfg = bp.get("balance", {})
    pos_frac = float(balance_cfg.get("pos_frac", 0.20))
    seed = int(balance_cfg.get("seed", 42))

    df_pos = df_train_pool[df_train_pool["Cancelled"] == 1]
    df_neg = df_train_pool[df_train_pool["Cancelled"] == 0]
    n_pos = len(df_pos)
    n_neg_target = int(n_pos * (1 - pos_frac) / pos_frac)

    if n_neg_target < len(df_neg):
        df_neg_sampled = df_neg.sample(n=n_neg_target, random_state=seed)
    else:
        df_neg_sampled = df_neg

    df_balanced = pd.concat([df_pos, df_neg_sampled], ignore_index=True)
    df_balanced = df_balanced.sample(frac=1, random_state=seed).reset_index(drop=True)

    bal_path = balance_cfg.get("output_path",
                               "data/processed/train_cancel_balanced_v9.parquet")
    bal_path = _abspath(bal_path)
    df_balanced.to_parquet(bal_path, index=False)
    actual_pos_rate = df_balanced["Cancelled"].mean()
    print(f"[SAVE] Balanced training ({len(df_balanced):,} rows, pos_rate={actual_pos_rate:.3f}) -> {bal_path}")

    print(f"[DONE] Cancellation dataset preparation complete.")
    print(f"  Unbalanced: {output_unbal}")
    if eval_cfg.get("enabled", True):
        print(f"  Eval:       {eval_path}")
    print(f"  Balanced:   {bal_path}")


if __name__ == "__main__":
    main()
