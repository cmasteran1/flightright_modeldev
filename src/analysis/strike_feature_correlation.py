#!/usr/bin/env python3
"""
Strike Feature Correlation Analysis for FlightRight.

Independently joins strike/labor-action data onto existing enriched flight
data (no need to re-run the pipeline), then measures correlation of each
proposed strike feature with actual departure/arrival delay.

Usage:
    python src/analysis/strike_feature_correlation.py

Reads:
    - Enriched flight parquet (intermediate/WN_100_v7g_unbalanced.parquet or similar)
    - History pool parquet (intermediate/history_pool_dep_WN_100_v7g.parquet)
    - Strike cache parquet (strike_cache/us_aviation_labor_actions.parquet)

Outputs:
    - Console summary table
    - data/analysis/strike_feature_correlations.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path.cwd()
DATA_ROOT = (REPO_ROOT.parent / "flightrightdata").resolve()

# ---------------------------------------------------------------------------
# Paths (defaults; can be overridden via CLI or env)
# ---------------------------------------------------------------------------
ENRICHED_PATH = DATA_ROOT / "data" / "intermediate" / "WN_100_v7g_unbalanced.parquet"
HISTORY_PATH = DATA_ROOT / "data" / "intermediate" / "history_pool_dep_WN_100_v7g.parquet"
STRIKE_CACHE = DATA_ROOT / "strike_cache" / "us_aviation_labor_actions.parquet"
OUTPUT_DIR = DATA_ROOT / "data" / "analysis"


# ---------------------------------------------------------------------------
# Strike enrichment (simplified, inline version of prepare_dataset logic)
# ---------------------------------------------------------------------------

def enrich_with_strike_status(df: pd.DataFrame, strikes: pd.DataFrame) -> pd.DataFrame:
    """Add strike status columns to flight data."""
    fd = pd.to_datetime(df["FlightDate"], errors="coerce").dt.normalize()
    airline = df["Reporting_Airline"].astype(str).str.upper().str.strip()
    origin = df["Origin"].astype(str).str.upper().str.strip()

    carrier_active = np.zeros(len(df), dtype=np.int8)
    origin_active = np.zeros(len(df), dtype=np.int8)
    max_severity = np.zeros(len(df), dtype=np.int8)
    best_worker_type = pd.Series("none", index=df.index, dtype=str)
    in_cool = np.zeros(len(df), dtype=np.int8)
    cool_days = np.full(len(df), np.nan, dtype=np.float32)
    days_to = pd.Series(np.nan, index=df.index, dtype=np.float32)
    days_since = pd.Series(np.nan, index=df.index, dtype=np.float32)
    spillover = np.zeros(len(df), dtype=np.int8)

    wt_priority = {"atc": 6, "pilot": 5, "mechanic": 4, "flight_attendant": 3, "ground_crew": 2, "dispatcher": 1}
    # Track best worker type priority as numeric array for vectorized updates
    best_wt_prio = np.zeros(len(df), dtype=np.int8)

    for _, evt in strikes.iterrows():
        evt_start = pd.Timestamp(evt["action_start_date"])
        evt_end = pd.Timestamp(evt["action_end_date"])
        evt_airline = str(evt.get("airline_iata", "")).upper().strip()
        evt_airports = str(evt.get("affected_airports", "ALL")).upper().strip()
        evt_severity = int(evt.get("severity", 1)) if pd.notna(evt.get("severity")) else 1
        evt_worker = str(evt.get("worker_type", "")).lower().strip()
        evt_scope = str(evt.get("scope", "")).lower().strip()

        if pd.isna(evt_start):
            continue
        if pd.isna(evt_end):
            evt_end = evt_start

        date_mask = (fd >= evt_start) & (fd <= evt_end)

        # Carrier match
        if evt_airline == "ALL" or evt_scope == "system_wide":
            carrier_mask = date_mask
        else:
            carrier_mask = date_mask & (airline == evt_airline)

        # Airport match
        if evt_airports == "ALL" or evt_scope in ("airline_wide", "system_wide"):
            airport_mask = date_mask
        else:
            airport_set = {a.strip() for a in evt_airports.split(",")}
            airport_mask = date_mask & origin.isin(airport_set)

        carrier_active[carrier_mask.values] = 1
        origin_active[airport_mask.values] = 1

        combined = carrier_mask | airport_mask
        max_severity[combined.values] = np.maximum(max_severity[combined.values], evt_severity)

        # Vectorized worker type update
        evt_prio = wt_priority.get(evt_worker, 0)
        upgrade_mask = combined.values & (best_wt_prio < evt_prio)
        best_wt_prio[upgrade_mask] = evt_prio
        best_worker_type[upgrade_mask] = evt_worker

        # Cooling-off
        cool_start = pd.Timestamp(evt.get("cooling_off_start"))
        if pd.notna(cool_start):
            cool_end = cool_start + pd.Timedelta(days=30)
            cool_mask = (fd >= cool_start) & (fd <= cool_end)
            if evt_airline != "ALL":
                cool_mask = cool_mask & (airline == evt_airline)
            in_cool[cool_mask.values] = 1
            remaining = (cool_end - fd).dt.days.astype(float)
            cool_update = cool_mask & (remaining >= 0)
            cool_days[cool_update.values] = np.fmin(cool_days[cool_update.values], remaining[cool_update].values)

        # Days to strike
        delta_to_start = (evt_start - fd).dt.days.astype(float)
        if evt_airline == "ALL" or evt_scope == "system_wide":
            cm = pd.Series(True, index=df.index)
        else:
            cm = airline == evt_airline
        future_mask = cm & (delta_to_start >= 0) & (delta_to_start <= 90)
        days_to = days_to.where(~future_mask, np.fmin(days_to[future_mask], delta_to_start[future_mask]))

        # Days since strike end
        delta_since_end = (fd - evt_end).dt.days.astype(float)
        past_mask = cm & (delta_since_end >= 0) & (delta_since_end <= 90)
        days_since = days_since.where(~past_mask, np.fmin(days_since[past_mask], delta_since_end[past_mask]))

        # Spillover
        if evt_airports == "ALL" or evt_scope in ("airline_wide", "system_wide"):
            am = date_mask
        else:
            airport_set_s = {a.strip() for a in evt_airports.split(",")}
            am = date_mask & origin.isin(airport_set_s)
        if evt_airline != "ALL":
            other_carrier = am & (airline != evt_airline)
            spillover[other_carrier.values] = 1

    df["strike_active_carrier"] = carrier_active
    df["strike_active_origin"] = origin_active
    df["strike_severity"] = max_severity
    df["strike_worker_type"] = best_worker_type
    df["in_cooling_off"] = in_cool
    df["cooling_off_days_remaining"] = cool_days
    df["days_to_strike"] = days_to
    df["days_since_strike_end"] = days_since
    df["strike_spillover_origin"] = spillover

    return df


def compute_carrier_delay_anomaly(df: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    """Compute carrier delay rate z-score anomaly feature."""
    if "CarrierDelay" not in hist.columns:
        df["carrier_delay_rate_anomaly_7d"] = np.nan
        return df

    h = hist.copy()
    h["FlightDate"] = pd.to_datetime(h.get("FlightDate"), errors="coerce").dt.normalize()
    h["Reporting_Airline"] = h.get("Reporting_Airline", "").astype(str).str.upper().str.strip()
    h["CarrierDelay"] = pd.to_numeric(h.get("CarrierDelay"), errors="coerce").fillna(0.0)
    h["_has_carrier_delay"] = (h["CarrierDelay"] > 0).astype(float)

    daily = (
        h.dropna(subset=["Reporting_Airline", "FlightDate"])
        .groupby(["Reporting_Airline", "FlightDate"], as_index=False)
        [["_has_carrier_delay"]]
        .mean()
        .sort_values(["Reporting_Airline", "FlightDate"])
    )

    daily["rate_7d"] = (
        daily.groupby("Reporting_Airline")["_has_carrier_delay"]
        .apply(lambda s: s.shift(1).rolling(window=7, min_periods=3).mean())
        .reset_index(level=0, drop=True)
    )
    daily["rate_60d_mean"] = (
        daily.groupby("Reporting_Airline")["_has_carrier_delay"]
        .apply(lambda s: s.shift(1).rolling(window=60, min_periods=14).mean())
        .reset_index(level=0, drop=True)
    )
    daily["rate_60d_std"] = (
        daily.groupby("Reporting_Airline")["_has_carrier_delay"]
        .apply(lambda s: s.shift(1).rolling(window=60, min_periods=14).std())
        .reset_index(level=0, drop=True)
    )
    daily["carrier_delay_rate_anomaly_7d"] = (
        (daily["rate_7d"] - daily["rate_60d_mean"]) / daily["rate_60d_std"].clip(lower=0.01)
    )

    df["FlightDate"] = pd.to_datetime(df.get("FlightDate"), errors="coerce").dt.normalize()
    df["Reporting_Airline"] = df.get("Reporting_Airline", "").astype(str).str.upper().str.strip()
    df = df.merge(
        daily[["Reporting_Airline", "FlightDate", "carrier_delay_rate_anomaly_7d"]],
        on=["Reporting_Airline", "FlightDate"],
        how="left",
    )
    return df


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def analyze_binary_feature(df: pd.DataFrame, feat: str, delay_col: str) -> dict:
    """Analyze a binary (0/1) strike feature against delay."""
    active = df[df[feat] == 1]
    inactive = df[df[feat] == 0]

    n_active = len(active)
    n_inactive = len(inactive)

    if n_active == 0:
        return {
            "feature": feat,
            "type": "binary",
            "n_active": 0,
            "n_inactive": n_inactive,
            "pct_active": 0.0,
            "mean_delay_active": np.nan,
            "mean_delay_inactive": np.nan,
            "mean_delay_diff": np.nan,
            "delay_rate_15_active": np.nan,
            "delay_rate_15_inactive": np.nan,
            "delay_rate_lift": np.nan,
            "point_biserial_r": np.nan,
            "p_value": np.nan,
            "cohens_d": np.nan,
            "ttest_p": np.nan,
        }

    mean_a = active[delay_col].mean()
    mean_i = inactive[delay_col].mean()
    std_pooled = df[delay_col].std()

    # Point-biserial correlation
    valid = df[[feat, delay_col]].dropna()
    if len(valid) > 2 and valid[feat].nunique() > 1:
        r, p_pb = stats.pointbiserialr(valid[feat], valid[delay_col])
    else:
        r, p_pb = np.nan, np.nan

    # T-test
    if n_active >= 2 and n_inactive >= 2:
        t_stat, t_p = stats.ttest_ind(active[delay_col].dropna(), inactive[delay_col].dropna(), equal_var=False)
    else:
        t_stat, t_p = np.nan, np.nan

    # Cohen's d
    cohens_d = (mean_a - mean_i) / std_pooled if std_pooled > 0 else np.nan

    # Delay rate (15+ min)
    rate_a = (active[delay_col] >= 15).mean() if n_active > 0 else np.nan
    rate_i = (inactive[delay_col] >= 15).mean() if n_inactive > 0 else np.nan
    lift = rate_a / rate_i if rate_i and rate_i > 0 else np.nan

    return {
        "feature": feat,
        "type": "binary",
        "n_active": n_active,
        "n_inactive": n_inactive,
        "pct_active": n_active / len(df) * 100,
        "mean_delay_active": round(mean_a, 2),
        "mean_delay_inactive": round(mean_i, 2),
        "mean_delay_diff": round(mean_a - mean_i, 2),
        "delay_rate_15_active": round(rate_a * 100, 2) if not np.isnan(rate_a) else np.nan,
        "delay_rate_15_inactive": round(rate_i * 100, 2) if not np.isnan(rate_i) else np.nan,
        "delay_rate_lift": round(lift, 3) if not np.isnan(lift) else np.nan,
        "point_biserial_r": round(r, 6) if not np.isnan(r) else np.nan,
        "p_value": p_pb,
        "cohens_d": round(cohens_d, 4) if not np.isnan(cohens_d) else np.nan,
        "ttest_p": t_p,
    }


def analyze_continuous_feature(df: pd.DataFrame, feat: str, delay_col: str) -> dict:
    """Analyze a continuous strike feature against delay."""
    valid = df[[feat, delay_col]].dropna()
    n_valid = len(valid)

    if n_valid < 10:
        return {
            "feature": feat,
            "type": "continuous",
            "n_valid": n_valid,
            "n_total": len(df),
            "pct_non_null": n_valid / len(df) * 100 if len(df) > 0 else 0,
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_r": np.nan,
            "spearman_p": np.nan,
            "mean_delay_q1": np.nan,
            "mean_delay_q4": np.nan,
            "delay_diff_q4_q1": np.nan,
        }

    pearson_r, pearson_p = stats.pearsonr(valid[feat], valid[delay_col])
    spearman_r, spearman_p = stats.spearmanr(valid[feat], valid[delay_col])

    # Quartile analysis
    try:
        quartiles = pd.qcut(valid[feat], 4, labels=False, duplicates="drop")
        q1_delay = valid[delay_col][quartiles == 0].mean()
        q4_delay = valid[delay_col][quartiles == quartiles.max()].mean()
    except ValueError:
        q1_delay = q4_delay = np.nan

    return {
        "feature": feat,
        "type": "continuous",
        "n_valid": n_valid,
        "n_total": len(df),
        "pct_non_null": round(n_valid / len(df) * 100, 2) if len(df) > 0 else 0,
        "pearson_r": round(pearson_r, 6),
        "pearson_p": pearson_p,
        "spearman_r": round(spearman_r, 6),
        "spearman_p": spearman_p,
        "mean_delay_q1": round(q1_delay, 2) if not np.isnan(q1_delay) else np.nan,
        "mean_delay_q4": round(q4_delay, 2) if not np.isnan(q4_delay) else np.nan,
        "delay_diff_q4_q1": round(q4_delay - q1_delay, 2) if not (np.isnan(q1_delay) or np.isnan(q4_delay)) else np.nan,
    }


def analyze_severity_feature(df: pd.DataFrame, delay_col: str) -> dict:
    """Analyze strike_severity ordinal feature."""
    result = analyze_continuous_feature(df, "strike_severity", delay_col)

    # Add per-severity breakdown
    for sev in sorted(df["strike_severity"].dropna().unique()):
        sev = int(sev)
        subset = df[df["strike_severity"] == sev]
        n = len(subset)
        mean_d = subset[delay_col].mean()
        rate = (subset[delay_col] >= 15).mean() * 100
        result[f"sev_{sev}_n"] = n
        result[f"sev_{sev}_mean_delay"] = round(mean_d, 2)
        result[f"sev_{sev}_delay_rate_15"] = round(rate, 2)

    return result


def temporal_analysis(df: pd.DataFrame, strikes: pd.DataFrame, delay_col: str):
    """Print daily delay rates around strike events."""
    print("\n" + "=" * 80)
    print("TEMPORAL ANALYSIS: Delay rates around strike events")
    print("=" * 80)

    fd = pd.to_datetime(df["FlightDate"], errors="coerce").dt.normalize()
    airline = df["Reporting_Airline"].astype(str).str.upper().str.strip()

    for _, evt in strikes.iterrows():
        evt_start = pd.Timestamp(evt["action_start_date"])
        evt_end = pd.Timestamp(evt["action_end_date"])
        evt_airline = str(evt.get("airline_iata", "")).upper().strip()
        evt_id = evt.get("event_id", "unknown")
        severity = evt.get("severity", "?")

        if pd.isna(evt_start):
            continue

        # Check if this event overlaps with our data
        data_start = fd.min()
        data_end = fd.max()
        window_start = evt_start - pd.Timedelta(days=7)
        window_end = (evt_end if pd.notna(evt_end) else evt_start) + pd.Timedelta(days=7)

        if window_end < data_start or window_start > data_end:
            continue

        # Filter to relevant flights
        if evt_airline == "ALL":
            mask = (fd >= window_start) & (fd <= window_end)
        else:
            mask = (fd >= window_start) & (fd <= window_end) & (airline == evt_airline)

        subset = df[mask].copy()
        if len(subset) == 0:
            continue

        print(f"\n--- {evt_id} (severity={severity}, {evt_airline}) ---")
        print(f"    Action window: {evt_start.date()} to {(evt_end.date() if pd.notna(evt_end) else evt_start.date())}")

        # Daily delay stats
        subset["_fd"] = pd.to_datetime(subset["FlightDate"]).dt.normalize()
        daily = subset.groupby("_fd").agg(
            flights=(delay_col, "count"),
            mean_delay=(delay_col, "mean"),
            delay_rate_15=(delay_col, lambda x: (x >= 15).mean() * 100),
        ).round(2)

        for dt, row in daily.iterrows():
            marker = " <-- ACTION" if evt_start <= dt <= (evt_end if pd.notna(evt_end) else evt_start) else ""
            print(f"    {dt.date()}: flights={int(row['flights']):>5}, "
                  f"mean_delay={row['mean_delay']:>6.1f}min, "
                  f"delay_rate_15={row['delay_rate_15']:>5.1f}%{marker}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("STRIKE FEATURE CORRELATION ANALYSIS")
    print("=" * 80)

    # Load data
    print(f"\n[INFO] Loading enriched flights from {ENRICHED_PATH}")
    cols_want = [
        "FlightDate", "Origin", "Dest", "Reporting_Airline",
        "DepDelayMinutes", "ArrDelayMinutes",
    ]
    import pyarrow.parquet as pq
    schema = pq.read_schema(ENRICHED_PATH)
    all_cols = schema.names
    read_cols = [c for c in cols_want if c in all_cols]
    df = pd.read_parquet(ENRICHED_PATH, columns=read_cols)

    df["FlightDate"] = pd.to_datetime(df["FlightDate"], errors="coerce")
    df["DepDelayMinutes"] = pd.to_numeric(df.get("DepDelayMinutes"), errors="coerce")
    print(f"[OK] {len(df):,} flights, date range {df['FlightDate'].min().date()} to {df['FlightDate'].max().date()}")

    # Load strike cache
    print(f"[INFO] Loading strike cache from {STRIKE_CACHE}")
    strikes = pd.read_parquet(STRIKE_CACHE)
    print(f"[OK] {len(strikes)} labor action events")

    # Load history pool for carrier delay anomaly
    print(f"[INFO] Loading history pool from {HISTORY_PATH}")
    hist_cols = ["FlightDate", "Reporting_Airline", "CarrierDelay", "DepDelayMinutes"]
    hist_schema = pq.read_schema(HISTORY_PATH)
    hist_read = [c for c in hist_cols if c in hist_schema.names]
    hist = pd.read_parquet(HISTORY_PATH, columns=hist_read)
    print(f"[OK] {len(hist):,} history pool rows")

    # Enrich with strike features
    print("\n[INFO] Computing strike features...")
    df = enrich_with_strike_status(df, strikes)
    df = compute_carrier_delay_anomaly(df, hist)

    # Feature summary
    print("\n" + "=" * 80)
    print("FEATURE VALUE DISTRIBUTIONS")
    print("=" * 80)

    strike_features = [
        "strike_active_carrier", "strike_active_origin", "strike_severity",
        "strike_worker_type", "in_cooling_off", "cooling_off_days_remaining",
        "days_to_strike", "days_since_strike_end", "strike_spillover_origin",
        "carrier_delay_rate_anomaly_7d",
    ]

    for feat in strike_features:
        if feat not in df.columns:
            print(f"  {feat}: NOT PRESENT")
            continue
        col = df[feat]
        if col.dtype == object or str(col.dtype).startswith("string"):
            print(f"\n  {feat} (categorical):")
            print(f"    {col.value_counts().to_dict()}")
        else:
            numeric_col = pd.to_numeric(col, errors="coerce")
            n_nonzero = (numeric_col != 0).sum() if col.dtype in [np.int8, np.int16, np.int32, np.int64] else numeric_col.notna().sum()
            print(f"\n  {feat}: non-null/non-zero={n_nonzero:,} ({n_nonzero/len(df)*100:.3f}%), "
                  f"mean={numeric_col.mean():.4f}, min={numeric_col.min():.2f}, max={numeric_col.max():.2f}")

    # --- Correlation Analysis ---
    delay_col = "DepDelayMinutes"
    results = []

    print("\n" + "=" * 80)
    print(f"CORRELATION WITH {delay_col}")
    print("=" * 80)

    # Binary features
    binary_features = ["strike_active_carrier", "strike_active_origin", "in_cooling_off", "strike_spillover_origin"]
    print("\n--- Binary Features ---")

    for feat in binary_features:
        if feat not in df.columns:
            continue
        r = analyze_binary_feature(df, feat, delay_col)
        results.append(r)
        p_str = f"{r['p_value']:.2e}" if r['p_value'] is not None and not np.isnan(r['p_value']) else "N/A"
        print(f"  {r['feature']:<28} n_active={r['n_active']:>8,} ({r['pct_active']:.2f}%)  "
              f"delay: {r['mean_delay_active']} vs {r['mean_delay_inactive']} (diff={r['mean_delay_diff']})  "
              f"rate15: {r['delay_rate_15_active']}% vs {r['delay_rate_15_inactive']}% (lift={r['delay_rate_lift']})  "
              f"r_pb={r['point_biserial_r']}  p={p_str}  d={r['cohens_d']}")

    # Severity feature
    print("\n--- Severity Feature (Ordinal) ---")
    sev_r = analyze_severity_feature(df, delay_col)
    results.append(sev_r)
    print(f"  strike_severity: pearson_r={sev_r.get('pearson_r', 'N/A')}, "
          f"spearman_r={sev_r.get('spearman_r', 'N/A')}, "
          f"p={sev_r.get('pearson_p', 'N/A')}")
    for key in sorted(sev_r.keys()):
        if key.startswith("sev_"):
            print(f"    {key}: {sev_r[key]}")

    # Continuous features
    continuous_features = [
        "days_to_strike", "days_since_strike_end", "cooling_off_days_remaining",
        "carrier_delay_rate_anomaly_7d",
    ]
    print("\n--- Continuous Features ---")

    for feat in continuous_features:
        if feat not in df.columns:
            continue
        r = analyze_continuous_feature(df, feat, delay_col)
        results.append(r)
        pp = r.get('pearson_p')
        pp_str = f"{pp:.2e}" if pp is not None and not np.isnan(pp) else "N/A"
        sp = r.get('spearman_p')
        sp_str = f"{sp:.2e}" if sp is not None and not np.isnan(sp) else "N/A"
        print(f"  {r['feature']:<33} n_valid={r['n_valid']:>10,} ({r['pct_non_null']:.2f}%)  "
              f"pearson={r.get('pearson_r', 'N/A')} (p={pp_str})  "
              f"spearman={r.get('spearman_r', 'N/A')} (p={sp_str})  "
              f"Q1={r.get('mean_delay_q1', 'N/A')} Q4={r.get('mean_delay_q4', 'N/A')} "
              f"diff={r.get('delay_diff_q4_q1', 'N/A')}")

    # Worker type analysis
    print("\n--- Worker Type Breakdown ---")
    for wt in sorted(df["strike_worker_type"].unique()):
        subset = df[df["strike_worker_type"] == wt]
        mean_d = subset[delay_col].mean()
        rate15 = (subset[delay_col] >= 15).mean() * 100
        print(f"  {wt:<20}: n={len(subset):>10,}, mean_delay={mean_d:>6.2f}min, delay_rate_15={rate15:>5.1f}%")

    # Temporal analysis
    temporal_analysis(df, strikes, delay_col)

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "strike_feature_correlations.csv"
    results_df = pd.DataFrame(results)
    results_df.to_csv(out_path, index=False)
    print(f"\n[OK] Results saved to {out_path}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 80)
    print("""
Key findings to evaluate:
1. Do binary strike features show statistically significant delay differences?
   (Look for p-value < 0.05 and meaningful Cohen's d > 0.1)
2. Does carrier_delay_rate_anomaly_7d correlate with delay?
   (This fires continuously, not just on labeled strike events)
3. Are there enough non-zero observations for rare features to be useful?
   (Features with < 100 active observations may lack training power)
4. Does the temporal analysis show delay spikes aligned with strike windows?
   (Visual confirmation of feature validity)
""")


if __name__ == "__main__":
    main()
