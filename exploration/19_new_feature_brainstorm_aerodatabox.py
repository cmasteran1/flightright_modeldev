#!/usr/bin/env python3
"""
19_new_feature_brainstorm_aerodatabox.py

Tier 1-2 evaluation of NEW candidate features derived from BTS historical data
that can be obtained near departure time via the Aerodatabox API.

We evaluate each candidate using:
  - Point-biserial correlation with binary delay targets
  - Mutual information score
  - Delay-rate-by-decile visualizations
  - Redundancy check vs. existing v9 features

All candidates must satisfy the constraint:
  "Knowable at least a few days before scheduled departure via Aerodatabox API"
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    sns = None
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

# ── paths ─────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT.parent / "flightrightdata" / "data"
RAW_ROOT  = DATA_ROOT / "raw"
PROC_ROOT = DATA_ROOT / "processed"
FIGURES   = Path(__file__).resolve().parent / "figures"
FIGURES.mkdir(exist_ok=True)

# Use the exploration utils for plot style
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import set_plot_style, save_fig

TARGETS_DEP = ["y_dep_ge15", "y_dep_ge30", "y_dep_ge45", "y_dep_ge60"]
TARGETS_ARR = ["y_arr_ge15", "y_arr_ge30", "y_arr_ge45", "y_arr_ge60"]
DELAY_COL_DEP = "DepDelayMinutes"
DELAY_COL_ARR = "ArrDelayMinutes"

AIRLINE = "WN"  # Southwest — largest sample for robust stats


# ══════════════════════════════════════════════════════════════════════
# STEP 0: Load raw BTS data with ALL available columns
# ══════════════════════════════════════════════════════════════════════
def load_raw_bts(airline="WN", max_files=None):
    """Load raw BTS parquet files with all columns for feature mining.
    Data is stored as Year=YYYY/Month=MM/part-0.parquet."""
    raw_path = DATA_ROOT / "raw_bts"
    if not raw_path.exists():
        print(f"  [WARN] raw_bts directory not found at {raw_path}")
        return pd.DataFrame()

    # Collect all parquet files from partitioned structure
    files = sorted(raw_path.rglob("*.parquet"))
    print(f"  Found {len(files)} raw BTS parquet files")
    if max_files:
        files = files[:max_files]

    # Columns we need for candidate feature computation
    need_cols = [
        "FlightDate", "Reporting_Airline", "Flight_Number_Reporting_Airline",
        "Origin", "Dest", "CRSDepTime", "CRSArrTime", "CRSElapsedTime",
        "DepDelayMinutes", "ArrDelayMinutes", "DepDel15", "ArrDel15",
        "Tail_Number", "AirTime", "TaxiOut", "TaxiIn",
        "WheelsOff", "WheelsOn", "DepTimeBlk", "Distance",
        "Cancelled", "Diverted", "ActualElapsedTime",
        "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay",
        "LateAircraftDelay", "TotalAddGTime", "LongestAddGTime",
        "Marketing_Airline_Network",
    ]

    dfs = []
    for fp in files:
        try:
            df = pd.read_parquet(fp)
            df.columns = [c.strip() for c in df.columns]
            # Keep only columns that exist
            avail = [c for c in need_cols if c in df.columns]
            df = df[avail]
            # Filter to airline
            for col_name in ["Reporting_Airline", "Marketing_Airline_Network",
                             "IATA_Code_Marketing_Airline"]:
                if col_name in df.columns:
                    df = df[df[col_name] == airline]
                    break
            if len(df) > 0:
                dfs.append(df)
        except Exception as e:
            print(f"  [WARN] Could not read {fp}: {e}")
    if not dfs:
        return pd.DataFrame()
    result = pd.concat(dfs, ignore_index=True)
    print(f"  Combined: {len(result):,} rows for {airline}")
    return result


def load_processed_features(airline="WN", target="dep"):
    """Load existing processed feature file for correlation checks."""
    pattern = f"features_{target}_{airline}_*_unbalanced.parquet"
    candidates = sorted(PROC_ROOT.glob(pattern))
    if not candidates:
        # Try alternate patterns
        pattern2 = f"features_{target}_{airline}_*.parquet"
        candidates = sorted(PROC_ROOT.glob(pattern2))
    if not candidates:
        print(f"  [WARN] No processed features found for {target}/{airline}")
        return pd.DataFrame()
    fp = candidates[-1]  # Most recent
    print(f"  Loading processed: {fp.name}")
    return pd.read_parquet(fp)


# ══════════════════════════════════════════════════════════════════════
# STEP 1: Define candidate features
# ══════════════════════════════════════════════════════════════════════

CANDIDATES = """
══════════════════════════════════════════════════════════════════════════
NEW FEATURE CANDIDATES: BTS-derived, Aerodatabox-obtainable
══════════════════════════════════════════════════════════════════════════

Each candidate below is:
  1. Derivable from BTS historical data (for training)
  2. Obtainable near departure via Aerodatabox API (for inference)
  3. NOT already in the v9 feature set

─────────────────────────────────────────────────────────────────────────
CANDIDATE 1: taxi_out_mean_origin_last{W}d  (W=1,7,14)
─────────────────────────────────────────────────────────────────────────
  What: Rolling mean TaxiOut time (minutes) at the origin airport
        over the previous W days.
  Why:  TaxiOut captures ground congestion, runway queuing, and
        de-icing delays. High taxi-out = stressed airport operations.
        This is a DIFFERENT signal from departure delay — a flight
        can push back on time but sit on the taxiway for 45 min.
  BTS:  TaxiOut column (already downloaded, in prepare_dataset keep_cols)
  Aerodatabox: Derivable from FIDS endpoint — compute difference between
        departure.scheduledTime and departure.runwayTime (wheels-off)
        from recent flights at the airport. The flight status endpoint
        returns scheduledTime + actual runway/gate times.
  Redundancy risk: Partial overlap with origin_depdelay_mean — but TaxiOut
        specifically captures the ground phase, not the gate delay.

─────────────────────────────────────────────────────────────────────────
CANDIDATE 2: taxi_in_mean_dest_last{W}d  (W=1,7,14)
─────────────────────────────────────────────────────────────────────────
  What: Rolling mean TaxiIn time at the destination airport.
  Why:  High taxi-in at destination = runway/taxiway congestion on
        arrival. Signals that even if the flight is on time in the air,
        arrival delay may accumulate on the ground.
  BTS:  TaxiIn column (downloaded, in prepare_dataset keep_cols)
  Aerodatabox: Derivable from wheels-on to gate-arrival time difference
        in flight status history at destination.
  Model: [ARR] primarily, but could also signal dest airport stress for [DEP]

─────────────────────────────────────────────────────────────────────────
CANDIDATE 3: cancellation_rate_origin_last{W}d  (W=1,7,14)
─────────────────────────────────────────────────────────────────────────
  What: Fraction of flights cancelled at origin in last W days.
  Why:  Cancellation spikes signal severe operational disruption
        (weather events, ATC issues, crew shortages). Surviving flights
        on high-cancellation days face rebooking chaos and are more
        likely to be delayed.
  BTS:  Cancelled column (already downloaded)
  Aerodatabox: Flight status endpoint returns status="Cancelled" —
        count cancelled / total at airport from FIDS history.
  Redundancy risk: Low — cancellations are not captured by any existing
        delay-based feature. A day with many cancellations but low
        mean delay for surviving flights would be missed entirely.

─────────────────────────────────────────────────────────────────────────
CANDIDATE 4: cancellation_rate_carrier_last{W}d  (W=1,7)
─────────────────────────────────────────────────────────────────────────
  What: Fraction of flights cancelled by this carrier in last W days.
  Why:  Carrier-wide cancellation spikes (crew shortages, fleet issues,
        IT outages) strongly predict delays on surviving flights.
  BTS:  Cancelled + Reporting_Airline
  Aerodatabox: Filter FIDS by airline, count status="Cancelled".

─────────────────────────────────────────────────────────────────────────
CANDIDATE 5: diversion_rate_origin_last{W}d  (W=7,14)
─────────────────────────────────────────────────────────────────────────
  What: Fraction of flights diverted from origin in last W days.
  Why:  Diversions signal extreme conditions (weather, security, ATC
        ground stops). Even after the event clears, ripple effects
        persist for 24-48h.
  BTS:  Diverted column (downloaded but never used)
  Aerodatabox: Status field includes diversion info in flight status.

─────────────────────────────────────────────────────────────────────────
CANDIDATE 6: elapsed_time_ratio_last14d
─────────────────────────────────────────────────────────────────────────
  What: Ratio of ActualElapsedTime / CRSElapsedTime for this flight
        number + OD over the last 14 days.
  Why:  Flights that consistently run long (ratio > 1.05) may have
        unrealistic schedules or face persistent headwinds/ATC routing.
        Flights running short may have padded schedules (low delay risk).
  BTS:  ActualElapsedTime (downloaded but unused) / CRSElapsedTime
  Aerodatabox: Compute from actual arrival - actual departure vs.
        scheduled times from flight history endpoint.

─────────────────────────────────────────────────────────────────────────
CANDIDATE 7: carrier_weatherdelay_rate_last{W}d  (W=1,7)
─────────────────────────────────────────────────────────────────────────
  What: Fraction of carrier's flights with WeatherDelay > 0 in last W days.
  Why:  Weather delay is a BTS-coded cause separate from NAS/carrier.
        A carrier with elevated weather delay rates may be operating
        in weather-impacted regions across its network.
  BTS:  WeatherDelay column (downloaded, used only for late-aircraft rate)
  Aerodatabox: Not directly available as cause-coded, but overall delay
        rates combined with weather data can approximate this. The
        Airport Delays endpoint returns delay index which correlates.
  NOTE: This is obtainable from BTS for training. For inference, we'd
        proxy via the Aerodatabox airport delay index + weather overlay.

─────────────────────────────────────────────────────────────────────────
CANDIDATE 8: origin_nasdelay_rate_last{W}d  (W=1,7)
─────────────────────────────────────────────────────────────────────────
  What: Fraction of flights at origin with NASDelay > 0 in last W days.
  Why:  NAS delays (air traffic control, ground stops, airspace issues)
        are a distinct mechanism from carrier or weather delays.
        High NAS rates signal systemic ATC congestion.
  BTS:  NASDelay column (downloaded)
  Aerodatabox: Similar proxy approach — airport delay statistics endpoint
        captures aggregate delay which includes NAS-driven delays.
  NOTE: Same inference limitation as candidate 7 — cause codes not
        directly available from Aerodatabox, but aggregate delay index
        serves as a reasonable proxy.

─────────────────────────────────────────────────────────────────────────
CANDIDATE 9: airport_delay_index_origin  (Aerodatabox native)
─────────────────────────────────────────────────────────────────────────
  What: Aerodatabox airport delay index (0.0 - 5.0) for the origin
        airport, queried at the most recent available time before
        prediction.
  Why:  This is a COMPOSITE signal that Aerodatabox computes from
        recent flight operations — it captures delay severity, cancellation
        rates, and overall airport stress in a single number. This is
        a real-time feature not derivable from BTS alone.
  BTS:  Can approximate for training by computing a synthetic delay index
        from BTS columns: combine delay rate, cancellation rate, and mean
        delay magnitude into a 0-5 composite score.
  Aerodatabox: Directly available from Airport Delays endpoint.
        GET /airports/delays/{airportIcao}
        Returns: delay index (0-5), median delay, percentile delays,
        cancellation count.
  Redundancy risk: Low — this is a multi-signal composite not captured
        by any single existing feature.

─────────────────────────────────────────────────────────────────────────
CANDIDATE 10: airport_delay_index_dest
─────────────────────────────────────────────────────────────────────────
  What: Same as candidate 9 but for the destination airport.
  Why:  Destination stress directly impacts arrival delays and can
        cause ground stops affecting departures.
  Model: [ARR] primarily, could also signal for [DEP] via ground holds.

─────────────────────────────────────────────────────────────────────────
CANDIDATE 11: schedule_buffer_minutes
─────────────────────────────────────────────────────────────────────────
  What: CRSElapsedTime minus median actual flight time for this route.
        Positive = padded schedule. Negative = tight schedule.
  Why:  Airlines pad schedules differently. Routes with thin buffers
        convert any small delay into an arrival delay. Routes with
        thick padding absorb delays gracefully.
  BTS:  CRSElapsedTime - median(ActualElapsedTime) per (Origin, Dest)
  Aerodatabox: Compute from flight history — scheduled vs actual durations
        from flight status endpoint.
  Redundancy: Partially correlated with CRSElapsedTime/Distance, but the
        RATIO matters more than the absolute value.

─────────────────────────────────────────────────────────────────────────
CANDIDATE 12: gate_hold_mean_origin_last{W}d  (W=1,7)
─────────────────────────────────────────────────────────────────────────
  What: Mean gate hold time at origin (TotalAddGTime from BTS).
        Gate holds are imposed by ATC when there's en-route or
        destination congestion. The flight sits at the gate with
        engines off instead of burning fuel on the taxiway.
  Why:  Gate holds indicate systemic congestion beyond the origin
        airport. They signal ATC flow control programs that affect
        multiple flights.
  BTS:  TotalAddGTime, LongestAddGTime (downloaded but never used)
  Aerodatabox: Can be approximated from the difference between revised
        departure time and scheduled departure time from flight status.
  NOTE: These BTS fields are sparse (only present when holds occur),
        but when present they're highly informative.
"""


# ══════════════════════════════════════════════════════════════════════
# STEP 2: Compute candidate features from raw BTS data
# ══════════════════════════════════════════════════════════════════════

def compute_candidates(df):
    """Compute all candidate feature columns from raw BTS data."""
    print("\n[STEP 2] Computing candidate features from BTS data...")

    # Ensure FlightDate is datetime
    if "FlightDate" in df.columns:
        df["FlightDate"] = pd.to_datetime(df["FlightDate"])
    else:
        print("  [ERROR] No FlightDate column found!")
        return df

    # Ensure numeric types
    for col in ["TaxiOut", "TaxiIn", "DepDelayMinutes", "ArrDelayMinutes",
                "Cancelled", "Diverted", "ActualElapsedTime", "CRSElapsedTime",
                "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay",
                "LateAircraftDelay", "TotalAddGTime", "LongestAddGTime",
                "AirTime", "Distance"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort for rolling computations
    df = df.sort_values(["Origin", "Dest", "FlightDate"]).reset_index(drop=True)

    # ── Binary targets (if not already present) ──
    if "DepDelayMinutes" in df.columns:
        for thr in [15, 30, 45, 60]:
            col = f"y_dep_ge{thr}"
            if col not in df.columns:
                df[col] = (df["DepDelayMinutes"] >= thr).astype(int)
    if "ArrDelayMinutes" in df.columns:
        for thr in [15, 30, 45, 60]:
            col = f"y_arr_ge{thr}"
            if col not in df.columns:
                df[col] = (df["ArrDelayMinutes"] >= thr).astype(int)

    # ── Helper: rolling mean by group over date ──
    def rolling_mean_by_date(group_cols, value_col, windows=[1, 7, 14], prefix=""):
        """Compute date-based rolling means with 1-day shift to avoid leakage."""
        # Aggregate to daily level first
        daily = (df.groupby(group_cols + ["FlightDate"])[value_col]
                 .mean().reset_index()
                 .rename(columns={value_col: "_val"}))
        daily = daily.sort_values(group_cols + ["FlightDate"])

        for w in windows:
            col_name = f"{prefix}_last{w}d"
            # Shifted rolling mean (shift 1 day)
            daily[col_name] = (
                daily.groupby(group_cols)["_val"]
                .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            )

        # Merge back to flight level
        merge_cols = group_cols + ["FlightDate"]
        result_cols = [f"{prefix}_last{w}d" for w in windows]
        return daily[merge_cols + result_cols]

    def rolling_rate_by_date(group_cols, flag_col, windows=[1, 7, 14], prefix=""):
        """Compute rolling rate of a binary flag."""
        daily = (df.groupby(group_cols + ["FlightDate"])[flag_col]
                 .mean().reset_index()
                 .rename(columns={flag_col: "_rate"}))
        daily = daily.sort_values(group_cols + ["FlightDate"])

        for w in windows:
            col_name = f"{prefix}_last{w}d"
            daily[col_name] = (
                daily.groupby(group_cols)["_rate"]
                .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            )

        merge_cols = group_cols + ["FlightDate"]
        result_cols = [f"{prefix}_last{w}d" for w in windows]
        return daily[merge_cols + result_cols]

    n_before = len(df.columns)

    # ── CANDIDATE 1: taxi_out_mean_origin ──
    print("  Computing: taxi_out_mean_origin...")
    if "TaxiOut" in df.columns:
        taxi_out_stats = rolling_mean_by_date(
            ["Origin"], "TaxiOut", windows=[1, 7, 14],
            prefix="taxi_out_mean_origin"
        )
        df = df.merge(taxi_out_stats, on=["Origin", "FlightDate"], how="left")

    # ── CANDIDATE 2: taxi_in_mean_dest ──
    print("  Computing: taxi_in_mean_dest...")
    if "TaxiIn" in df.columns:
        taxi_in_stats = rolling_mean_by_date(
            ["Dest"], "TaxiIn", windows=[1, 7, 14],
            prefix="taxi_in_mean_dest"
        )
        df = df.merge(taxi_in_stats, on=["Dest", "FlightDate"], how="left")

    # ── CANDIDATE 3: cancellation_rate_origin ──
    print("  Computing: cancellation_rate_origin...")
    if "Cancelled" in df.columns:
        cancel_origin = rolling_rate_by_date(
            ["Origin"], "Cancelled", windows=[1, 7, 14],
            prefix="cancel_rate_origin"
        )
        df = df.merge(cancel_origin, on=["Origin", "FlightDate"], how="left")

    # ── CANDIDATE 4: cancellation_rate_carrier ──
    print("  Computing: cancellation_rate_carrier...")
    if "Cancelled" in df.columns and "Reporting_Airline" in df.columns:
        cancel_carrier = rolling_rate_by_date(
            ["Reporting_Airline"], "Cancelled", windows=[1, 7],
            prefix="cancel_rate_carrier"
        )
        df = df.merge(cancel_carrier, on=["Reporting_Airline", "FlightDate"], how="left")

    # ── CANDIDATE 5: diversion_rate_origin ──
    print("  Computing: diversion_rate_origin...")
    if "Diverted" in df.columns:
        divert_origin = rolling_rate_by_date(
            ["Origin"], "Diverted", windows=[7, 14],
            prefix="divert_rate_origin"
        )
        df = df.merge(divert_origin, on=["Origin", "FlightDate"], how="left")

    # ── CANDIDATE 6: elapsed_time_ratio ──
    print("  Computing: elapsed_time_ratio...")
    if "ActualElapsedTime" in df.columns and "CRSElapsedTime" in df.columns:
        df["_elapsed_ratio"] = df["ActualElapsedTime"] / df["CRSElapsedTime"].replace(0, np.nan)
        et_ratio = rolling_mean_by_date(
            ["Origin", "Dest"], "_elapsed_ratio", windows=[14],
            prefix="elapsed_time_ratio"
        )
        df = df.merge(et_ratio, on=["Origin", "Dest", "FlightDate"], how="left")
        df.drop(columns=["_elapsed_ratio"], inplace=True)

    # ── CANDIDATE 7: carrier_weatherdelay_rate ──
    print("  Computing: carrier_weatherdelay_rate...")
    if "WeatherDelay" in df.columns and "Reporting_Airline" in df.columns:
        df["_has_weather_delay"] = (df["WeatherDelay"].fillna(0) > 0).astype(int)
        wd_rate = rolling_rate_by_date(
            ["Reporting_Airline"], "_has_weather_delay", windows=[1, 7],
            prefix="carrier_wxdelay_rate"
        )
        df = df.merge(wd_rate, on=["Reporting_Airline", "FlightDate"], how="left")
        df.drop(columns=["_has_weather_delay"], inplace=True)

    # ── CANDIDATE 8: origin_nasdelay_rate ──
    print("  Computing: origin_nasdelay_rate...")
    if "NASDelay" in df.columns:
        df["_has_nas_delay"] = (df["NASDelay"].fillna(0) > 0).astype(int)
        nas_rate = rolling_rate_by_date(
            ["Origin"], "_has_nas_delay", windows=[1, 7],
            prefix="origin_nasdelay_rate"
        )
        df = df.merge(nas_rate, on=["Origin", "FlightDate"], how="left")
        df.drop(columns=["_has_nas_delay"], inplace=True)

    # ── CANDIDATE 11: schedule_buffer_minutes ──
    print("  Computing: schedule_buffer_minutes...")
    if "ActualElapsedTime" in df.columns and "CRSElapsedTime" in df.columns:
        # Compute median actual elapsed per OD pair from historical data
        od_median = (df.groupby(["Origin", "Dest"])["ActualElapsedTime"]
                     .median().reset_index()
                     .rename(columns={"ActualElapsedTime": "_od_median_elapsed"}))
        df = df.merge(od_median, on=["Origin", "Dest"], how="left")
        df["schedule_buffer_min"] = df["CRSElapsedTime"] - df["_od_median_elapsed"]
        df.drop(columns=["_od_median_elapsed"], inplace=True)

    # ── CANDIDATE 12: gate_hold_mean_origin ──
    print("  Computing: gate_hold_mean_origin...")
    if "TotalAddGTime" in df.columns:
        # Gate hold is sparse — fill NaN with 0 (no hold)
        df["_gate_hold"] = df["TotalAddGTime"].fillna(0)
        gh_stats = rolling_mean_by_date(
            ["Origin"], "_gate_hold", windows=[1, 7],
            prefix="gate_hold_mean_origin"
        )
        df = df.merge(gh_stats, on=["Origin", "FlightDate"], how="left")
        df.drop(columns=["_gate_hold"], inplace=True)

    n_after = len(df.columns)
    print(f"  Added {n_after - n_before} candidate feature columns.\n")
    return df


# ══════════════════════════════════════════════════════════════════════
# STEP 3: Tier 1 evaluation — correlation + stats
# ══════════════════════════════════════════════════════════════════════

def evaluate_tier1(df, candidate_cols, targets, delay_col):
    """Compute point-biserial correlation and basic stats for each candidate."""
    print(f"\n[STEP 3] Tier 1 evaluation: point-biserial correlation with {len(targets)} targets")
    results = []

    for feat in candidate_cols:
        if feat not in df.columns:
            continue
        valid = df[[feat] + targets + [delay_col]].dropna()
        if len(valid) < 100:
            print(f"  SKIP {feat}: only {len(valid)} valid rows")
            continue

        row = {
            "feature": feat,
            "n_valid": len(valid),
            "missing_pct": 100 * (1 - len(valid) / len(df)),
            "mean": valid[feat].mean(),
            "std": valid[feat].std(),
            "skew": valid[feat].skew(),
        }

        # Point-biserial correlation with each target
        for tgt in targets:
            r, p = stats.pointbiserialr(valid[tgt], valid[feat])
            row[f"corr_{tgt}"] = r
            row[f"pval_{tgt}"] = p

        # Spearman correlation with continuous delay
        rho, p_rho = stats.spearmanr(valid[feat], valid[delay_col])
        row["spearman_delay"] = rho
        row["spearman_pval"] = p_rho

        results.append(row)

    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════
# STEP 4: Tier 2 evaluation — mutual information
# ══════════════════════════════════════════════════════════════════════

def evaluate_tier2_mi(df, candidate_cols, target_col, sample_n=50000):
    """Compute mutual information scores."""
    print(f"\n[STEP 4] Tier 2: Mutual information vs {target_col}")
    valid_cols = [c for c in candidate_cols if c in df.columns]
    sub = df[valid_cols + [target_col]].dropna()
    if len(sub) > sample_n:
        sub = sub.sample(n=sample_n, random_state=42)

    X = sub[valid_cols].values
    y = sub[target_col].values

    mi_scores = mutual_info_classif(X, y, random_state=42, n_neighbors=5)
    return pd.DataFrame({
        "feature": valid_cols,
        f"MI_{target_col}": mi_scores
    }).sort_values(f"MI_{target_col}", ascending=False)


# ══════════════════════════════════════════════════════════════════════
# STEP 5: Visualization — delay rate by feature decile
# ══════════════════════════════════════════════════════════════════════

def plot_delay_rate_by_decile(df, feat_col, target_col, title=None):
    """Bar chart of delay rate by feature decile."""
    valid = df[[feat_col, target_col]].dropna()
    if len(valid) < 100:
        return None

    try:
        valid["decile"] = pd.qcut(valid[feat_col], q=10, duplicates="drop")
    except ValueError:
        valid["decile"] = pd.cut(valid[feat_col], bins=10, duplicates="drop")

    rates = valid.groupby("decile", observed=True)[target_col].agg(["mean", "count"])
    rates = rates.reset_index()

    fig, ax1 = plt.subplots(figsize=(12, 5))
    x = range(len(rates))

    ax1.bar(x, rates["mean"], color="#1565C0", alpha=0.8, label="Delay Rate")
    ax1.set_ylabel(f"P({target_col}=1)", color="#1565C0")
    ax1.set_xlabel(f"{feat_col} (deciles, low→high)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(d) for d in rates["decile"]], rotation=45, ha="right", fontsize=7)

    ax2 = ax1.twinx()
    ax2.plot(x, rates["count"], "o--", color="#FF5722", alpha=0.7, label="Sample Count")
    ax2.set_ylabel("N flights", color="#FF5722")

    ax1.set_title(title or f"{feat_col} → {target_col}: Delay Rate by Decile")
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════
# STEP 6: Redundancy check — correlation with existing features
# ══════════════════════════════════════════════════════════════════════

def check_redundancy(df, candidate_cols, existing_num_cols, top_n=5):
    """Check each candidate's max correlation with existing features."""
    print(f"\n[STEP 6] Redundancy check: candidates vs {len(existing_num_cols)} existing features")
    results = []

    for cand in candidate_cols:
        if cand not in df.columns:
            continue
        max_corr = 0.0
        max_feat = ""
        for exist in existing_num_cols:
            if exist not in df.columns:
                continue
            valid = df[[cand, exist]].dropna()
            if len(valid) < 100:
                continue
            r, _ = stats.spearmanr(valid[cand], valid[exist])
            if abs(r) > abs(max_corr):
                max_corr = r
                max_feat = exist
        results.append({
            "candidate": cand,
            "most_correlated_existing": max_feat,
            "spearman_r": max_corr,
            "redundant": abs(max_corr) > 0.85
        })

    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    set_plot_style()

    print("=" * 70)
    print("NEW FEATURE CANDIDATE EVALUATION")
    print(f"Airline: {AIRLINE}")
    print("=" * 70)

    # ── Load data ──
    print("\n[STEP 0] Loading raw BTS data...")
    df = load_raw_bts(AIRLINE)
    if df.empty:
        print("[ERROR] No raw data found. Trying processed data instead...")
        df = load_processed_features(AIRLINE, "dep")

    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    if "FlightDate" in df.columns:
        print(f"  Date range: {df['FlightDate'].min()} to {df['FlightDate'].max()}")
    else:
        print("  No FlightDate column")

    # ── Compute candidates ──
    df = compute_candidates(df)

    # ── Identify candidate columns ──
    candidate_cols = [c for c in df.columns if any(c.startswith(p) for p in [
        "taxi_out_mean_origin", "taxi_in_mean_dest",
        "cancel_rate_origin", "cancel_rate_carrier",
        "divert_rate_origin", "elapsed_time_ratio",
        "carrier_wxdelay_rate", "origin_nasdelay_rate",
        "schedule_buffer_min", "gate_hold_mean_origin",
    ])]
    print(f"\n  Candidate columns found: {len(candidate_cols)}")
    for c in sorted(candidate_cols):
        print(f"    {c}")

    if not candidate_cols:
        print("\n[ERROR] No candidate columns were computed. Check data availability.")
        return

    # ── Tier 1: Correlation ──
    tier1_dep = evaluate_tier1(df, candidate_cols, TARGETS_DEP, DELAY_COL_DEP)
    if not tier1_dep.empty:
        print("\n" + "=" * 70)
        print("TIER 1 RESULTS: DEPARTURE DELAY CORRELATION")
        print("=" * 70)
        display_cols = ["feature", "n_valid", "missing_pct", "spearman_delay"]
        display_cols += [c for c in tier1_dep.columns if c.startswith("corr_y_dep")]
        print(tier1_dep[display_cols].to_string(index=False, float_format="%.4f"))

    tier1_arr = evaluate_tier1(df, candidate_cols, TARGETS_ARR, DELAY_COL_ARR)
    if not tier1_arr.empty:
        print("\n" + "=" * 70)
        print("TIER 1 RESULTS: ARRIVAL DELAY CORRELATION")
        print("=" * 70)
        display_cols = ["feature", "n_valid", "missing_pct", "spearman_delay"]
        display_cols += [c for c in tier1_arr.columns if c.startswith("corr_y_arr")]
        print(tier1_arr[display_cols].to_string(index=False, float_format="%.4f"))

    # ── Tier 2: Mutual Information ──
    if TARGETS_DEP[0] in df.columns:
        mi_dep = evaluate_tier2_mi(df, candidate_cols, TARGETS_DEP[0])
        print("\n" + "=" * 70)
        print(f"TIER 2 RESULTS: MUTUAL INFORMATION vs {TARGETS_DEP[0]}")
        print("=" * 70)
        print(mi_dep.to_string(index=False, float_format="%.5f"))

    if TARGETS_DEP[2] in df.columns:
        mi_dep30 = evaluate_tier2_mi(df, candidate_cols, TARGETS_DEP[2])
        print(f"\nMI vs {TARGETS_DEP[2]}:")
        print(mi_dep30.to_string(index=False, float_format="%.5f"))

    # ── Visualizations ──
    print("\n[STEP 5] Generating decile plots...")
    for feat in candidate_cols:
        if feat not in df.columns:
            continue
        for tgt in [TARGETS_DEP[0], TARGETS_DEP[2]]:
            if tgt not in df.columns:
                continue
            fig = plot_delay_rate_by_decile(df, feat, tgt)
            if fig:
                safe_name = feat.replace("/", "_")
                save_fig(fig, f"candidate_{safe_name}_vs_{tgt}")

    # ── Redundancy check ──
    print("\n[STEP 6] Checking redundancy with existing features...")
    proc_df = load_processed_features(AIRLINE, "dep")
    if not proc_df.empty:
        # Get existing numeric feature columns
        existing_num = [c for c in proc_df.columns
                       if proc_df[c].dtype in ["float64", "float32", "int64", "int32"]
                       and c not in TARGETS_DEP + TARGETS_ARR + ["DepDelayMinutes", "ArrDelayMinutes"]]

        # Merge candidates into processed df for correlation
        merge_keys = ["FlightDate", "Origin", "Dest"]
        available_keys = [k for k in merge_keys if k in df.columns and k in proc_df.columns]
        if available_keys:
            candidates_sub = df[available_keys + candidate_cols].drop_duplicates(subset=available_keys)
            merged = proc_df.merge(candidates_sub, on=available_keys, how="inner")

            redundancy = check_redundancy(merged, candidate_cols, existing_num)
            print("\n" + "=" * 70)
            print("REDUNDANCY CHECK: Candidate vs Most-Correlated Existing Feature")
            print("=" * 70)
            print(redundancy.to_string(index=False, float_format="%.4f"))

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 70)
    print("""
Each candidate has been evaluated at Tier 1 (correlation) and Tier 2 (MI).
Review the numbers above and the decile plots in exploration/figures/ to
decide which candidates merit Tier 3 evaluation (SHAP / permutation importance).

RECOMMENDATION CRITERIA:
  ADD to Tier 3  -> |corr| > 0.05 with any target AND MI > 0.005 AND not redundant
  DROP           -> |corr| < 0.02 across all targets AND MI < 0.002
  INVESTIGATE    -> Mixed signals or high redundancy with existing feature

See exploration/figures/candidate_*.png for visual evidence.
""")


if __name__ == "__main__":
    main()
