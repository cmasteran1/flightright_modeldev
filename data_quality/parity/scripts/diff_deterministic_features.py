"""
Side-by-side diff of *deterministic* features (schedule, calendar, weather)
between training (BTS-derived parquet) and production (predict_departure-
served via cached AeroDataBox + Open-Meteo).

These three groups should have **zero** train/serve divergence: schedule
features are pure functions of (flight_date, scheduled_time, airport_tz);
calendar features are pure functions of date; weather features come from
the same Open-Meteo archive on both sides. Any non-zero delta is a bug.

Outputs two CSVs under data_quality/parity/reports/:
  - deterministic_diff_summary.csv (per-feature aggregate)
  - deterministic_diff_rows.csv    (per-(flight, feature) row, for drill-in)
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
PROD = REPO.parent / "flightright"
DATA = REPO.parent / "flightrightdata"
PARITY = REPO / "data_quality" / "parity"
sys.path.insert(0, str(PROD / "src"))
sys.path.insert(0, str(PARITY / "scripts"))


SCHEDULE_FEATURES = [
    "sched_dep_hour", "dep_dow", "CRSDepTime", "CRSElapsedTime",
    "Distance", "is_peak_hour", "flight_month",
    # Categorical (string) features — compared as exact match, no delta:
    "DepTimeBlk", "Origin", "Dest", "od_pair", "Reporting_Airline",
    "aircraft_type",
]

CALENDAR_FEATURES = [
    "is_holiday", "is_spring_break", "days_to_strike", "strike_severity",
    # NOTE: carrier_delay_rate_anomaly_7d / airline_cancel_rate_anomaly_7d
    # are *not* deterministic — they're anomaly-vs-rolling-mean. They live
    # in the rolling-stats SQLite, so they belong with rolling, not here.
]

WEATHER_FEATURES = [
    # Daily
    "origin_temp_max_K", "origin_temp_min_K",
    "origin_daily_precip_sum_mm",
    "origin_daily_windspeed_max_kmh", "origin_daily_windgusts_max_kmh",
    "origin_daily_weathercode",
    # Hourly at dep
    "origin_dep_temp_K", "origin_dep_precip_mm",
    "origin_dep_windspeed_kmh", "origin_dep_windgusts_kmh",
    "origin_dep_visibility_m", "origin_dep_cape_jkg", "origin_dep_log1p_cape",
    "origin_dep_cloudcover_pct",
    "origin_dep_hour_weathercode",
    # Derived
    "wind_x_precip",
]

CATEGORICAL = {"DepTimeBlk", "Origin", "Dest", "od_pair", "Reporting_Airline",
               "aircraft_type"}

GROUP_OF = {}
for f in SCHEDULE_FEATURES:
    GROUP_OF[f] = "schedule"
for f in CALENDAR_FEATURES:
    GROUP_OF[f] = "calendar"
for f in WEATHER_FEATURES:
    GROUP_OF[f] = "weather"


def _load_env():
    env_file = PROD / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def make_cfg(rolling_sqlite_path: Path):
    from flightright.service.predictor import (
        DataPaths, ModelSpecDefaults, RemoteModelConfig, RollingStatsConfig, ServiceConfig,
    )
    e2 = os.environ.get("E2_ENDPOINT", "")
    if e2 and not e2.startswith("http"):
        e2 = f"https://{e2}"
    return ServiceConfig(
        aeroapi_key=os.environ.get("AEROAPI_KEY", ""),
        aerodatabox_key=os.environ.get("AERODATABOX_KEY", ""),
        provider_name="aerodatabox",
        data_paths=DataPaths(
            airports_csv=DATA / "data" / "meta" / "airports.csv",
            airport_rankings_dir=DATA / "data" / "meta" / "airport_rankings",
        ),
        remote_models=RemoteModelConfig(
            use_remote_models=True,
            s3_bucket="flightright-models",
            s3_prefix="models/",
            s3_endpoint=e2 or None,
            remote_cache_dir=Path.home() / ".flightright" / "remote_models",
        ),
        model_defaults=ModelSpecDefaults(airports_n=100, model_version="v11"),
        rolling_sqlite_path=rolling_sqlite_path,
        rolling_stats=RollingStatsConfig(enabled=False),
    )


def _to_float(x):
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _to_str(x):
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    return str(x)


def main():
    _load_env()
    os.environ["FLIGHTRIGHT_LATE_AIRCRAFT_CAL"] = "1.0"

    from seed_adb_cache import install_disk_cache
    install_disk_cache(PARITY / "cache" / "adb_responses",
                       allow_network=False, log_misses=False)

    from flightright.service.predictor import predict_departure  # noqa: E402

    cfg = make_cfg(Path.home() / ".flightright" / "rolling_stats_bts_dec2025_v7g_v2.db")
    flights = json.loads((PARITY / "test_set" / "flights_dec_400.json").read_text())["flights"]

    bts = pd.read_parquet(PARITY / "test_set" / "bts_features_dec_400.parquet")
    bts["FlightDate"] = pd.to_datetime(bts["FlightDate"])
    bts["key"] = (bts["Reporting_Airline"].astype(str) + "/"
                  + pd.to_numeric(bts["Flight_Number_Reporting_Airline"], errors="coerce").astype("Int64").astype(str)
                  + "/" + bts["Origin"].astype(str) + "/" + bts["Dest"].astype(str)
                  + "/" + bts["FlightDate"].dt.strftime("%Y-%m-%d"))
    bts = bts.set_index("key")

    all_features = SCHEDULE_FEATURES + CALENDAR_FEATURES + WEATHER_FEATURES

    rows = []
    n_scored = 0
    print(f"Scoring {len(flights)} flights × {len(all_features)} features...", flush=True)
    for i, f in enumerate(flights, 1):
        fkey = f"{f['carrier']}/{f['flight_number']}/{f['origin']}/{f['dest']}/{f['flight_date']}"
        if fkey not in bts.index:
            continue
        bts_row = bts.loc[fkey]
        try:
            resp = predict_departure(
                airline=f["carrier"], flightnum=str(f["flight_number"]),
                dep_date=date.fromisoformat(f["flight_date"]),
                origin=f["origin"], dest=f["dest"],
                cfg=cfg, include_features=True, public_mode=False,
                model_version_override="v11",
            )
        except Exception:
            continue
        served = resp.get("features", {}) or {}
        n_scored += 1

        for feat in all_features:
            t_raw = bts_row.get(feat) if feat in bts_row.index else None
            s_raw = served.get(feat)
            row = {
                "flight_key": fkey,
                "carrier": f["carrier"],
                "feature": feat,
                "group": GROUP_OF[feat],
            }
            if feat in CATEGORICAL:
                ts, ss = _to_str(t_raw), _to_str(s_raw)
                row["trained"] = ts
                row["served"] = ss
                row["match"] = (ts == ss) if (ts is not None and ss is not None) else None
            else:
                tn, sn = _to_float(t_raw), _to_float(s_raw)
                row["trained"] = tn
                row["served"] = sn
                if tn is not None and sn is not None:
                    row["delta"] = sn - tn
            rows.append(row)
        if i % 50 == 0 or i == len(flights):
            print(f"  [{i}/{len(flights)}] processed", flush=True)

    df = pd.DataFrame(rows)
    print(f"\nDiagnosed {len(df)} (flight, feature) cells across {n_scored} flights\n")

    # Per-feature summary
    agg_rows = []
    for feat, g in df.groupby("feature"):
        is_cat = feat in CATEGORICAL
        n = len(g)
        n_both = int(g.dropna(subset=["trained", "served"]).shape[0])
        only_trn = int((g["trained"].notna() & g["served"].isna()).sum())
        only_srv = int((g["trained"].isna() & g["served"].notna()).sum())
        if is_cat:
            matches = g.dropna(subset=["match"])
            n_match = int(matches["match"].sum())
            n_mismatch = int((~matches["match"]).sum())
            agg_rows.append({
                "group": g["group"].iloc[0],
                "feature": feat,
                "n_total": n, "n_both_present": n_both,
                "n_only_trained": only_trn, "n_only_served": only_srv,
                "n_match": n_match, "n_mismatch": n_mismatch,
                "match_rate": (round(n_match / max(n_both, 1), 4)),
                "delta_mean": None, "abs_delta_mean": None, "abs_delta_p95": None,
                "abs_delta_max": None,
            })
        else:
            deltas = g.dropna(subset=["delta"])["delta"]
            n_exact = int((deltas.abs() < 1e-9).sum()) if not deltas.empty else 0
            n_diff = int((deltas.abs() >= 0.001).sum()) if not deltas.empty else 0
            agg_rows.append({
                "group": g["group"].iloc[0],
                "feature": feat,
                "n_total": n, "n_both_present": n_both,
                "n_only_trained": only_trn, "n_only_served": only_srv,
                "n_match": n_exact, "n_mismatch": n_diff,
                "match_rate": (round(n_exact / max(n_both, 1), 4)),
                "delta_mean": (round(float(deltas.mean()), 6) if not deltas.empty else None),
                "abs_delta_mean": (round(float(deltas.abs().mean()), 6) if not deltas.empty else None),
                "abs_delta_p95": (round(float(deltas.abs().quantile(0.95)), 6) if not deltas.empty else None),
                "abs_delta_max": (round(float(deltas.abs().max()), 6) if not deltas.empty else None),
            })

    agg = pd.DataFrame(agg_rows)
    agg["group_order"] = agg["group"].map({"schedule": 0, "calendar": 1, "weather": 2})
    agg = agg.sort_values(["group_order", "match_rate", "abs_delta_mean"],
                          ascending=[True, True, False], na_position="last")
    agg = agg.drop(columns=["group_order"])

    # Print compact table
    print(f"  {'group':<10} {'feature':<35} {'match%':>8} {'n_diff':>7} "
          f"{'mean_Δ':>10} {'|Δ|_mean':>10} {'|Δ|_p95':>10} {'|Δ|_max':>10}")
    print("  " + "-" * 110)
    for _, r in agg.iterrows():
        mr = f"{r['match_rate']*100:.1f}%" if pd.notna(r['match_rate']) else "N/A"
        dm = f"{r['delta_mean']:+.4f}" if r['delta_mean'] is not None else "N/A"
        am = f"{r['abs_delta_mean']:.4f}" if r['abs_delta_mean'] is not None else "N/A"
        ap = f"{r['abs_delta_p95']:.4f}" if r['abs_delta_p95'] is not None else "N/A"
        ax = f"{r['abs_delta_max']:.4f}" if r['abs_delta_max'] is not None else "N/A"
        print(f"  {r['group']:<10} {r['feature']:<35} {mr:>8} {int(r['n_mismatch']):>7} "
              f"{dm:>10} {am:>10} {ap:>10} {ax:>10}")

    out_summary = PARITY / "reports" / "deterministic_diff_summary.csv"
    out_rows = PARITY / "reports" / "deterministic_diff_rows.csv"
    agg.to_csv(out_summary, index=False)
    df.to_csv(out_rows, index=False)
    print(f"\n  Outputs:")
    print(f"    {out_summary}")
    print(f"    {out_rows}")


if __name__ == "__main__":
    main()
