"""
Feature-source ablation: which feature group is responsible for the
residual production-vs-training AUC gap?

For each test flight:
  1. Run predict_departure to get the production feature dict.
  2. For each ablation config, patch the served feature dict by replacing
     features in `override_groups` with their BTS-derived counterparts
     from the training feature parquet.
  3. Run model_predict with the patched features against the same v11
     bundle. Get p_ge for thresholds 15/30/60/120.

Aggregate AUC per config + airline + threshold; compute the lift each
group adds when swapped from production-served values to training values.

Cost: ~6 min wall clock for 400 flights × 7 configs (offline, cached).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[3]
PROD = REPO.parent / "flightright"
DATA = REPO.parent / "flightrightdata"
PARITY = REPO / "data_quality" / "parity"
sys.path.insert(0, str(PROD / "src"))
sys.path.insert(0, str(PARITY / "scripts"))

THRESHOLDS = [15, 30, 60, 120]
TRAINING_AUC = {
    "WN": {15: 0.789, 30: 0.798, 60: 0.811, 120: 0.835},
    "AA": {15: 0.736, 30: 0.735, 60: 0.733, 120: 0.726},
    "DL": {15: 0.744, 30: 0.755, 60: 0.765, 120: 0.781},
    "UA": {15: 0.752, 30: 0.762, 60: 0.773, 120: 0.791},
}

# Feature group definitions (prefix-based + explicit lists). When a config
# requests a group, every feature matching is replaced with its BTS value.
def feature_group(name: str) -> str:
    """Categorize a feature name into one of the ablation groups."""
    if name.startswith("flightnum_od_"):
        return "flightnum_od"
    if name in {"is_holiday", "is_spring_break", "days_to_strike",
                "strike_severity", "carrier_delay_rate_anomaly_7d",
                "airline_cancel_rate_anomaly_7d"}:
        return "calendar"
    if (name.startswith("origin_") and ("temp" in name or "precip" in name
        or "wind" in name or "weather" in name or "visibility" in name
        or "cape" in name or "cloudcover" in name or "daily" in name
        or "dep_" in name and not "depdelay" in name)) \
       or name in {"origin_temp_max_K", "origin_temp_min_K", "wind_x_precip"} \
       or name == "origin_dep_hour_weathercode":
        return "weather"
    if name.startswith(("carrier_", "origin_depdelay", "origin_lateaircraft",
                        "origin_weather_rate", "origin_nas_rate",
                        "dest_", "hub_", "cancel_rate_origin",
                        "divert_rate_origin")):
        return "rolling"
    if name in {"sched_dep_hour", "dep_dow", "DepTimeBlk", "CRSDepTime",
                "CRSElapsedTime", "Distance", "is_peak_hour", "flight_month",
                "Origin", "Dest", "od_pair", "Reporting_Airline",
                "aircraft_type"}:
        return "schedule"
    return "other"


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
        DataPaths, ModelSpecDefaults, RemoteModelConfig,
        RollingStatsConfig, ServiceConfig,
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


def _safe_auc(y, p):
    if len(set(y)) < 2:
        return None
    return float(roc_auc_score(y, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flights-json", type=Path,
                    default=PARITY / "test_set" / "flights_dec_400.json")
    ap.add_argument("--bts-parquet", type=Path,
                    default=PARITY / "test_set" / "bts_features_dec_400.parquet")
    ap.add_argument("--cache-dir", type=Path,
                    default=PARITY / "cache" / "adb_responses")
    ap.add_argument("--rolling-sqlite-path", type=str,
                    default=str(Path.home() / ".flightright" / "rolling_stats_bts_dec2025.db"),
                    help="May contain `{carrier}` for per-airline DB routing.")
    ap.add_argument("--late-aircraft-cal", type=float, default=1.0)
    ap.add_argument("--out", type=Path,
                    default=PARITY / "reports" / "auc_ablation.json")
    args = ap.parse_args()

    _load_env()
    os.environ["FLIGHTRIGHT_LATE_AIRCRAFT_CAL"] = str(args.late_aircraft_cal)
    from seed_adb_cache import install_disk_cache
    install_disk_cache(args.cache_dir, allow_network=False, log_misses=False)

    flights = json.loads(args.flights_json.read_text())["flights"]
    bts = pd.read_parquet(args.bts_parquet)
    bts["FlightDate"] = pd.to_datetime(bts["FlightDate"])
    bts["key"] = (bts["Reporting_Airline"].astype(str)
                  + "/" + pd.to_numeric(bts["Flight_Number_Reporting_Airline"], errors="coerce").astype("Int64").astype(str)
                  + "/" + bts["Origin"].astype(str)
                  + "/" + bts["Dest"].astype(str)
                  + "/" + bts["FlightDate"].dt.strftime("%Y-%m-%d"))
    bts = bts.set_index("key")

    from flightright.service.predictor import predict_departure  # noqa: E402
    from flightright.cli.predict import model_predict  # noqa: E402

    cfg_by_carrier: Dict[str, Any] = {}
    rolling_path = args.rolling_sqlite_path
    if "{carrier}" in rolling_path:
        for c in ("AA", "DL", "UA", "WN"):
            p = Path(rolling_path.replace("{carrier}", c)).expanduser()
            cfg_by_carrier[c] = make_cfg(p)
            print(f"[INFO] rolling DB[{c}]: {p}", flush=True)
        cfg = None
    else:
        cfg = make_cfg(Path(rolling_path).expanduser())

    # Load each airline's bundle once (cached locally).
    bundle_by_airline = {}
    bundle_dir = DATA / "data" / "models"
    for airline in ("WN", "AA", "DL", "UA"):
        path = bundle_dir / f"dep_{airline}_100_v11" / f"dep_delay_bins_bundle_{airline}_100_v11.joblib"
        bundle_by_airline[airline] = joblib.load(path)
        print(f"[INFO] loaded bundle for {airline}: {len(bundle_by_airline[airline].get('numeric_features', []))} num + "
              f"{len(bundle_by_airline[airline].get('categorical_features', []))} cat features", flush=True)

    CONFIGS = ["none", "schedule", "calendar", "weather", "rolling",
               "flightnum_od", "all"]
    started = time.monotonic()

    # Each row: {flight_key, carrier, actual_delay, p_ge<thr>_<config>}
    rows: List[Dict[str, Any]] = []
    n_total = len(flights)

    for i, f in enumerate(flights, 1):
        fkey = f"{f['carrier']}/{f['flight_number']}/{f['origin']}/{f['dest']}/{f['flight_date']}"
        if fkey not in bts.index:
            continue
        bts_row = bts.loc[fkey]

        flight_cfg = cfg_by_carrier.get(f["carrier"], cfg) if cfg_by_carrier else cfg
        try:
            resp = predict_departure(
                airline=f["carrier"], flightnum=str(f["flight_number"]),
                dep_date=date.fromisoformat(f["flight_date"]),
                origin=f["origin"], dest=f["dest"],
                cfg=flight_cfg, include_features=True, public_mode=False,
                model_version_override="v11",
            )
        except Exception as e:
            continue
        served = resp.get("features", {}) or {}

        bundle = bundle_by_airline[f["carrier"]]
        feat_list = list(bundle.get("numeric_features", [])) + list(bundle.get("categorical_features", []))

        row: Dict[str, Any] = {
            "flight_key": fkey,
            "carrier": f["carrier"],
            "actual_delay": f["bts_dep_delay_minutes"],
        }

        # Compute predictions for each config
        for cfg_name in CONFIGS:
            if cfg_name == "none":
                groups = set()
            elif cfg_name == "all":
                groups = {"schedule", "calendar", "weather", "rolling", "flightnum_od"}
            else:
                groups = {cfg_name}

            patched = dict(served)
            for feat in feat_list:
                if feature_group(feat) not in groups:
                    continue
                if feat not in bts_row.index:
                    continue
                bts_val = bts_row[feat]
                # Skip if BTS value is unusable (NaN for numerics is OK; we still want
                # to patch with NaN to actually represent BTS).
                if isinstance(bts_val, (list, dict, np.ndarray)):
                    continue
                patched[feat] = bts_val

            try:
                pred = model_predict(bundle, patched)
                p_ge = pred.get("p_ge", {}) if isinstance(pred, dict) else {}
                for thr in THRESHOLDS:
                    v = p_ge.get(str(thr)) if str(thr) in p_ge else p_ge.get(thr)
                    if v is not None:
                        row[f"p_ge{thr}_{cfg_name}"] = float(v)
            except Exception:
                pass

        rows.append(row)
        if i % 50 == 0 or i == n_total:
            print(f"  [{i}/{n_total}] processed", flush=True)

    elapsed = time.monotonic() - started
    df = pd.DataFrame(rows)
    print(f"\n[INFO] Scored {len(df)} flights × {len(CONFIGS)} configs in {elapsed:.1f}s")

    # Compute mean gap per config across (airline × threshold) cells
    print(f"\n{'config':<14} {'mean_serve_AUC':>15} {'mean_gap':>10} {'lift_vs_none':>14}")
    print("-" * 60)
    config_means = {}
    config_gaps = {}
    for cfg_name in CONFIGS:
        deltas = []
        aucs = []
        for airline in ("WN", "AA", "DL", "UA"):
            sub = df[df["carrier"] == airline]
            if sub.empty:
                continue
            for thr in THRESHOLDS:
                col = f"p_ge{thr}_{cfg_name}"
                if col not in sub.columns:
                    continue
                ok = sub.dropna(subset=[col])
                if ok.empty:
                    continue
                y = (ok["actual_delay"] >= thr).astype(int).to_numpy()
                p = ok[col].to_numpy()
                auc = _safe_auc(y, p)
                if auc is None:
                    continue
                aucs.append(auc)
                tauc = TRAINING_AUC.get(airline, {}).get(thr)
                if tauc is not None:
                    deltas.append(auc - tauc)
        m = float(np.mean(aucs)) if aucs else None
        g = float(np.mean(deltas)) if deltas else None
        config_means[cfg_name] = m
        config_gaps[cfg_name] = g

    base_auc = config_means.get("none")
    for cfg_name in CONFIGS:
        m = config_means.get(cfg_name)
        g = config_gaps.get(cfg_name)
        lift = (m - base_auc) if (m is not None and base_auc is not None) else None
        m_s = f"{m:.4f}" if m is not None else "N/A"
        g_s = f"{g:+.4f}" if g is not None else "N/A"
        l_s = f"{lift:+.4f}" if lift is not None else "N/A"
        print(f"{cfg_name:<14} {m_s:>15} {g_s:>10} {l_s:>14}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_flights": len(df),
        "elapsed_s": round(elapsed, 1),
        "config_mean_auc": config_means,
        "config_mean_gap": config_gaps,
    }, default=str, indent=2))
    print(f"\n  Detail -> {args.out}")


if __name__ == "__main__":
    main()
