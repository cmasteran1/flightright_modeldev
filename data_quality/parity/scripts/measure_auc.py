"""
Production AUC + calibration on the parity test set.

For each flight in the canonical test set:
  1. Run production prediction (cached AeroDataBox + Open-Meteo, real
     v11 bundle from S3).
  2. Compare predicted P(>=T) to actual BTS dep_delay outcome.
  3. Aggregate per-airline and overall AUC + calibration metrics.

Reports the gap to training-eval AUC so we can quantify the production
shortfall.

By default uses the 40-flight test set with cached fixtures (offline,
~30s). For a larger sample, point at a separate test-set JSON; the
caller is responsible for seeding its cache.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

REPO = Path(__file__).resolve().parents[3]
PROD = REPO.parent / "flightright"
PARITY = REPO / "data_quality" / "parity"
sys.path.insert(0, str(PROD / "src"))
sys.path.insert(0, str(PARITY / "scripts"))


# Training AUC for v11 (from MLflow / training logs we just ran). Compared
# against to compute the train/serve AUC gap. Update this dict on retrain.
TRAINING_AUC = {
    "WN": {15: 0.789, 30: 0.798, 60: 0.811, 120: 0.835},
    "AA": {15: 0.736, 30: 0.735, 60: 0.733, 120: 0.726},
    "DL": {15: 0.744, 30: 0.755, 60: 0.765, 120: 0.781},
    "UA": {15: 0.752, 30: 0.762, 60: 0.773, 120: 0.791},
}
THRESHOLDS = [15, 30, 60, 120]


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
    DATA = REPO.parent / "flightrightdata"
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


def _rolling_db_meta(path: Path):
    """Read date range + row count from a rolling_stats SQLite, if present."""
    if not path.exists():
        return {"present": False}
    try:
        import sqlite3
        with sqlite3.connect(str(path)) as conn:
            cur = conn.execute("SELECT MIN(date), MAX(date), COUNT(*) FROM daily_rollup")
            mn, mx, n = cur.fetchone()
            return {"present": True, "min_date": mn, "max_date": mx, "row_count": int(n)}
    except Exception as e:
        return {"present": True, "error": str(e)}


def _safe_auc(y, p):
    if len(set(y)) < 2:
        return None
    return float(roc_auc_score(y, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flights-json", type=Path,
                    default=PARITY / "test_set" / "flights.json")
    ap.add_argument("--cache-dir", type=Path,
                    default=PARITY / "cache" / "adb_responses")
    ap.add_argument("--allow-network", action="store_true",
                    help="Permit API calls on cache miss (default: hard fail).")
    ap.add_argument("--rolling-sqlite-path", type=str,
                    default=str(Path.home() / ".flightright" / "rolling_stats.db"),
                    help="Path to the rolling-stats SQLite. Default is the production-synced "
                         "DB (covers ~50 days of recent data only). Point at a BTS-seeded DB "
                         "to populate the rolling-stats family for historical parity tests. "
                         "May contain `{carrier}` to route per-carrier DBs (e.g. "
                         "~/.flightright/rolling_stats_bts_dec2025_{carrier}_top100.db); "
                         "this matches training's per-airline top-100 airport filter.")
    ap.add_argument("--late-aircraft-cal", type=float, default=0.51,
                    help="Calibration factor for `_lateaircraft_rate_*` features. 0.51 (default) "
                         "is correct for the production ADB-derived rolling DB which counts "
                         "dep_delay>=15 of any cause. Use 1.0 with the BTS-seeded DB whose "
                         "dep_late_aircraft_count already counts LateAircraftDelay>=15.")
    ap.add_argument("--out", type=Path,
                    default=PARITY / "reports" / "auc_calibration.json")
    args = ap.parse_args()

    _load_env()
    # CRITICAL: set the env var BEFORE importing predict_departure (which
    # imports batch_rollups, which reads FLIGHTRIGHT_LATE_AIRCRAFT_CAL at
    # module-import time into a module-level constant). Setting it later
    # has no effect.
    os.environ["FLIGHTRIGHT_LATE_AIRCRAFT_CAL"] = str(args.late_aircraft_cal)

    from seed_adb_cache import install_disk_cache
    install_disk_cache(args.cache_dir, allow_network=args.allow_network, log_misses=False)

    spec = json.loads(args.flights_json.read_text())
    flights = spec["flights"]

    from flightright.service.predictor import predict_departure  # noqa: E402

    cfg_by_carrier: Dict[str, object] = {}
    db_meta_by_carrier: Dict[str, dict] = {}
    rolling_path_template = args.rolling_sqlite_path
    if "{carrier}" in rolling_path_template:
        for c in ("AA", "DL", "UA", "WN"):
            p = Path(rolling_path_template.replace("{carrier}", c)).expanduser()
            cfg_by_carrier[c] = make_cfg(p)
            db_meta_by_carrier[c] = _rolling_db_meta(p)
            print(f"[INFO] rolling DB[{c}]: {p}  meta={db_meta_by_carrier[c]}")
        cfg = None
        db_meta = db_meta_by_carrier
    else:
        rp = Path(rolling_path_template).expanduser()
        cfg = make_cfg(rp)
        db_meta = _rolling_db_meta(rp)
        print(f"[INFO] rolling-stats DB: {rp}  meta={db_meta}")
    print(f"[INFO] late-aircraft cal: {args.late_aircraft_cal}")
    started = time.monotonic()

    rows: List[Dict] = []
    print(f"Scoring {len(flights)} flights...", flush=True)
    for i, f in enumerate(flights, 1):
        flight_cfg = cfg_by_carrier.get(f["carrier"], cfg) if cfg_by_carrier else cfg
        try:
            resp = predict_departure(
                airline=f["carrier"], flightnum=str(f["flight_number"]),
                dep_date=date.fromisoformat(f["flight_date"]),
                origin=f["origin"], dest=f["dest"],
                cfg=flight_cfg, include_features=False, public_mode=False,
                model_version_override="v11",
            )
        except Exception as e:
            print(f"  [{i:3d}/{len(flights)}] FAIL {f['carrier']}{f['flight_number']}: {e}",
                  flush=True)
            continue
        # predict_departure returns {"prediction": {"p_ge": {"15": 0.04, ...}}}
        ge_probs: Dict[int, float] = {}
        prediction = resp.get("prediction") or {}
        p_ge = prediction.get("p_ge") or {}
        for thr in THRESHOLDS:
            v = p_ge.get(str(thr)) if str(thr) in p_ge else p_ge.get(thr)
            if v is not None:
                ge_probs[thr] = float(v)
        rows.append({
            "carrier": f["carrier"],
            "flight_number": f["flight_number"],
            "origin": f["origin"], "dest": f["dest"],
            "date": f["flight_date"],
            "actual_delay": f["bts_dep_delay_minutes"],
            **{f"p_ge{thr}": ge_probs.get(thr) for thr in THRESHOLDS},
            "raw_response_keys": sorted(list(resp.keys()))[:8],
        })

    elapsed = time.monotonic() - started
    df = pd.DataFrame(rows)
    print(f"\n[INFO] Scored {len(df)} flights in {elapsed:.1f}s")
    if df.empty:
        sys.exit("No predictions returned.")

    if df["p_ge15"].isna().all():
        # Print response shape so we know what to extract
        print("\n[DEBUG] Probabilities not found in response. First-row keys:")
        print(f"  {df.iloc[0]['raw_response_keys']}")
        sys.exit(1)

    print(f"\n{'='*78}")
    print(" PRODUCTION AUC + CALIBRATION (v11 bundles, 40-flight test set)")
    print(f"{'='*78}\n")

    overall_metrics = []
    by_airline = {}

    for thr in THRESHOLDS:
        col = f"p_ge{thr}"
        ok = df.dropna(subset=[col, "actual_delay"]).copy()
        if ok.empty:
            continue
        ok[f"y_{thr}"] = (ok["actual_delay"] >= thr).astype(int)
        y = ok[f"y_{thr}"].to_numpy()
        p = ok[col].to_numpy()

        auc = _safe_auc(y, p)
        cal_pred = float(p.mean())
        cal_actual = float(y.mean())
        brier = float(brier_score_loss(y, p))

        # Per-airline
        per_air = {}
        for airline in sorted(df["carrier"].unique()):
            sub = ok[ok["carrier"] == airline]
            if len(sub) < 3:
                continue
            ya = sub[f"y_{thr}"].to_numpy()
            pa = sub[col].to_numpy()
            per_air[airline] = {
                "n": int(len(sub)),
                "actual_rate": round(float(ya.mean()), 3),
                "pred_rate": round(float(pa.mean()), 3),
                "auc": (None if _safe_auc(ya, pa) is None else round(_safe_auc(ya, pa), 3)),
                "training_auc": TRAINING_AUC.get(airline, {}).get(thr),
            }
            by_airline.setdefault(airline, {})[thr] = per_air[airline]

        overall_metrics.append({
            "threshold": thr,
            "n": int(len(ok)),
            "actual_rate": round(cal_actual, 3),
            "pred_rate": round(cal_pred, 3),
            "calibration_gap": round(cal_pred - cal_actual, 3),
            "brier": round(brier, 4),
            "auc": (None if auc is None else round(auc, 3)),
            "by_airline": per_air,
        })

    # Print table
    print(f"  {'thr':>5} {'n':>4} {'actual':>7} {'pred':>7} {'cal_gap':>8} "
          f"{'brier':>7} {'AUC':>7}")
    print(f"  {'-'*5} {'-'*4} {'-'*7} {'-'*7} {'-'*8} {'-'*7} {'-'*7}")
    for m in overall_metrics:
        print(f"  ≥{m['threshold']:<4} {m['n']:>4} {m['actual_rate']:>7.3f} "
              f"{m['pred_rate']:>7.3f} {m['calibration_gap']:>+8.3f} "
              f"{m['brier']:>7.4f} {(m['auc'] if m['auc'] else 'N/A'):>7}")

    print(f"\n  Per-airline AUC vs training-eval AUC:")
    print(f"  {'airline':<8} {'thr':>4} {'n':>3} {'serve_AUC':>10} {'train_AUC':>10} "
          f"{'Δ':>7} {'pred':>7} {'actual':>7}")
    for airline in sorted(by_airline.keys()):
        for thr in THRESHOLDS:
            cell = by_airline[airline].get(thr)
            if cell is None:
                continue
            tauc = cell.get("training_auc")
            sauc = cell.get("auc")
            delta = (sauc - tauc) if (sauc is not None and tauc is not None) else None
            print(f"  {airline:<8} {thr:>4} {cell['n']:>3} "
                  f"{(f'{sauc:.3f}' if sauc else 'N/A'):>10} "
                  f"{(f'{tauc:.3f}' if tauc else 'N/A'):>10} "
                  f"{(f'{delta:+.3f}' if delta is not None else 'N/A'):>7} "
                  f"{cell['pred_rate']:>7.3f} {cell['actual_rate']:>7.3f}")

    # Compute the headline mean gap across (airline × threshold) cells where
    # both sides have an AUC. This is the user's stopping criterion.
    deltas = []
    for airline_cells in by_airline.values():
        for cell in airline_cells.values():
            sa = cell.get("auc")
            ta = cell.get("training_auc")
            if sa is not None and ta is not None:
                deltas.append(sa - ta)
    mean_gap = round(float(sum(deltas) / len(deltas)), 4) if deltas else None
    print(f"\n  Mean (airline × threshold) AUC gap = {mean_gap} (target: ≥ -0.04)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_flights": len(df),
        "elapsed_s": round(elapsed, 1),
        "rolling_sqlite_path": rolling_path_template,
        "rolling_sqlite_meta": db_meta,
        "late_aircraft_cal": args.late_aircraft_cal,
        "mean_auc_gap": mean_gap,
        "n_cells": len(deltas),
        "thresholds": overall_metrics,
        "training_auc_reference": TRAINING_AUC,
    }, default=str, indent=2))
    print(f"\n  Detail -> {args.out}")


if __name__ == "__main__":
    main()
