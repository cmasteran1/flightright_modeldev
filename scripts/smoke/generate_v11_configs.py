"""
Generate v11 config files from v10 templates.

Changes applied to every v10 config:
  - Replace "v10" with "v11" in filenames and all internal path references.
  - Bump delay thresholds [15,30,45,60] -> [15,30,60,120] where present.
  - Update per_threshold_train_paths keys accordingly.
  - Turn off NAS delay feature flag (nas_rate_last1d.enabled = false).
  - Drop origin_nasdelay_rate_last1d from numeric_features.
  - Add flightnum_od_otp_rate_last14d + airline_cancel_rate_anomaly_7d to
    numeric_features and to blueprint feature flags per production/smoke policy.
  - Update deploy_bundle_joblib_name.
  - Update _comment field to describe v11 changes.
"""

from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any

REPO = Path("/Users/connermasteran/software/mpqc/flightright_modeldev")
DATA = REPO / "data"

# All v10 configs to clone
V10_FILES = [
    # Departure blueprints
    "blueprint_dep_WN_v10.json",
    "blueprint_dep_AA_v10.json",
    "blueprint_dep_DL_v10.json",
    "blueprint_dep_UA_v10.json",
    # Arrival blueprints
    "blueprint_arr_WN_v10.json",
    "blueprint_arr_AA_v10.json",
    "blueprint_arr_DL_v10.json",
    "blueprint_arr_UA_v10.json",
    # Cancel blueprint
    "blueprint_cancel_v10.json",
    # Departure training
    "dep_train_WN_100_v10.json",
    "dep_train_AA_100_v10.json",
    "dep_train_DL_100_v10.json",
    "dep_train_UA_100_v10.json",
    # Arrival training
    "arr_train_WN_100_v10.json",
    "arr_train_AA_100_v10.json",
    "arr_train_DL_100_v10.json",
    "arr_train_UA_100_v10.json",
    # Cancel training
    "cancel_train_v10.json",
    # Smoke variants
    "blueprint_dep_WN_v10_smoke.json",
    "dep_train_WN_100_v10_smoke.json",
]


def bump_path_strings(obj: Any) -> Any:
    """Recursively replace 'v10' -> 'v11' in all string values."""
    if isinstance(obj, str):
        return obj.replace("_v10", "_v11").replace("v10_", "v11_")
    if isinstance(obj, list):
        return [bump_path_strings(x) for x in obj]
    if isinstance(obj, dict):
        return {k: bump_path_strings(v) for k, v in obj.items()}
    return obj


def bump_thresholds(obj: Any) -> Any:
    """Replace [15, 30, 45, 60] with [15, 30, 60, 120] wherever it appears."""
    if isinstance(obj, list) and obj == [15, 30, 45, 60]:
        return [15, 30, 60, 120]
    if isinstance(obj, list):
        return [bump_thresholds(x) for x in obj]
    if isinstance(obj, dict):
        return {k: bump_thresholds(v) for k, v in obj.items()}
    return obj


def update_per_threshold_paths(cfg: dict) -> dict:
    """Rekey per_threshold_train_paths from 15/30/45/60 to 15/30/60/120 and
    rename parquet filenames from ge45/ge60 -> ge60/ge120.
    Also swaps numeric 45->60, 60->120 in values if they appear as ge{N}.
    """
    ptp = cfg.get("per_threshold_train_paths")
    if not isinstance(ptp, dict):
        return cfg
    new_ptp = {}
    # Map old-key -> new-key
    key_map = {"15": "15", "30": "30", "45": "60", "60": "120"}
    for k, v in ptp.items():
        new_key = key_map.get(str(k), str(k))
        # Rewrite filename: ge{old_key} -> ge{new_key}
        if isinstance(v, str):
            old_str = f"ge{k}"
            new_str = f"ge{new_key}"
            v = v.replace(old_str, new_str)
        new_ptp[new_key] = v
    cfg["per_threshold_train_paths"] = new_ptp
    return cfg


def update_numeric_features(cfg: dict, smoke: bool) -> dict:
    """Drop NAS feature, add OTP (+ cancel anomaly if not smoke) to numeric_features."""
    nf = cfg.get("numeric_features")
    if not isinstance(nf, list):
        return cfg
    # Drop origin_nasdelay_rate_last1d
    nf = [f for f in nf if f != "origin_nasdelay_rate_last1d"]
    # Add OTP rate and (conditionally) cancel anomaly
    if "flightnum_od_otp_rate_last14d" not in nf:
        nf.append("flightnum_od_otp_rate_last14d")
    if not smoke and "airline_cancel_rate_anomaly_7d" not in nf:
        nf.append("airline_cancel_rate_anomaly_7d")
    cfg["numeric_features"] = nf
    return cfg


def update_feature_flags(cfg: dict, smoke: bool) -> dict:
    """Turn off nas_rate_last1d, turn on otp_rate_last14d + cancel_anomaly_7d per policy."""
    fd = cfg.get("features_dep")
    if not isinstance(fd, dict):
        return cfg
    # NAS off
    fd["nas_rate_last1d"] = {"enabled": False}
    # OTP on (always)
    fd["otp_rate_last14d"] = {"enabled": True}
    # Cancel anomaly: prod on, smoke off
    fd["cancel_anomaly_7d"] = {"enabled": False if smoke else True}
    cfg["features_dep"] = fd
    return cfg


def update_comment(cfg: dict, name: str) -> dict:
    """Prefix or replace the _comment field to call out v11 changes."""
    v11_blurb = (
        "v11: starts from v10. Changes: (1) late-aircraft threshold aligned with Aerodatabox at >=15 min "
        "(was >0 in v10) across all *_lateaircraft_rate_* features, (2) delay buckets [15,30,45,60] -> "
        "[15,30,60,120] to capture severe-delay risk, (3) origin_nasdelay_rate_last1d dropped (no "
        "Aerodatabox source without approximation), (4) flightnum_od_otp_rate_last14d added (BTS-offline, "
        "univariate AUC 0.685 at ge15, +1.5-2.2% log-loss lift). (5) airline_cancel_rate_anomaly_7d added "
        "in production configs (disabled on smoke because 60-day baseline requires longer history). "
        "See exploration/v11_feature_spec.md."
    )
    cfg["_comment"] = v11_blurb
    return cfg


def is_smoke(name: str) -> bool:
    return "smoke" in name


def main() -> None:
    for v10_name in V10_FILES:
        v10_path = DATA / v10_name
        if not v10_path.exists():
            print(f"[WARN] missing v10 source: {v10_path}")
            continue

        v11_name = v10_name.replace("v10", "v11")
        v11_path = DATA / v11_name

        with open(v10_path) as f:
            cfg = json.load(f)

        smoke = is_smoke(v10_name)

        # Pipeline of transformations
        cfg = bump_path_strings(cfg)
        cfg = bump_thresholds(cfg)
        cfg = update_per_threshold_paths(cfg)
        cfg = update_numeric_features(cfg, smoke)
        cfg = update_feature_flags(cfg, smoke)
        cfg = update_comment(cfg, v11_name)

        # Update deploy_bundle_joblib_name if present
        if "deploy_bundle_joblib_name" in cfg:
            cfg["deploy_bundle_joblib_name"] = cfg["deploy_bundle_joblib_name"].replace("v10", "v11")

        with open(v11_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"[OK] wrote {v11_path.name}")


if __name__ == "__main__":
    main()
