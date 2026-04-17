"""
Extend v11 dep and arr blueprints to train on full available BTS history
(2022-01-01 through 2025-12-31). BTS publishes with a 1-2 month lag, so
the latest complete calendar year is 2025; 2026 data is not yet available.

Changes applied to each dep blueprint:
  - start_date: "2022-01-01"
  - end_date:   "2025-12-31"
  - input_flights_path:   globs for Year=2022..2025
  - history_flights_path: globs for Year=2021 (late-year for warmup) through 2025

Changes applied to each arr blueprint:
  - start_date: "2022-01-01"
  - end_date:   "2025-12-31"
  (arr blueprints read from the dep pipeline's intermediate parquets; no input path
  extension needed.)

Run from REPO_ROOT.
"""
from __future__ import annotations
import json
from pathlib import Path

REPO = Path("/Users/connermasteran/software/mpqc/flightright_modeldev")
DATA = REPO / "data"

NEW_START = "2022-01-01"
NEW_END = "2025-12-31"

# Glob patterns covering 2022 through 2025. History patterns start earlier
# (Nov-Dec 2021) to provide warmup for 60-day lookback windows at 2022-01-01.
NEW_INPUT_GLOBS = [
    "../flightrightdata/data/raw_bts/Year=2022/Month=*/flights.parquet",
    "../flightrightdata/data/raw_bts/Year=2023/Month=*/flights.parquet",
    "../flightrightdata/data/raw_bts/Year=2024/Month=*/flights.parquet",
    "../flightrightdata/data/raw_bts/Year=2025/Month=*/flights.parquet",
]
NEW_HISTORY_GLOBS = [
    "../flightrightdata/data/raw_bts/Year=2021/Month=1[1-2]/flights.parquet",
    "../flightrightdata/data/raw_bts/Year=2022/Month=*/flights.parquet",
    "../flightrightdata/data/raw_bts/Year=2023/Month=*/flights.parquet",
    "../flightrightdata/data/raw_bts/Year=2024/Month=*/flights.parquet",
    "../flightrightdata/data/raw_bts/Year=2025/Month=*/flights.parquet",
]

DEP_BLUEPRINTS = [
    "blueprint_dep_WN_v11.json",
    "blueprint_dep_AA_v11.json",
    "blueprint_dep_DL_v11.json",
    "blueprint_dep_UA_v11.json",
]
ARR_BLUEPRINTS = [
    "blueprint_arr_WN_v11.json",
    "blueprint_arr_AA_v11.json",
    "blueprint_arr_DL_v11.json",
    "blueprint_arr_UA_v11.json",
]


def bump_comment(cfg: dict) -> None:
    c = cfg.get("_comment", "")
    if "2022-01-01" in c:
        return
    cfg["_comment"] = (
        c + " [v11-full date range extended to 2022-01-01..2025-12-31 covering "
        "the full available BTS history for production training (BTS publishes "
        "with a lag; 2026 is not yet available). Input/history globs include "
        "Year=2022..2025; locally missing 2022 data globs to nothing and the bigger "
        "machine picks up the full set.]"
    )


def extend_dep(path: Path) -> None:
    cfg = json.loads(path.read_text())
    cfg["start_date"] = NEW_START
    cfg["end_date"] = NEW_END
    cfg["input_flights_path"] = list(NEW_INPUT_GLOBS)
    cfg["history_flights_path"] = list(NEW_HISTORY_GLOBS)
    bump_comment(cfg)
    path.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"[OK] extended dep blueprint: {path.name}")


def extend_arr(path: Path) -> None:
    cfg = json.loads(path.read_text())
    cfg["start_date"] = NEW_START
    cfg["end_date"] = NEW_END
    bump_comment(cfg)
    path.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"[OK] extended arr blueprint: {path.name}")


def main() -> None:
    for name in DEP_BLUEPRINTS:
        extend_dep(DATA / name)
    for name in ARR_BLUEPRINTS:
        extend_arr(DATA / name)
    print()
    print("Done. 8 v11 blueprints extended to 2022-01-01 .. 2025-12-31.")


if __name__ == "__main__":
    main()
