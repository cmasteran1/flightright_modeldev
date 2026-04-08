# v10 Feature Specification

**Date:** 2026-04-07
**Models:** departure delay (per airline), arrival delay (per airline), cross-airline cancellation
**Airlines covered:** WN, UA, AA, DL

## Why v10 exists

v9 added 11 tail-number-derived features (aircraft_type, has_recent_arrival_turn_5h, turn_time_hours, tail_leg_num_day, tail_depdelay_mean_last{1,14}, tail_lateaircraft_rate_last{1,14}, tail_n_legs_scheduled, tail_min_turn_time, tail_has_tight_turn) on top of v8. In production v9 became dramatically pessimistic (`v9_feature_outlier_analysis.md`, 2026-04-06): the top-importance features were either implementation-broken (`tail_n_legs_scheduled` values 230–847 versus a training max of ~12, `tail_min_turn_time` pinned at 0.25h, `tail_has_tight_turn` always 1, `tail_leg_num_day` always 5), and the remaining rolling-mean features were extremely sensitive to a small number of recent severe delays (carrier_depdelay_std_last7/14 hit 6–10× their training p99 on bad ops days at CLT/EWR/MIA, dragging the model into pessimism).

The new production source is **AeroDataBox**, which:

- Does not publish tail numbers for upcoming flights, so any feature that requires the tail at predict time is off the table.
- Does not report actual gate-out / gate-in or wheels-off times in a usable way, so any feature derived from `WheelsOff`/`WheelsOn`/`AirTime` is off the table even when it could be computed offline from BTS.

v10 therefore starts from **v8** (which performed better in production than v9) and:

1. Strips every tail/aircraft-derived feature.
2. Strips every wheels-off / actual-times-derived historical aggregate (`wo_slip_*`, `airtime_*_d_*`).
3. Reduces sensitivity to recent outliers by promoting the 1-day `*_mean_last1` baselines for carrier / carrier_origin / hub / dest delays to true `*_median_last1` features (computed over the previous day's individual flight rows, not the daily mean). The 7d/14d windows stay as means.
4. Adds the strong AeroDataBox-compatible candidates from `aerodatabox_feature_candidates.md` that do not need wheels-off: `origin_nasdelay_rate_last1d`, `cancel_rate_origin_last1d`, `divert_rate_origin_last14d`, plus `elapsed_time_ratio_last14d` (arrival only).

## AeroDataBox compatibility audit

Every v10 feature is producible from data we can serve at predict time without a tail number and without any actual gate or wheels-off observation:

- BTS history (offline) supplies all historical aggregates. The cells we now read additionally include `Cancelled`, `Diverted`, `ActualElapsedTime`, and `CRSElapsedTime`, all of which are gate-based or status flags — none of them are wheels-off-derived.
- AeroDataBox at predict time supplies the schedule fields (`CRSDepTime`, `CRSElapsedTime`, `CRSArrTime`, `Origin`, `Dest`, `DepTimeBlk`, `Distance`).
- Open-Meteo supplies the day-before-forecast weather features at the origin and destination.
- The strike cache supplies `strike_severity` / `days_to_strike` / `carrier_delay_rate_anomaly_7d`.

There are zero tail-number features and zero wheels-off-derived features. This is enforced by:

- Disabling `features_dep.tail_history` in every v10 blueprint.
- Disabling `features_dep.tail_and_flightnum_time` in the cancellation blueprint.
- Setting `recent_airtime.enabled` and `wheels_off_slip.enabled` to false in every v10 arrival blueprint.
- Gating the `add_turn_time_hours` and `add_tail_cascade_features` calls in `features_dep.py` on `tail_history.enabled` (they used to run unconditionally).
- A static feature-list audit (see `Verification` below) that greps the v10 training configs for forbidden substrings.

## Departure model

### Categorical (4)

`Origin`, `Dest`, `DepTimeBlk`, `origin_dep_hour_weathercode`

### Numeric (76)

#### Schedule & route (6)

`CRSDepTime`, `CRSElapsedTime`, `Distance`, `dep_dow`, `sched_dep_hour`, `is_peak_hour`

#### Flight-number / OD history (6)

| Feature | Window | Stat | Notes |
|---|---|---|---|
| `flightnum_od_depdelay_mean_last7` | 7d | mean | v8 mean |
| `flightnum_od_depdelay_mean_last14` | 14d | mean | v8 mean |
| `flightnum_od_depdelay_median_last1` | 1d | median | v8 already exposed; only median variant in v10 |
| `flightnum_od_depdelay_median_last7` | 7d | median | v8 already exposed |
| `flightnum_od_depdelay_median_last14` | 14d | median | v8 already exposed |
| `flightnum_od_support_count_last14d` | 14d | count | v8 |

`flightnum_od_depdelay_mean_last1` is dropped (replaced by its median sibling per the hybrid rule).

#### Carrier / origin baselines (6)

| Feature | Window | Stat |
|---|---|---|
| `carrier_depdelay_median_last1` | 1d | median over flight rows (NEW) |
| `carrier_depdelay_mean_last7` | 7d | mean of daily means |
| `carrier_origin_depdelay_median_last1` | 1d | median over flight rows (NEW) |
| `carrier_origin_depdelay_mean_last7` | 7d | mean of daily means |
| `carrier_origin_depdelay_mean_last14` | 14d | mean of daily means |
| `origin_depdelay_mean_last14` | 14d | mean of daily means |

#### Late-aircraft rates (4)

`origin_lateaircraft_rate_last1`, `origin_lateaircraft_rate_last7`, `carrier_lateaircraft_rate_last1`, `carrier_lateaircraft_rate_last7`. These stay as means — they are rates, not delay magnitudes, and the outlier analysis did not flag them as the median-fix target.

#### Hub spillover (31)

For each hub `i ∈ {0..4}`:

- `hub_{i}_depdelay_median_last1` (NEW, replaces v8's `hub_{i}_depdelay_mean_last1`)
- `hub_{i}_depdelay_mean_last7`, `hub_{i}_depdelay_mean_last14`
- `hub_{i}_lateaircraft_rate_last1`, `hub_{i}_lateaircraft_rate_last7`, `hub_{i}_lateaircraft_rate_last14`

Plus the `hub_max_lateaircraft_last1` worst-hub signal. Hubs are airline-specific (WN: DEN/PHX/BWI/MDW/BNA, UA: EWR/IAH/ORD/DEN/SFO, AA: DFW/CLT/MIA/ORD/PHX, DL: ATL/BOS/DTW/LAX/JFK).

#### Destination baselines (4)

| Feature | Window |
|---|---|
| `dest_depdelay_median_last1` | 1d (NEW, replaces v8's `dest_depdelay_mean_last1`) |
| `dest_lateaircraft_rate_last1` | 1d |
| `dest_lateaircraft_rate_last7` | 7d |
| `dest_lateaircraft_rate_last14` | 14d |

#### Congestion (2)

`origin_congestion_3h_total`, `origin_airline_congestion_3h_total`

#### Origin weather (12)

Daily (4): `origin_temp_max_K`, `origin_temp_min_K`, `origin_daily_precip_sum_mm`, `origin_daily_windgusts_max_kmh`
Hourly at departure (7): `origin_dep_temp_K`, `origin_dep_precip_mm`, `origin_dep_windgusts_kmh`, `origin_dep_visibility_m`, `origin_dep_cape_jkg`, `origin_dep_log1p_cape`, `origin_dep_cloudcover_pct`
Derived (1): `wind_x_precip`

#### Strike (3)

`strike_severity`, `days_to_strike`, `carrier_delay_rate_anomaly_7d`

#### NEW v10 AeroDataBox aggregates (3)

| Feature | Definition |
|---|---|
| `origin_nasdelay_rate_last1d` | Fraction of flights at origin with `NASDelay > 0` over the previous 1 day. Strongest single-feature signal in `aerodatabox_feature_candidates.md` (point-biserial 0.115 vs `y_dep_ge15`). |
| `cancel_rate_origin_last1d` | Fraction of flights at origin marked Cancelled over the previous 1 day. Captures a disruption mechanism orthogonal to delay magnitudes. |
| `divert_rate_origin_last14d` | Fraction of flights at origin marked Diverted over the previous 14 days. Captures lingering ripple effects of severe disruption events. |

### Departure totals: 4 categorical + 76 numeric ≈ 80 features (v8 = 76, v9 = 85)

## Arrival model

The arrival model inherits all departure numerics + categoricals via the `merge_dep_features` step in `features_arr.py`, then adds arrival-specific features.

### Schedule (2)

`CRSArrTime`, `sched_arr_hour`

### Route arrival-delay history (8)

| Feature | Window | Stat |
|---|---|---|
| `arrdelay_mean_7d_fn_od` / `arrdelay_n_7d_fn_od` | 7d | mean + count |
| `arrdelay_mean_14d_fn_od` / `arrdelay_n_14d_fn_od` | 14d | mean + count |
| `arrdelay_median_7d_fn_od` | 7d | median |
| `arrdelay_median_14d_fn_od` | 14d | median (NEW) |
| `arrdelay_mean_7d_car_od` / `arrdelay_n_7d_car_od` | 7d | mean + count |
| `arrdelay_mean_14d_car_od` / `arrdelay_n_14d_car_od` | 14d | mean + count |
| `arrdelay_median_7d_car_od` | 7d | median |
| `arrdelay_median_14d_car_od` | 14d | median (NEW) |

### Destination arrival congestion (4)

`dest_arrivals_pm60_sched`, `dest_airline_arrivals_pm60_sched`, `dest_arrivals_pm60_eta`, `dest_airline_arrivals_pm60_eta`. The ETA shift continues to use the surviving `arrdelay_mean_14d_car_od` (then 7d) as in v8.

### Destination weather at scheduled arrival (7)

`dest_arr_temperature_2m`, `dest_arr_precipitation`, `dest_arr_windspeed_10m`, `dest_arr_windgusts_10m`, `dest_arr_visibility`, `dest_arr_cape`, `dest_arr_cloudcover`. (`dest_arr_weathercode` remains deferred, same as v8.)

### NEW v10 elapsed-time signal (1)

| Feature | Definition |
|---|---|
| `elapsed_time_ratio_last14d` | Rolling mean of `ActualElapsedTime / CRSElapsedTime` for `(Airline, FlightNum, Origin, Dest)` over the previous 14 days. Both inputs are gate-based BTS fields, so this feature does not depend on `WheelsOff`/`WheelsOn` and is safe under the AeroDataBox constraint. |

### Dropped from v8 → v10 arrival

- All eight `wo_slip_*` features (wheels-off slip family).
- All eight `airtime_*_d_*` features (mean and count, both fn_od and car_od grains) — they aggregate `AirTime`, which is `WheelsOn − WheelsOff`.

### Arrival totals: ≈ 80 inherited dep numerics + 22 arrival-specific numerics + 4 categoricals ≈ ~106 features (v8 = 114, v9 = 123)

## Cancellation model

The cancellation model only had a v9 baseline (`blueprint_cancel_v9.json`, `cancel_train_v9.json`). v10 synthesizes a v8-equivalent feature list by stripping every tail / wheels-off feature and applying the same hybrid-median rules as the dep model.

- All seven v9 tail features are dropped: `has_recent_arrival_turn_5h`, `turn_time_hours`, `tail_leg_num_day`, `tail_depdelay_mean_last1`, `tail_depdelay_mean_last14`, `tail_lateaircraft_rate_last1`, `tail_lateaircraft_rate_last14`. Both `features_dep.tail_history.enabled` and `features_dep.tail_and_flightnum_time` are disabled in the blueprint.
- The same hybrid median rule (1-day window switches to median over flight rows) is applied to `carrier_depdelay`, `carrier_origin_depdelay`, `hub_{0..4}_depdelay`, and `dest_depdelay`.
- The same three new AeroDataBox aggregates are added.
- Cross-airline categoricals retained: `Origin`, `Dest`, `Reporting_Airline`, `DepTimeBlk`.
- Hub list and the 2022-start training window are kept from v9 — cancellations need the longer history for class balance.
- Bundle name → `cancellation_bundle_v10.joblib`.

## Files added or modified

### New files

- `data/blueprint_dep_{WN,UA,AA,DL}_v10.json`
- `data/blueprint_arr_{WN,UA,AA,DL}_v10.json`
- `data/blueprint_cancel_v10.json`
- `data/dep_train_{WN,UA,AA,DL}_100_v10.json`
- `data/arr_train_{WN,UA,AA,DL}_100_v10.json`
- `data/cancel_train_v10.json`
- `exploration/v10_feature_spec.md` (this file)

### Modified

- `src/fetch_prune/prepare_dataset.py` — added `Diverted`, `ActualElapsedTime` to the upstream BTS load and to the history-pool keep list (along with `CRSElapsedTime` and `Cancelled`).
- `src/fetch_prune/prepare_cancel_dataset.py` — added the same columns to `_KEEP_COLS` and `_HIST_COLS`, imported the six new aggregator functions from `features_dep`, and added gated calls for the median-last1 and AeroDataBox features. The tail/flightnum-time block now logs when it is disabled.
- `src/fetch_prune/features_dep.py` — added `_daily_median_by`, `_attach_shifted_daily_median`, `add_carrier_dep_delay_median_last1`, `add_dest_depdelay_median_last1`, `add_hub_depdelay_median_last1`, `add_origin_nasdelay_rate_last1d`, `add_cancel_rate_origin_last1d`, `add_divert_rate_origin_last14d`. Wired all six into `main()` behind config flags. Gated the previously-unconditional `add_turn_time_hours` and `add_tail_cascade_features` on `tail_history.enabled`.
- `src/fetch_prune/features_arr.py` — added `MEDIAN_WINDOWS_V10 = [7, 14]`, threaded an optional `median_windows` parameter through `add_route_arrival_delay_stats`, added `add_elapsed_time_ratio_stats`, and gated `add_recent_airtime_stats` / `add_recent_wheels_off_slip_stats` behind config flags. The history projection now also reads `ActualElapsedTime` and `CRSElapsedTime` when present.

## Verification

1. **Static feature-list audit.** Grep every v10 training config and confirm none of the strings appear: `tail_`, `aircraft_type`, `has_recent_arrival_turn`, `turn_time_hours`, `wo_slip_`, `airtime_mean_`, `taxi_out`, `gate_hold`. Run via:

   ```
   .venv/bin/python - <<'PY'
   import json, pathlib
   FORBIDDEN = ["tail_", "aircraft_type", "has_recent_arrival_turn",
                "turn_time_hours", "wo_slip_", "airtime_mean_",
                "taxi_out", "gate_hold"]
   for cfg_path in sorted(pathlib.Path("data").glob("*v10*.json")):
       blob = cfg_path.read_text()
       hits = [tok for tok in FORBIDDEN if tok in blob]
       if hits:
           print(f"FAIL  {cfg_path}: {hits}")
       else:
           print(f"PASS  {cfg_path}")
   PY
   ```

2. **Pipeline smoke test (one airline at a time).**

   ```
   .venv/bin/python src/fetch_prune/prepare_dataset.py data/blueprint_dep_WN_v10.json
   .venv/bin/python src/fetch_prune/features_dep.py    data/blueprint_dep_WN_v10.json
   .venv/bin/python src/fetch_prune/features_arr.py    data/blueprint_arr_WN_v10.json
   ```

   Confirm the output parquets contain every column listed in the matching v10 training config and no extras from v8/v9 we meant to drop.

3. **Health-check report.** Run `src/health/health_checks.py` on the v10 enriched parquet for WN. Expect zero rows of tail / wo_slip / airtime columns, populated `*_median_last1` columns with reasonable training-distribution stats (medians should track means within ~10–20%), and the three new AeroDataBox columns with non-trivial coverage (>95% non-null after the lookback warmup).

4. **Train one threshold end-to-end** for WN dep ≥15 to confirm the training pipeline accepts the new feature list:

   ```
   .venv/bin/python src/training/train_dep_bins_ordinal_catboost.py data/dep_train_WN_100_v10.json
   ```

5. **Outlier replay.** Re-score the three AA2365 / UA1777 / AA2318 production payloads from `v9_feature_outlier_analysis.md` against the new v10 models and confirm `P(≥60)` and `P(≥120)` drop substantially from the v9 numbers, and that monotonicity (`P(≥15) ≥ P(≥30) ≥ P(≥60)`) holds.

6. **Cancellation regression.** Train the v10 cancel model and compare AUC / Brier / calibration against the v9 cancel model on the same eval split. Expect roughly equal headline metrics with improved tail-of-distribution behaviour on bad ops days.

## Out of scope for v10

- Re-tuning CatBoost hyperparameters. v8 settings (`depth=8/9`, `iterations=6000`, `learning_rate=0.03`, `l2_leaf_reg=3.0/5.0`) are reused.
- Backfilling AeroDataBox snapshots for offline evaluation. v10 trains on BTS as before.
- Promoting `dest_arr_weathercode` to a categorical (deferred from v9). Revisit in v11.
