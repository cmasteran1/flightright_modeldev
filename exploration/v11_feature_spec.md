# v11 Feature Specification

**Date:** 2026-04-16
**Models:** departure delay (per airline), arrival delay (per airline), cross-airline cancellation
**Airlines covered:** WN, UA, AA, DL
**Authoritative predecessor:** `exploration/v10_feature_spec.md`

---

## Why v11 exists

v11 is a semantics + scope iteration on top of v10:

1. **Late-aircraft definition alignment.** v10 used `LateAircraftDelay > 0` (any minute of prior-leg tardiness); the industry standard (BTS OTP reporting, Aerodatabox, IATA on-time performance) uses 15+ minutes. v10 therefore had a train/serve definitional mismatch. v11 fixes this across all `*_lateaircraft_rate_*` features via the module constant `LATE_AIRCRAFT_THRESHOLD_MIN = 15` in `src/fetch_prune/features_dep.py`.
2. **NAS approximation removed.** `origin_nasdelay_rate_last1d` has no direct Aerodatabox source — the `/airports/delays/{icao}` endpoint returns only aggregate delay counts, not a NAS-specific breakdown. Per user rule "no approximations," the feature is **disabled in all v11 blueprints and dropped from all v11 training configs**. The feature function stays in `features_dep.py` for offline research.
3. **Prediction range extended.** Delay thresholds change from v10's `[15, 30, 45, 60]` to v11's `[15, 30, 60, 120]`. This replaces the tight 45-min bucket with a meaningful severe-delay bucket at 2 hours — more useful for consumer-facing "will I miss my connection / make a rebooking" risk outputs. Matches the original v9 thresholds.
4. **Two new BTS-offline features added** (no Aerodatabox dependency):
   - `flightnum_od_otp_rate_last14d`
   - `airline_cancel_rate_anomaly_7d` (flag-gated: enabled in production, disabled in smoke)

5. **v10 training scope preserved.** v11 trains on BTS + Open-Meteo only. The Aerodatabox airport delay index candidate (`airport_delay_index_origin/dest` from the research phase) is **dropped from v11 scope** because of training-time API cost. The encoding-change candidate (`dep_dow` / `sched_dep_hour` / `is_peak_hour` numeric → categorical) is **dropped from v11 scope** because empirical testing showed CatBoost performs slightly worse with categorical encoding (`exploration/reports/2026-04-16-numeric-vs-categorical-encoding.md`).

All four Phase-1 agent reports are preserved:
- `research/2026-04-16-v11-new-feature-candidates.md`
- `feature_transferability/reports/2026-04-16-v11-aerodatabox-screening.md`
- `exploration/reports/2026-04-16-v11-feature-signal-tests.md`
- `exploration/reports/2026-04-16-numeric-vs-categorical-encoding.md`

---

## Diff from v10

### Code changes in `src/fetch_prune/features_dep.py`

- **New module constant** near the top of the file: `LATE_AIRCRAFT_THRESHOLD_MIN = 15`.
- **Four late-aircraft threshold sites** changed from `> 0` to `>= LATE_AIRCRAFT_THRESHOLD_MIN`:
  - `add_tail_rolling_history` (disabled in all blueprints but code kept correct)
  - `add_dest_rolling_stats`
  - `add_wn_hub_spillover_from_history`
  - `add_delay_cause_rates_from_history`
- **Two new feature functions** following the v10 `add_cancel_rate_origin_last1d` / `add_divert_rate_origin_last14d` pattern:
  - `add_flightnum_od_otp_rate_last14d(df, hist)` — emits `flightnum_od_otp_rate_last14d` in [0, 1].
  - `add_airline_cancel_rate_anomaly_7d(df, hist)` — emits `airline_cancel_rate_anomaly_7d` (z-score, typical [-3, +3]).
- **Two new blueprint flags** wired into `main()`:
  - `features_dep.otp_rate_last14d.enabled` → calls `add_flightnum_od_otp_rate_last14d`
  - `features_dep.cancel_anomaly_7d.enabled` → calls `add_airline_cancel_rate_anomaly_7d`

### Code change in `src/training/train_dep_bins_ordinal_catboost.py`

- Default `bin_weights_minutes` changed from `[7.5, 22.5, 45.0, 90.0, 150.0]` (v10 buckets) to `[7.5, 22.5, 45.0, 90.0, 180.0]` (v11 buckets: midpoints of `<15`, `15–30`, `30–60`, `60–120`, `>=120`).

### Code changes in `src/health/health_checks.py`

- All `*_lateaircraft_rate_*` description strings annotated with "(>=15min)".
- `hub_max_lateaircraft_last1` description annotated.
- Two new registrations: `flightnum_od_otp_rate_last14d` in [0, 1]; `airline_cancel_rate_anomaly_7d` in [-10, 10].
- `origin_nasdelay_rate_last1d` registration retained (function still exists) but description clarifies v11 configs drop it.

### Config changes (22 new files)

Each v10 JSON config has a v11 twin in `data/`. Changes propagated across all of them:

| Change | Scope |
|--------|-------|
| Replace `v10` with `v11` in filenames and all internal paths | all 22 files |
| `balance.thresholds` / `feature_balance.thresholds`: `[15,30,45,60]` → `[15,30,60,120]` | blueprints |
| `thresholds` and `per_threshold_train_paths` keys/filenames: `{45, 60}` → `{60, 120}` | training configs |
| `features_dep.nas_rate_last1d.enabled: false` | blueprints |
| Remove `origin_nasdelay_rate_last1d` from `numeric_features` | training configs |
| Add `features_dep.otp_rate_last14d.enabled: true` (blueprints), add `flightnum_od_otp_rate_last14d` to `numeric_features` | all blueprints + training configs |
| Add `features_dep.cancel_anomaly_7d.enabled: true` (prod) / `false` (smoke); add `airline_cancel_rate_anomaly_7d` to `numeric_features` in prod only | prod blueprints + training configs |

---

## Departure model feature inventory

### Categorical (4; unchanged from v10)

`Origin`, `Dest`, `DepTimeBlk`, `origin_dep_hour_weathercode`

### Numeric (production departure ≈ 78)

Identical to v10 departure numerics with these deltas:
- **Removed:** `origin_nasdelay_rate_last1d`
- **Added:** `flightnum_od_otp_rate_last14d`, `airline_cancel_rate_anomaly_7d`
- **Semantics-only change:** all `*_lateaircraft_rate_*` features (threshold aligned to >=15 min)

Net feature count: v10 76 → v11 ~78 numeric.

### Smoke-variant departure (77)

Same as production but **excludes** `airline_cancel_rate_anomaly_7d` (the smoke 1-month slice has insufficient history for the 60-day baseline rolling std).

### Delay buckets

Output bucket labels change: `[<15, 15-30, 30-45, 45-60, >=60]` → `[<15, 15-30, 30-60, 60-120, >=120]`. Model heads: `p_ge15`, `p_ge30`, `p_ge60`, `p_ge120`.

---

## New-feature definitions

### `flightnum_od_otp_rate_last14d`

- **Keys:** `(Reporting_Airline, Flight_Number_Reporting_Airline, Origin, Dest, FlightDate)`
- **Computation:** Fraction of same-identity flights in the prior 14 days with `DepDelayMinutes <= 15`. Cancelled flights excluded. Shift 1 day forward to prevent same-day leakage.
- **Range:** [0, 1]
- **Missingness:** ~7% on 1-month smoke slices (new flight-number-OD identities without 14-day history); CatBoost handles natively.
- **Signal:** Univariate AUC 0.685 at `y_dep_ge15`. After controlling for `flightnum_od_depdelay_median_last14`, partial Spearman grows with severity (+0.099 at ge15, +0.389 at ge60, +0.491 at ge120). Logistic log-loss lift +1.5% to +2.2% across v11 thresholds.
- **Rationale:** Median captures magnitude; OTP rate captures frequency / variance. The partial-correlation growth for severe buckets indicates OTP rate encodes high-variance/bimodal patterns that drive tail events.

### `airline_cancel_rate_anomaly_7d`

- **Keys:** `(Reporting_Airline, FlightDate)`
- **Computation:** `(cancel_rate_7d - cancel_rate_60d) / std(cancel_rate_60d)`. Both windows are per-carrier, `shift(1)` before rolling to prevent same-day leakage.
- **Range:** unbounded z-score; typical [-3, +3]; registered bounds [-10, 10] for extreme disruption events.
- **Signal:** Near-zero on smoke data (too little disruption variance in a calm 1-month window). Prior full-year work (`research/2026-04-13-correlations-novel-features.md`) found Spearman 0.084 vs `y_dep_ge60` and 0.156 vs cancellation label.
- **Rationale:** Operationally distinct from `carrier_delay_rate_anomaly_7d` (delays vs cancellations). Aerial meltdown events typically show cancel-rate spikes days before the worst delay day.
- **Enablement policy:** Production only. Smoke configs disable the flag because 1-month windows lack the 60-day baseline.

---

## Verification Checklist

1. **No forbidden features:** grep v11 configs for `tail_`, `aircraft_type`, `has_recent_arrival_turn`, `turn_time_hours`, `wo_slip_`, `airtime_mean_`, `taxi_out`, `gate_hold`. All must be absent.
2. **NAS drop:** grep v11 training configs for `origin_nasdelay_rate_last1d`. Must be absent.
3. **Threshold buckets:** all v11 configs have `thresholds: [15, 30, 60, 120]` (not 45, not 60 as v10's fourth entry).
4. **Late-aircraft threshold in code:** `grep 'LateAircraftDelay > 0' src/fetch_prune/features_dep.py` must return zero hits. The constant `LATE_AIRCRAFT_THRESHOLD_MIN = 15` must exist.
5. **Smoke test:** WN v11 smoke produces p_ge15/30/60/120 bundles, `flightnum_od_otp_rate_last14d` has [0, 1] range and ~90-95% coverage, `airline_cancel_rate_anomaly_7d` is absent from the smoke training matrix.
6. **Feature counts:** v11 departure ~78 (v10 = 76), v11 smoke departure ~77 (without cancel anomaly).
