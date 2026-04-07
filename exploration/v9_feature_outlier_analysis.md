# V9 Feature Outlier Analysis — Production vs Training Distribution

**Date:** 2026-04-06
**Source data:** `v9-feature-dump-2026-04-04.json` (3 flights: AA2365 DCA→MIA, UA1777 EWR→SFO, AA2318 DFW→LAX)
**Issue:** V9 models are highly pessimistic in production — predicting severe delays for flights that landed on time or nearly on time.

## Summary

We compared the feature values from 3 real production predictions (2026-04-04) against the distributions of the V9 training data (2024-01 through 2025-11). Multiple features are far outside the training distribution, all biased toward the "more delay" direction. The pessimism is caused by a combination of broken tail-number features and a genuinely bad day at certain hubs.

---

## Tier 1 — Broken Features (must fix)

These features have implementation bugs in the production serving path. Their values are mechanically wrong.

### `tail_n_legs_scheduled`
- **Production values:** 230, 390, 847
- **Training distribution:** mean=3.5–4.1, p99=6–8, max≈12
- **Z-scores:** 153x, 330x, 571x (hundreds of standard deviations out)
- **Bug:** The production code counts ALL same-carrier departures at the origin airport on that day, not the number of legs scheduled for the specific tail number. A correct value would be 3–8.
- **Impact:** This is the #2 most important feature for the dep ≥60 and ≥120 thresholds. Values 30–100x the max training value will push predictions to extreme pessimism.
- **Status:** Fixed to NaN in deployed code post-incident.

### `tail_min_turn_time`
- **Production value:** 0.25 hours (15 minutes) in all flights
- **Training distribution:** mean=1.7–1.9, p1=0.60–0.67, min≈0.5
- **Bug:** Derived from airport-wide carrier statistics, not from the actual aircraft's schedule. The value 0.25 is below every observation in training data.
- **Status:** Fixed to NaN in deployed code post-incident.

### `tail_has_tight_turn`
- **Production value:** 1 (always true) in all flights
- **Training distribution:** only ~2–10% of training rows have this set to 1
- **Bug:** Always 1 because `tail_min_turn_time` is always 0.25 (below any reasonable threshold). Derived from the broken `tail_min_turn_time`.
- **Status:** Fixed to NaN in deployed code post-incident.

### `tail_leg_num_day`
- **Production value:** 5 (always) in all flights
- **Training distribution:** mean=2.2–2.5, mode=1, values range 1–12
- **Bug:** The production code iterates up to 4 times matching ANY same-carrier arrival at the origin, not the specific tail number. It always hits the max ceiling (1 + 4 = 5). A correct value for an 8am departure might be 1-2; for a 5pm departure, 3-5.
- **Impact:** This is the **#1 most important feature** across ALL thresholds for departure models. Having it pinned at 5 (which is a high value) for every flight — including early morning flights — systematically biases predictions upward.
- **Status:** Still computed in production but suspect. Value is plausible for late-day flights but wrong for morning flights.

---

## Tier 2 — Genuinely Out-of-Distribution (real-world values, but beyond training range)

These features have correct implementations but the values on 2026-04-04 were extreme compared to training history. This can happen during network disruptions.

### Hub late-aircraft rates (multiple features)

| Feature | Flight | Value | Train p99 | Z-score |
|---------|--------|-------|-----------|---------|
| `hub_1_lateaircraft_rate_last1` | AA flights (CLT) | 0.75 | 0.43 | 6.5 |
| `hub_0_lateaircraft_rate_last1` | UA (EWR) | 0.75 | 0.36 | 7.9 |
| `hub_0_lateaircraft_rate_last14` | UA (EWR) | 0.40 | 0.21 | 6.8 |
| `hub_0_lateaircraft_rate_last7` | UA (EWR) | 0.42 | 0.24 | 6.3 |
| `hub_4_lateaircraft_rate_last7` | AA (MIA) | 0.42 | 0.23 | 8.0 |
| `hub_4_lateaircraft_rate_last14` | AA (MIA) | 0.34 | 0.20 | 7.0 |
| `hub_max_lateaircraft_last1` | all | 0.75 | 0.47–0.49 | 4.6–5.4 |

**Context:** April 3 was a bad operations day at CLT and EWR. The hub late-aircraft rates reflect real disruption, but the magnitudes exceed what the model saw during training. The 14-day rolling averages being elevated too suggests a rough period, not just a one-day spike.

### Carrier delay volatility

| Feature | Value | Train p99 | Z-score |
|---------|-------|-----------|---------|
| `carrier_depdelay_std_last14` (AA) | 51.1 | 26.1 | 10.3 |
| `carrier_depdelay_std_last7` (AA) | 48.7 | 21.1 | 8.9 |
| `carrier_depdelay_std_last14` (UA) | 45.2 | 38.8 | 6.6 |
| `carrier_depdelay_std_last7` (UA) | 47.0 | 27.9 | 6.6 |

**Context:** The standard deviation of carrier delays reflects high variance in recent operations — some flights very late, some on time. The model has never seen this level of volatility during training.

### `origin_dep_visibility_m` (AA2365 DCA→MIA)
- **Value:** 49,300m
- **Training p99:** 25,000m, z=5.0
- **Context:** This is an Open-Meteo visibility value. The API may have changed its max visibility cap, or DCA weather station is reporting differently than during training. The value is nearly 2x the training max. Since high visibility = good weather, this feature being out of range actually pushes AGAINST pessimism — but CatBoost may extrapolate unpredictably.

### `origin_daily_precip_sum_mm` (AA2318 DFW→LAX)
- **Value:** 47.8mm
- **Training p99:** 36.8mm, z=5.8
- **Context:** Heavy rain at DFW on April 4. This is a real weather event, but the intensity exceeded what training data captured.

### `turn_time_hours` (all flights)
- **Values:** 0.50–0.53 hours
- **Training p1:** 0.63–0.67
- **Context:** The production API returns very short turn times. These could be real quick-turn operations, or may reflect the same tail-number matching bug (matching any carrier aircraft, finding unrealistically short gaps).

---

## Tier 3 — Elevated but within or near training range

These features are high but not dramatically out of distribution. They contribute to pessimism additively.

| Feature | Typical value | Train p95 | Note |
|---------|--------------|-----------|------|
| `carrier_lateaircraft_rate_last1` | 0.34–0.35 | ~0.28 | Elevated but near edge |
| `dest_lateaircraft_rate_last1` | 0.45–0.47 | ~0.38 | Bad day at destinations |
| `origin_lateaircraft_rate_last14` | 0.31 | ~0.25 | DCA running hot for 2 weeks |
| `strike_severity` | 3 | varies | Present in all flights |
| `is_spring_break` | 1 | rare (~9%) | Calendar-driven, correct |

---

## Compounding Effect

The core problem is that these outliers **all push in the same direction** (toward more delay), and the most extreme outliers are in the **most important features**:

1. **tail_leg_num_day** (feature importance rank #1 across all thresholds): pinned at 5 for all flights
2. **tail_n_legs_scheduled** (rank #2 for ≥60 and ≥120): values 30–100x the training max
3. **has_recent_arrival_turn_5h** (rank #3 for ≥60 and ≥120): always 1
4. Hub late-aircraft rates: multiple features beyond p99 simultaneously

When the top 3 features by importance are all at extreme-delay levels AND dozens of secondary features are also beyond p99, the model produces extreme delay probabilities. The monotonic violation in AA2365 (P(≥30) > P(≥15)) confirms the model is in an extrapolation regime it was never trained for.

---

## Recommended Fixes

### Immediate (already done)
- `tail_n_legs_scheduled`, `tail_min_turn_time`, `tail_has_tight_turn` → set to NaN

### Needs investigation
- `tail_leg_num_day` → The current production code does not filter by actual tail number. It matches any same-carrier arrival at the origin airport. This needs to either: (a) use actual tail number from FlightLabs/AeroDataBox, or (b) be set to NaN until real tail data is available.
- `turn_time_hours` → Verify if production values reflect real aircraft turnaround or the same carrier-level matching bug.
- `origin_dep_visibility_m` → Cap at training max (25,000m) or investigate Open-Meteo API change.

### Model robustness
- Consider clipping extreme feature values to training p1/p99 at inference time as a safety net
- Consider retraining with the broken features removed entirely (since production can't reliably compute them) rather than relying on NaN imputation for the 4 most important features
- The carrier_depdelay_std features being 2–3x their p99 suggests the training data (2024–2025) may not include enough disruption days. Consider extending to 2022–2023 for more variety.

---

## Raw Outlier Data

### AA2365 DCA→MIA (30 outlier features)

| Feature | Value | Z-score | Percentile | Train p1 | Train p99 | Train Mean |
|---------|-------|---------|------------|----------|-----------|------------|
| tail_n_legs_scheduled | 230.00 | 152.9 | 100.0% | 1.00 | 8.00 | 4.07 |
| carrier_depdelay_std_last14 | 51.11 | 10.3 | 100.0% | 2.36 | 26.07 | 8.38 |
| carrier_depdelay_std_last7 | 48.73 | 8.9 | 100.0% | 1.99 | 21.08 | 7.42 |
| hub_4_lateaircraft_rate_last7 | 0.42 | 8.0 | 100.0% | 0.04 | 0.23 | 0.11 |
| hub_4_lateaircraft_rate_last14 | 0.34 | 7.0 | 100.0% | 0.04 | 0.20 | 0.11 |
| hub_1_lateaircraft_rate_last1 | 0.75 | 6.5 | 100.0% | 0.02 | 0.43 | 0.15 |
| hub_4_lateaircraft_rate_last1 | 0.47 | 6.2 | 100.0% | 0.02 | 0.26 | 0.11 |
| origin_depdelay_std_last7 | 62.42 | 5.7 | 99.7% | 1.60 | 44.61 | 9.75 |
| origin_depdelay_std_last14 | 55.30 | 5.2 | 99.5% | 2.37 | 41.91 | 10.96 |
| origin_dep_visibility_m | 49300.00 | 5.0 | 100.0% | 200.00 | 25000.00 | 20911.24 |
| hub_max_lateaircraft_last1 | 0.75 | 4.6 | 100.0% | 0.04 | 0.49 | 0.22 |
| dest_lateaircraft_rate_last7 | 0.42 | 4.3 | 99.9% | 0.03 | 0.32 | 0.13 |
| hub_1_lateaircraft_rate_last14 | 0.40 | 4.1 | 100.0% | 0.05 | 0.36 | 0.15 |
| hub_1_lateaircraft_rate_last7 | 0.42 | 4.0 | 100.0% | 0.05 | 0.37 | 0.15 |
| tail_lateaircraft_rate_last14 | 0.50 | 4.0 | 99.9% | 0.00 | 0.39 | 0.14 |
| dest_lateaircraft_rate_last1 | 0.47 | 3.7 | 99.2% | 0.01 | 0.46 | 0.13 |
| dest_lateaircraft_rate_last14 | 0.34 | 3.5 | 99.7% | 0.04 | 0.31 | 0.13 |
| hub_2_lateaircraft_rate_last1 | 0.48 | 3.5 | 100.0% | 0.02 | 0.43 | 0.14 |
| hub_1_depdelay_mean_last1 | 68.39 | 3.5 | 98.9% | 3.81 | 68.48 | 19.55 |
| origin_lateaircraft_rate_last14 | 0.31 | 3.3 | 99.6% | 0.04 | 0.29 | 0.12 |
| tail_lateaircraft_rate_last7 | 0.50 | 3.2 | 99.5% | 0.00 | 0.46 | 0.14 |
| is_spring_break | 1.00 | 3.2 | 90.9% | 0.00 | 1.00 | 0.09 |
| tail_has_tight_turn | 1.00 | 3.0 | 90.2% | 0.00 | 1.00 | 0.10 |
| carrier_lateaircraft_rate_last1 | 0.34 | 2.6 | 99.4% | 0.03 | 0.32 | 0.15 |
| hub_2_lateaircraft_rate_last14 | 0.28 | 2.4 | 100.0% | 0.05 | 0.26 | 0.14 |
| turn_time_hours | 0.53 | -0.9 | 0.1% | 0.67 | 44.37 | 10.00 |
| tail_depdelay_mean_last1 | -13.00 | -0.5 | 0.0% | 0.00 | 247.00 | 24.74 |
| flightnum_od_depdelay_mean_last1 | -13.00 | -0.5 | 0.0% | 0.00 | 287.00 | 22.51 |
| flightnum_od_depdelay_median_last1 | -13.00 | -0.5 | 0.0% | 0.00 | 287.00 | 22.51 |
| tail_min_turn_time | 0.25 | -0.4 | 0.0% | 0.67 | 19.57 | 1.71 |

### UA1777 EWR→SFO (25 outlier features)

| Feature | Value | Z-score | Percentile | Train p1 | Train p99 | Train Mean |
|---------|-------|---------|------------|----------|-----------|------------|
| tail_n_legs_scheduled | 390.00 | 329.5 | 100.0% | 1.00 | 6.00 | 3.54 |
| hub_0_lateaircraft_rate_last1 | 0.75 | 7.9 | 100.0% | 0.01 | 0.36 | 0.11 |
| hub_0_lateaircraft_rate_last14 | 0.40 | 6.8 | 100.0% | 0.04 | 0.21 | 0.11 |
| carrier_depdelay_std_last14 | 45.24 | 6.6 | 100.0% | 1.72 | 38.83 | 7.84 |
| carrier_depdelay_std_last7 | 46.99 | 6.6 | 99.0% | 1.27 | 27.91 | 6.91 |
| tail_lateaircraft_rate_last14 | 0.50 | 6.4 | 100.0% | 0.00 | 0.27 | 0.09 |
| tail_has_tight_turn | 1.00 | 6.3 | 97.5% | 0.00 | 1.00 | 0.02 |
| hub_0_lateaircraft_rate_last7 | 0.42 | 6.3 | 100.0% | 0.04 | 0.24 | 0.11 |
| hub_max_lateaircraft_last1 | 0.75 | 5.4 | 100.0% | 0.03 | 0.47 | 0.20 |
| carrier_lateaircraft_rate_last1 | 0.35 | 5.4 | 99.9% | 0.02 | 0.22 | 0.09 |
| tail_lateaircraft_rate_last1 | 1.00 | 4.5 | 98.4% | 0.00 | 1.00 | 0.09 |
| carrier_lateaircraft_rate_last14 | 0.22 | 4.2 | 100.0% | 0.04 | 0.17 | 0.09 |
| carrier_lateaircraft_rate_last7 | 0.23 | 4.1 | 100.0% | 0.03 | 0.18 | 0.09 |
| dest_lateaircraft_rate_last1 | 0.45 | 4.0 | 99.4% | 0.00 | 0.41 | 0.11 |
| origin_depdelay_std_last14 | 44.36 | 3.9 | 99.5% | 2.17 | 37.66 | 10.51 |
| dest_lateaircraft_rate_last7 | 0.32 | 3.7 | 99.8% | 0.03 | 0.28 | 0.11 |
| tail_lateaircraft_rate_last7 | 0.40 | 3.7 | 99.6% | 0.00 | 0.35 | 0.09 |
| is_spring_break | 1.00 | 3.2 | 90.9% | 0.00 | 1.00 | 0.09 |
| hub_4_lateaircraft_rate_last14 | 0.22 | 3.1 | 100.0% | 0.05 | 0.20 | 0.11 |
| CRSElapsedTime | 380.00 | 2.8 | 99.2% | 64.00 | 375.00 | 177.43 |
| hub_4_depdelay_mean_last7 | 5.53 | -1.7 | 0.0% | 6.83 | 39.83 | 15.67 |
| hub_3_depdelay_mean_last14 | 7.63 | -1.4 | 0.0% | 7.88 | 33.35 | 16.85 |
| origin_depdelay_mean_last7 | 5.53 | -1.2 | 0.7% | 5.89 | 49.38 | 17.03 |
| tail_min_turn_time | 0.25 | -0.7 | 0.0% | 0.60 | 3.95 | 1.91 |
| turn_time_hours | 0.52 | -0.6 | 0.1% | 0.63 | 16.20 | 2.16 |

### AA2318 DFW→LAX (25 outlier features)

| Feature | Value | Z-score | Percentile | Train p1 | Train p99 | Train Mean |
|---------|-------|---------|------------|----------|-----------|------------|
| tail_n_legs_scheduled | 847.00 | 570.6 | 100.0% | 1.00 | 8.00 | 4.07 |
| carrier_depdelay_std_last14 | 51.11 | 10.3 | 100.0% | 2.36 | 26.07 | 8.38 |
| carrier_depdelay_std_last7 | 48.73 | 8.9 | 100.0% | 1.99 | 21.08 | 7.42 |
| hub_4_lateaircraft_rate_last7 | 0.42 | 8.0 | 100.0% | 0.04 | 0.23 | 0.11 |
| hub_4_lateaircraft_rate_last14 | 0.34 | 7.0 | 100.0% | 0.04 | 0.20 | 0.11 |
| hub_1_lateaircraft_rate_last1 | 0.75 | 6.5 | 100.0% | 0.02 | 0.43 | 0.15 |
| hub_4_lateaircraft_rate_last1 | 0.47 | 6.2 | 100.0% | 0.02 | 0.26 | 0.11 |
| origin_daily_precip_sum_mm | 47.78 | 5.8 | 99.6% | 0.00 | 36.80 | 2.97 |
| hub_max_lateaircraft_last1 | 0.75 | 4.6 | 100.0% | 0.04 | 0.49 | 0.22 |
| origin_lateaircraft_rate_last1 | 0.48 | 4.1 | 99.7% | 0.01 | 0.42 | 0.12 |
| hub_1_lateaircraft_rate_last14 | 0.40 | 4.1 | 100.0% | 0.05 | 0.36 | 0.15 |
| hub_1_lateaircraft_rate_last7 | 0.42 | 4.0 | 100.0% | 0.05 | 0.37 | 0.15 |
| origin_depdelay_std_last14 | 42.59 | 3.7 | 99.1% | 2.37 | 41.91 | 10.96 |
| origin_daily_windspeed_max_kmh | 40.30 | 3.5 | 99.5% | 6.30 | 36.40 | 17.31 |
| hub_2_lateaircraft_rate_last1 | 0.48 | 3.5 | 100.0% | 0.02 | 0.43 | 0.14 |
| hub_1_depdelay_mean_last1 | 68.39 | 3.5 | 98.9% | 3.81 | 68.48 | 19.55 |
| origin_depdelay_std_last7 | 40.87 | 3.4 | 98.8% | 1.60 | 44.61 | 9.75 |
| tail_lateaircraft_rate_last7 | 0.50 | 3.2 | 99.5% | 0.00 | 0.46 | 0.14 |
| is_spring_break | 1.00 | 3.2 | 90.9% | 0.00 | 1.00 | 0.09 |
| origin_daily_windgusts_max_kmh | 73.40 | 3.1 | 99.2% | 15.50 | 70.90 | 36.48 |
| tail_has_tight_turn | 1.00 | 3.0 | 90.2% | 0.00 | 1.00 | 0.10 |
| carrier_lateaircraft_rate_last1 | 0.34 | 2.6 | 99.4% | 0.03 | 0.32 | 0.15 |
| hub_2_lateaircraft_rate_last14 | 0.28 | 2.4 | 100.0% | 0.05 | 0.26 | 0.14 |
| turn_time_hours | 0.50 | -0.9 | 0.0% | 0.67 | 44.37 | 10.00 |
| tail_min_turn_time | 0.25 | -0.4 | 0.0% | 0.67 | 19.57 | 1.71 |
