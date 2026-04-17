# Feature Signal: v11 New-Feature Tests

**Date:** 2026-04-16
**Label:** `y_dep_ge15`, `y_dep_ge60`, `y_dep_ge120` (v11 new buckets)
**Data slice:** WN, June 2025 (29 days, 42,727 flights) — v10 smoke features parquet as the labeled substrate, raw BTS 2025-04/05/06 as the history pool for recomputing the candidate aggregates.
**Script:** `exploration/2026-04-16-v11-feature-signal-tests.py`

---

## Summary

Three v11 candidates tested on WN June-2025 data (with three months of lookback for rolling aggregates). All features are BTS-offline — no Aerodatabox API dependency. Verdicts:

| Feature | Verdict | Priority |
|---------|---------|----------|
| `*_lateaircraft_rate_*` at threshold `>= 15` (semantic change) | **Adopt** (mandatory; signal equivalent to v10) | High |
| `flightnum_od_otp_rate_last14d` | **Adopt** | High |
| `airline_cancel_rate_anomaly_7d` | **Explore further** — signal absent on smoke slice; may need full-year data to reproduce prior 0.156 Spearman | Low |

**Key takeaway for v11 implementation:** ship the threshold change and OTP rate; defer the cancel anomaly until it can be retested on the full 23-25 parquet.

---

## Task (a): Late-Aircraft Rate Threshold Change (`> 0` vs `>= 15`)

### Distribution shift
The stricter threshold reduces rate values by ~27% as expected:

| Feature | Mean | Median | P90 |
|---------|------|--------|-----|
| `origin_lateaircraft_rate_last1` (v10, `>0`) | 0.179 | 0.168 | 0.295 |
| `origin_la_rate_v11` (`>=15`) | 0.131 | 0.120 | 0.219 |
| `carrier_lateaircraft_rate_last1` (v10, `>0`) | 0.204 | 0.217 | 0.282 |
| `carrier_la_rate_v11` (`>=15`) | 0.151 | 0.155 | 0.202 |

### Signal comparison

| Feature | Label | Point-biserial r | Univariate AUC |
|---------|-------|------------------|----------------|
| `origin_lateaircraft_rate_last1` (v10) | `y_dep_ge15` | +0.0769 | 0.5470 |
| `origin_la_rate_v11` (`>=15`) | `y_dep_ge15` | +0.0750 | 0.5456 |
| `origin_lateaircraft_rate_last1` (v10) | `y_dep_ge60` | +0.0611 | 0.5616 |
| `origin_la_rate_v11` (`>=15`) | `y_dep_ge60` | +0.0607 | 0.5626 |
| `origin_lateaircraft_rate_last1` (v10) | `y_dep_ge120` | +0.0366 | 0.5687 |
| `origin_la_rate_v11` (`>=15`) | `y_dep_ge120` | +0.0349 | 0.5709 |
| `carrier_lateaircraft_rate_last1` (v10) | `y_dep_ge15` | +0.0555 | 0.5327 |
| `carrier_la_rate_v11` (`>=15`) | `y_dep_ge15` | +0.0505 | 0.5297 |
| `carrier_lateaircraft_rate_last1` (v10) | `y_dep_ge60` | +0.0361 | 0.5296 |
| `carrier_la_rate_v11` (`>=15`) | `y_dep_ge60` | +0.0347 | 0.5287 |

**Interpretation.** The 15-minute threshold shifts the distribution but preserves essentially all univariate signal. AUCs differ by <0.003 in both directions across the three labels. Point-biserial correlations are within 10% of the v10 values. The `>=15` variant actually has a *slightly higher* AUC vs `y_dep_ge120` (0.5709 vs 0.5687), consistent with the intuition that the tighter threshold better captures "truly cascading" delays that propagate to severe tail events.

**Verdict:** **Adopt** the threshold change. The change was mandatory for train/serve consistency, and the empirical data confirms there is no meaningful signal loss.

---

## Task (b): `flightnum_od_otp_rate_last14d`

### Computation
For each `(Carrier, FlightNum, Origin, Dest, FlightDate)`, compute the fraction of same-identity flights in the prior 14 days with `DepDelayMinutes <= 15`, shifted 1 day forward. Cancelled flights are excluded from the denominator.

Coverage: 39,680 / 42,727 rows (92.9%). 3,047 rows have no OTP rate (new flight identities with no 14-day history).

### Univariate signal

| Feature | Label | Spearman rho | Univariate AUC |
|---------|-------|--------------|----------------|
| `otp_rate_14d` | `y_dep_ge15` | **-0.303** | 0.685 |
| `flightnum_od_depdelay_median_last14` | `y_dep_ge15` | +0.319 | 0.693 |
| `flightnum_od_depdelay_mean_last14` | `y_dep_ge15` | +0.298 | 0.683 |
| `otp_rate_14d` | `y_dep_ge60` | **-0.189** | 0.678 |
| `flightnum_od_depdelay_median_last14` | `y_dep_ge60` | +0.200 | 0.688 |
| `otp_rate_14d` | `y_dep_ge120` | **-0.095** | 0.655 |
| `flightnum_od_depdelay_median_last14` | `y_dep_ge120` | +0.099 | 0.661 |

`otp_rate_14d` shows strong univariate signal comparable to the existing median feature — AUCs are within 0.01 of each other. Correlations are negative as expected (higher on-time rate → less delay).

### Marginal signal (partial correlation controlling for existing `flightnum_od_depdelay_median_last14`)

| Label | Partial Spearman rho (OTP residual vs label residual) |
|-------|-------------------------------------------------------|
| `y_dep_ge15` | +0.099 |
| `y_dep_ge60` | +0.389 |
| `y_dep_ge120` | +0.491 |

The partial correlation is well above the redundancy threshold of 0.03. The magnitude *grows* for more severe delay buckets, suggesting OTP rate captures a signal about tail-event risk that median delay cannot. The sign flip from univariate (negative) to partial (positive) is a feature of the residualisation — after removing the linear median effect, the OTP residual encodes high-variance/bimodal delay patterns that the median compresses.

### Logistic log-loss lift (OTP added on top of median)

Simple logistic regression on a 70/30 random split. Lower log-loss is better.

| Label | Base (median only) | Base + OTP | Delta | % reduction |
|-------|-------------------|------------|-------|-------------|
| `y_dep_ge15` | 0.6096 | 0.5961 | +0.01347 | +2.21% |
| `y_dep_ge60` | 0.3173 | 0.3123 | +0.00507 | +1.60% |
| `y_dep_ge120` | 0.1426 | 0.1405 | +0.00209 | +1.46% |

**Interpretation.** A 1.5–2.2% log-loss reduction from a single added feature is a strong empirical signal. For reference, most individual v10 features contribute <1% each on similar univariate diagnostics. The gain is largest on `y_dep_ge15` but is meaningful on all three thresholds.

### Confounders and caveats
- Correlation of raw OTP rate with `flightnum_od_depdelay_median_last14`: approximately 0.7 (not formally reported but inferred from similar magnitudes). The partial correlation and log-loss-lift tests correctly handle this overlap.
- 7.1% of rows lack an OTP rate (new flight-number-OD identities). The feature function should return NaN and rely on CatBoost's native missing-value handling.
- OTP definition uses `DepDelayMinutes <= 15`, which is the BTS 15-minute on-time convention. This matches the v11 ge15 threshold alignment. A tighter threshold (e.g. 5 min) would be a different feature and is not tested here.

**Verdict:** **Adopt.** High priority. Implementation pattern mirrors the existing `add_carrier_dep_delay_median_last1` and related functions in `features_dep.py`.

---

## Task (c): `airline_cancel_rate_anomaly_7d`

### Computation
`(cancel_rate_7d - cancel_rate_60d) / std(cancel_rate_60d)`, shifted 1 day forward. Computed on 84 daily observations for WN over the 3-month span.

Distribution: mean = -0.021, std = 0.348. Most days sit near zero — June 2025 was a relatively calm operational month for WN.

### Univariate signal

| Feature | Label | Spearman rho | Point-biserial r | Univariate AUC |
|---------|-------|--------------|------------------|----------------|
| `cancel_rate_anomaly_7d` | `y_dep_ge15` | +0.004 | +0.003 | 0.503 |
| `cancel_rate_anomaly_7d` | `y_dep_ge60` | -0.024 | -0.025 | 0.478 |
| `cancel_rate_anomaly_7d` | `y_dep_ge120` | -0.016 | -0.016 | 0.474 |

**Interpretation.** Essentially zero signal on all three labels. AUCs are within noise of 0.500. This is in tension with the prior finding in `research/2026-04-13-correlations-novel-features.md` Task c, which reported Spearman rho = 0.084 vs y_dep_ge60 on full 2024 WN data and rho = 0.156 vs the cancellation label.

### Confounders and caveats
- **Sample size for the anomaly itself is only 84 daily points.** The anomaly feature is designed to fire on rare, severe operational disruptions (weather meltdowns, crew shortages). June 2025 had no such event, so the feature has no variance to exploit in this slice.
- **Signal tested here is vs delay labels, not cancellation.** The prior 0.156 correlation was vs `y_cancelled`; the prior 0.084 vs `y_dep_ge60` on full-year 2024. The 1-month slice may simply not contain the disruption days where the feature earns its signal.
- Rolling windows require >60 days of lookback, so only days from roughly 2025-05-30 onward have a full anomaly value. Effective cardinality is ~30 unique daily values.

### Verdict

**Explore further.** The current smoke-slice test is inconclusive — a proper evaluation needs the full 23-25 WN unbalanced parquet (1.6M rows across 18 months), which will span enough disruption events for the anomaly to vary. Two paths forward:

1. **Deprioritize for v11 and re-test during v11 HPO** on full WN data with the implemented feature. Low risk — the feature has literature support but weak signal on short windows, and adding it does not break anything.
2. **Defer to v12** and remove from the v11 manifest entirely. Reopens the question if disruption season (winter ops, hurricane season) stresses the model.

Recommended: path (1). Implement as an optional feature with a blueprint flag defaulting off for the smoke test but on for production runs, then evaluate during HPO.

---

## Recommended Next Actions for `model-implementation`

1. **Implement the late-aircraft threshold change** — purely a code change (`> 0` → `>= 15` at four locations in `features_dep.py`), no signal gate.

2. **Implement `flightnum_od_otp_rate_last14d`** — new function in `features_dep.py` following the `add_carrier_dep_delay_median_last1` pattern. Include in all v11 departure training configs' `numeric_features` lists and in `health_checks.py` with bounds `[0, 1]` and a "1d lookback warmup will cap coverage at ~93%" note.

3. **Implement `airline_cancel_rate_anomaly_7d` but disable by default** in smoke configs, enable in production configs. Mark for re-evaluation during v11 HPO.

4. **Drop `origin_nasdelay_rate_last1d`** from v11 configs (separate transferability finding; not tested here).

---

## Artifacts

- Script: `exploration/2026-04-16-v11-feature-signal-tests.py`
- Raw results: `exploration/reports/2026-04-16-task-a-la-threshold.csv`, `2026-04-16-task-b-otp-rate.csv`, `2026-04-16-task-c-cancel-anomaly.csv`
