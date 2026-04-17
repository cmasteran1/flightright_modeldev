# Feature Signal: Numeric vs Categorical Encoding (`dep_dow`, `sched_dep_hour`, `is_peak_hour`)

**Date:** 2026-04-16
**Label:** `y_dep_ge15`, `y_dep_ge60`
**Data slice:** WN smoke parquet (June 2025, 39,950 rows after dropna, 50k sample cap)
**Script:** `exploration/2026-04-16-numeric-vs-categorical-encoding.py`

---

## Summary

Tested whether `dep_dow` (0-6), `sched_dep_hour` (0-23), and `is_peak_hour` (0/1) perform better as CatBoost categoricals than as numerics. **The empirical answer is: keep them numeric.** All three features are already well-handled by CatBoost's native numeric splits (aided by `DepTimeBlk` which is already categorical). Switching to categorical produces no meaningful gain and a small degradation on `y_dep_ge60`.

**Verdict: Reject the encoding change.** Drop this item from the v11 plan.

---

## Univariate Signal (Grouped Delay Rates)

### `dep_dow`
Delay rate by day of week (0=Mon, 6=Sun):

| DOW | N | rate_ge15 | rate_ge60 |
|-----|-----|-----------|-----------|
| 0 (Mon) | 6,289 | 0.322 | 0.089 |
| 1 (Tue) | 5,624 | 0.309 | 0.096 |
| 2 (Wed) | 5,793 | 0.311 | 0.096 |
| 3 (Thu) | 4,664 | **0.385** | **0.121** |
| 4 (Fri) | 6,177 | 0.335 | 0.106 |
| 5 (Sat) | 4,519 | 0.280 | 0.081 |
| 6 (Sun) | 6,884 | 0.361 | 0.121 |

Non-monotonic — Thursday is peak, Saturday is trough — so the hypothesis that categorical might help is reasonable a priori. But the non-monotonicity is subtle (range 0.08, mean 0.33) and CatBoost captures it with one or two tree splits even as a numeric.

### `sched_dep_hour`
Delay rate by scheduled hour:

| Hour | N | rate_ge15 | rate_ge60 |
|------|-----|-----------|-----------|
| 0 | 611 | 0.458 | 0.133 |
| 6 | 1,969 | 0.060 | 0.014 |
| 9 | 2,871 | 0.128 | 0.023 |
| 12 | 2,018 | 0.229 | 0.050 |
| 15 | 2,094 | 0.390 | 0.104 |
| 18 | 2,302 | 0.513 | 0.176 |
| 21 | 2,414 | 0.549 | 0.206 |
| 23 | 1,484 | 0.483 | 0.176 |

Mostly **monotonic increase** from morning to late evening, with a small anomaly at midnight (hr=0 delay rate ~0.46 is higher than hr=1-5). The late-evening peak + early-morning trough is the classic delay-cascade pattern. This is exactly the shape CatBoost handles well as a numeric.

### `is_peak_hour`
Already binary, trivially representable either way.

---

## CatBoost Benchmark

Quick CatBoost (200 iter, depth 6, L2=3.0). 25% held-out test, stratified split, same seed across variants. Feature set includes all four v10 categoricals (`Origin`, `Dest`, `DepTimeBlk`, `origin_dep_hour_weathercode`) plus a representative subset of numeric features.

### `y_dep_ge15`

| Variant | Log-loss | AUC | Δ log-loss vs baseline |
|---------|----------|-----|------------------------|
| baseline (all numeric) | 0.51710 | 0.78320 | — |
| `dep_dow` categorical | 0.51623 | 0.78437 | **-0.087%** (small improvement) |
| `sched_dep_hour` categorical | 0.51860 | 0.78134 | +0.290% (degradation) |
| `is_peak_hour` categorical | 0.51702 | 0.78321 | -0.015% (noise) |
| all three categorical | 0.51775 | 0.78229 | +0.126% (degradation) |

### `y_dep_ge60`

| Variant | Log-loss | AUC | Δ log-loss vs baseline |
|---------|----------|-----|------------------------|
| baseline (all numeric) | 0.27134 | 0.80293 | — |
| `dep_dow` categorical | 0.27154 | 0.80164 | +0.074% (degradation) |
| `sched_dep_hour` categorical | 0.27174 | 0.80035 | +0.147% (degradation) |
| `is_peak_hour` categorical | 0.27168 | 0.80167 | +0.125% (degradation) |
| all three categorical | 0.27169 | 0.80073 | +0.129% (degradation) |

---

## Interpretation

1. **CatBoost handles low-cardinality ordinal numerics well.** Day of week (7 values) and hour of day (24 values) can be split cleanly by the gradient-boosted trees. The tight monotonicity of hour-of-day makes numeric encoding actually *better* — the model uses one split instead of categorical target encoding which trades some signal for noise on small subgroup sizes.

2. **Redundancy with `DepTimeBlk`.** `DepTimeBlk` is already in the feature set as a categorical (e.g., `0600-0659`, `1500-1559`). Adding `sched_dep_hour` as another categorical creates overlapping information that the model splits redundantly, slightly degrading signal-to-noise.

3. **`dep_dow` as categorical is a wash.** Marginal +0.001 AUC gain on `y_dep_ge15`, marginal -0.001 AUC loss on `y_dep_ge60`. Well within noise on a 50k sample.

4. **Sample-size caveat.** Test is on a 1-month slice. A larger test on full-year data could reveal seasonal interactions. But the direction is clear enough that further testing is not warranted — the current data gives no reason to make the change.

---

## Recommended Next Action

**Update the v11 plan to drop the encoding change.** Keep `dep_dow`, `sched_dep_hour`, and `is_peak_hour` as numeric features, matching v10. No changes to any training config's `categorical_features` or `numeric_features` lists on this item.

This simplifies v11 implementation — no need to cast these to strings in `features_dep.py`, and the existing CatBoost feature-type routing in the trainer continues to work unchanged.

---

## Artifacts

- Script: `exploration/2026-04-16-numeric-vs-categorical-encoding.py`
- Raw results CSV: `exploration/reports/2026-04-16-encoding-results.csv`
