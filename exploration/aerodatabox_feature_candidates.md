# New Feature Candidates: BTS-Derived, Aerodatabox-Obtainable

**Generated**: 2026-04-01
**Airline tested**: WN (Southwest), 5.4M flights, Jan 2022 - Nov 2025
**Evaluation tiers**: Tier 1 (correlation) + Tier 2 (mutual information) + redundancy check

---

## Executive Summary

12 candidate feature families were proposed, 17 concrete features were computed and evaluated.
The analysis identified **6 strong ADD candidates** and **2 to DROP**.

### Top Recommendations (ranked by signal strength)

| Rank | Feature | Dep corr (ge15) | Arr corr (ge15) | MI (dep ge15) | Redundant? | Verdict |
|------|---------|-----------------|-----------------|---------------|------------|---------|
| 1 | `origin_nasdelay_rate_last1d` | **0.1153** | **0.1361** | 0.01300 | No (r=0.82 with existing nas_rate, but 1d window is new) | **ADD** |
| 2 | `cancel_rate_origin_last1d` | **0.0702** | **0.0674** | 0.00649 | No (max r=0.32) | **ADD** |
| 3 | `divert_rate_origin_last14d` | 0.0510 | 0.0510 | 0.01460 | No (max r=0.43) | **ADD** |
| 4 | `taxi_out_mean_origin_last1d` | 0.0657 | **0.0754** | 0.01048 | No (max r=0.40) | **ADD** |
| 5 | `gate_hold_mean_origin_last7d` | 0.0363 | 0.0381 | **0.01547** | No (max r=0.36) | **ADD** |
| 6 | `elapsed_time_ratio_last14d` | 0.0393 | **0.0875** | ~0 (dep), 0.002 (ge45) | No (max r=0.54 with Distance) | **ADD (arr)** |
| 7 | `taxi_in_mean_dest_last1d` | 0.0307 | 0.0454 | 0.00736 | No (max r=0.30) | **INVESTIGATE** |
| 8 | `schedule_buffer_min` | -0.0344 | **-0.0719** | ~0 | No | **INVESTIGATE** |
| -- | `origin_nasdelay_rate_last7d` | 0.1159 | 0.1302 | 0.01666 | **Yes** (r=0.88 with origin_nas_rate_last7) | **DROP** (redundant) |
| -- | `cancel_rate_origin_last7d/14d` | 0.02-0.03 | 0.02-0.03 | 0.01-0.015 | No | **DROP** (1d window dominates) |

---

## Detailed Analysis

### 1. origin_nasdelay_rate -- STRONG ADD (1d window)

**What**: Fraction of flights at origin with NAS (air traffic control) delay > 0 over the previous 1 day.

**Signal**: Strongest individual correlations of ALL candidates tested:
- Spearman with dep delay: **0.1529**
- Spearman with arr delay: **0.1730**
- Point-biserial with y_dep_ge15: **0.1153**
- Point-biserial with y_arr_ge15: **0.1361**

**Redundancy**: The 7d window (r=0.88 with existing origin_nas_rate_last7) is redundant. But the **1d window is new** (r=0.82, below 0.85 threshold) and captures day-over-day ATC disruption spikes the 7d smooths away.

**Aerodatabox obtainability**: Not directly available as cause-coded. Proxy via:
- Airport Delays endpoint delay index (captures ATC-driven delays)
- High overall delay + no severe weather at the airport = likely NAS cause

**Recommendation**: **ADD `origin_nasdelay_rate_last1d`** to both dep and arr models. Skip the 7d window (redundant with existing feature).

---

### 2. cancel_rate_origin -- STRONG ADD (1d window)

**What**: Fraction of flights cancelled at origin airport in the previous 1 day.

**Signal**:
- Spearman with dep delay: **0.1027** (strong)
- corr y_dep_ge15: **0.0702**
- corr y_dep_ge60: **0.0629** (signal persists at severe thresholds)
- MI (dep ge15): 0.00649

**Why this matters**: Cancellations capture a DIFFERENT disruption mechanism than delays. A day with 15% cancellations but average delay for surviving flights would be completely invisible to current features. The surviving flights on those days face gate shortages, crew reshuffling, and passenger rebooking chaos.

**Redundancy**: max r = 0.32 with carrier_origin_depdelay_mean_last1. Not redundant.

**Aerodatabox obtainability**: Directly available -- FIDS endpoint returns flight status including "Cancelled". Count cancelled / total flights at airport.

**Recommendation**: **ADD `cancel_rate_origin_last1d`**. The 7d/14d windows add diminishing signal (corr drops to 0.02-0.03) -- the 1d spike is what matters.

---

### 3. divert_rate_origin -- ADD (7d and/or 14d)

**What**: Fraction of flights diverted from origin in the previous 7 or 14 days.

**Signal**:
- Spearman with dep delay: 0.10 (7d), 0.096 (14d)
- corr y_dep_ge15: 0.046 (7d), 0.051 (14d)
- MI (dep ge15): 0.013 (7d), **0.015** (14d)

**Why this matters**: Diversions signal extreme events (severe weather, security incidents, ground stops). The 14d window captures the lingering ripple effects of disruption events that take days to fully resolve.

**Redundancy**: max r = 0.43 with days_to_strike (14d window). Interesting -- diversions and labor disruptions may co-occur, but the mechanisms differ.

**Aerodatabox obtainability**: Flight status endpoint includes diversion information.

**Recommendation**: **ADD `divert_rate_origin_last14d`**. Consider 7d too if model capacity allows.

---

### 4. taxi_out_mean_origin -- ADD (1d window)

**What**: Rolling mean taxi-out time (minutes) at origin airport.

**Signal**:
- Spearman with dep delay: **0.0755** (1d)
- corr y_dep_ge15: **0.0657** (1d)
- corr y_arr_ge15: **0.0754** (1d) -- carries through to arrival
- MI (dep ge15): 0.01048

**Why this matters**: Taxi-out captures GROUND congestion specifically -- runway queuing, de-icing, ramp congestion. A flight can push back from the gate on time but sit on the taxiway for 30+ minutes. This is a different signal from departure delay mean, which blends gate delay and ground delay together.

**Redundancy**: max r = 0.40 with origin_nas_rate_last1. Moderate but NOT redundant.

**Aerodatabox obtainability**: Derivable from flight status -- compute (actual runway time - scheduled departure time) or (wheels-off - gate departure) from recent flights at the airport.

**Recommendation**: **ADD `taxi_out_mean_origin_last1d`**. The 7d/14d windows have weaker signal.

---

### 5. gate_hold_mean_origin -- ADD (7d window)

**What**: Rolling mean ATC gate hold time at origin (from BTS TotalAddGTime field).

**Signal**:
- Spearman with dep delay: 0.0853 (7d)
- corr y_dep_ge15: 0.0363
- MI (dep ge15): **0.01547** -- third-highest MI of all candidates!

**Why this matters**: Gate holds are imposed by ATC when there's en-route or destination congestion. The flight stays at the gate with engines off instead of taxiing. This is a leading indicator of systemic air traffic flow control programs (GDPs, ground stops) that affect multiple airports simultaneously. The high MI with relatively modest correlation suggests a non-linear relationship -- gate holds may be rare but highly informative when they occur.

**Redundancy**: max r = 0.36 with carrier_origin_depdelay_mean. Not redundant.

**Aerodatabox obtainability**: Approximate from the difference between revised/actual departure time and scheduled departure time. When ATC imposes holds, airlines update departure times in their systems.

**Recommendation**: **ADD `gate_hold_mean_origin_last7d`**. The 7d window provides more stable signal than 1d for this sparse feature.

---

### 6. elapsed_time_ratio -- ADD for arrival model

**What**: Ratio of ActualElapsedTime / CRSElapsedTime for this OD pair over 14 days.

**Signal**:
- Departure: modest (corr 0.039, MI near 0)
- **Arrival: STRONG** (corr y_arr_ge15 = **0.0875**, Spearman = **0.1266**)
- Captures routes where flights consistently run long due to headwinds, ATC routing, or unrealistic schedules

**Redundancy**: r = 0.54 with Distance. Correlated but captures something different -- Distance is static, elapsed_time_ratio captures dynamic operational reality.

**Aerodatabox obtainability**: Directly from flight history -- compute actual flight duration / scheduled duration from past flights.

**Recommendation**: **ADD `elapsed_time_ratio_last14d` to ARRIVAL model only.** Signal for departure is too weak.

---

### 7. taxi_in_mean_dest -- INVESTIGATE

**What**: Rolling mean taxi-in time at destination airport.

**Signal**: Moderate for arrival (corr y_arr_ge15 = 0.045), weak for departure. MI = 0.007.

**Aerodatabox obtainability**: Available from flight status history.

**Recommendation**: **INVESTIGATE at Tier 3** (SHAP analysis) for the arrival model. Cost estimate: ~5 min with existing trained model.

---

### 8. schedule_buffer_min -- INVESTIGATE

**What**: CRSElapsedTime minus median actual flight time for this OD pair.

**Signal**: Weak for departure (corr -0.034), moderate for arrival (Spearman **-0.112**, corr y_arr_ge15 = -0.072). MI near zero suggests the relationship is very linear -- MI doesn't add much beyond what correlation captures.

**Aerodatabox obtainability**: Derivable from flight history -- scheduled vs actual durations.

**Recommendation**: **INVESTIGATE for arrival model only.** The negative correlation makes intuitive sense (more buffer = less delay), but the near-zero MI is concerning. May add signal as an interaction with weather or congestion features. Cost estimate for Tier 3 SHAP: ~5 min.

---

## Features NOT Evaluated (Aerodatabox-native, no BTS equivalent)

Two candidates require Aerodatabox data for both training AND inference:

### 9. airport_delay_index_origin (Aerodatabox native)

**What**: Aerodatabox airport delay index (0.0-5.0) composite score.

**Why promising**: Combines delay rate, cancellation rate, and delay magnitude into one signal. Could capture complex disruption patterns not visible in any single BTS feature.

**Training approach**: Synthesize from BTS data by combining: delay_rate * 2 + cancel_rate * 2 + mean_delay_magnitude_norm * 1, scaled to 0-5.

**NOT evaluated here** because it requires defining the synthetic formula first. Recommend building a synthetic index from BTS data and evaluating at Tier 1-2.

### 10. airport_delay_index_dest

Same concept for destination. Especially relevant for arrival model.

---

## Aerodatabox API Cost Impact

Adding the recommended features would require these additional API calls:

| Feature | API Endpoint | Additional Calls | Frequency |
|---------|-------------|-----------------|-----------|
| cancel_rate_origin_last1d | FIDS (airport departures) | 0 (already pulling) | Nightly |
| divert_rate_origin_last14d | FIDS | 0 (derive from existing pulls) | Nightly |
| taxi_out_mean_origin_last1d | FIDS / Flight Status | 0 (derive from existing pulls) | Nightly |
| origin_nasdelay_rate_last1d | Airport Delays endpoint | +50 calls (1 per airport) | Nightly |
| gate_hold_mean_origin_last7d | FIDS / Flight Status | 0 (derive from existing) | Nightly |
| elapsed_time_ratio_last14d | Flight History | 0 (derive from existing Tier 2 pulls) | Nightly |
| airport_delay_index | Airport Delays endpoint | +50 calls | Nightly or per-prediction |

**Total new API cost**: ~50 additional Aerodatabox calls per airline per night for the airport delay endpoint. All other features derive from data already being pulled.

---

## Next Steps

1. **Implement** the 6 ADD candidates in `features_dep.py` / `features_arr.py`
2. **Run Tier 3 evaluation** (SHAP permutation importance) on `taxi_in_mean_dest_last1d` and `schedule_buffer_min` for the arrival model
3. **Design synthetic airport_delay_index** formula from BTS columns and evaluate
4. **Retrain** with new features and measure AUC delta (Tier 4)

---

## Visualization Reference

All decile plots saved to `exploration/figures/candidate_*.png`:
- 34 plots total (17 features x 2 targets: y_dep_ge15 and y_dep_ge45)
