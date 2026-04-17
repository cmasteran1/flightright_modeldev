# Research: v11 New Feature Candidates

**Date:** 2026-04-16
**Question:** What new features — beyond the v10 set — can improve flight delay prediction at predict time, given the Aerodatabox constraint (no tail numbers, no wheels-off times)?
**Scope:** features
**Builds on:** `research/2026-04-13-approach-audit-novel-features-granular-causes.md`, `research/2026-04-13-correlations-novel-features.md`

---

## Executive Summary

1. **The highest-ROI new feature family is airport-level operational stress composites** — combining airport delay index, congestion metrics, and prior-day disruption signals into a richer operational picture than v10's single-variable rates. The AeroDataBox `/airports/delays/{icao}` endpoint now supports historical queries, making this viable for both training and inference.

2. **En-route convective weather (CAPE sampled along the great-circle route)** is the strongest untested weather signal. It is orthogonal to origin/dest weather, free via Open-Meteo, and identified in the 2026-04-13 audit as High priority but deferred due to sandbox limitations. v11 should implement it.

3. **Airline cancellation-rate anomaly (z-score vs 60d baseline)** was empirically validated in `research/2026-04-13-correlations-novel-features.md` with Spearman rho=0.156 vs cancellation label and moderate orthogonality (r=0.38) to existing `carrier_delay_rate_anomaly_7d`. Recommend adoption.

4. **Day-of-week and holiday interaction features** are cheap, universally found in the literature as top-10 importance features, and currently only present as raw `dep_dow`. Interaction terms with peak-hour and weather amplify signal.

5. **Flight-number historical on-time performance (OTP) rate** — a simple binary "was this flight number late >=15 min" rate over the last 14 days — is consistently the #1 or #2 feature in published work (Berkeley 2025 capstone, arxiv 2601.00875). v10 has delay magnitudes but not the binary OTP rate.

---

## Candidate Features

### 1. `airport_delay_index_origin` / `airport_delay_index_dest`

- **Description:** AeroDataBox composite airport stress index (0.0–5.0) queried at predict time for origin and destination. Incorporates median delay, P75/P90 delays, and cancellation count into a single score. AeroDataBox confirmed historical queries are now supported (blog: "historical-airport-and-global-delays").
- **Rationale:** This is the live-at-predict-time proxy for what v10 approximates with offline BTS aggregates (`origin_nasdelay_rate_last1d`, `cancel_rate_origin_last1d`). It is a richer, multi-factor stress signal updated in near-real-time.
- **Evidence:** Identified as High priority in 2026-04-13 audit (§2a: `airport_delay_index_last6h`). AeroDataBox documentation confirms the endpoint returns departures/arrivals delay info, `numTotal`, delay index, and now supports historical lookback.
- **Availability signal:** Plausible — AeroDataBox endpoint confirmed. Historical support means we can build training features from it. Rate limits apply (respect API cost).
- **Leakage risk:** Low. Query window must be strictly before the target flight's scheduled departure. Use a 6h-lookback ending at prediction time.
- **Priority:** **High.**
- **Next step:** `feature-transferability` to verify exact endpoint schema, rate limits, and historical data availability depth. Then `correlations-and-interactions` to test redundancy vs existing NAS/cancel/divert rates.

### 2. `enroute_cape_max`

- **Description:** Maximum CAPE (Convective Available Potential Energy, J/kg) sampled at 3 points along the great-circle route (origin, midpoint, destination) at the scheduled departure hour. Captures en-route thunderstorm risk.
- **Rationale:** v10 only has origin/dest weather. Convective weather en route drives tactical rerouting, holds, and added block time. Standard ATM knowledge; referenced in Li 2025 delay propagation review.
- **Evidence:** Audit report §2c rated this High priority. Correlations report (Task d) was blocked by sandbox limitations but estimated Spearman rho 0.15–0.25 based on literature. Open-Meteo provides free hourly CAPE at arbitrary lat/lon.
- **Availability signal:** Plausible. Open-Meteo hourly forecast API is free. Great-circle midpoint computation is trivial. The existing weather cache infrastructure can be extended.
- **Leakage risk:** None — uses day-before forecast, same as existing weather features.
- **Priority:** **High.**
- **Next step:** `correlations-and-interactions` on a 10k-flight sample (compute from Open-Meteo API). Then `model-implementation` to integrate into the weather cache pipeline.

### 3. `airline_cancel_rate_anomaly_7d`

- **Description:** Z-score of the airline's rolling 7-day cancellation rate vs its 60-day baseline: `(cancel_rate_7d - cancel_rate_60d) / std(cancel_rate_60d)`.
- **Rationale:** Cancellation spikes are operationally distinct from delay spikes. A carrier that is proactively cancelling flights is in crisis mode — crew shortages, maintenance issues, or weather recovery. The signal is orthogonal to `carrier_delay_rate_anomaly_7d`.
- **Evidence:** Empirically validated in `research/2026-04-13-correlations-novel-features.md` Task (c). Spearman rho = 0.156 vs cancellation label, 0.084 vs y_dep_ge60. Pearson r = 0.38 vs existing delay anomaly (86% unique variance). **Verdict from prior analysis: Adopt, Low-Medium priority.**
- **Availability signal:** Plausible — computed from BTS offline `Cancelled` column (already in v10 history pool). At predict time, derivable from AeroDataBox flight status (cancelled flights visible in FIDS).
- **Leakage risk:** None with 1-day shift.
- **Priority:** **Medium** (validated signal but moderate effect size).
- **Next step:** `model-implementation` — straightforward to compute alongside existing `carrier_delay_rate_anomaly_7d` in the strike/anomaly feature block.

### 4. `flightnum_od_otp_rate_last14d`

- **Description:** Binary on-time performance rate: fraction of times this specific flight number on this OD pair departed within 15 minutes of schedule over the last 14 days.
- **Rationale:** This is the single most predictive feature in multiple published studies. The Berkeley 2025 capstone found "Safe Previous Departure Delay" (which encodes this concept) at 17.4% feature importance. arxiv 2601.00875 found "historical delay rate of the flight number" was the most influential prediction feature. v10 has delay *magnitudes* (mean, median) for flight-number-OD but not the *binary OTP rate*, which captures a different signal — a flight that is consistently 5 minutes late has good OTP but moderate mean delay.
- **Evidence:** [Berkeley 2025 capstone](https://www.ischool.berkeley.edu/projects/2025/air-travel-delay-prediction-feature-engineering-and-ml-approaches); [arxiv 2601.00875](https://arxiv.org/html/2601.00875v1); consistent with BTS on-time reporting methodology.
- **Availability signal:** Plausible — computed from BTS history. Same keys as existing `flightnum_od_depdelay_*` features. At predict time, derivable from AeroDataBox flight history endpoint (new in 2025).
- **Leakage risk:** None with 1-day shift.
- **Priority:** **High.**
- **Next step:** `correlations-and-interactions` to test marginal signal beyond existing `flightnum_od_depdelay_median_last14`. Then `model-implementation`.

### 5. `dep_dow_is_monday`, `dep_dow_is_friday`, `dep_dow_is_sunday` (day-of-week dummies)

- **Description:** Binary indicators for high-delay days of the week. v10 has `dep_dow` as a single numeric (0-6) which CatBoost treats as ordinal, missing the non-monotonic pattern where Monday/Friday/Sunday have distinctly higher delay rates.
- **Rationale:** Day-of-week is universally a top-10 importance feature in delay prediction literature. Expanded dummies allow the model to capture the U-shaped weekly pattern without requiring tree depth.
- **Evidence:** PMC flight delay prediction study (2025) found day-of-week among top features. BTS data consistently shows Monday/Friday/Sunday as highest-delay days due to business travel peaks and weekend recovery patterns.
- **Availability signal:** Trivial — derived from `FlightDate`.
- **Leakage risk:** None.
- **Priority:** **Medium** (cheap to add, marginal improvement over CatBoost's ability to split on `dep_dow`).
- **Next step:** `correlations-and-interactions` to quantify marginal signal beyond raw `dep_dow`. If CatBoost already handles it via splits, skip.

### 6. `is_holiday_window`

- **Description:** Binary flag for whether the flight date falls within ±2 days of a US federal holiday or peak travel period (Thanksgiving week, Christmas week, July 4th, Memorial Day, Labor Day).
- **Rationale:** Holiday periods have systematically higher load factors, reduced slack in crew scheduling, and elevated delay rates. This is a structural signal distinct from day-of-week.
- **Evidence:** Standard in airline ops planning; BTS delay rates spike during holiday windows. Nature Scientific Reports 2024 hybrid model includes holiday features.
- **Availability signal:** Trivial — static calendar lookup.
- **Leakage risk:** None.
- **Priority:** **Medium.**
- **Next step:** `correlations-and-interactions` to test against delay labels, controlling for weather (which also peaks in summer holidays).

### 7. `origin_dep_crosswind_component_kmh`

- **Description:** Crosswind component at origin = `windspeed × sin(wind_direction - runway_heading)`, using the dominant runway heading and hourly wind forecast.
- **Rationale:** Crosswind, not gust magnitude, drives tower-imposed runway-capacity downgrades and approach delays. v10 uses raw `windgusts_kmh` which doesn't capture the directional component.
- **Evidence:** ICAO/EUROCONTROL standard ATM explanation. Audit report §2c rated Medium priority.
- **Availability signal:** Plausible. Open-Meteo provides wind direction; a static runway-heading lookup per airport is needed (OurAirports or FAA 5010 data).
- **Leakage risk:** None (forecast).
- **Priority:** **Medium.**
- **Next step:** `feature-transferability` to source runway heading data. Then `correlations-and-interactions`.

### 8. `origin_dep_visibility_below_3sm`

- **Description:** Binary flag for IFR/IMC conditions at origin at scheduled departure hour: visibility < 4,800m (3 statute miles) or cloud ceiling < 300m (1,000 ft AGL).
- **Rationale:** ATC spacing rules change fundamentally below IFR minima, causing disproportionate delays. v10 has continuous visibility but the step-function at IFR threshold captures a qualitative regime change.
- **Evidence:** Standard ATM knowledge; FAA instrument approach spacing requirements.
- **Availability signal:** Plausible. Open-Meteo provides visibility and cloud base. Already in weather cache.
- **Leakage risk:** None (forecast).
- **Priority:** **Medium-Low** (may be redundant with continuous visibility + CatBoost splits).
- **Next step:** `correlations-and-interactions` to test marginal signal.

### 9. `upstream_hub_delay_composite`

- **Description:** Weighted average of prior-day delay rates at the top 3 airports that feed the origin (weighted by historical rotation frequency from BTS). Extends v10's hardcoded 5-hub spillover to a dynamic, origin-specific upstream signal.
- **Rationale:** v10's hub spillover is per-airline and hardcoded. A dynamic upstream signal captures inter-airline propagation and non-hub-to-hub rotations. The correlations report (Task b) found simple lagged mean had partial r=0.042 after controlling for carrier median — modest but the weighted/dynamic version should do better.
- **Evidence:** Edge-based GNN papers (MDPI 2024), Aeolus benchmark, Li 2025 review all confirm graph-based propagation signals outperform handcrafted hubs.
- **Availability signal:** Plausible from BTS offline (rotation frequencies computable from tail-number history even though tails aren't used at predict time — only the aggregate weights matter). At predict time, the origin-keyed lookup table provides the score.
- **Leakage risk:** Low — uses strictly lagged data.
- **Priority:** **Medium** (implementation complexity higher than other candidates).
- **Next step:** `correlations-and-interactions` on a simple 3-feeder-airport weighted delay score.

### 10. `month_sin`, `month_cos` (cyclical month encoding)

- **Description:** Sine and cosine encoding of the month (1-12 → sin(2π·month/12), cos(2π·month/12)). Captures seasonal patterns as continuous features.
- **Rationale:** v10 has no explicit seasonal feature. `dep_dow` and weather features partially capture seasonality, but the cyclical encoding directly represents the annual delay pattern (summer thunderstorms, winter ice, holiday peaks).
- **Evidence:** Standard feature engineering technique; used in multiple recent delay prediction papers.
- **Availability signal:** Trivial — derived from `FlightDate`.
- **Leakage risk:** None.
- **Priority:** **Low** (CatBoost may learn this from weather features; test first).
- **Next step:** `correlations-and-interactions` to test marginal signal.

---

## Features Explicitly NOT Recommended

These were considered and rejected:

| Candidate | Reason for rejection |
|-----------|---------------------|
| `schedule_padding_ratio` | Pearson r=0.71 with existing `elapsed_time_ratio_last14d` — HIGH redundancy. Rejected in correlations report Task (a). |
| `upstream_airport_delay_yesterday_mean` (simple version) | Partial r drops from 0.095 to 0.042 after controlling for carrier median — signal mostly captured. Rejected in correlations report Task (b). Recommend dynamic weighted version (#9) instead. |
| Tail-number MX history | Tail not available at predict time. v9 lesson. |
| Real-time TSA wait time | Affects consumer experience, not aircraft delay. |
| Social media sentiment | Causality reversed — sentiment follows delays, doesn't predict them. |
| OPSNET facility-level features | Requires FAA ASPM registration; no API; suitable for offline enrichment only. Defer to v12 if registration succeeds. |
| NOTAM runway-closure flags | High parsing complexity; no reliable bulk API. Defer to v12. |

---

## Recommended Priority Tiers for v11

### Tier 1: Implement in v11 (validated or high-confidence signal)

| # | Feature | Signal evidence | Implementation cost |
|---|---------|----------------|-------------------|
| 4 | `flightnum_od_otp_rate_last14d` | #1 feature in Berkeley 2025 + arxiv 2601.00875 | Low — same computation pattern as existing flightnum features |
| 3 | `airline_cancel_rate_anomaly_7d` | Validated: Spearman 0.156 vs cancellation | Low — parallels existing `carrier_delay_rate_anomaly_7d` |
| 2 | `enroute_cape_max` | High literature support; orthogonal to O/D weather | Medium — requires great-circle sampling in weather cache |
| 1 | `airport_delay_index_origin/dest` | High priority in audit; AeroDataBox native | Medium — requires API integration + historical backfill |

### Tier 2: Test empirically, implement if signal confirmed

| # | Feature | Test needed | Implementation cost |
|---|---------|------------|-------------------|
| 6 | `is_holiday_window` | Signal vs weather confound | Trivial |
| 5 | `dep_dow_is_monday/friday/sunday` | Marginal vs raw `dep_dow` | Trivial |
| 7 | `origin_dep_crosswind_component_kmh` | Marginal vs windgusts | Low (need runway heading table) |
| 9 | `upstream_hub_delay_composite` | Marginal vs v10 hub spillover | Medium |

### Tier 3: Defer to v12

| # | Feature | Reason |
|---|---------|--------|
| 10 | `month_sin/cos` | Likely redundant with weather |
| 8 | `origin_dep_visibility_below_3sm` | Likely redundant with continuous visibility |
| — | OPSNET facility features | Requires FAA registration |
| — | NOTAM runway-closure | Requires parsing pipeline |
| — | GNN airport embeddings | Architectural change; v12 scope |
| — | AbsorbScore two-stage | Architectural change; v12 scope |

---

## Open Questions

1. **Does the AeroDataBox airport delay index endpoint provide enough historical depth (>=2 years) to train on?** If only recent history, we need a hybrid approach: train on BTS-derived proxy, serve AeroDataBox live.
2. **What is the API cost per query for airport delay index?** At 4 airlines × 100 airports × 730 days, how many API calls does historical backfill require?
3. **Does `flightnum_od_otp_rate_last14d` carry marginal signal beyond `flightnum_od_depdelay_median_last14`?** Need empirical test — the binary rate captures a different distributional moment but may be redundant in practice.
4. **Is the en-route CAPE signal strong enough to justify extending the weather cache pipeline?** Need the blocked correlations test (Task d from 2026-04-13) to be completed.

---

## Sources

1. [UC Berkeley Air Travel Delay Prediction (2025)](https://www.ischool.berkeley.edu/projects/2025/air-travel-delay-prediction-feature-engineering-and-ml-approaches)
2. [Prediction of airport on-time performance (arxiv 2601.00875)](https://arxiv.org/html/2601.00875v1)
3. [AeroDataBox Airport Delays API](https://aerodatabox.com/api-airport-delays/)
4. [AeroDataBox Historical Airport and Global Delays](https://aerodatabox.com/historical-airport-and-global-delays/)
5. [AeroDataBox Flight Delays with Percentiles](https://aerodatabox.com/flight-delays-with-percentiles/)
6. [AeroDataBox Flight History endpoint](https://aerodatabox.com/flight-history/)
7. [Li 2025: A Review of Research on Flight Delay Propagation](https://onlinelibrary.wiley.com/doi/full/10.1155/atr/4851103)
8. [Edge-based GNN for Network Delay Prediction (MDPI 2024)](https://www.mdpi.com/2226-4310/13/2/161)
9. [Aeolus: Multi-structural Flight Delay Dataset (arxiv 2510.26616)](https://arxiv.org/abs/2510.26616)
10. [Integrating Delay-Absorption Capability (arxiv 2512.08197)](https://arxiv.org/abs/2512.08197)
11. [Flight delay prediction: Evaluating ML algorithms (PMC 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12685205/)
12. [Hybrid ML model for flight delay (Nature 2024)](https://www.nature.com/articles/s41598-024-55217-z)
13. [Delay predictive analytics for airport capacity (ScienceDirect 2024)](https://www.sciencedirect.com/science/article/pii/S0968090X24004686)
14. [AeroDataBox Official Documentation](https://doc.aerodatabox.com/)
15. Prior research in this repo: `research/2026-04-13-approach-audit-novel-features-granular-causes.md`
16. Prior correlations in this repo: `research/2026-04-13-correlations-novel-features.md`

---

## Recommended Next Actions

1. **`feature-transferability`:** Verify AeroDataBox `/airports/delays/{icao}` historical depth, exact JSON schema, and rate limit cost. This gates feature #1 (airport_delay_index).
2. **`feature-transferability`:** Confirm AeroDataBox flight history endpoint can serve `flightnum_od_otp_rate_last14d` at predict time.
3. **`correlations-and-interactions`:** Test Tier 1 features (#2, #4) against delay labels on WN data. Test Tier 2 features (#5, #6, #7, #9) if time permits.
4. **`correlations-and-interactions`:** Complete the deferred en-route CAPE analysis (Task d from 2026-04-13 correlations report).
5. **`model-implementation`:** Implement validated Tier 1 features + mandatory v11 threshold changes (late aircraft >= 15 min, NAS delay disposition).
