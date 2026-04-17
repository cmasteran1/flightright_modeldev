# Research: Approach Audit, Novel Pre-Departure Features, and Granular Delay Causes

**Date:** 2026-04-13
**Question:** (1) Is our consumer-facing travel-risk approach sound? (2) What novel features obtainable days before departure could improve v10? (3) Where can we find granular, specific delay-cause data rather than the five BTS buckets?
**Scope:** approach audit | features | data-source discovery

---

## Executive Summary

1. **The product thesis is sound but the modeling framing is slightly misaligned.** v10 is optimized as a per-flight on-day delay predictor, but the consumer question is really "what's the *travel risk* of this itinerary at booking / day-before-check-in?" That framing argues for (a) *itinerary-level* (multi-leg, missed-connection) risk, (b) *outcome-level* predictions users actually care about (missed connection, cancellation, >2 h disruption, rebooking chaos), and (c) *calibrated probabilistic* output with explicit uncertainty, not just ordinal thresholds. See Audit below.

2. **The strongest untapped pre-departure signals are structural, not aircraft-specific.** Published work and FAA reporting converge on: ATC-facility staffing/stress, airport delay-index composites, NOTAM/runway-closure state, time-of-day convective weather indices at hub airports *en route* (not just O/D), crew-duty stress proxies, and graph-based upstream-airport delay propagation scores. Most are obtainable free or cheaply days in advance.

3. **Delay-propagation graph features (edge/node GNN embeddings or simpler summary statistics) consistently beat "more features of the same kind"** in 2024-2025 papers. v10 approximates this with hub spillover; a proper *airport-pair upstream delay graph* at 24-48 h horizon is a high-ROI next experiment.

4. **For truly granular delay causes, three data sources are underused in this repo:** (a) FAA OPSNET/ASPM, which attributes reportable delays to *Weather / Volume / Runway / Equipment / Other* with secondary-cause decomposition per facility per hour; (b) European CODA IATA delay-code data (the 80+ two-digit IATA AHM 730 codes: e.g., 36=fueling, 15=late boarding, 63=late crew boarding, 41=tech/aircraft defect); (c) NTSB CAROL narrative text for mechanical causes. None are fully BTS-aligned for US commercial flights, but each gives qualitative pattern data to guide feature engineering.

5. **New probabilistic-prediction methods (quantile regression, mixture density networks) reach MAE <15 min on per-flight delay and produce calibrated intervals.** This is worth considering as a v11 direction alongside the ordinal-binary stack CatBoost approach.

---

## 1. Audit of Current Approach

### What v10 does well

- **Honest post-mortem discipline.** The v9 → v10 rewrite (dropping tail-number features, wheels-off-derived aggregates, mean→median for 1-day baselines) is the right response to the v9 outlier-sensitivity failure and shows disciplined iteration.
- **Production-source discipline.** Locking v10 to what AeroDataBox + Open-Meteo + BTS-derived offline aggregates can actually serve prevents a recurrence of v9's "trained on features we cannot produce at predict time" problem.
- **Calibrated ordinal head.** Isotonically calibrated binary heads at 15/30/45/60 stitched into a severity distribution is sensible and gives the product team something to threshold.
- **Leakage hygiene.** Explicit 1-day-shift rules, forbidden-substring audit, and the explicit "gate-based not wheels-off" distinction for `ActualElapsedTime` are well thought out.

### Where the current approach is weak or misaligned with the stated product goal

The stated goal is: *help regular consumers understand the travel risks they face before check-in.* Read strictly, that is not the same as "predict the delay bucket for a single leg." Three gaps:

**(a) Wrong unit of analysis.** Most consumers buy itineraries, not legs. A 20-minute departure delay at CLT that misses a UA tight connection at IAH is catastrophic; a 90-minute delay on a non-connecting non-stop is annoying. The v10 models predict legs, not itineraries. Recommended: add an *itinerary-risk scorer* on top of the per-leg models that combines (i) P(leg1 arrives late by X), (ii) minimum-connect-time (MCT) at the connection airport, (iii) P(leg2 departs on time | leg1 arrives late), and (iv) rebooking-inventory proxy (number of same-airline alternatives in a ±4 h window at that OD pair). Even a hand-built scorer is a step change for the consumer.

**(b) Wrong target family.** Consumers don't care about "was this ≥15 min late?" They care about: "will I miss my connection", "will this be cancelled", "will I be diverted", "will my delay exceed 2 h (making rebooking realistic) or 6 h (disruptive but not overnight)." The current ≥15/≥30/≥45/≥60 thresholds are fine for operations research but only partly map to consumer utility. Adding heads at the *consumer-meaningful* thresholds (miss-connection bool, ≥120 min, ≥360 min, cancellation-or-≥240) may help the product team without re-architecting.

**(c) Uncertainty communication.** A calibrated severity distribution is information-rich but awkward for consumers. The literature on probabilistic forecasting (see §3) supports presenting a prediction interval ("likely 0-25 min late; 10% chance of ≥120 min; 2% chance cancelled") rather than a bucket. Mixture density networks and quantile regression both produce calibrated intervals with MAE <15 min in published work (arc.aiaa.org 2023; mdpi.com 2021).

### Alternative architectural options worth considering

- **Two-stage / causal-style split.** Stage 1 predicts upstream-system "stress" (airport delay index, NAS delay rate, carrier delay-rate anomaly) at 24 h horizon. Stage 2 uses that stress as an input to per-flight risk. This matches the Aviate AI / FlightAware mental model and the arxiv "delay-absorption capability" two-stage paper (arxiv 2512.08197), which shows an explicit "AbsorbScore" as an intermediate target improves downstream delay prediction.
- **Graph-based model for hub spillover.** v10 handcrafts hub features; recent GNN papers (2024) outperform tabular on network-level delays by modeling the airport graph directly. Not recommended as a from-scratch rewrite, but *GNN-derived embeddings* could be precomputed offline for each origin and merged in as 16-32 extra numerics.
- **Per-airline stratification is expensive; consider an airline embedding in a single model.** The 4×3 = 12 CatBoost trainers are operationally painful and prevent cross-airline transfer (e.g., WN's delay pattern at DEN informs UA's delay pattern at DEN). A single model with a learned airline embedding is a trade-off worth evaluating for v11.

### What *not* to do

- Do not re-introduce tail-number features unless and until AeroDataBox exposes registration for upcoming flights. The v9 failure was not "tail features are bad" — it was "tail features computed on stale/fake tails at predict time are bad." Don't chase that ghost without a reliable source.
- Do not chase features that correlate highly *ex post* with delay if they are only observable on the day of. Several candidates below are explicitly flagged for leakage.

---

## 2. Candidate Features (Novel, Obtainable Days Before Departure)

Grouped by *where the signal comes from*. Each entry follows the skill's candidate-feature template.

### 2a. ATC / Airspace Structural Signals

#### `tracon_staffing_stress_forecast_24h`

- **Description:** Composite stress score for the TRACON governing the origin airport, derived from FAA staffing-level indicators and recent ground-delay-program history.
- **Rationale:** The FAA has publicly reported that 19 of 30 largest facilities operate below 85% of their staffing targets, 19 facilities account for ~40% of all NAS delays, and Q4-2025 saw a 14.3% YoY increase in ATC-attributable delays (Fortune 2025; Altitudes Magazine 2026). Structural understaffing is a slow-moving signal available days ahead.
- **Evidence:** [FAA Workforce Plan 2025-2028](https://www.faa.gov/sites/faa.gov/files/fy25-air-traffic-controller-workforce-plan_0.pdf); [USAFacts controller shortage](https://usafacts.org/articles/is-there-a-shortage-of-air-traffic-controllers/); [Fortune Nov 2025](https://fortune.com/2025/11/01/faa-air-traffic-control-facilities-staffing-shortages-government-shutdown-flight-delays/).
- **Availability signal:** Plausible from trusted free sources. FAA NAS status feed (nasstatus.faa.gov) and OPSNET publish facility-level delay indices. Not a direct AeroDataBox field.
- **Leakage risk:** None. Staffing levels are trailing indicators; operating ground delay programs (GDPs) are publicly advertised in advance.
- **Priority:** **High.**
- **Next step:** `feature-transferability` to confirm nasstatus.faa.gov scrapable / stable; then `correlations-and-interactions` vs. `y_dep_ge60` and cancellations.

#### `active_gdp_or_ground_stop_24h`

- **Description:** Binary/ordinal flag for whether the origin or destination airport has an active or announced Ground Delay Program, Ground Stop, or AFP (Airspace Flow Program) within a ±24 h window around the flight.
- **Rationale:** GDPs/GSs are the single most reliable explicit signal of impending delay — the FAA is literally metering arrivals. They are announced hours ahead.
- **Evidence:** [FAA NAS Status](https://nasstatus.faa.gov/) publishes these in real time. Covered in most delay-prediction reviews (e.g., Li 2025 Journal of Advanced Transportation).
- **Availability signal:** Plausible, free (FAA feed). AeroDataBox has an "airport delays" endpoint that returns a composite delay index correlated with but not identical to NAS GDP state.
- **Leakage risk:** Low. Announced in advance; Ground Stops are shorter-fuse but still observable before the affected flight's departure.
- **Priority:** **High.**
- **Next step:** `feature-transferability` to check AeroDataBox airport-delays endpoint payload; `correlations-and-interactions` on historical NAS status archives.

#### `airport_delay_index_last6h` (origin and destination)

- **Description:** AeroDataBox `GET /airports/delays/{icao}` returns a composite airport stress index (0-5) plus median / P75 / P90 delays and cancellation count.
- **Rationale:** This is a near-real-time proxy for the composite of NAS stress + weather + ground handling capacity that v10 only approximates with offline BTS aggregates. It is explicitly available at predict time.
- **Evidence:** Noted in `v10_feature_transferability_guide.md` §5c as a candidate proxy; not yet trained on.
- **Availability signal:** Plausible (AeroDataBox); v10 already identified the endpoint.
- **Leakage risk:** None if queried <6 h before scheduled departure. Medium if queried on the day of, since it already folds in that day's disruption — be careful the offline training label never overlaps with the query window.
- **Priority:** **High.**
- **Next step:** `feature-transferability` to lock the exact endpoint schema; `correlations-and-interactions` against `origin_nasdelay_rate_last1d` (redundancy check).

### 2b. NOTAMs and Airport Physical State

#### `origin_runway_closure_notam_active`, `dest_runway_closure_notam_active`

- **Description:** Binary flag (and minute-count) indicating whether a runway at the origin or destination is closed by active NOTAM during the flight's scheduled window. Additional: whether the closed runway is the dominant wind-aligned runway for the day's forecast wind direction (interaction with weather).
- **Rationale:** A closed primary runway can cut airport capacity 30-50%. NOTAMs are issued well in advance (often weeks for planned construction) and are publicly available via the FAA NOTAM Search / FNS.
- **Evidence:** FAA NOTAM guidance (faa.gov chap5_section_1); FAA runway-construction advisory circular AC 150/5200-28F/G.
- **Availability signal:** Plausible but non-trivial to parse free. FAA NOTAM Search is text-heavy. Third-party libraries (e.g., `fns-aim-faa` scrapers) exist. AeroDataBox does not expose NOTAMs cleanly.
- **Leakage risk:** None.
- **Priority:** **Medium-High** (signal strong; parsing cost non-trivial).
- **Next step:** `feature-transferability` to evaluate NOTAM parsing pipelines; if feasible, `correlations-and-interactions` on historical runway-closure days.

### 2c. Weather — Beyond Origin/Destination

#### `enroute_convective_index_max`

- **Description:** Maximum CAPE / convective SIGMET coverage fraction along the great-circle route (not just origin and destination) at scheduled flight time, sampled every ~50 nm.
- **Rationale:** Convective weather *en route* drives tactical rerouting and adds block time / holds that are not visible in O/D weather. The v10 arrival model's `dest_arr_cape` and `origin_dep_cape` capture endpoints only; thunderstorm lines over the Ohio Valley don't show up at MIA or LAX.
- **Evidence:** Standard ATM knowledge; delay-propagation reviews (Li 2025) list en-route weather as a distinct bucket.
- **Availability signal:** Plausible — Open-Meteo CAPE forecasts at arbitrary lat/lon are free; great-circle sampling is trivial.
- **Leakage risk:** None (forecast, day-before).
- **Priority:** **High.** Cheap to implement, plausibly orthogonal to existing origin/dest weather.
- **Next step:** `correlations-and-interactions` on a small en-route CAPE aggregate vs. delay residual after controlling for origin/dest weather.

#### `origin_dep_crosswind_component_kmh`

- **Description:** Crosswind component = forecast wind speed × sin(wind_dir − runway_heading), using the dominant runway heading at the origin and the hourly wind-direction forecast.
- **Rationale:** Crosswind, not gust magnitude, drives tower-imposed runway-capacity downgrades. v10 uses windgusts; crosswind is a more mechanistic signal.
- **Evidence:** ICAO / EUROCONTROL standard explanation (ansperformance.eu). Not flight-delay-paper-famous, but well known in ATM.
- **Availability signal:** Plausible. Open-Meteo provides wind direction; a static runway-heading lookup per airport is manageable.
- **Leakage risk:** None.
- **Priority:** **Medium.**
- **Next step:** `feature-transferability` (runway-heading table availability); `correlations-and-interactions` vs. wind_gust residual.

#### `ceiling_visibility_imc_flag`

- **Description:** Binary flag for forecast IMC conditions at origin / destination at the scheduled hour (ceiling <1000 ft AGL or visibility <3 sm).
- **Rationale:** ATC spacing rules fundamentally change below IFR minima, which drives larger delays than the marginal rise in existing weather features would suggest.
- **Availability signal:** Plausible (Open-Meteo provides ceiling and visibility; derivation is a threshold).
- **Leakage risk:** None (forecast).
- **Priority:** **Medium.**

### 2d. Structural / Schedule Signals

#### `schedule_padding_ratio`

- **Description:** `CRSElapsedTime / route-minimum-block-time-baseline`. The route-minimum can be estimated as the 10th percentile of `ActualElapsedTime` for that OD pair in the last 180 days.
- **Rationale:** A heavily padded schedule absorbs upstream stress and lowers on-time-arrival risk even when pushed back at the gate. Schedule padding is a strategic airline choice; it varies by route and carrier.
- **Evidence:** Standard in academic flight-delay literature; Aeolus includes `scheduled flight durations` among its 14 continuous features (arxiv 2510.26616).
- **Availability signal:** Plausible. Both inputs are already in v10 feeds.
- **Leakage risk:** None.
- **Priority:** **Medium.** Partially redundant with the new `elapsed_time_ratio_last14d` (arrival) but is a departure-side signal and available from the schedule alone without BTS history.
- **Next step:** `correlations-and-interactions` vs. both `y_dep_ge15` and `y_arr_ge15`.

#### `rebooking_inventory_sameday_±4h`

- **Description:** Count of same-airline, same-OD scheduled flights within ±4 h of the target flight's scheduled departure. (Different from congestion — congestion *hurts*; rebooking inventory *helps the consumer*.)
- **Rationale:** Not a delay predictor per se. But it is *the single most actionable risk communication input for the consumer*: a delayed 0600 flight with four same-airline alternatives before noon is low-risk to the trip; a delayed last flight of the day is high-risk. This is a *product-layer feature* that should be exposed alongside delay probability.
- **Availability signal:** Plausible. AeroDataBox schedule endpoints return same-airport same-airline schedules.
- **Leakage risk:** None.
- **Priority:** **High** — but as a *product-layer* score, not necessarily a model feature. Discuss with product.
- **Next step:** Surface to the product team as a companion metric. Optionally feed into an itinerary-risk wrapper.

### 2e. Propagation and Network Signals (Graph)

#### `upstream_origin_delay_graph_score`

- **Description:** For the origin airport, an importance-weighted sum of the previous few hours' delay rates at airports that frequently *feed* that origin (aircraft rotations end here from there). Edge weights derived from historical shared-aircraft graphs.
- **Rationale:** v10's hub-spillover features handcraft this for a hardcoded 5-hub list per airline. A graph-based origin score generalizes it. Aeolus builds exactly this as a flight-network-graph modality; multiple 2024-2025 GNN papers (MDPI edge-based GNN, CausalNet, AAGNN, CHAMFormer) show graph features outperform tabular-only baselines on network-wide delay tasks.
- **Evidence:** [Aeolus (arxiv 2510.26616)](https://arxiv.org/abs/2510.26616); [Edge-based GNN, MDPI 2024](https://www.mdpi.com/2226-4310/13/2/161); [Li 2025 review](https://onlinelibrary.wiley.com/doi/full/10.1155/atr/4851103).
- **Availability signal:** Plausible from BTS offline (same-tail sequencing is derivable there even if we can't serve tail at predict time — you only need the *aggregate edge weight*, not the tail). Serving live at predict time requires a lookup table keyed by origin airport.
- **Leakage risk:** Low. Use strictly lagged edges.
- **Priority:** **High.**
- **Next step:** `correlations-and-interactions` on a simple weighted-sum version before going full GNN.

#### `delay_absorption_probability` (AbsorbScore)

- **Description:** Predicted probability that a given flight will absorb any upstream delay, conditional on schedule slack, time of day, airline, origin/dest, and weather. Two-stage architecture: absorbance model → delay model.
- **Rationale:** Directly from [arxiv 2512.08197](https://arxiv.org/abs/2512.08197). The authors show treating upstream delays as static inputs to an XGBoost is strictly worse than feeding in a learned absorption-probability first.
- **Availability signal:** Plausible but requires training an auxiliary model. All inputs (schedule, weather, airline) are already pre-predict.
- **Leakage risk:** Low if the absorption label is defined carefully from *past* legs only.
- **Priority:** **Medium** — architectural change rather than a drop-in feature.
- **Next step:** `research` deeper (read the paper in full) → `model-implementation` prototype on one airline.

### 2f. Crew and Operational Signals (Indirect)

Direct crew-duty-time data is not public. But there are indirect proxies:

#### `last_flight_of_day_ratio_tight_connection`

- **Description:** Heuristic flag: is this flight the 2nd-to-last or last flight of the day for its OD + airline, following a long duty-day pattern observable in the schedule (e.g., the airline's typical crew-pairing chain ends here)?
- **Rationale:** FAR Part 117 duty limits mean end-of-day crews are more likely to time out on moderate delays, causing cancellations. The GAO has extensively documented crew scheduling impact on cancellations (GAO-08-1041R).
- **Evidence:** [GAO-08-1041R Crew Scheduling](https://www.govinfo.gov/content/pkg/GAOREPORTS-GAO-08-1041R/html/GAOREPORTS-GAO-08-1041R.htm); [14 CFR Part 117](https://www.ecfr.gov/current/title-14/chapter-I/subchapter-G/part-117).
- **Availability signal:** Plausible. Full-day schedule data from AeroDataBox.
- **Leakage risk:** None.
- **Priority:** **Medium.** The signal is there but the heuristic is imperfect.
- **Next step:** `correlations-and-interactions` on end-of-day schedule slot vs. cancellation rate.

#### `airline_operational_anomaly_score_7d`

- **Description:** Z-score of the airline's rolling 7-day cancellation rate vs. its 60-day baseline. Orthogonal to `carrier_delay_rate_anomaly_7d` in v10 — that measures *delays*; this measures *cancellations*, which lag and indicate system-level crew/maintenance stress.
- **Rationale:** Southwest Dec-2022, American Oct-2019 — cancellation-rate anomaly jumps days before the worst day of an operational meltdown.
- **Availability signal:** Plausible from BTS offline + near-real-time AeroDataBox status.
- **Leakage risk:** None with 1-day shift.
- **Priority:** **High.**
- **Next step:** `correlations-and-interactions`.

### 2g. Features to explicitly push back on (leakage or infeasibility)

- **"Incoming aircraft's actual arrival time"** — leaks if queried on day-of. Safe only if queried <90 min before scheduled departure, which changes the prediction horizon contract.
- **"Tail-number MX history"** — tail not available at predict time (v9 lesson).
- **"Real-time TSA wait time"** — affects consumer experience but not departure delay of the aircraft itself; avoid mixing these into the delay model.
- **"Social media sentiment about airline"** — correlates with delays *after* they happen (causality reversed); do not use.

---

## 3. Granular Delay-Cause Data Sources

The BTS 5-bucket classification (Carrier / Weather / NAS / Late Aircraft / Security) is too coarse for root-cause analysis. Here are the underused granular sources, ranked by usefulness.

### 3a. IATA AHM 730 delay codes — the gold standard for granularity

The International Air Transport Association (IATA) maintains 80+ two-digit numeric codes covering every conceivable delay cause. Examples relevant to this project:

- **11** — late check-in, acceptance after deadline
- **15 (PS)** — boarding discrepancies / missing pax
- **21-26** — cargo handling delays (26 = late preparation in warehouse)
- **31-37** — ramp handling (32 = late baggage, **36 (GF) = fueling/defueling**)
- **41 (TD)** — **aircraft defect / technical**
- **42** — scheduled maintenance, late release
- **43** — non-scheduled maintenance
- **51-52** — damage to aircraft / EDP failure
- **61-69** — **crew / flight operations (63 = late crew boarding, 67 = crew awaiting aircraft/ground transport)**
- **71-77** — weather sub-causes
- **81-86** — ATFM / ATC (83 = ATFM en-route, 84 = ATFM due to weather at destination)
- **89** — airport / government authorities
- **91-99** — reactionary / miscellaneous (91 = load connection, **93 = aircraft rotation — the specific, numeric "late aircraft" code**)

**Availability for US commercial flights:** IATA codes are *not* routinely published by US carriers to BTS. They are the standard in Europe (EUROCONTROL CODA publishes anonymized IATA-code breakdowns quarterly) and internally at almost every airline. A commercial route to them is OAG's flight-status feed, which includes carrier delay reason text that often maps to IATA codes.

**Sources:** [IATA delay codes (Wikipedia summary)](https://en.wikipedia.org/wiki/IATA_delay_codes); [Cosmos practical guide](https://usecosmos.com/blog/decoding-delays-a-practical-guide-to-iata-delay-codes); [Aviation Intelligence Portal](https://ansperformance.eu/definition/iata-delay-codes/).

### 3b. FAA OPSNET — reportable NAS delays with facility and secondary cause

OPSNET is the FAA's own reportable-delay system. A reportable delay is any ATC-induced delay ≥15 min. Each record has:

- Facility (ARTCC / TRACON / tower) constraining the flight
- **Primary cause** (JO 7210.55F, 5 categories: Weather / Volume / Runway / Equipment / Other)
- **Secondary cause** (further decomposition within primary)
- Traffic Management Initiative used (GDP / GS / AFP / MIT / CFR)
- Duration

OPSNET is granular and US-wide. It covers *ATC-induced* delays only, not airline-induced. Detail Data Download is available through ASPM.

**Sources:** [OPSNET Detail Data Download](https://www.aspm.faa.gov/aspmhelp/index/OPSNET_Delays__Detail_Data_Download.html); [Delay By Cause](https://www.aspm.faa.gov/aspmhelp/index/OPSNET_Delays__Delay_By_Cause_Report.html); [Delay By Secondary Cause](https://www.aspm.faa.gov/aspmhelp/index/OPSNET_Delays__Delay_By_Secondary_Cause_Report.html); [ASPM overview](https://www.aspm.faa.gov/aspmhelp/index/Aviation_System_Performance_Metrics_(ASPM).html).

**Practical note:** OPSNET requires ASPM registration; downloads are CSV. It is *per-facility per-hour*, not per-flight. This makes it excellent for enriching *airport-level* features (which is what we need for pre-departure prediction) but not per-flight attribution.

### 3c. NTSB CAROL — narrative mechanical causes

For the "did the aircraft fail safety inspection" question: the NTSB publishes full investigation narratives for civil aviation accidents and selected incidents from 1962 onwards. CAROL (Case Analysis and Reporting Online) has free-text probable-cause findings. This is the only public source that gets to "engine #2 fan blade liberated during takeoff roll" granularity for US commercial aviation. However, (a) it is limited to NTSB-investigated events (serious incidents and accidents only, not normal daily delays), and (b) it's text data requiring NLP.

**Sources:** [NTSB Aviation Investigation Search](https://www.ntsb.gov/Pages/AviationQueryv2.aspx); [NTSB Accident Data index](https://www.ntsb.gov/safety/data/Pages/Data_Stats.aspx); the [Kaggle NTSB synopses dataset up to 2023](https://www.kaggle.com/datasets/khsamaha/aviation-accident-database-synopses) is a convenience copy.

**Practical note:** Use CAROL only to *calibrate intuition* about mechanical failure distributions, not to train delay models. A tail number that appears in a CAROL narrative from 2023 says nothing predictive about a tail number flying today unless you happen to have a mechanical-fleet panel. Treat this as ethnography.

### 3d. Aeolus — academic dataset with schedule + weather + delay-chain structure

The NeurIPS 2025 Datasets & Benchmarks paper *Aeolus* provides 50 million flights with 22 features across three modalities (tabular, flight-chain, flight-network graph). Notably, its label is still a conventional delay / on-time outcome, not IATA-coded causes — but its flight-chain modality is a ready-made testbed for upstream-delay-graph features.

**Citation:** Xu, L. et al. (2025). *Aeolus: A Multi-structural Flight Delay Dataset*. NeurIPS 2025 Datasets & Benchmarks Track. [arxiv 2510.26616](https://arxiv.org/abs/2510.26616). Code: https://github.com/Flnny/Delay-data.

### 3e. Summary: what level of granularity is realistically obtainable?

| Source | Granularity | US commercial coverage | Effort | Usefulness to this project |
|---|---|---|---|---|
| BTS 5-category (current) | Airline-reported bucket minutes | 100% of reporting carriers | Already have | Baseline |
| OPSNET primary+secondary | Per-facility per-hour, 5+sub causes | ATC-induced only | Low (CSV download) | **High** for feature engineering |
| IATA AHM 730 codes | 80+ specific causes per flight | Near-zero public for US | High (commercial feed, e.g., OAG) | **Highest** if purchasable |
| NTSB CAROL narrative | Per-incident full text | Investigated incidents only | Medium (NLP) | Low for training, medium for intuition |
| Aeolus graph modality | Tabular + chain + graph | Global flights | Low (GitHub) | Medium for method benchmarking |

**Recommendation:** Pull OPSNET Detail by secondary cause for the five v10 hubs per airline, cross-reference with our BTS NASDelay flags for 2023-2025, and see what secondary-cause patterns dominate. This is the highest-ROI granular-cause study we can do without paying OAG or IATA.

---

## 4. Modeling Strategy Notes

### Probabilistic prediction and calibration (relevant to audit point 3)

Two directly applicable works:

- *Probabilistic Pretactical Arrival and Departure Flight Delay Prediction with Quantile Regression* (Journal of Air Transportation, 2023). Uses gradient-boosted quantile regression; reports calibrated prediction intervals and supports per-flight uncertainty at 24+ h horizon.
- *Probabilistic Flight Delay Predictions Using Machine Learning* (MDPI Aerospace 2021). Compares Mixture Density Networks and Random Forest regression; reports MAE <15 min and demonstrates downstream use for flight-to-gate assignment with 74% fewer conflicts vs. deterministic baseline.

**Relevance to flightright-modeldev:** v11 could add a quantile-regression head (same CatBoost backbone; CatBoost supports quantile loss natively) to emit a (P10, P50, P90) triple per flight, in addition to the ordinal bucket heads. This is additive work, not a rewrite.

### Delay-propagation graph / GNN

The 2024-2025 review literature is unanimous that delay propagates through three mechanisms: (1) aircraft rotation, (2) passenger/crew connection, (3) airport congestion. Since AeroDataBox blocks us from tail numbers, mechanism (1) is partially off-limits at the per-flight grain but fully available at the *airport-pair* grain (aggregate rotation flows are public from BTS history). A GNN is overkill as v11 — a weighted upstream-airport delay score is the 80/20 move.

### Datasets worth bookmarking

- [Aeolus](https://arxiv.org/abs/2510.26616) — code + data at Flnny/Delay-data.
- Published features list: 8 categorical + 14 continuous; largely overlapping with v10 but notably *does not use BTS causes* as either features or labels.

---

## 5. Paper Digests

### Xu et al. (2025). *Aeolus: A Multi-structural Flight Delay Dataset*. NeurIPS 2025 D&B.

- **Citation:** Xu, L. et al. (2025). [arxiv.org/abs/2510.26616](https://arxiv.org/abs/2510.26616).
- **Problem:** Lack of a standard multi-modal benchmark for flight delay prediction that exposes tabular, sequence (flight chain), and graph (flight network) structure on the same 50 M-flight substrate.
- **Method:** Aligns three modalities from the same raw flights, with a temporal 6:2:2 split and explicit leakage prevention. Offers regression, classification, and uncertainty tasks. 8 categorical + 14 continuous features per flight.
- **Results:** Baselines reported across modalities; the paper's contribution is the dataset, not a SOTA model. (I did not find a leaderboard number to cite without fabricating.)
- **Relevance to flightright-modeldev:**
  - Benchmarks future v10/v11 feature-engineering decisions against a common substrate.
  - Flight-chain modality could serve as pretraining for a graph-feature extractor that we import into v11 as pre-computed origin/destination embeddings.
  - Their choice of only 22 features says our 80-feature v10 is *information-rich*, not feature-poor — the return is diminishing on more handcrafted features; graph structure is the next axis.

### Gui et al. (2025, Arxiv preprint). *Integrating Delay-Absorption Capability into Flight Departure Delay Prediction*.

- **Citation:** [arxiv.org/abs/2512.08197](https://arxiv.org/abs/2512.08197).
- **Problem:** Traditional ML treats upstream delay as a static input; in reality whether it *propagates* depends on a flight's absorption capability (schedule slack, time-of-day, crew/aircraft state).
- **Method:** Two-stage. Stage 1: CatBoost classifier predicts P(absorb upstream delay) = AbsorbScore. Stage 2: XGBoost classifier predicts P(departure ≥15 min late) using AbsorbScore as an extra feature.
- **Results:** The paper reports improved precision on ≥15 min classification relative to a single-stage XGBoost baseline (exact numbers not cited here to avoid paraphrase error — read before implementing).
- **Relevance to flightright-modeldev:**
  - Directly motivates adding an absorption-style intermediate target to v11.
  - Compatible with the existing CatBoost stack.
  - **Handoff:** re-read end-to-end before implementation (open question #2 below).

### Monteiro et al. (2024) *Edge-based GNN for Network Delay Prediction*. MDPI Aerospace 13(2), 161.

- **Citation:** [mdpi.com/2226-4310/13/2/161](https://www.mdpi.com/2226-4310/13/2/161).
- **Problem:** Airport-to-airport delay propagation through directional flow paths, captured via edge rather than node representation.
- **Method:** Edge-based GNN where edges are directional OD flows; upstream punctuality on connected edges feeds downstream edge's prediction.
- **Results:** Claim superior predictive accuracy over baselines; I did not extract numbers.
- **Relevance to flightright-modeldev:** Confirms the ROI of upstream-delay-graph features beyond handcrafted hub spillover.

---

## 6. Open Questions

1. **What fraction of our v10 `NASDelay > 0` flights can be matched to OPSNET facility-level primary/secondary causes?** A simple join study would tell us whether adding OPSNET as an offline feature substrate adds information beyond `origin_nasdelay_rate_last1d`.
2. **Is the AbsorbScore two-stage architecture measurably better than a single-stage CatBoost on our data?** The arxiv paper trains on different data; we owe ourselves an apples-to-apples test.
3. **Does a simple upstream-airport weighted delay score beat the five hardcoded hubs?** Easy, cheap experiment.
4. **Would a quantile-regression head on CatBoost be accepted by the consumer product team as a "prediction interval" vs. today's buckets?** Not an engineering question — a product one. Flag to stakeholders.
5. **Is AeroDataBox's `/airports/delays/{icao}` endpoint actually stable and historical enough to train on, or is it real-time-only?** Critical for treating it as a live feature.

---

## 7. Sources

1. [Flight delay prediction: Evaluating machine learning algorithms for enhanced accuracy (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12685205/)
2. [A hybrid machine learning-based model for predicting flight delay (Nature Sci Rep 2024)](https://www.nature.com/articles/s41598-024-55217-z)
3. [Predictive Modeling of Flight Delays at an Airport Using Machine Learning Methods (MDPI 2024)](https://www.mdpi.com/2076-3417/14/13/5472)
4. [Predicting Flight Delays with Machine Learning: Saudi Arabian Airlines (Wiley 2024)](https://onlinelibrary.wiley.com/doi/10.1155/2024/3385463)
5. [Dynamically forecasting airline departure delay probability distributions (Elsevier 2025)](https://www.sciencedirect.com/science/article/pii/S0969699725000511)
6. [Aeolus: A Multi-structural Flight Delay Dataset (arxiv 2510.26616)](https://arxiv.org/abs/2510.26616)
7. [Integrating Delay-Absorption Capability into Flight Departure Delay Prediction (arxiv 2512.08197)](https://arxiv.org/abs/2512.08197)
8. [A Review of Research on Flight Delay Propagation (Li 2025)](https://onlinelibrary.wiley.com/doi/full/10.1155/atr/4851103)
9. [Systemic delay propagation in the US airport network (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3557445/)
10. [FAA NAS Status](https://nasstatus.faa.gov/)
11. [FAA Air Traffic Controller Workforce Plan 2025-2028](https://www.faa.gov/sites/faa.gov/files/fy25-air-traffic-controller-workforce-plan_0.pdf)
12. [USAFacts: Is there a shortage of air traffic controllers?](https://usafacts.org/articles/is-there-a-shortage-of-air-traffic-controllers/)
13. [Fortune: FAA controller shortages Nov 2025](https://fortune.com/2025/11/01/faa-air-traffic-control-facilities-staffing-shortages-government-shutdown-flight-delays/)
14. [OPSNET Detail Data Download](https://www.aspm.faa.gov/aspmhelp/index/OPSNET_Delays__Detail_Data_Download.html)
15. [OPSNET Delay By Cause Report](https://www.aspm.faa.gov/aspmhelp/index/OPSNET_Delays__Delay_By_Cause_Report.html)
16. [OPSNET Delay By Secondary Cause Report](https://www.aspm.faa.gov/aspmhelp/index/OPSNET_Delays__Delay_By_Secondary_Cause_Report.html)
17. [Aviation System Performance Metrics (ASPM) index](https://www.aspm.faa.gov/aspmhelp/index/Aviation_System_Performance_Metrics_(ASPM).html)
18. [ASPM Types of Delay](https://www.aspm.faa.gov/aspmhelp/index/Types_of_Delay.html)
19. [BTS: Understanding the Reporting of Causes of Flight Delays](https://www.bts.gov/topics/airlines-and-airports/understanding-reporting-causes-flight-delays-and-cancellations)
20. [FAA NOTAMs chap 5 sec 1 — Movement Area NOTAMs](https://www.faa.gov/air_traffic/publications/atpubs/notam_html/chap5_section_1.html)
21. [FAA Airport Construction Notices](https://www.faa.gov/air_traffic/flight_info/aeronav/aero_data/apt_constr_notices/)
22. [AC 150/5200-28F Advisory Circular on NOTAMs](https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_150_5200-28F.pdf)
23. [14 CFR Part 117 — Flight and Duty Limitations](https://www.ecfr.gov/current/title-14/chapter-I/subchapter-G/part-117)
24. [GAO-08-1041R: Airline Crew Scheduling Impact on Delays and Cancellations](https://www.govinfo.gov/content/pkg/GAOREPORTS-GAO-08-1041R/html/GAOREPORTS-GAO-08-1041R.htm)
25. [IATA delay codes (Wikipedia)](https://en.wikipedia.org/wiki/IATA_delay_codes)
26. [Cosmos practical guide to IATA delay codes](https://usecosmos.com/blog/decoding-delays-a-practical-guide-to-iata-delay-codes)
27. [Aviation Intelligence Portal: IATA delay codes](https://ansperformance.eu/definition/iata-delay-codes/)
28. [OAG Flight Status Data](https://www.oag.com/flight-status-data)
29. [NTSB Aviation Investigation Search](https://www.ntsb.gov/Pages/AviationQueryv2.aspx)
30. [NTSB Accident Data](https://www.ntsb.gov/safety/data/Pages/Data_Stats.aspx)
31. [Probabilistic Pretactical Flight Delay Prediction with Quantile Regression (AIAA J. Air Transportation)](https://arc.aiaa.org/doi/10.2514/1.D0406)
32. [Probabilistic Flight Delay Predictions Using ML (MDPI Aerospace 2021)](https://www.mdpi.com/2226-4310/8/6/152)
33. [Edge-Based GNN for Network Delay Prediction (MDPI Aerospace 2024)](https://www.mdpi.com/2226-4310/13/2/161)
34. [A Spatio-Temporal Approach with Self-Corrective Causal Inference (arxiv 2407.15185)](https://arxiv.org/html/2407.15185v1)
35. [AAGNN: Adaptive Airport Graph Neural Network (ESWA 2024)](https://www.sciencedirect.com/science/article/abs/pii/S0957417424018803)
36. [Airline on-time performance and consumer choice (Transportation Research)](https://www.sciencedirect.com/science/article/abs/pii/S0739885917300148)
37. [EUROCONTROL Turnaround Time standard inputs](https://ansperformance.eu/economics/cba/standard-inputs/latest/chapters/turnaround_time.html)

---

## Recommended Next Actions

1. **`feature-transferability`:** Confirm AeroDataBox `/airports/delays/{icao}` endpoint returns historical data (not just real-time) — gates every "airport delay index" feature.
2. **`feature-transferability`:** Verify NOTAM pipeline options (FAA FNS scraper or free mirror). Runway-closure flags only work if we can reliably parse them.
3. **`correlations-and-interactions`:**
   a. Test `schedule_padding_ratio = CRSElapsedTime / p10(ActualElapsedTime over last 180d for OD)` against `y_dep_ge15` and `y_arr_ge15`.
   b. Test a simple weighted-upstream-airport delay score (BTS-derived, lagged) against `y_dep_ge30` after controlling for existing hub features.
   c. Test an `airline_cancel_rate_anomaly_7d` orthogonal to the existing carrier-delay anomaly.
   d. Test `enroute_max_cape` (sampled along great circle from Open-Meteo) against arrival-delay residual after controlling for origin/dest weather.
4. **`research` (follow-up):** Full digest of *Integrating Delay-Absorption Capability into Flight Departure Delay Prediction* (arxiv 2512.08197) with extracted hyperparameters and absorption-label definition before any implementation.
5. **`research` (follow-up):** Pull OPSNET secondary-cause breakdowns for v10 hub facilities (DEN/PHX/BWI/MDW/BNA for WN; EWR/IAH/ORD/DEN/SFO for UA; DFW/CLT/MIA/ORD/PHX for AA; ATL/BOS/DTW/LAX/JFK for DL) for 2023-2025 and match against v10 `origin_nasdelay_rate_last1d` peaks.
6. **Product discussion (out of skill scope):** Agree with product on whether to expose `rebooking_inventory_sameday_±4h` and whether the model's public output should shift from bucket probabilities to a (P10, P50, P90) interval + miss-connection probability + cancellation probability stack.
