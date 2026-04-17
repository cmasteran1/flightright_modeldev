# Research: IATA AHM 730 Delay Codes — Deep Dive on Obtainability

**Date:** 2026-04-13
**Question:** Can we obtain IATA AHM 730 delay-code data (or equivalent granular per-flight cause codes) anywhere — public, historical, any era — in order to uncover hidden correlations between specific delay-cause domains and our v10 predictors?
**Scope:** features (granular cause-code data sourcing)

## Executive Summary

- **AHM 730 per-flight data is nowhere publicly downloadable as a primary feed.** IATA owns the code standard, and the two bodies that actually collect flight-by-flight AHM-coded records — EUROCONTROL CODA in Europe and the airlines themselves — treat the per-flight files as confidential benchmarking inputs. CODA publishes only aggregated Digest reports in PDF. [1][2][3]
- **There is exactly one sizeable public dataset with per-flight delay-cause codes at airline/airport granularity: the ANAC Brazil VRA feed and its academic successor, the Brazilian Flight Dataset (BFD).** BFD ships 15.5M rows across 2000–2024 with decoded "justificativa" delay reason codes (ANAC's own taxonomy — conceptually parallel to, not identical to, IATA AHM 730). This is the most realistic path to exposing granular cause-class signal for correlation discovery. [4][5][6]
- **Even though BFD uses ANAC codes, many of them map 1:1 or 1:several to IATA AHM 730 families** (e.g., crew-related, tech-defect, fueling, late-inbound, ATC-flow, weather-at-origin). That mapping is enough for domain-level correlation work, which is what the user asked for. [5][7]
- **US BTS five-bucket data (CarrierDelay / WeatherDelay / NASDelay / SecurityDelay / LateAircraftDelay) remains the only national-scale public granular breakdown for US flights** — and that is already in our pipeline. No US dataset exposes AHM-level subcodes. [8]
- **Recommended play:** stop trying to source IATA AHM 730 directly, pull the Brazilian BFD for exploratory correlation analysis against our feature set (even on a Brazilian-only sample), and hand this report to `correlations-and-interactions` to quantify whether specific delay-domain codes correlate with our existing v10 features. If signal shows up, it becomes justification to engineer proxies for those domains from AeroDataBox/Open-Meteo/strike-cache at inference time.

## Candidate Data Sources

### 1. EUROCONTROL CODA — per-flight AHM 730 feed

- **Description:** CODA (Central Office for Delay Analysis) collects flight-by-flight post-operations data from ~100 airlines and 55 airports under IR2019/317, using IATA AHM 730 + AHM 731 delay codes. [2][3]
- **What's public:** Aggregated Digest reports (quarterly and annual PDFs) plus the Aviation Intelligence Portal / CID dashboard. Digests include per-code breakdowns in aggregate tables (e.g., "reactionary (code 93) share of total delay minutes"), but flight-level records are **not** published. [2][3][9]
- **What's gated:** The actual per-flight AHM-coded dataset goes only to data contributors in exchange for benchmarking reports. There is no public API, no bulk download, no academic-access portal that I was able to locate. [2]
- **Availability signal:** Likely unavailable at per-flight granularity. Aggregates are available.
- **Leakage risk:** N/A (post-ops data).
- **Priority:** Low for per-flight, Medium for reading aggregated AHM-code shares as population priors.
- **Next step:** Use CODA Digests as a reference for expected prior frequencies of each AHM category (e.g., "how much of total delay minutes is reactionary vs. ATC vs. weather in Europe") when designing proxies. Do not try to source the raw feed.

### 2. ANAC VRA (Voo Regular Ativo) — Brazil — per-flight with justification codes

- **Description:** ANAC (Brazilian civil aviation authority) requires every scheduled flight to register departure, arrival, and a justification code for delays/cancellations. Data is published monthly via ANAC's portal, continuous since 2000. [4][10]
- **What's in it:** Flight number, airline, origin/destination (ICAO), scheduled vs. actual block times, status, and a delay **justification code** drawn from ANAC's own taxonomy (closely parallel to IATA AHM 730 but with Brazilian-specific additions). [5][7][11]
- **Availability signal:** Plausible (open, historical, per-flight). Raw VRA is posted as monthly CSVs on ANAC's site; the `flightsbr` R package and the Kaggle mirror provide download automation. [11][12][13]
- **Leakage risk:** None (post-ops).
- **Priority:** **High.** This is the one realistic public granular-cause-code dataset.
- **Next step:** `feature-transferability` to confirm a stable download mechanism; `correlations-and-interactions` to join BFD/VRA against our v10 feature set on overlapping Brazilian routes, if any. If overlap is thin, use BFD as an exploratory side-study to test which cause domains are predictable from schedule/weather/congestion features alone.

### 3. Brazilian Flight Dataset (BFD) — enhanced VRA with weather + decoded codes

- **Description:** Academic integrated dataset by Teixeira et al. (CEFET/RJ DAL group) that combines raw VRA with IEM/ASOS hourly airport weather and **expands the ANAC justification codes into human-readable descriptions**. 15,505,922 rows × 45 features, 2000–2024. Published on IEEE DataPort (doi 10.21227/k10b-qn21) and mirrored on GitHub (`cefet-rj-dal/bfd`). Paper: arxiv 2102.13330. [5][6][14]
- **Why it matters for this project:** BFD is the closest publicly available analog to what we would build if we had AHM 730 codes on our BTS+AeroDataBox data. It gives us a ready-made sandbox to test the core hypothesis: *do specific cause domains correlate with features we can compute from schedule + weather + historical airport performance?*
- **Availability signal:** Plausible (IEEE DataPort, requires free account; GitHub repo has code to rebuild).
- **Leakage risk:** None.
- **Priority:** **High.**
- **Next step:** `feature-transferability` to download BFD and confirm column-by-column content; `correlations-and-interactions` to run the core correlation study (see "Hidden Correlations" section below).

### 4. US BTS five-bucket cause breakdown

- **Description:** Carrier / Weather / NAS / Security / LateAircraft delay minutes on every US flight >15min late in the BTS On-Time Performance file. [8]
- **What we already do:** v10 aggregates these into rolling rates (`origin_nasdelay_rate_last1d`, etc.). [v10 spec in `exploration/v10_feature_spec.md`]
- **Gap:** The five buckets are coarser than AHM 730. LateAircraft hides the reason the upstream leg was late; Carrier hides crew-timeout vs. maintenance vs. boarding vs. fueling. We cannot refine this with BTS alone.
- **Availability signal:** Already ingested.
- **Leakage risk:** Addressed in current pipeline via `last1d` / `last14d` windows.
- **Priority:** Already exploited. No further action.

### 5. NTSB CAROL / FAA SDR — incident-level (not delay-level)

- **Description:** NTSB Case Analysis and Reporting Online and FAA Service Difficulty Reports expose aircraft incident / maintenance-finding records. Useful for *safety-inspection-failure* and *mechanical-event* signal, but these report incidents, not routine delays.
- **Availability signal:** Plausible (public), but the signal is rare events only.
- **Leakage risk:** Low (records are post-event and dated).
- **Priority:** Medium — already covered in the prior research report. Worth revisiting if BFD correlation study flags tech-defect codes as high-signal.

### 6. Kaggle "Flight Delay and Causes" and similar compilations

- **Description:** Several Kaggle datasets with "causes" in the name (Giovamata Airlines Delay, undersc0re Flight Delay and Causes, 2015 Flight Delays). Scanned each. All are BTS-derived five-bucket data repackaged; none contain AHM 730 codes. [15][16][17]
- **Availability signal:** N/A (no new granular signal).
- **Priority:** None — already subsumed by our BTS pipeline.

### 7. DGAC France / UK CAA / AENA / CAAC

- **Finding:** Nothing public at flight+code granularity. DGAC and UK CAA publish punctuality summaries only. UK CAA's punctuality notes describe cause codes but do not release per-flight data. CAAC publishes annual aggregates only. AENA has no public delay-cause feed I could locate. [18][19]
- **Priority:** None.

## Hidden Correlations Hypotheses (for `correlations-and-interactions` to test on BFD)

The user's stated goal is *"uncover potentially hidden correlations with specific delay type domains."* The following pairings are worth testing on BFD; each asks "does the feature we already compute in v10 carry signal for a specific narrow cause domain that BTS's five buckets obscure?"

| V10 feature | Hypothesized cause domain | Rationale |
|---|---|---|
| `late_aircraft_rate_origin_last1d` | ANAC "rotação" (aircraft rotation) / IATA 93 reactionary | Propagation chain; if our feature is really capturing this, it should correlate more strongly with rotation-coded delays than with first-leg crew delays |
| `elapsed_time_ratio_last14d` | ANAC ATC-flow / IATA 81–89 ATFM | Airborne holding shows up as block-time inflation before it shows up in BTS NAS |
| `divert_rate_origin_last14d` | Weather-at-destination / IATA 71–75 | Divert-per-arrival is a downstream proxy for IMC |
| Departure slot (hour-of-day, end-of-day flag) | Crew duty limits / IATA 61–69 | Late-day slots accumulate duty-hour risk; this should correlate with crew codes, not weather codes |
| Origin historical cancel rate | IATA 41 tech-defect + 87 airport/runway closure | Tech groundings and runway closures drive cancels, not just delays |
| Low-cost carrier flag | IATA 11 check-in, 15 boarding, 36 fueling | Tight turns at LCCs should over-index on turnaround operational codes |

If any of these correlations come back empirically strong on BFD, it justifies engineering narrow proxies for those domains in v11 — not by sourcing AHM 730 directly, but by building better synthetic features (e.g., a "turnaround-stress index" that loads on IATA 11/15/36 territory).

## Key Notes on Code Mapping

- ANAC's justification codes and IATA AHM 730 are **parallel but not identical.** ANAC maintains extra Brazil-specific codes; BFD's paper explicitly states "the ANAC also determines other delay codes that are not on the IATA list." [5]
- For cross-market correlation work we need to work at the **domain** level, not the two-digit code level. Mapping tables in BFD's documentation and in CODA's AHM 730 annex let us roll both taxonomies up to ~6 domains: Airline-ramp, Aircraft-tech, Crew, ATC/Flow, Weather, Airport-infrastructure, Reactionary. [2][5]
- The Erau Correct Delay Code Assignment thesis [7] documents the mislabeling problem — airlines often misapply AHM 730 codes under commercial pressure. Any per-flight code is noisy. Domain-level aggregation is more trustworthy than single-code fidelity.

## Paper Digests

- **Citation:** Teixeira, C., Tavares, L., Soares, J., dos Santos, J., Amorim, G., Ogasawara, E. (2021). *Integrated Dataset of Brazilian Flights*. BreSci / arxiv 2102.13330. [5][14]
  - **Problem:** VRA alone is operationally useful but analytically shallow — raw codes, no weather, no decoded descriptions.
  - **Method:** Join VRA with IEM/ASOS hourly METARs on airport × hour; expand ANAC justification codes; normalize airline / airport identifiers.
  - **Results:** 15.5M flight observations × 45 columns, 2000–2024 coverage, delay justification codes decoded into natural-language categories.
  - **Relevance to flightright-modeldev:** BFD is a drop-in sandbox for testing whether our v10 feature concepts carry granular cause signal on a second, independent market. If a feature correlates with IATA-92-analog codes on BFD, that's cross-market evidence it's genuinely capturing that cause domain.

## Open Questions

- Exact column list in the 2024 BFD release — does it include flight-level weather at scheduled departure time (so we can replicate our Open-Meteo feature join) or only hourly airport weather? Hand off to `feature-transferability`.
- What's the ANAC→AHM 730 crosswalk table? The BFD paper references a mapping; need the actual table to group codes into domains consistently with what we'd see in Europe. Hand off to `feature-transferability`.
- Is there any EUROCONTROL research collaboration path (academic access to CODA) that would yield per-flight European data? Out of scope for this project's compute posture, but worth noting.

## Sources

1. EUROCONTROL. "CODA Digest — All-Causes Delay" (various quarterly/annual PDFs). <https://www.eurocontrol.int/publication/all-causes-delays-air-transport-europe-annual-2024>
2. EUROCONTROL. "All-Causes Delay — Aviation Intelligence Portal." <https://ansperformance.eu/capacity/tot_dly/>
3. EUROCONTROL. "All-causes delay analysis interactive dashboard (CID)." <https://www.eurocontrol.int/dashboard/all-causes-delay-analysis-interactive-dashboard>
4. Kaggle mirror of ANAC VRA. <https://www.kaggle.com/datasets/rdewes/voo-regular-ativo-vra-anac>
5. Teixeira et al. "Brazilian Flights Dataset (BFD)." IEEE DataPort. <https://ieee-dataport.org/documents/brazilian-flights-dataset>
6. CEFET-RJ DAL. "Brazilian Flight Dataset." <https://eic.cefet-rj.br/~dal/brazilian-flight-dataset-description/>
7. Embry-Riddle. "Correct Delay Code Assignment" (Brazil graduate thesis). <https://commons.erau.edu/cgi/viewcontent.cgi?article=1011&context=brazil-graduate-works>
8. US DOT BTS. "On-Time Performance — Delay Cause." <https://www.transtats.bts.gov/OT_Delay/OT_DelayCause1.asp>
9. EUROCONTROL. "CODA Digest Q2 2019." <https://www.eurocontrol.int/sites/default/files/2019-09/coda-digest_q2-2019.pdf>
10. ANAC. "National Civil Aviation Agency." <https://www.gov.br/anac/en>
11. IPEA. `flightsbr` R package. <https://ipeagit.github.io/flightsbr/>
12. GitHub. `cefet-rj-dal/bfd`. <https://github.com/cefet-rj-dal/bfd>
13. GitHub. `alvarofpp/dataset-flights-brazil`. <https://github.com/alvarofpp/dataset-flights-brazil>
14. arxiv 2102.13330. *Integrated Dataset of Brazilian Flights*. <https://arxiv.org/pdf/2102.13330>
15. Kaggle. "Flight Delay and Causes." <https://www.kaggle.com/datasets/undersc0re/flight-delay-and-causes>
16. Kaggle. "Airlines Delay." <https://www.kaggle.com/datasets/giovamata/airlinedelaycauses>
17. Kaggle. "Flight Delay Data." <https://www.kaggle.com/datasets/sriharshaeedala/airline-delay>
18. UK CAA. "UK flight punctuality statistics notes." <https://www.caa.co.uk/data-and-analysis/uk-aviation-market/flight-punctuality/notes/>
19. DGAC France profile. <https://www.ecologie.gouv.fr/en/french-civil-aviation-authority-dgac>

## Recommended Next Actions

1. `feature-transferability`: download the Brazilian Flight Dataset (IEEE DataPort doi 10.21227/k10b-qn21 and/or GitHub `cefet-rj-dal/bfd`), confirm its exact schema, and produce an ANAC-code → domain crosswalk table (Airline-ramp / Tech / Crew / ATC-flow / Weather / Airport-infra / Reactionary). Verify whether hourly airport METAR joins are pre-computed so we can mimic our v10 weather-at-scheduled-departure feature without re-joining.
2. `correlations-and-interactions`: once BFD is local, replicate the v10 feature set that is *constructible from BFD columns alone* (schedule-slot, LCC flag, late-aircraft-rate proxy, elapsed-time-ratio proxy, cancel-rate, divert-rate-if-present, weather-at-origin) and run domain-level correlation tests per the hypothesis table above. Report which cause domains are predictable from current-style features and which are not — the unpredictable ones are where v11 must add new proxies.
3. Do **not** continue pursuing AHM 730 direct sourcing from CODA, DGAC, UK CAA, or any European authority — per-flight access is closed.
4. Park US-side granularity at BTS five-bucket. Any sub-bucket refinement in the US will have to come from engineered proxies, not from authoritative code data.
