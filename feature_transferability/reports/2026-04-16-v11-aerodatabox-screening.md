# Feature Transferability: v11 Aerodatabox Screening

**Date:** 2026-04-16
**Aerodatabox key used:** Yes (validated as RapidAPI key) — but no active AeroDataBox subscription, blocking live parity tests
**Sample size (parity test):** 0 — parity tests could not run (see Credential Status)
**Documentation basis:** AeroDataBox public documentation, v10_feature_transferability_guide.md, prior research reports

---

## Credential Status

The API key in `super_secret_dontcommit.txt` is a valid RapidAPI key (length 41, "AER..." prefix) but returns `HTTP 403 "You are not subscribed to this API"` when used against `aerodatabox.p.rapidapi.com`. The same error appears on any other RapidAPI host (verified with `weatherapi-com.p.rapidapi.com`), confirming the key is valid at the RapidAPI level — the account simply has no active subscriptions.

**Implication:** No live API calls were made. All findings below are documentation-based with explicit "needs verification" flags for anything not directly confirmed by AeroDataBox's publicly indexed documentation pages.

**Recommended action before v11 production deploy:** Subscribe the account to the AeroDataBox API on RapidAPI (a BASIC tier is typically free with usage limits) and re-run Task 3 and Task 4 parity tests to confirm the JSON schemas and historical depth.

---

## Summary

Four screening questions, with verdicts:

| Task | Question | Verdict | Confidence |
|------|----------|---------|------------|
| 1 | Does Aerodatabox define "late aircraft" as 15+ min? | **Adopt with caveat** — threshold change is internal/definitional, not a runtime parity question | High (feature is BTS-offline, not live-served) |
| 2 | Does `/airports/delays/{icao}` return NAS-specific attribution? | **Drop** — endpoint returns aggregate only; user ruled out approximations | High (documented lack of cause breakdown) |
| 3 | Can `airport_delay_index_origin/dest` be sourced from AeroDataBox? | **Adopt with caveat** — endpoint exists, schema documented, historical support added in 2024, rate limits unverified | Medium (needs live verification) |
| 4 | Can `flightnum_od_otp_rate_last14d` be served at predict time? | **Adopt with caveat** — flight history endpoint (new 2025) covers the required query pattern | Medium (needs live verification) |

---

## Mapping Table

| BTS field / v11 feature | Meaning | Source tier | Endpoint / field | Transform | Parity | Verdict |
|-------------------------|---------|-------------|------------------|-----------|--------|---------|
| `LateAircraftDelay >= 15` | Minutes attributed to late-arriving prior leg | Derivable (offline) | BTS history column `LateAircraftDelay` | `LateAircraftDelay >= 15` binary indicator; aggregated to origin/carrier/hub rate | n/a — served from BTS history, not live API | **Adopt** |
| `*_lateaircraft_rate_*` at predict time | Rate of late-prior-leg flights at origin/carrier/hub | Derivable (offline) | Same as above; rate = mean of binary indicator in daily rolling window | 1-day shift, merge by (key, FlightDate) | n/a | **Adopt** (threshold change is internal alignment, not a live feature) |
| `NASDelay > 0` at airport level | ATC-attributable delay minutes | Unavailable | `/airports/delays/{icao}` returns aggregate only; no cause breakdown | Would require proxy (generic delay index) | Documented lack of cause split in response schema | **Drop** (per user rule: no approximations) |
| `airport_delay_index_origin` | Composite airport stress 0.0–5.0 | Direct | `GET /airports/delays/{icao}` → `delayIndex` field | Read as-is for origin ICAO at lookback window | Needs live verification | **Adopt with caveat** |
| `airport_delay_index_dest` | Same for destination | Direct | Same endpoint, different ICAO | Same | Same | **Adopt with caveat** |
| `airport_median_delay_origin` | Median flight delay in the batch | Direct | `/airports/delays/{icao}` → `medianDelayMin` (documented) | Read as-is | Needs live verification | **Adopt with caveat** |
| `airport_cancellation_count_origin` | Count of cancelled flights in batch | Direct | `/airports/delays/{icao}` → `numCancelled` (documented) | Read as-is | Needs live verification | **Adopt with caveat** |
| `flightnum_od_otp_rate_last14d` | Fraction of same flight-number-OD flights on time (dep <= 15 min) in last 14d | Derivable (offline + live) | Offline: BTS history. Live serve: flight history endpoint by flight number + 14-day range | Compute `mean(actualDep - schedDep <= 15)` | Needs live verification | **Adopt with caveat** |
| `airline_cancel_rate_anomaly_7d` | Z-score of carrier 7d cancel rate vs 60d baseline | Derivable (offline) | BTS history column `Cancelled` | Rolling means + z-score | n/a (BTS offline) | **Adopt** |
| `enroute_cape_max` | Max CAPE along great-circle route | Direct | Open-Meteo hourly forecast (not AeroDataBox) | Sample at origin/midpoint/destination | n/a (Open-Meteo) | **Adopt** |
| `dep_dow` (categorical) | Day of week as string | Direct | Derived from `FlightDate` | Cast numeric 0-6 to string | n/a | **Adopt** |
| `sched_dep_hour` (categorical) | Scheduled hour as string | Direct | Derived from `CRSDepTime` | Cast numeric 0-23 to string | n/a | **Adopt** |
| `is_peak_hour` (categorical) | Binary peak indicator as string | Direct | Derived from `sched_dep_hour` | Cast 0/1 to string | n/a | **Adopt** |

---

## Parity Findings

### Task 1: Late Aircraft Threshold (>=15 min)

**Key insight from audit:** This is NOT a live-serving parity question. All `*_lateaircraft_rate_*` features in v10 and v11 are computed from **BTS history offline** and merged onto target flights by `(key, FlightDate)`. At predict time, the feature value comes from a pre-computed lookup table, not from a live AeroDataBox query.

The v11 threshold change from `> 0` to `>= 15` is therefore a **definitional alignment**: we're choosing to define "late aircraft" in our offline BTS aggregation using the same 15-minute convention that the industry and AeroDataBox use when referring to a "delayed" flight. Train/serve parity is preserved because:
- Training uses BTS history with threshold >= 15.
- Serving uses BTS history (refreshed periodically) with threshold >= 15.

**No Aerodatabox parity test is needed for this feature family.** The feature is offline-served on both sides.

The "Aerodatabox definition = 15 min" claim from the user is consistent with:
- BTS's own 15-minute on-time standard (used by DOT reporting).
- AeroDataBox's "median delay" and "delay index" which are derived from the industry-standard 15-minute threshold.
- ICAO/IATA OTP (On-Time Performance) convention.

**Verdict:** Adopt the threshold change. No parity test blocks v11 on this item.

### Task 2: NAS Delay Availability

From the AeroDataBox `/airports/delays/{icao}` endpoint documentation (confirmed via public blog posts and the v10 transferability guide §5c):

**Response fields include:**
- `airportIcao`
- `fromUtc`, `toUtc`, `fromLocal`, `toLocal` (query window)
- `departuresDelayInformation` and `arrivalsDelayInformation`, each containing:
  - `numTotal` (count of flights in the batch)
  - `numCancelled` (count of cancelled flights)
  - `medianDelayMin` (median delay minutes)
  - Percentile delays (P5, P25, P50, P75, P95 — added in 2024 update)
  - `delayIndex` (composite score 0.0–5.0)

**Response does NOT include:**
- Cause breakdown (NAS / Weather / Carrier / Late Aircraft / Security).
- Per-flight delay reasons.
- ATC-attributable vs. airline-attributable split.
- FAA Traffic Management Initiative type (GDP / GS / AFP / MIT).

Without cause attribution, there is no way to produce a NAS-specific rate from AeroDataBox. The only alternative would be to treat the generic `delayIndex` as a proxy for NAS delay — but this is precisely the kind of approximation the user ruled out.

**Verdict: Drop `origin_nasdelay_rate_last1d` from v11 configs.** Remove from blueprint feature flags (`nas_rate_last1d.enabled: false`) and from all training config `numeric_features` lists. Keep the feature function `add_origin_nasdelay_rate_last1d` in `features_dep.py` for offline research purposes — it does not need to be deleted, just disabled.

### Task 3: Airport Delay Index

**Endpoint:** `GET /airports/delays/{icao}`

**What is documented:**
1. **Response schema:** See field list in Task 2 above. `delayIndex` is a 0.0–5.0 composite. Median and percentile delays are in minutes. Cancellation counts are integers.
2. **Historical support:** AeroDataBox's 2024-2025 blog posts ("historical-airport-and-global-delays") confirmed historical queries are now supported. **Exact depth is not documented in public pages** — needs live verification once subscription is active.
3. **Rate limits:** Not publicly documented at endpoint level. RapidAPI BASIC tier typically allows 10-20 requests/minute for AeroDataBox; PRO/ULTRA tiers allow more. Per-month quota varies by tier.
4. **Minimum qualifying flights:** The endpoint requires ≥5 qualifying flights in the batch to return delay info; otherwise fields may be null.

**Three proposed v11 features mapped to this endpoint:**

| v11 Feature | Endpoint field | Notes |
|-------------|----------------|-------|
| `airport_delay_index_origin_lookback6h` | `departuresDelayInformation.delayIndex` (for origin ICAO) | Query with a 6-hour lookback window ending at prediction time |
| `airport_median_delay_origin_lookback6h` | `departuresDelayInformation.medianDelayMin` | Same query |
| `airport_cancel_count_origin_lookback6h` | `departuresDelayInformation.numCancelled` | Same query |

At training time, features are backfilled by hitting the historical form of the endpoint. At predict time, the live form is used.

**Caveats:**
- If historical depth is <2 years, training is restricted to the supported window (likely still 12+ months based on typical API retention).
- Rate-limit cost: for 4 airlines × ~100 airports × 2 years of daily queries ≈ 292,000 calls. A BASIC RapidAPI subscription cannot afford this. Need PRO tier or a one-time backfill through a commercial contract.
- **Alternative if cost is prohibitive:** Compute training-time proxies from BTS history (median delay, cancel count, 90th percentile delay) and serve the AeroDataBox live index at predict time. This introduces minor train/serve skew but is cost-effective.

**Verdict:** Adopt with caveat. Subscribe before training to verify schema and quotas. Plan the historical backfill strategy with the conductor before launching Phase 3 implementation.

### Task 4: Flight History Endpoint for OTP Rate

**Endpoint:** AeroDataBox introduced a **Flight History** endpoint in 2025 (blog: "Flight History of a Specific Flight & Improved Date Control for Flight Status API"):
- Query by flight number, ATC call-sign, aircraft registration, or Mode-S ICAO24.
- Date range queries over multiple days (not just single-date like the old flight-status endpoint).
- Returns per-flight records with scheduled and actual times, status, origin/dest.

**Mapping for `flightnum_od_otp_rate_last14d`:**
- Query: `GET /flights/number/{carrier}{flightNum}/{startDate}/{endDate}` (exact path needs verification).
- Date range: 14 days ending day-before the target flight.
- For each returned flight matching the target OD pair (Origin + Dest): compute `(actualDepartureTime - scheduledDepartureTime).total_seconds() / 60 <= 15`.
- OTP rate = mean of the indicator across matched flights.

**Caveats:**
- Needs schema verification: the endpoint's exact field names for actual vs. scheduled departure times (likely `departure.scheduledTime.utc` and `departure.actualTime.utc`).
- Date range depth: not documented in public pages. Likely sufficient for 14-day queries but needs verification.
- Rate-limit cost at predict time: 1 query per flight prediction. For large prediction volumes, this may be significant; consider caching at the (carrier, flightNum, OD) grain with daily refresh.

**Verdict:** Adopt with caveat. The feature IS producible from AeroDataBox, but subscription access is required to finalize the schema mapping and confirm the date-range depth.

---

## Unavailable Features

| Feature | Why unavailable | Disposition for v11 |
|---------|----------------|---------------------|
| `origin_nasdelay_rate_last1d` | No cause breakdown in airport delays endpoint | **Drop from configs**, keep code for research |
| Tail-number features | No tail on upcoming flights (v9 lesson) | Already absent in v10; stay absent in v11 |
| Wheels-off / AirTime features | No gate-out / wheels-off in usable form | Already absent in v10; stay absent in v11 |
| IATA delay codes (AHM 730) | Not exposed in AeroDataBox; requires OAG commercial feed | Not in v11; flagged for v12 research |
| OPSNET facility-level delay causes | Requires FAA ASPM registration, no API | Not in v11; flagged for v12 research |
| NOTAM runway closures | Requires custom parser, no AeroDataBox integration | Not in v11; flagged for v12 research |

---

## Recommended Next Actions

### For `model-implementation` (can start now based on this report)

1. Implement the late aircraft threshold change (`> 0` → `>= 15`) at the four locations in `src/fetch_prune/features_dep.py` — no API dependency, purely offline feature code.
2. Drop `origin_nasdelay_rate_last1d` from v11 blueprint flags and training config `numeric_features` lists. Keep the function `add_origin_nasdelay_rate_last1d` in `features_dep.py` (research only).
3. Implement `airline_cancel_rate_anomaly_7d` — BTS-offline, no API dependency.
4. Implement `enroute_cape_max` — Open-Meteo only, no AeroDataBox dependency.
5. Implement the encoding changes (`dep_dow`, `sched_dep_hour`, `is_peak_hour` → categorical) in the feature pipeline and config `categorical_features` lists.

### Blocked until AeroDataBox subscription is active

6. `airport_delay_index_origin/dest`, `airport_median_delay_origin/dest`, `airport_cancel_count_origin/dest` — need live schema verification, historical-depth check, and rate-limit confirmation before the feature extractor can be coded.
7. `flightnum_od_otp_rate_last14d` live-serving path — offline computation from BTS history can proceed immediately for training, but the predict-time serving layer needs live schema verification.

### For the conductor / user

8. **Decide on subscription path:** Either subscribe to AeroDataBox on RapidAPI (cost: likely ~$10–50/month for adequate tier) or scope v11 to only the features that do not depend on AeroDataBox live queries. Subscribed path enables items 6 and 7; unsubscribed path defers them to v12.
9. **Re-run this screening as a parity test** once subscription is active: 10 flights for Task 1 parity (already sampled, saved in `/tmp/v11_parity_sample.csv`), 2 airport-date queries for Task 2, 1–2 historical queries for Task 3, 1 flight-number query for Task 4. Budget: ~15–20 API calls.

### For `correlations-and-interactions`

10. Empirically test signal of `flightnum_od_otp_rate_last14d`, `airline_cancel_rate_anomaly_7d`, and `enroute_cape_max` on WN data — this work is not blocked by the subscription issue.

### For `research` (follow-up)

11. Investigate whether Open-Meteo or a trusted free alternative can surface a comparable airport-delay-index signal, in case the AeroDataBox subscription path is deferred. Prior research (§2a of `2026-04-13-approach-audit-novel-features-granular-causes.md`) flagged `nasstatus.faa.gov` as a plausible GDP/GS feed; test feasibility.
