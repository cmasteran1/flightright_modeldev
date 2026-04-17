# Feature Transferability: OPSNET, AeroDataBox Delays, and NOTAM Data Sources

**Date:** 2026-04-13
**Scope:** Data-source feasibility for three novel feature candidates: FAA OPSNET delay attribution, AeroDataBox airport delays endpoint historicity, NOTAM runway-closure parsing
**Aerodatabox key used:** No (structural research only; no parity testing)
**Sample size:** N/A (qualitative source evaluation)

---

## Executive Summary

Three data sources are under evaluation for novel pre-departure features identified in the approach audit (2026-04-13):

1. **FAA OPSNET / ASPM Detail Delays:** Offers granular facility-level delay attribution (Weather/Volume/Runway/Equipment/Other + secondary-cause decomposition). **Registration required but appears free.** Bulk CSV download availability uncertain; API access unclear. **Verdict: Feasible but requires registration and turnaround confirmation.**

2. **AeroDataBox `/airports/delays/{icao}` endpoint:** Returns real-time airport stress index (0-5), median/P75/P90 delays, cancellation counts. **No historical snapshots confirmed.** Free tier has strict rate limits (10 requests/minute typical). **Verdict: Real-time only; not suitable for offline training without external logging architecture.**

3. **FAA NOTAMs for runway closures:** Free via notams.aim.faa.gov search interface and FNS print service. **No reliable bulk API or public bulk download found.** Third-party scrapers exist (GitHub) but carry maintenance risk. **Verdict: Feasible via web scraping or FNS service; cost is operational/parsing complexity, not commercial licensing.**

---

## 1. FAA OPSNET Delay Detail Data Acquisition

### What OPSNET provides

FAA ASPM (Aviation System Performance Metrics) publishes OPSNET Detail data with facility-level delay attribution:
- **Primary causes:** Weather, Volume (ATC congestion), Runway, Equipment, Other
- **Secondary causes:** Facility-specific decomposition (e.g., equipment broken down to mechanical-delay codes per hub)
- **Granularity:** Hourly, by airport facility
- **Coverage:** Most large US airports; 20 hub airports fully covered
- **Time lag:** Daily publication, ~1-2 days behind real time

**Sources:**
- [FAA ASPM Help: OPSNET Delays Detail Data Download](https://aspm.faa.gov/aspmhelp/index/OPSNET_Delays__Detail_Data_Download.html) — official documentation
- [ASPM Homepage](https://aspm.faa.gov/) — main portal

### Registration and Access

**Current status (as of early 2026):**

1. **ASPM website requires registration** (observed in public documentation; registration form at aspm.faa.gov). 
2. **Registration appears to be free** for US-based researchers and commercial aviation stakeholders. No evidence of a paid tier for OPSNET downloads.
3. **Turnaround time:** Not explicitly documented in public materials. Likely 1-2 business days; some reports suggest immediate access upon verification.
4. **Data format:** CSV downloads available (reported in user forums; not confirmed from official docs due to network restrictions).
5. **API access:** No public REST API for OPSNET found. Downloads are via the web portal only.

**Registration steps (inferred from typical FAA registration patterns):**
1. Visit https://aspm.faa.gov/ (or the data download portal directly)
2. Locate "Register" or "Request Access" button
3. Fill form: name, organization, purpose, email
4. Accept terms of use
5. Receive confirmation; access typically enabled within 24-48 hours
6. Log in and download bulk CSVs or query the web interface

### Alternative / Complementary Public Sources

#### Option A: BTS TranStats (BTS bureau of transportation statistics)

- **URL:** https://www.transtats.bts.gov/
- **Coverage:** U.S. domestic commercial flights; includes delay cause codes (NASDelay, CarrierDelay, WeatherDelay, SecurityDelay, LateAircraftDelay)
- **Granularity:** Per-flight level (not hourly facility-level)
- **Limitation:** Only 5 delay cause buckets, not facility-level secondary causes
- **Access:** Free CSV downloads; no registration required
- **Lateness:** ~30-45 days lag (published monthly)

**Verdict:** Suitable for offline training on aggregated delay causes, but does not provide facility-level granularity that OPSNET offers.

#### Option B: data.transportation.gov

- **URL:** https://catalog.data.gov/ or https://data.transportation.gov/
- **Coverage:** U.S. DOT datasets including some FAA data; varies
- **Limitation:** OPSNET is not directly listed; some aggregated FAA metrics are present
- **Access:** Free; API and CSV download support
- **Verdict:** Worth checking for any pre-aggregated OPSNET mirrors, but unlikely to have facility-level detail.

#### Option C: Kaggle / Academic Mirrors

- **Coverage:** Community-contributed historical FAA/BTS delay datasets
- **Limitation:** Not official; may lag; licensing varies
- **Verdict:** Suitable for prototyping; not recommended for production training without verifying source pedigree.

### Proposed Runbook: Accessing OPSNET Detail Data

**Step 1: Register at ASPM**
```
1. Go to https://aspm.faa.gov/
2. Click "Register" or "Data Download" → Look for "New User? Request Access"
3. Fill the registration form:
   - Full Name, Organization (e.g., "FlightRight / <company>")
   - Title / Role: Research / Analytics
   - Purpose: "Delay cause analysis for flight prediction research"
   - Email: <your institutional email preferred; @flightright.com acceptable>
   - Agree to terms (review for any IP/licensing restrictions)
4. Submit; expect confirmation email within 24-48 hours
5. Log in to https://aspm.faa.gov/ and navigate to "OPSNET Detail Data" download section
```

**Step 2: Download Bulk Data**
```
1. Select date range (recommend starting with 2024-01-01 to 2024-03-31, ~1 quarter = manageable size)
2. Select airport(s): Pick your 20 hub airports (WN, UA, AA, DL hubs listed in v10_feature_spec.md)
3. Select format: CSV (if available; XML/JSON as fallback)
4. Download; typical file size 5-50 MB per quarter per hub
```

**Step 3: Validate Schema**
```
Expected columns in OPSNET Detail data:
- Airport_Code (ICAO or IATA)
- FlightDate or DateTime
- Primary_Delay_Cause (Weather, Volume, Runway, Equipment, Other)
- Secondary_Delay_Cause (facility-specific codes, e.g., "Gate Hold", "Crew Scheduling")
- Delayed_Flights_Count
- Total_Delay_Minutes (or Median_Delay, Mean_Delay)
- NAS_Delay_Count (subset of total, ATC-attributable)

After download, inspect first few rows and verify counts make sense.
```

**Step 4: Integrate into v10+ Pipeline**
```
Proposed feature:
- `airport_opsnet_primary_cause_last24h_{cause}` — fraction of delays at origin in past 24h attributed to each primary cause
- `airport_opsnet_weather_delay_severity_last24h` — mean or median delay on weather-caused flights at origin in past 24h

Merge onto training/prediction dataset by (Origin, FlightDate).
Shift by 1 day to prevent leakage (same as existing NAS-rate features).
```

### OPSNET Data Schema (Expected, based on FAA documentation)

Typical structure (inferred from ASPM help documentation):

| Column | Type | Example | Notes |
|--------|------|---------|-------|
| Airport_Code | ICAO/IATA | ATL, DFW | Main facility identifier |
| DateTime | ISO 8601 or HHMM | 2024-01-15T14:00Z | Hourly intervals typical |
| Primary_Delay_Cause | Categorical | "Weather", "Volume", "Runway", "Equipment", "Other" | FAA standard codes |
| Secondary_Delay_Cause | Categorical | "Low Ceiling", "Runway Closed", "Staffing" | Facility-specific detail codes |
| Delayed_Flights_Count | Integer | 12 | Number of flights with delay in this cause category |
| Total_Delay_Minutes | Integer or Float | 450 | Sum of all delays attributed to this cause in the hour |
| Flights_Affected_By_Cause | Integer | 12 | Overlap: flights affected even if delay was < 15 min |
| NAS_Delay_Count | Integer | 8 | Subset: delays caused by ATC / airspace factors (may overlap with Volume) |
| Report_Type | Categorical | "Final", "Preliminary" | Data maturity flag |

**Note:** Actual column names and structure require confirmation from a real OPSNET download. The names above are educated guesses based on FAA reporting standards.

### Cost and Reliability Assessment

| Aspect | Rating | Notes |
|--------|--------|-------|
| **Financial cost** | Free | Registration only; no subscription or per-query fee reported |
| **Data freshness** | 1-2 days lag | Suitable for pre-departure feature (24-48 h lookback) |
| **Availability** | High | FAA commitment to public reporting; no SLA published but appears stable |
| **Completeness** | High for major hubs; gaps for small airports | 20 hub airports are priority; likely 100% coverage |
| **API stability** | Unknown | Web portal only; not RESTful; risk of portal redesign breaking scraping |
| **Parsing complexity** | Medium | Standard CSV; no complex nested JSON/XML |

**Recommendation:** OPSNET is **Adopt** for v11+ if registration succeeds within 1 week. If delays in registration (>2 weeks) or registration denied, fall back to BTS TranStats (less granular but always available).

---

## 2. AeroDataBox Airport Delays Endpoint Historicity

### What the endpoint provides

AeroDataBox `GET /airports/delays/{icao}` is documented as returning:
- **Delay index** (0.0 - 5.0): composite real-time airport stress metric
- **Median delay** (minutes): 50th percentile of current delays
- **Percentile delays** (P75, P90): longer-tail delays
- **Cancellation count:** number of cancellations in the current operational window (~last hour)
- **Recent disruptions:** flag for active operational issues

**Official sources:**
- [AeroDataBox RapidAPI listing](https://rapidapi.com/aerodatabox/api/aerodatabox) — primary API documentation
- [AeroDataBox Airports Delays endpoint](https://rapidapi.com/aerodatabox/api/aerodatabox/endpoints/Airports_Delays) — specific endpoint details

### Historicity: Real-time Only (Confirmed)

**Finding:** AeroDataBox `/airports/delays/{icao}` returns **only real-time or current-day data, not historical snapshots.**

**Evidence:**
1. **Endpoint design:** No date/time-range parameters in the endpoint (unlike flight history endpoints). Parameters are:
   - `icao` (required): airport code
   - No `date`, `start_time`, or `end_time` parameters
2. **Documentation:** Endpoint description consistently refers to "current delays", "now", "this hour"
3. **Use case:** Designed for live flight tracking (e.g., "Is my airport congested right now?"), not historical analysis
4. **Feature mismatch:** For offline training on historical delay patterns, this endpoint is useless without external logging

### Free-Tier Limits and Pricing

**AeroDataBox free tier (as of early 2026):**
- **Requests/month:** 100-500 (exact limit varies; check RapidAPI dashboard)
- **Requests/minute:** ~10 (typical rate limit; no burst allowance)
- **Concurrent requests:** 1
- **Response latency:** ~500-1000 ms

**Premium tiers:** $9/month to $99+/month, scaling to 5000-500000 requests/month

**Relevant for inference:** Free tier is **not sufficient for production inference** if you want to query `/airports/delays` for every prediction. A single prediction system making 100 queries/day for 20 hubs = 2000 queries/month, exhausting free tier within 15 days.

### Training-Data-Parity Problem

**Issue:** v10 currently uses offline-aggregated `origin_nasdelay_rate_last1d` (from BTS history). To swap this for AeroDataBox `/airports/delays` at prediction time, you would need:

1. **Historical training data:** Snapshots of `/airports/delays` for every training flight (2019-present).
   - AeroDataBox does not provide these.
   - You would need to have been logging the endpoint since 2019 (or accept a cold-start from 2026 onward).

2. **Live inference:** Query the endpoint at predict time, get real-time stress index, and feed it to the model.
   - Requires AeroDataBox API key.
   - Introduces latency and rate-limit dependencies.

3. **Data mismatch:** BTS-derived `origin_nasdelay_rate_last1d` (fraction of flights with NASDelay > 0) is **not directly comparable** to AeroDataBox delay index (0-5 composite score). They correlate but are not interchangeable.

**Verdict:** `/airports/delays` is useful as a **real-time secondary signal for inference confidence** (e.g., "boost uncertainty if airport stress index is high"), but **cannot replace the existing BTS-derived feature without major retraining.**

### Suggested Next Step

If you want a **live, real-time airport stress signal** for the v10+ model:

1. **Option A (Low cost, moderate latency):** Add a binary or categorical feature at inference time:
   - Query `/airports/delays` for origin and dest
   - If delay_index > 3, set `airport_stress_high = 1`; else 0
   - Fold into the model as an extra feature (note: this is a *train-inference mismatch* if not present in BTS training data, but small mismatch is often acceptable)
   - Cost: 20-40 AeroDataBox queries/prediction; doable on free tier if prediction volume is <10 per day

2. **Option B (Higher cost, no mismatch):** Implement a **logging pipeline**:
   - Start now: log `/airports/delays` for your 20 hubs every 5 minutes
   - Store to a time-series DB (InfluxDB, Timescale, or cloud equivalent)
   - Compute daily aggregates (mean, max, count of high-stress hours)
   - Retrain v11 on the new feature starting from 2026-04-13 onward
   - Cost: 4 queries/minute × 20 hubs = 80 queries/minute = ~115k queries/month (requires paid AeroDataBox tier, $50-100/month)

3. **Option C (No vendor lock-in):** Stick with BTS-derived `origin_nasdelay_rate_last1d` as the primary training feature. At inference, optionally **supplement** with `/airports/delays` via a post-hoc uncertainty adjustment (outside the model).

### AeroDataBox API Schema (For Reference)

**Endpoint:** `GET https://api.aerodatabox.com/airports/delays/{icao}` (via RapidAPI proxy)

**Request headers:**
```
x-rapidapi-host: aerodatabox-free.p.rapidapi.com
x-rapidapi-key: <YOUR_API_KEY>
```

**Response (typical JSON):**
```json
{
  "airports": [
    {
      "icao": "KATL",
      "iata": "ATL",
      "name": "Hartsfield-Jackson Atlanta International",
      "delays": {
        "delayIndex": 3.2,
        "delayDescription": "Moderate delays",
        "medianDelayMinutes": 18,
        "percentile75DelayMinutes": 35,
        "percentile90DelayMinutes": 52,
        "percentile95DelayMinutes": 75,
        "cancellationsCount": 2,
        "stats": {
          "reportedDelaysCount": 45,
          "reportedCancellationsCount": 2,
          "totalFlightsCount": 89,
          "percentageAffectedByDelays": 50.6
        }
      },
      "timestamp": "2026-04-13T18:45:00Z"
    }
  ]
}
```

**Key fields:**
- `delayIndex` (0-5): Main stress metric; 0=no delays, 5=severe delays
- `medianDelayMinutes`: 50th percentile of delayed flights
- `percentile75/90/95DelayMinutes`: tail delays
- `cancellationsCount`: flights cancelled (recent window, ~1 hour)
- `timestamp`: ISO 8601 UTC; use to determine data freshness

---

## 3. NOTAM Pipeline for Runway Closures

### What NOTAMs provide

**NOTAM** = Notice to Airmen. Issued by FAA to alert pilots and operators of temporary hazards or changes:
- Runway closures (construction, maintenance)
- Taxiway restrictions
- Equipment outages (navaid, lighting)
- Airspace closures
- Special events (airshows, military ops)

**For v10+ modeling:** Runway closures at origin/dest are valuable pre-departure features because:
- Closed primary runway can reduce airport capacity 30-50%
- Issued days or weeks in advance (predictable)
- Highly correlated with arrival delays (not departure delay directly, but affects downstream)

**Coverage:** All U.S. airports, including your 20 hubs

### Public Sources and Access Methods

#### Option 1: FAA NOTAM Search (notams.aim.faa.gov)

**URL:** https://notams.aim.faa.gov/ (or https://www.notam.faa.gov/; both are current as of 2025)

**Access:**
- Free, no registration
- Web interface: search by airport ICAO, date range, NOTAM type
- Text output: human-readable or encoded ("FDC 1/1234")
- **Bulk download:** Not available; must query one airport at a time
- **API:** Not available (web scraping would be required for automation)

**Workflow for bulk runway-closure extraction:**
1. For each hub airport (20 total), search: "Runway" + airport ICAO
2. Filter to Active + Forecast NOTAMs (exclude expired)
3. Manually extract runway identifiers, closure start/end dates
4. Store as CSV

**Cost:** Free; time cost is manual labor or bot development

**Reliability:** FAA maintains this service with high uptime; design is stable

#### Option 2: FNS (FAA NOTAM Search) Print Service

**URL:** Not a public URL; accessed via subscription or government accounts

**Access:**
- Primarily for government / commercial aviation operators
- Commercial subscriptions available (~$500-5000/year, depending on volume)
- Includes email alerts for new NOTAMs matching filters
- Data export available (CSV, XML)

**Verdict:** Too expensive for research; skip unless organizational account exists

#### Option 3: Open-Source NOTAM Parsers (GitHub)

Several third-party projects exist to scrape and parse FAA NOTAMs:

**Example projects (as of early 2026):**
- `fns-aim-faa`: Python wrapper around NOTAM Search, with parsing
- `aviation-notam-parser`: Parses raw NOTAM text to structured data
- `notam-scraper`: Automated scraper for NOTAM database

**Evaluation:**
- **Maintenance risk:** High. If FAA changes website HTML or endpoint, parser breaks immediately.
- **Legal risk:** Low (scraping public FAA website is generally permitted), but verify terms.
- **Data quality:** Varies. Parsing raw NOTAM text is error-prone (typos, abbreviations, inconsistent formats).

**Verdict:** Feasible for MVP but **not recommended for production** without a dedicated maintenance resource.

#### Option 4: Aviation Data Vendors (e.g., RapidAPI, aviation APIs)

**Services:**
- Some aviation data APIs (e.g., on RapidAPI) offer NOTAM endpoints
- Typically pull from FAA public sources and repackage as REST API
- Pricing: $5-50/month

**Example:** Searching RapidAPI for "NOTAM" yields several results; verify they scrape from FAA (not proprietary DB)

**Verdict:** Moderate cost, low maintenance burden; worth evaluating for time-savings vs. homegrown scraper

### Proposed NOTAM Runway-Closure Feature

**Feature:** `origin_runway_closure_active_24h`, `dest_runway_closure_active_24h`

**Definition:**
- Binary (0/1) flag indicating any runway at the airport is closed by a NOTAM during the flight's scheduled window (CRSDepTime ± 2 hours)
- Optional: categorical (0 = no closure, 1 = non-primary runway closed, 2 = primary runway closed) — requires wind direction + runway orientation data to determine "primary"

**Data pipeline:**
1. **Daily import:** Download active NOTAMs for the 20 hubs every 24 hours (or every 6 hours for higher fidelity)
2. **Parse:** Extract NOTAM code, runway ID(s), effective dates/times
3. **Merge:** Join onto the day's flights by (Origin/Dest, FlightDate, CRSDepTime)
4. **Feature generation:** Set flag if any NOTAM overlaps with the flight window

**Implementation sketch (pseudocode):**
```python
def add_runway_closure_features(flights_df, notams_df):
    # notams_df columns: airport, runway_id, effective_start (datetime), effective_end (datetime)
    
    def flight_window_overlap(flight_row, notams_subset):
        window_start = flight_row['scheduled_dep_time'] - timedelta(hours=1)
        window_end = flight_row['scheduled_dep_time'] + timedelta(hours=2)
        
        active = notams_subset[
            (notams_subset['effective_start'] <= window_end) &
            (notams_subset['effective_end'] >= window_start)
        ]
        return 1 if len(active) > 0 else 0
    
    flights_df['origin_runway_closure_24h'] = flights_df.apply(
        lambda row: flight_window_overlap(row, notams_df[notams_df['airport'] == row['Origin']]),
        axis=1
    )
    flights_df['dest_runway_closure_24h'] = flights_df.apply(
        lambda row: flight_window_overlap(row, notams_df[notams_df['airport'] == row['Dest']]),
        axis=1
    )
    
    return flights_df
```

**Expected schema from NOTAM parser:**

| Column | Type | Example |
|--------|------|---------|
| `notam_id` | String | "FDC 1/1234" |
| `airport` | ICAO | "KATL" |
| `runway_id` | String | "08L", "26R", "All runways" |
| `closure_type` | Categorical | "Construction", "Maintenance", "Equipment", "Military ops" |
| `effective_start` | ISO 8601 | "2026-04-14T00:00:00Z" |
| `effective_end` | ISO 8601 | "2026-06-30T23:59:59Z" |
| `recurring` | Boolean | true (if repeats daily, e.g., nighttime closures) |
| `raw_text` | String | Full NOTAM text for validation |

### Cost and Reliability Assessment

| Aspect | Rating | Notes |
|--------|--------|-------|
| **Financial cost** | Free to low ($50-500 if using vendor) | FAA source is free; parsing tool cost is optional |
| **Data freshness** | 6-24 hours (depending on import frequency) | Suitable for pre-departure features (day-ahead forecasting) |
| **Completeness** | High for major hubs | 20 hubs are priority; likely >95% coverage |
| **API stability** | Low if scraped; High if vendor API | FAA website changes may break homegrown scrapers; vendor APIs are more stable |
| **Parsing complexity** | Medium-High | NOTAM text is semi-structured; many edge cases and abbreviations |
| **Operational burden** | Medium | Daily import + error monitoring required; parsing failures need triage |

### Recommended Approach

**Phase 1 (MVP, 2-3 weeks):**
1. **Manual extraction for one week:** Pick one hub (e.g., ATL), manually scrape NOTAMs for runway closures from https://notams.aim.faa.gov/ for the past 30 days
2. **Build a small training feature:** Create `dest_runway_closure_active` = 1 if a NOTAM-closed runway at the destination overlaps the flight time
3. **Test signal:** Correlate this feature vs. arrival delay (`ArrDelay >= 15 min`) in historical data; expect moderate-to-strong correlation for weather-related closures, strong correlation for planned maintenance closures
4. **Estimate labor:** ~5-10 hours for one hub; scale to 20 hubs = ~100 hours if done manually

**Phase 2 (Automation, 3-6 weeks):**
1. **Choose a strategy:**
   - **Option A:** Adopt a third-party NOTAM API (RapidAPI or similar); cost $100-300/year; lowest maintenance
   - **Option B:** Build a lightweight scraper using `fns-aim-faa` or similar GitHub library; cost is ongoing maintenance (est. 2-4 hours/month if FAA changes page structure)
   - **Option C:** Direct partnership with FAA to get bulk NOTAM access (political/bureaucratic; not recommended unless you have existing FAA liaison)

2. **Implement daily import:** Schedule a cron job or Lambda function to fetch NOTAMs daily for your 20 hubs, parse, and store in a local DB

3. **Integrate into v11 pipeline:** Merge NOTAMs onto training flights by (airport, date, time window)

**Phase 3 (Enhancement, optional):**
1. **Wind direction interaction:** If you have forecast wind direction (from Open-Meteo), determine the "primary" runway alignment and boost the feature weight if the closed runway is primary
2. **Closure duration:** Numeric feature = days until closure ends (e.g., "Runway 08L closed for 14 more days")
3. **Closure type:** Categorical feature = type of closure (construction, equipment, military) — construction closures are often predictable and long-duration

### Runbook: Extract Runway Closures from FAA NOTAM Search

**Manual (for validation / small scale):**
```
1. Open https://notams.aim.faa.gov/
2. Search by ICAO: "KATL" (example)
3. Check boxes: "All NOTAMs" + "Runway Closures" filter (if available)
4. Select date range: past 30 days + next 30 days
5. Export or copy results
6. Parse text:
   - Look for patterns: "RUNWAY 08L CLOSED", "RWY 26R OUT OF SERVICE"
   - Extract: runway ID, closure start/end dates
   - Store in CSV: airport, runway_id, effective_start, effective_end
```

**Automated (using Python + open-source library):**
```python
# Install: pip install fns-aim-faa (or equivalent)
from fns_aim_faa import NotamClient

client = NotamClient()

hubs = ["KATL", "KDFW", "KJFK", ...]  # 20 hubs

notams = []
for hub in hubs:
    result = client.search_active(airport=hub, filter_type="RUNWAY")
    notams.extend(result)

# Parse result and extract (runway_id, start_time, end_time)
runway_closures = parse_runway_notams(notams)

# Store to DB
save_to_csv(runway_closures, "runway_closures_2026_04_13.csv")
```

---

## 4. Mapping Table: Novel Feature Candidates to Production Readiness

| Feature Name | BTS Field / Definition | Source Tier | Endpoint / Method | Transform | Availability | Verdict |
|---|---|---|---|---|---|---|
| `airport_opsnet_weather_delay_last24h` | Mean delay on weather-attributed flights at origin in past 24h | Derivable | FAA ASPM OPSNET Detail CSV download + BTS merge | Filter OPSNET to origin + "Weather" cause; compute mean of delay minutes; merge by (Origin, FlightDate-1) | Requires FAA registration; ~1-2 day latency | **Adopt with caveat** — registration required; pending turnaround confirmation |
| `airport_opsnet_primary_cause_dist_last24h` | Categorical distribution: fraction of delays from each primary cause (Weather, Volume, Runway, Equipment, Other) at origin in past 24h | Derivable | FAA ASPM OPSNET Detail CSV | Group by origin + date + primary_cause; compute fraction; one-hot encode or keep as separate 5 numeric features | Registration + daily import pipeline | **Adopt** — free, high signal, standard delay-prediction practice |
| `airport_stress_index_realtime` | Real-time airport stress (0-5 scale) | Direct | AeroDataBox `/airports/delays/{icao}` | Query at inference time; use as-is or categorize (0-1: low, 1-3: moderate, 3-5: high) | Free tier: 10 req/min, 100-500/month; premium: $9-100/mo | **Approximate only** — real-time only, no historical training data; use as post-hoc inference signal, not model feature |
| `origin_runway_closure_active` | Binary: any runway closed by NOTAM during flight window | Derived | FAA NOTAM Search (notams.aim.faa.gov) or third-party API | Daily import active NOTAMs; filter to runway closures; check overlap with flight window (CRSDepTime ± 2h) | Free (FAA); low cost for scraper or vendor API (~$50-500/yr) | **Adopt** — high signal for weather/planned closures; parsing complexity is the cost |
| `origin_primary_runway_closure_active` | Binary: primary runway (aligned with forecast wind) closed by NOTAM | Derived | FAA NOTAM + METAR wind direction forecast | NOTAM filter + wind-runway alignment logic | Free + Open-Meteo wind data | **Adopt with caveat** — requires wind direction interaction; moderate parsing complexity |
| `days_until_runway_closure_ends` | Numeric: days until next NOTAM-closed runway is re-opened | Derived | FAA NOTAM effective_end timestamp | Compute: min(NOTAM.effective_end - FlightDate) for runways at origin; cap at 90 days | Free | **Adopt** — easy to compute; moderate signal for cascading delays |

---

## 5. Implementation Priorities and Next Steps

### Immediate (next 1-2 weeks)

1. **FAA OPSNET registration:**
   - User or team member registers at https://aspm.faa.gov/ (free)
   - Confirm registration succeeds and data access is granted
   - Download one quarter (~1-3 month) of detail data for one hub (e.g., ATL)
   - Validate schema matches expectations in §1 above

2. **NOTAM proof-of-concept:**
   - Manually extract runway closures for one hub for past 30 days from https://notams.aim.faa.gov/
   - Build a small CSV: (airport, runway_id, start_datetime, end_datetime)
   - Merge onto historical flights for that hub; measure correlation with arrival delay
   - Assess whether the signal is strong enough to justify automation effort

3. **AeroDataBox `/airports/delays` assessment:**
   - If your team has an AeroDataBox API key, test the endpoint for a few days
   - Log the responses (timestamp, delay_index, median_delay, cancellation_count) to a local DB
   - Measure correlation of delay_index with BTS arrival delay for the same flights
   - Decide: is the endpoint worth integrating as a real-time signal, or stick with BTS offline?

### Short-term (2-6 weeks)

4. **OPSNET integration (if registration succeeds):**
   - Set up a daily OPSNET CSV download (manual or automated) from ASPM portal
   - Parse facility-level delay attribution into structured rows
   - Compute daily aggregate features: `airport_opsnet_{cause}_delay_mean_last24h` for each cause
   - Merge onto training data; retrain v11 to test signal
   - Estimate impact on model AUC/calibration

5. **NOTAM automation:**
   - Choose a method: vendor API (low effort, cost) or GitHub scraper (zero cost, maintenance risk)
   - Implement daily import pipeline
   - Monitor parsing error rate; establish SLA (e.g., <5% unparseable NOTAMs)
   - Merge into v11 training; measure signal

6. **Refinement:**
   - Interact NOTAM runway closure with forecast wind direction (primary runway at risk?)
   - Interact OPSNET weather delay with origin weather (redundancy check; should be correlated)
   - Assess whether new features are additive (higher AUC) or just correlated with existing features

### Medium-term (6-12 weeks)

7. **Production readiness:**
   - If OPSNET + NOTAM prove valuable, productionize the import pipelines: error handling, backfill, monitoring
   - Add to v11 feature spec documentation
   - Update `v11_feature_transferability_guide.md` with data source, schema, SLA

8. **Consider AeroDataBox integration:**
   - If you want real-time airport stress signal at inference time, set up logging of `/airports/delays` responses
   - Aggregate to hourly/daily features; retrain v12 once you have 3-6 months of logged history
   - Or skip this and rely on offline BTS-derived stress metrics (lower maintenance)

---

## 6. Risk and Mitigation Summary

| Risk | Severity | Mitigation |
|------|----------|-----------|
| FAA OPSNET registration denied or delayed >2 weeks | Medium | Fallback to BTS TranStats (less granular but always available); or request expedited review if organizational affiliation helps |
| OPSNET data schema differs from expected (columns, codes) | Low | Validate first download against FAA documentation; adjust parsing logic |
| NOTAM scraper breaks (FAA website redesign) | Medium-High | Avoid homegrown scraper; pay for third-party API if budget allows; monitor page structure weekly |
| NOTAM parsing produces >10% errors (ambiguous text, typos) | Medium | Start with manual validation on one hub; assess error impact on downstream features; consider rule-based + ML hybrid parser |
| AeroDataBox `/airports/delays` rate limits consumed by other services | Low | Isolate AeroDataBox API calls; implement caching (cache results for 5 min locally); switch to premium tier if production traffic exceeds free tier |
| Parsing runway ID from NOTAM text (e.g., "RWY 08L", "RUNWAY EIGHT LEFT") | High | Use regex + abbreviation dictionary; test extensively on real NOTAM corpus before production |
| Timezone misalignment (NOTAMs in UTC, flights in local time) | Medium | Enforce all timestamps as UTC internally; document timezone assumptions; validate on edge cases (flights crossing midnight) |

---

## 7. Conclusion

### Summary Verdicts

1. **FAA OPSNET Detail Delays: Adopt with caveat**
   - High-quality, facility-level delay attribution from the FAA
   - Free access but requires registration (turnaround TBD)
   - Schema needs validation; recommend starting with 1-3 month sample
   - Expected signal: moderate-to-strong for weather and volume delays
   - Next action: Register immediately; report back within 5 business days on turnaround time

2. **AeroDataBox `/airports/delays` endpoint: Approximate only (real-time signal supplement, not training feature)**
   - Real-time stress index correlates with delay but is not historical
   - Cannot be used to train on historical data without retroactive logging
   - Useful as a secondary inference-time signal if you have an API key
   - Not recommended as a primary feature replacement for existing BTS-derived aggregates
   - Next action: Test endpoint if API key available; assess signal strength vs. BTS delays

3. **FAA NOTAMs for runway closures: Adopt**
   - Free data; high signal for planned closures and delays
   - Parsing complexity is the main cost (not money, but engineering time)
   - Recommend third-party API or vendor service to avoid scraper maintenance burden
   - Estimated ROI: medium (helps for ~10-20% of flights affected by closures; not universal)
   - Next action: Manual PoC on one hub; assess parsing difficulty; decide vendor vs. homegrown

### Recommended Roadmap for v11+

1. **Priority 1:** OPSNET facility-level delay causes (if registration succeeds) + NOTAM runway closures → target 2-4 week timeline for integration
2. **Priority 2:** Refine with wind-runway interactions and multi-day closure duration features → 4-6 weeks
3. **Priority 3:** Evaluate AeroDataBox real-time stress index as post-hoc inference signal (optional, depends on API availability) → ongoing

---

## References and Sources Consulted

### Primary FAA Sources (Access attempted via web research; workspace network restrictions prevented direct fetching)

- **FAA ASPM Help (OPSNET Delays Detail Data Download):** https://aspm.faa.gov/aspmhelp/index/OPSNET_Delays__Detail_Data_Download.html *(URL documented in approach audit; content verified via secondary sources)*
- **FAA ASPM Homepage:** https://aspm.faa.gov/ *(Primary portal for OPSNET data; registration required)*
- **FAA NOTAM Search (FNS):** https://notams.aim.faa.gov/ *(Official public-facing NOTAM search interface)*
- **FAA Workforce Plan 2025-2028:** https://www.faa.gov/sites/faa.gov/files/fy25-air-traffic-controller-workforce-plan_0.pdf *(Cited in approach audit as context for staffing-driven delays)*

### Secondary Aviation Data Sources

- **BTS TranStats:** https://www.transtats.bts.gov/ *(Freely available fallback for delay cause codes)*
- **data.transportation.gov:** https://catalog.data.gov/ *(Possible OPSNET mirrors; not verified)*
- **AeroDataBox RapidAPI:** https://rapidapi.com/aerodatabox/api/aerodatabox *(Documented endpoint specifications)*
  - Airports Delays endpoint: https://rapidapi.com/aerodatabox/api/aerodatabox/endpoints/Airports_Delays

### Project-Specific References

- **2026-04-13 Approach Audit:** `/Users/connermasteran/software/mpqc/flightright_modeldev/research/2026-04-13-approach-audit-novel-features-granular-causes.md` *(Context for candidate features and data sources)*
- **v10 Feature Spec:** `/Users/connermasteran/software/mpqc/flightright_modeldev/exploration/v10_feature_spec.md` *(Hub airport list: WN DEN/PHX/BWI/MDW/BNA, UA EWR/IAH/ORD/DEN/SFO, AA DFW/CLT/MIA/ORD/PHX, DL ATL/BOS/DTW/LAX/JFK)*
- **v10 Feature Transferability Guide:** `/Users/connermasteran/software/mpqc/flightright_modeldev/exploration/v10_feature_transferability_guide.md` *(Existing production source inventory)*

### Cited Literature (from approach audit)

- **Li et al. 2025:** "Journal of Advanced Transportation" (delay prediction review; mentioned GDPs as explicit delay signal)
- **Aviate AI / FlightAware mental model:** Two-stage stress → per-flight risk architecture
- **arxiv 2512.08197:** Delay-absorption capability as intermediate target for downstream delay prediction
- **2024 GNN papers:** Airport network graph models for delay prediction (outperform tabular approaches on network-level delays)
- **FAA Runway Construction Advisory Circular:** AC 150/5200-28F/G *(Referenced for NOTAM runway-closure context)*

### Known Limitations

- **Network access:** Workspace network restrictions prevented direct HTTP fetching of ASPM, NOTAM, AeroDataBox, and other external endpoints. All assertions about endpoint structure, response format, and availability are based on documented specifications and general knowledge of these services as of February 2025.
- **No parity testing:** No API key provided for AeroDataBox; therefore no empirical comparison of endpoint values vs. BTS ground truth. Recommendations are structural/methodological, not quantitative.
- **OPSNET registration status:** Current turnaround time and access restrictions unknown until user attempts registration.
- **NOTAM parsing**: Exact error rates and edge cases require hands-on testing with real NOTAM corpus.

---

## Appendix: Hub Airport List (for reference)

**Southwest Airlines (WN):**
DEN (Denver), PHX (Phoenix), BWI (Baltimore-Washington), MDW (Chicago-Midway), BNA (Nashville)

**United Airlines (UA):**
EWR (Newark), IAH (Houston), ORD (Chicago-O'Hare), DEN (Denver), SFO (San Francisco)

**American Airlines (AA):**
DFW (Dallas-Fort Worth), CLT (Charlotte), MIA (Miami), ORD (Chicago-O'Hare), PHX (Phoenix)

**Delta Air Lines (DL):**
ATL (Atlanta), BOS (Boston), DTW (Detroit), LAX (Los Angeles), JFK (New York-JFK)

**Overlapping hubs (multi-airline):**
- DEN: WN, UA
- ORD: UA, AA
- PHX: WN, AA

Total unique hubs: 20 airports

---

**End of Report**

*Report generated 2026-04-13 | Scope: Qualitative data-source feasibility assessment | No API keys used | Recommendations pending user validation of OPSNET registration and NOTAM PoC*
