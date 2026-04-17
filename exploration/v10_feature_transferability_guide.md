# v10 Feature Transferability Guide

**Purpose:** Teach another agent or team how to produce every v10 feature using AeroDataBox as the production data source.
**Date:** 2026-04-10
**Authoritative spec:** `exploration/v10_feature_spec.md`

---

## 1. Why v10 Exists

v9 added 11 tail-number-derived features on top of v8. In production, v9 became dramatically pessimistic because:

- **AeroDataBox does not publish tail numbers** for upcoming flights. Every tail-derived feature (`aircraft_type`, `has_recent_arrival_turn_5h`, `turn_time_hours`, `tail_leg_num_day`, `tail_depdelay_mean_last{1,14}`, `tail_lateaircraft_rate_last{1,14}`, `tail_n_legs_scheduled`, `tail_min_turn_time`, `tail_has_tight_turn`) is impossible to compute at predict time.
- **AeroDataBox does not report actual gate-out/gate-in or wheels-off times** in a usable way. Every feature derived from `WheelsOff`/`WheelsOn`/`AirTime` (`wo_slip_*`, `airtime_*_d_*`) is also off the table.
- The v9 rolling-mean features were extremely sensitive to recent severe delays, causing pessimistic predictions on bad ops days.

**v10 starts from v8** (which performed better in production) and:
1. Strips every tail/aircraft-derived feature.
2. Strips every wheels-off/actual-times-derived historical aggregate.
3. Promotes 1-day delay baselines from `*_mean_last1` to `*_median_last1` (computed over individual flight rows, not daily means) to reduce outlier sensitivity.
4. Adds 4 new AeroDataBox-compatible features that don't need tail or wheels-off data.

---

## 2. The AeroDataBox Constraint

Every v10 feature must satisfy this rule:

> **Producible at predict time without a tail number and without any actual gate or wheels-off observation.**

At predict time, four data sources are available:

| Source | What it provides | When |
|--------|-----------------|------|
| **AeroDataBox** (live) | Schedule fields: `CRSDepTime`, `CRSElapsedTime`, `CRSArrTime`, `Origin`, `Dest`, `DepTimeBlk`, `Distance`, airline, flight number | At predict time |
| **BTS historical data** (offline) | All rolling aggregates (delay means/medians, late-aircraft rates, NAS rates, cancel rates, divert rates, congestion counts, elapsed-time ratios) | Pre-computed daily from BTS, merged by (keys + FlightDate) |
| **Open-Meteo** (live) | Origin and destination weather forecasts (daily + hourly) | Day-before forecast query |
| **Strike cache** (offline) | `strike_severity`, `days_to_strike`, `carrier_delay_rate_anomaly_7d` | Refreshed periodically from labor action tracking |

**Forbidden at predict time:** Tail numbers, WheelsOff, WheelsOn, AirTime, TaxiOut, TaxiIn, ActualElapsedTime (for the target flight), gate-out/gate-in times.

---

## 3. What Changed: v8 to v10

### 3a. Features ADDED in v10 (4 new features)

#### `origin_nasdelay_rate_last1d` (departure + arrival)

- **What:** Fraction of flights at the origin airport with `NASDelay > 0` over the previous 1 day.
- **Why:** Captures day-over-day spikes in ATC-driven delays (ground stops, airspace congestion). Strongest new single-feature signal (point-biserial 0.115 vs `y_dep_ge15`).
- **BTS source column:** `NASDelay`
- **Computation:**
  1. From BTS history, filter to the origin airport.
  2. For each day, compute: `count(NASDelay > 0) / count(all flights)`.
  3. Shift forward by 1 day (so today's feature sees yesterday's data only).
  4. Merge onto target flight by `(Origin, FlightDate)`.
- **Output range:** 0.0 to 1.0 (rate).
- **Implementation:** `src/fetch_prune/features_dep.py:add_origin_nasdelay_rate_last1d()` (line 1114)
- **Blueprint flag:** `features_dep.nas_rate_last1d.enabled`
- **AeroDataBox inference:** At predict time, this is pre-computed from BTS history (or approximated via the AeroDataBox Airport Delays endpoint `GET /airports/delays/{airportIcao}`, which returns an aggregate delay index correlated with NAS delays).

#### `cancel_rate_origin_last1d` (departure + arrival)

- **What:** Fraction of flights at the origin airport marked Cancelled over the previous 1 day.
- **Why:** Cancellation spikes signal severe operational disruption (weather, ATC, crew shortages). Surviving flights on high-cancellation days face rebooking chaos. This is orthogonal to delay magnitudes -- a day can have many cancellations but low mean delay for surviving flights.
- **BTS source column:** `Cancelled` (binary 0/1)
- **Computation:**
  1. From BTS history, filter to origin airport.
  2. For each day, compute: `mean(Cancelled)` (= fraction cancelled).
  3. Shift forward by 1 day.
  4. Merge onto target flight by `(Origin, FlightDate)`.
- **Output range:** 0.0 to 1.0 (rate).
- **Implementation:** `src/fetch_prune/features_dep.py:add_cancel_rate_origin_last1d()` (line 1142)
- **Blueprint flag:** `features_dep.cancel_rate_last1d.enabled`
- **AeroDataBox inference:** Flight status endpoint returns `status="Cancelled"`. Count cancelled vs total at the airport from FIDS history.

#### `divert_rate_origin_last14d` (departure + arrival)

- **What:** Fraction of flights at the origin airport marked Diverted over the previous 14 days.
- **Why:** Diversions signal extreme conditions (severe weather, ground stops, security events). Ripple effects persist for days after the event clears.
- **BTS source column:** `Diverted` (binary 0/1)
- **Computation:**
  1. From BTS history, group by `(Origin, FlightDate)`, compute daily diversion rate = `mean(Diverted)`.
  2. Sort by `(Origin, FlightDate)`.
  3. Within each Origin group, apply `shift(1).rolling(window=14, min_periods=1).mean()` to get the 14-day rolling mean of daily rates, shifted by 1 day.
  4. Merge onto target flight by `(Origin, FlightDate)`.
- **Output range:** 0.0 to ~0.05 typically (diversions are rare).
- **Implementation:** `src/fetch_prune/features_dep.py:add_divert_rate_origin_last14d()` (line 1170)
- **Blueprint flag:** `features_dep.divert_rate_last14d.enabled`
- **AeroDataBox inference:** Flight status field includes diversion info. Compute from recent flight status history.

#### `elapsed_time_ratio_last14d` (arrival model ONLY)

- **What:** Rolling mean of `ActualElapsedTime / CRSElapsedTime` for `(Airline, FlightNum, Origin, Dest)` over the previous 14 days.
- **Why:** Flights that consistently run long (ratio > 1.05) may have unrealistic schedules or face persistent headwinds/ATC routing. Flights running short have padded schedules (low delay risk).
- **BTS source columns:** `ActualElapsedTime`, `CRSElapsedTime` -- both are gate-based (gate-out to gate-in elapsed time), NOT wheels-off-derived.
- **Computation:**
  1. From BTS history, compute `ratio = ActualElapsedTime / CRSElapsedTime` per flight row.
  2. Group by `(Reporting_Airline, Flight_Number_Reporting_Airline, Origin, Dest)`.
  3. Rolling 14-day mean with 1-day shift to avoid leakage.
  4. Merge onto target flight by the same 4 keys + FlightDate.
- **Output range:** Typically 0.8 to 1.3.
- **Implementation:** `src/fetch_prune/features_arr.py:add_elapsed_time_ratio_stats()` (line 233)
- **Blueprint flag:** Controlled in the arrival blueprint (enabled when `ActualElapsedTime` and `CRSElapsedTime` are present in history).
- **AeroDataBox inference:** Compute from flight history endpoint: `(actual arrival - actual departure) / (scheduled arrival - scheduled departure)`.

### 3b. Median-Last1 Replacements (v10 hybrid rule)

v10 replaces the 1-day rolling MEAN baselines with MEDIAN baselines for delay magnitudes. The 7-day and 14-day windows remain as means.

**Why:** A single extremely delayed flight (e.g., 300-minute delay) would dominate the daily mean and cause pessimistic predictions for all flights the next day. The median is robust to these outliers.

**How it works:**
1. Helper `_daily_median_by(hist, group_cols, value_col)` computes the median of `value_col` over individual flight rows grouped by `group_cols + FlightDate`.
2. Helper `_attach_shifted_daily_median(df, daily_med, group_cols, out_col)` shifts the result forward by 1 day and merges onto the target dataframe.

| Old feature (v8) | New feature (v10) | Group keys |
|---|---|---|
| `carrier_depdelay_mean_last1` | `carrier_depdelay_median_last1` | `Reporting_Airline` |
| `carrier_origin_depdelay_mean_last1` | `carrier_origin_depdelay_median_last1` | `Reporting_Airline, Origin` |
| `hub_{i}_depdelay_mean_last1` | `hub_{i}_depdelay_median_last1` | `Origin` (filtered to hub IATA) |
| `dest_depdelay_mean_last1` | `dest_depdelay_median_last1` | `Origin` in history, merged by `Dest` |
| `flightnum_od_depdelay_mean_last1` | `flightnum_od_depdelay_median_last1` | (Already existed in v8 as median) |

**Implementation:**
- `src/fetch_prune/features_dep.py:add_carrier_dep_delay_median_last1()` (line 1051) -- emits both `carrier_depdelay_median_last1` and `carrier_origin_depdelay_median_last1`
- `src/fetch_prune/features_dep.py:add_dest_depdelay_median_last1()` (line 1062)
- `src/fetch_prune/features_dep.py:add_hub_depdelay_median_last1()` (line 1076)
- Blueprint flag: `features_dep.delay_median_last1.enabled`

For arrival models, v10 also adds median variants for route arrival-delay history:
- `arrdelay_median_14d_fn_od` (NEW)
- `arrdelay_median_14d_car_od` (NEW)

### 3c. Features REMOVED from v9 (all AeroDataBox-incompatible)

**Tail/aircraft features (11 removed):**
- `aircraft_type` (categorical)
- `has_recent_arrival_turn_5h`, `turn_time_hours`, `tail_leg_num_day`
- `tail_depdelay_mean_last1`, `tail_depdelay_mean_last14`
- `tail_lateaircraft_rate_last1`, `tail_lateaircraft_rate_last14`
- `tail_n_legs_scheduled`, `tail_min_turn_time`, `tail_has_tight_turn`

**Wheels-off/airtime features (16 removed from arrival):**
- `wo_slip_mean_{7,14}d_fn_od`, `wo_slip_n_{7,14}d_fn_od`
- `wo_slip_mean_{7,14}d_origin_blk`, `wo_slip_n_{7,14}d_origin_blk`
- `airtime_mean_{7,14}d_fn_od`, `airtime_n_{7,14}d_fn_od`
- `airtime_mean_{7,14}d_car_od`, `airtime_n_{7,14}d_car_od`

---

## 4. Complete v10 Feature Inventory

### 4a. Departure Model (80 features: 4 categorical + 76 numeric)

#### Categorical (4)

| Feature | Source | How to obtain |
|---------|--------|---------------|
| `Origin` | AeroDataBox schedule | Origin airport IATA code |
| `Dest` | AeroDataBox schedule | Destination airport IATA code |
| `DepTimeBlk` | Derived from `CRSDepTime` | Map HHMM to block string (e.g., "0600-0659") |
| `origin_dep_hour_weathercode` | Open-Meteo | WMO weather code at origin at departure hour |

#### Numeric: Schedule & Route (6)

| Feature | Source | How to obtain |
|---------|--------|---------------|
| `CRSDepTime` | AeroDataBox schedule | Scheduled departure HHMM integer |
| `CRSElapsedTime` | AeroDataBox schedule | Scheduled block time in minutes |
| `Distance` | AeroDataBox schedule or airport DB | Great-circle distance in statute miles |
| `dep_dow` | Derived from `FlightDate` | Day of week (0=Mon, 6=Sun) |
| `sched_dep_hour` | Derived from `CRSDepTime` | Hour component (0-23) |
| `is_peak_hour` | Derived from `sched_dep_hour` | 1 if hour in {16,17,18,19,20}, else 0 |

#### Numeric: Flight-Number / OD History (6)

All computed from BTS history grouped by `(Reporting_Airline, Flight_Number_Reporting_Airline, Origin, Dest)`:

| Feature | Window | Statistic |
|---------|--------|-----------|
| `flightnum_od_depdelay_mean_last7` | 7 days | Mean DepDelayMinutes |
| `flightnum_od_depdelay_mean_last14` | 14 days | Mean DepDelayMinutes |
| `flightnum_od_depdelay_median_last1` | 1 day | Median DepDelayMinutes |
| `flightnum_od_depdelay_median_last7` | 7 days | Median DepDelayMinutes |
| `flightnum_od_depdelay_median_last14` | 14 days | Median DepDelayMinutes |
| `flightnum_od_support_count_last14d` | 14 days | Count of flights |

All shifted by 1 day to prevent leakage.

#### Numeric: Carrier / Origin Baselines (6)

| Feature | Window | Statistic | Group keys |
|---------|--------|-----------|------------|
| `carrier_depdelay_median_last1` | 1 day | **Median** over flight rows (NEW v10) | `Reporting_Airline` |
| `carrier_depdelay_mean_last7` | 7 days | Mean of daily means | `Reporting_Airline` |
| `carrier_origin_depdelay_median_last1` | 1 day | **Median** over flight rows (NEW v10) | `Reporting_Airline, Origin` |
| `carrier_origin_depdelay_mean_last7` | 7 days | Mean of daily means | `Reporting_Airline, Origin` |
| `carrier_origin_depdelay_mean_last14` | 14 days | Mean of daily means | `Reporting_Airline, Origin` |
| `origin_depdelay_mean_last14` | 14 days | Mean of daily means | `Origin` |

#### Numeric: Late-Aircraft Rates (4)

| Feature | Window | Group keys |
|---------|--------|------------|
| `origin_lateaircraft_rate_last1` | 1 day | `Origin` |
| `origin_lateaircraft_rate_last7` | 7 days | `Origin` |
| `carrier_lateaircraft_rate_last1` | 1 day | `Reporting_Airline` |
| `carrier_lateaircraft_rate_last7` | 7 days | `Reporting_Airline` |

Computed as: fraction of flights with `LateAircraftDelay > 0`. These stay as means (rates are not outlier-sensitive like delay magnitudes).

#### Numeric: Hub Spillover (31)

Hub airports are airline-specific:
- **WN:** DEN, PHX, BWI, MDW, BNA (hub_0 through hub_4)
- **UA:** EWR, IAH, ORD, DEN, SFO
- **AA:** DFW, CLT, MIA, ORD, PHX
- **DL:** ATL, BOS, DTW, LAX, JFK

For each hub `i` in {0..4}:

| Feature | Window | Statistic |
|---------|--------|-----------|
| `hub_{i}_depdelay_median_last1` | 1 day | **Median** over flight rows (NEW v10) |
| `hub_{i}_depdelay_mean_last7` | 7 days | Mean of daily means |
| `hub_{i}_depdelay_mean_last14` | 14 days | Mean of daily means |
| `hub_{i}_lateaircraft_rate_last1` | 1 day | Rate |
| `hub_{i}_lateaircraft_rate_last7` | 7 days | Rate |
| `hub_{i}_lateaircraft_rate_last14` | 14 days | Rate |

Plus: `hub_max_lateaircraft_last1` = max of `hub_{0..4}_lateaircraft_rate_last1`.

Total: 5 hubs x 6 features + 1 = 31.

#### Numeric: Destination Baselines (4)

| Feature | Window | Statistic |
|---------|--------|-----------|
| `dest_depdelay_median_last1` | 1 day | **Median** over flight rows (NEW v10, replaces mean) |
| `dest_lateaircraft_rate_last1` | 1 day | Rate |
| `dest_lateaircraft_rate_last7` | 7 days | Rate |
| `dest_lateaircraft_rate_last14` | 14 days | Rate |

Note: `dest_depdelay_median_last1` is built from BTS history keyed on `Origin` (since the historical pool sees an airport as an Origin), then merged onto `df` by `Dest`.

#### Numeric: Congestion (2)

| Feature | How to compute |
|---------|---------------|
| `origin_congestion_3h_total` | Count of all scheduled departures from origin within +/-1.5h of this flight's CRSDepTime |
| `origin_airline_congestion_3h_total` | Same, filtered to same airline only |

Source: Computed from the day's full schedule (AeroDataBox FIDS or BTS).

#### Numeric: Origin Weather -- Daily (4)

| Feature | Open-Meteo field | Units |
|---------|-----------------|-------|
| `origin_temp_max_K` | `temperature_2m_max` | Kelvin (C + 273.15) |
| `origin_temp_min_K` | `temperature_2m_min` | Kelvin |
| `origin_daily_precip_sum_mm` | `precipitation_sum` | mm |
| `origin_daily_windgusts_max_kmh` | `windgusts_10m_max` | km/h |

#### Numeric: Origin Weather -- Hourly at Departure (7)

| Feature | Open-Meteo field | Notes |
|---------|-----------------|-------|
| `origin_dep_temp_K` | `temperature_2m` at dep hour | Kelvin |
| `origin_dep_precip_mm` | `precipitation` at dep hour | mm |
| `origin_dep_windgusts_kmh` | `windgusts_10m` at dep hour | km/h |
| `origin_dep_visibility_m` | `visibility` at dep hour | meters, clipped to 25000 |
| `origin_dep_cape_jkg` | `cape` at dep hour | J/kg (convective instability) |
| `origin_dep_log1p_cape` | `log1p(cape)` | Compresses heavy tail |
| `origin_dep_cloudcover_pct` | `cloudcover` at dep hour | Percentage |

#### Numeric: Derived Weather (1)

| Feature | Computation |
|---------|------------|
| `wind_x_precip` | `origin_dep_windgusts_kmh * origin_dep_precip_mm` |

#### Numeric: Strike (3)

| Feature | Source |
|---------|--------|
| `strike_severity` | Strike cache: severity score (0-10) of nearest active/announced labor action |
| `days_to_strike` | Days until nearest announced strike (0-90 window) |
| `carrier_delay_rate_anomaly_7d` | Z-score: (carrier's 7-day delay rate - 60-day baseline) / 60-day std |

#### Numeric: NEW v10 AeroDataBox Aggregates (3)

| Feature | Definition | Source column | Window |
|---------|-----------|---------------|--------|
| `origin_nasdelay_rate_last1d` | Fraction of flights at origin with NASDelay > 0 | `NASDelay` | 1 day |
| `cancel_rate_origin_last1d` | Fraction of flights at origin marked Cancelled | `Cancelled` | 1 day |
| `divert_rate_origin_last14d` | Fraction of flights at origin marked Diverted | `Diverted` | 14 days (rolling) |

**Departure totals: 4 categorical + 76 numeric = 80 features.**

---

### 4b. Arrival Model (~106 features)

The arrival model inherits ALL 80 departure features (merged via `merge_dep_features` step), then adds:

#### Schedule (2)

| Feature | Source |
|---------|--------|
| `CRSArrTime` | AeroDataBox schedule (HHMM integer) |
| `sched_arr_hour` | Hour component of CRSArrTime |

#### Route Arrival-Delay History (8)

Grouped by `(Reporting_Airline, Flight_Number_Reporting_Airline, Origin, Dest)` (fn_od) or `(Reporting_Airline, Origin, Dest)` (car_od):

| Feature | Window | Statistic |
|---------|--------|-----------|
| `arrdelay_mean_7d_fn_od` / `arrdelay_n_7d_fn_od` | 7 days | Mean + count |
| `arrdelay_mean_14d_fn_od` / `arrdelay_n_14d_fn_od` | 14 days | Mean + count |
| `arrdelay_median_7d_fn_od` | 7 days | Median |
| `arrdelay_median_14d_fn_od` | 14 days | Median (NEW v10) |
| `arrdelay_mean_7d_car_od` / `arrdelay_n_7d_car_od` | 7 days | Mean + count |
| `arrdelay_mean_14d_car_od` / `arrdelay_n_14d_car_od` | 14 days | Mean + count |
| `arrdelay_median_7d_car_od` | 7 days | Median |
| `arrdelay_median_14d_car_od` | 14 days | Median (NEW v10) |

#### Destination Arrival Congestion (4)

| Feature | How to compute |
|---------|---------------|
| `dest_arrivals_pm60_sched` | Scheduled arrivals at Dest within +/-60 min of CRSArrTime |
| `dest_airline_arrivals_pm60_sched` | Same, filtered to same airline |
| `dest_arrivals_pm60_eta` | Arrivals within +/-60 min of ETA (CRSArrTime + `arrdelay_mean_14d_car_od`) |
| `dest_airline_arrivals_pm60_eta` | Same, filtered to same airline |

#### Destination Weather at Arrival (7)

| Feature | Open-Meteo field |
|---------|-----------------|
| `dest_arr_temperature_2m` | `temperature_2m` at dest at arrival hour |
| `dest_arr_precipitation` | `precipitation` at dest |
| `dest_arr_windspeed_10m` | `windspeed_10m` at dest |
| `dest_arr_windgusts_10m` | `windgusts_10m` at dest |
| `dest_arr_visibility` | `visibility` at dest |
| `dest_arr_cape` | `cape` at dest |
| `dest_arr_cloudcover` | `cloudcover` at dest |

#### NEW v10 Elapsed-Time Signal (1)

| Feature | Definition |
|---------|-----------|
| `elapsed_time_ratio_last14d` | Rolling 14-day mean of `ActualElapsedTime / CRSElapsedTime` for (Airline, FlightNum, Origin, Dest) |

**Arrival totals: ~80 inherited dep + 22 arrival-specific + 4 categoricals = ~106 features.**

---

### 4c. Cancellation Model

Uses the v8-equivalent feature list (stripped of all tail features) plus:
- Same median-last1 replacements as departure model
- Same three new AeroDataBox aggregates
- Cross-airline categoricals: `Origin`, `Dest`, `Reporting_Airline`, `DepTimeBlk`
- Training window starts 2022 (longer history for class balance)

---

## 5. AeroDataBox API Mapping

### 5a. Schedule fields (live at predict time)

AeroDataBox provides these directly from its flight/airport schedule endpoints:

| v10 Feature | AeroDataBox field |
|-------------|-------------------|
| `Origin` | Departure airport IATA code |
| `Dest` | Arrival airport IATA code |
| `CRSDepTime` | `departure.scheduledTime` (convert to HHMM integer) |
| `CRSArrTime` | `arrival.scheduledTime` (convert to HHMM integer) |
| `CRSElapsedTime` | Compute: `CRSArrTime - CRSDepTime` adjusted for timezone (minutes) |
| `Distance` | Not directly in AeroDataBox; use airport coordinates or a lookup table |
| `DepTimeBlk` | Derive from CRSDepTime |
| `Reporting_Airline` | Airline IATA code from the flight record |
| `Flight_Number_Reporting_Airline` | Flight number from the flight record |

### 5b. Rolling aggregates (pre-computed from BTS, served at predict time)

These are NOT fetched live from AeroDataBox. They are pre-computed offline from BTS historical data and stored in a lookup table. At predict time, you merge them onto the incoming flight by the appropriate keys + date.

However, in a pure AeroDataBox production system, you could rebuild these aggregates from AeroDataBox flight history data:

| Aggregate family | AeroDataBox source | Notes |
|-----------------|-------------------|-------|
| Delay means/medians (carrier, origin, hub, dest, flightnum_od) | Flight status endpoint: compare `actual` vs `scheduled` times | Compute `DepDelayMinutes = actual_dep - scheduled_dep` in minutes |
| Late-aircraft rates | Not directly available | Would need to infer from cascading delay patterns or use the BTS-derived offline values |
| NAS delay rate | Airport Delays endpoint: `GET /airports/delays/{airportIcao}` | The delay index (0-5) correlates with NAS delays; for exact rates, use BTS offline |
| Cancel rate | Flight status endpoint: count `status="Cancelled"` | Directly available from FIDS history |
| Divert rate | Flight status endpoint: diversion info in status | Available from flight status |
| Elapsed-time ratio | Flight history: `(actual_arr - actual_dep) / (sched_arr - sched_dep)` | Gate-based times are available |
| Congestion | Airport FIDS: count scheduled flights in time window | Directly computable from schedule data |

### 5c. AeroDataBox Airport Delays Endpoint

`GET /airports/delays/{airportIcao}`

Returns:
- Delay index (0.0 - 5.0): composite airport stress score
- Median delay (minutes)
- Percentile delays (P75, P90, etc.)
- Cancellation count

This endpoint provides a real-time composite signal. While v10 does not use it as a direct feature, it could serve as a proxy for `origin_nasdelay_rate_last1d` or `cancel_rate_origin_last1d` in a live system.

### 5d. Weather (Open-Meteo, live)

All weather features are obtained from Open-Meteo's forecast API, queried the day before for the flight date. AeroDataBox is not involved in weather features.

### 5e. Strike data (offline cache)

Strike features come from an offline labor action tracking cache. AeroDataBox is not involved.

---

## 6. Leakage Prevention Rules

Every rolling aggregate follows the same leakage prevention pattern:

1. **1-day shift:** The feature for FlightDate `D` is computed from data up to and including `D-1`. The code achieves this by adding `pd.Timedelta(days=1)` to the FlightDate in the lookup table before merging.
2. **No same-day data:** A flight on 2024-03-15 only sees history through 2024-03-14.
3. **Lookback warmup:** The first ~14-60 days of the training window will have incomplete rolling windows. The pipeline uses `min_periods=1` so these rows get values (from fewer days), not NaN.

---

## 7. Blueprint Configuration

Each feature group in v10 is gated by a flag in the blueprint JSON. Example from `data/blueprint_dep_WN_v10.json`:

```json
{
  "features_dep": {
    "tail_history": { "enabled": false },
    "delay_median_last1": { "enabled": true },
    "nas_rate_last1d": { "enabled": true },
    "cancel_rate_last1d": { "enabled": true },
    "divert_rate_last14d": { "enabled": true }
  },
  "add_aircraft_type": false
}
```

For arrival blueprints, wheels-off features are gated by:
- `recent_airtime.enabled: false`
- `wheels_off_slip.enabled: false`

The `elapsed_time_ratio` feature is enabled when `ActualElapsedTime` and `CRSElapsedTime` are present in the history pool.

---

## 8. Implementation Reference

| File | Purpose |
|------|---------|
| `src/fetch_prune/features_dep.py` | All departure feature computation |
| `src/fetch_prune/features_arr.py` | All arrival-specific feature computation |
| `src/fetch_prune/prepare_dataset.py` | Loads BTS data, filters columns, builds history pool |
| `src/fetch_prune/prepare_cancel_dataset.py` | Same for cancellation model |
| `data/blueprint_dep_{airline}_v10.json` | Departure pipeline config per airline |
| `data/blueprint_arr_{airline}_v10.json` | Arrival pipeline config per airline |
| `data/blueprint_cancel_v10.json` | Cancellation pipeline config |
| `exploration/v10_feature_spec.md` | Authoritative v10 feature specification |
| `exploration/19_new_feature_brainstorm_aerodatabox.py` | Feature candidate research and evaluation |

### Key functions for v10-specific features:

| Function | File:Line | What it does |
|----------|-----------|-------------|
| `_daily_median_by()` | `features_dep.py:994` | Computes median of a value column over flight rows, grouped by keys + FlightDate |
| `_attach_shifted_daily_median()` | `features_dep.py:1020` | Shifts daily medians forward 1 day and merges onto target dataframe |
| `add_carrier_dep_delay_median_last1()` | `features_dep.py:1051` | Emits `carrier_depdelay_median_last1` and `carrier_origin_depdelay_median_last1` |
| `add_dest_depdelay_median_last1()` | `features_dep.py:1062` | Emits `dest_depdelay_median_last1` |
| `add_hub_depdelay_median_last1()` | `features_dep.py:1076` | Emits `hub_{i}_depdelay_median_last1` for each hub |
| `add_origin_nasdelay_rate_last1d()` | `features_dep.py:1114` | Emits `origin_nasdelay_rate_last1d` |
| `add_cancel_rate_origin_last1d()` | `features_dep.py:1142` | Emits `cancel_rate_origin_last1d` |
| `add_divert_rate_origin_last14d()` | `features_dep.py:1170` | Emits `divert_rate_origin_last14d` |
| `add_elapsed_time_ratio_stats()` | `features_arr.py:233` | Emits `elapsed_time_ratio_last14d` (arrival only) |

---

## 9. BTS Columns Required for v10

The BTS history pool must include these columns (beyond what v8 already used):

| Column | Used by | Already in v8? |
|--------|---------|----------------|
| `NASDelay` | `origin_nasdelay_rate_last1d` | Downloaded but not used in features |
| `Cancelled` | `cancel_rate_origin_last1d` | Downloaded but not used in dep features |
| `Diverted` | `divert_rate_origin_last14d` | Downloaded but never used |
| `ActualElapsedTime` | `elapsed_time_ratio_last14d` | Downloaded but never used |
| `CRSElapsedTime` | `elapsed_time_ratio_last14d` | Already used for schedule features |

These columns were added to the history pool keep-list in `prepare_dataset.py` and `prepare_cancel_dataset.py` as part of the v10 changes.

---

## 10. Verification Checklist

To confirm a correct v10 implementation:

1. **No forbidden features:** Grep all v10 configs for: `tail_`, `aircraft_type`, `has_recent_arrival_turn`, `turn_time_hours`, `wo_slip_`, `airtime_mean_`, `taxi_out`, `gate_hold`. All must be absent.
2. **Median-last1 populated:** Check that `*_median_last1` columns are non-null with reasonable distributions (medians should track means within ~10-20%).
3. **New aggregates populated:** `origin_nasdelay_rate_last1d`, `cancel_rate_origin_last1d`, `divert_rate_origin_last14d` should have >95% non-null coverage after lookback warmup.
4. **Elapsed-time ratio (arrival):** `elapsed_time_ratio_last14d` should be in range ~0.8-1.3, non-null after warmup.
5. **Feature counts:** Departure = 80, Arrival = ~106.
6. **No leakage:** All rolling features use data strictly before the target FlightDate.
