# v9 Feature Specification

## Changes from v8

### Added (11 features for dep, 12 for arr)

| # | Feature | Dep | Arr | Rationale |
|---|---------|-----|-----|-----------|
| 1 | `aircraft_type` | cat | cat | +9.4 bp alone; strongest new categorical. New API provides tail number days ahead; aircraft type derived via FAA registry lookup. |
| 2 | `has_recent_arrival_turn_5h` | num | num | +19.7 bp. Binary: did this tail arrive at Origin within the last 5 hours? Signals quick-turn pressure. Requires tail number + schedule. |
| 3 | `turn_time_hours` | num | num | +8.3 bp. Hours between this tail's previous arrival and current departure. Shorter turn = less recovery buffer. Requires tail number + schedule. |
| 4 | `tail_leg_num_day` | num | num | Which leg of the day this aircraft is on (1st, 2nd, ...). Later legs accumulate cascade risk. Requires tail number + schedule. |
| 5 | `tail_depdelay_mean_last1` | num | num | Tail-specific rolling 1-day mean departure delay. Individually marginal but contributes to +55 bp combined package. Requires tail number + BTS history. |
| 6 | `tail_depdelay_mean_last14` | num | num | Tail-specific rolling 14-day mean departure delay. More stable signal. |
| 7 | `tail_lateaircraft_rate_last1` | num | num | Tail-specific 1-day late-aircraft occurrence rate. Captures same-day propagation signal. |
| 8 | `tail_lateaircraft_rate_last14` | num | num | Tail-specific 14-day late-aircraft rate. Captures chronic aircraft issues. |
| 9 | `tail_n_legs_scheduled` | num | num | NEW. Number of legs this tail is scheduled for today. More legs = more cascade opportunity. Top cascade predictor (importance 22.4). |
| 10 | `tail_min_turn_time` | num | num | NEW. Minimum scheduled turn time in this tail's daily rotation. Tightest turn = weakest link for delay absorption. |
| 11 | `tail_has_tight_turn` | num | num | NEW. Binary: does this tail have any turn < 45 min today? Single tight turn can cascade the rest of the day. |
| 12 | `dest_arr_weathercode` | -- | cat | Destination hourly weather code at scheduled arrival. Already computed in v8 pipeline but not included in v8 training list. |

### Dropped (5 features)

| # | Feature | Model | Reason |
|---|---------|-------|--------|
| 1 | `tail_depdelay_mean_last7` | both | Redundant with last1+last14. Rolling 7d window adds noise between the reactive (1d) and stable (14d) signals. |
| 2 | `tail_lateaircraft_rate_last7` | both | Same reasoning. |
| 3 | `origin_daily_windspeed_max_kmh` | dep | Removed in v8 already; confirming it stays dropped. SHAP near zero; windgusts subsumes it. |
| 4 | `flightnum_od_low_support_last14d` | dep | Removed in v8 already; confirming. Near-zero importance. |
| 5 | `is_holiday` | dep | Removed in v8 already; confirming. Near-zero SHAP. |

### Unchanged from v8

All other v8 features are retained as-is. The hub spillover block (hub_0 through hub_4, all windows, both depdelay_mean and lateaircraft_rate), origin/carrier/dest baselines, weather features, congestion features, strike features, schedule features, and the v8 analyst-recommended features (is_peak_hour, hub_max_lateaircraft_last1, wind_x_precip) all carry over.

---

## Complete v9 Feature List: DEPARTURE

### Categorical (5)

| Feature | How to Calculate | Knowable Day Before? |
|---------|-----------------|---------------------|
| `Origin` | BTS/API: origin airport IATA code | Yes -- from schedule |
| `Dest` | BTS/API: destination airport IATA code | Yes -- from schedule |
| `DepTimeBlk` | BTS time block string (e.g., "0600-0659"). Derive from CRSDepTime: map HHMM to block. | Yes -- from schedule |
| `origin_dep_hour_weathercode` | Open-Meteo hourly archive: WMO weather code at origin airport lat/lon at the UTC hour containing scheduled departure. Integer code (0-99). | Yes -- from day-before forecast |
| `aircraft_type` | **NEW.** Look up Tail_Number in FAA MASTER.txt registry to get MFR-MDL-SERIES, then join ACFTREF.txt for aircraft description. | Yes -- tail number known from new API |

### Numeric (80)

#### Schedule & Route (6)

| Feature | How to Calculate | Knowable Day Before? |
|---------|-----------------|---------------------|
| `CRSDepTime` | Scheduled departure in HHMM integer format from schedule/API. | Yes |
| `CRSElapsedTime` | Scheduled block time in minutes (CRSArrTime - CRSDepTime adjusted for timezone). From schedule. | Yes |
| `Distance` | Great-circle distance in statute miles between Origin and Dest. From BTS or airport coordinates. | Yes |
| `dep_dow` | Day of week (0=Mon, 6=Sun) from FlightDate. | Yes |
| `sched_dep_hour` | Hour component of CRSDepTime (0-23). | Yes |
| `is_peak_hour` | Binary: 1 if sched_dep_hour in {16, 17, 18, 19, 20}, else 0. | Yes |

#### Flight Number / OD History (7)

| Feature | How to Calculate | Knowable Day Before? |
|---------|-----------------|---------------------|
| `flightnum_od_depdelay_mean_last1` | Mean DepDelayMinutes for same (Airline, FlightNum, Origin, Dest) over previous 1 calendar day. Shift by 1 day to avoid leakage. | Yes -- uses yesterday's data |
| `flightnum_od_depdelay_mean_last7` | Same, 7-day rolling window. | Yes |
| `flightnum_od_depdelay_mean_last14` | Same, 14-day rolling window. | Yes |
| `flightnum_od_depdelay_median_last1` | Median variant of above, 1-day. | Yes |
| `flightnum_od_depdelay_median_last7` | Median, 7-day. | Yes |
| `flightnum_od_depdelay_median_last14` | Median, 14-day. | Yes |
| `flightnum_od_support_count_last14d` | Count of flights for this (Airline, FlightNum, Origin, Dest) in last 14 days. Low counts flag unreliable history. | Yes |

#### Carrier Baselines (6)

| Feature | How to Calculate | Knowable Day Before? |
|---------|-----------------|---------------------|
| `carrier_depdelay_mean_last1` | Mean DepDelayMinutes for entire carrier over previous 1 day. | Yes |
| `carrier_depdelay_mean_last7` | Same, 7-day. | Yes |
| `carrier_origin_depdelay_mean_last1` | Mean DepDelayMinutes for (Carrier, Origin) pair over previous 1 day. | Yes |
| `carrier_origin_depdelay_mean_last7` | Same, 7-day. | Yes |
| `carrier_origin_depdelay_mean_last14` | Same, 14-day. | Yes |
| `origin_depdelay_mean_last14` | Mean DepDelayMinutes at origin airport over previous 14 days (all carriers). | Yes |

#### Delay Cause Rates (4)

| Feature | How to Calculate | Knowable Day Before? |
|---------|-----------------|---------------------|
| `origin_lateaircraft_rate_last1` | Fraction of flights at origin with LateAircraftDelay > 0 over previous 1 day. | Yes |
| `origin_lateaircraft_rate_last7` | Same, 7-day. | Yes |
| `carrier_lateaircraft_rate_last1` | Fraction of carrier's flights with LateAircraftDelay > 0 over previous 1 day. | Yes |
| `carrier_lateaircraft_rate_last7` | Same, 7-day. | Yes |

#### Hub Spillover (30)

For each hub index (hub_0=DEN, hub_1=PHX, hub_2=BWI, hub_3=MDW, hub_4=BNA):

| Feature | How to Calculate | Knowable Day Before? |
|---------|-----------------|---------------------|
| `hub_{i}_depdelay_mean_last{W}` | Mean DepDelayMinutes at hub airport over previous W days (W=1,7,14). | Yes |
| `hub_{i}_lateaircraft_rate_last{W}` | Late-aircraft rate at hub over previous W days (W=1,7,14). | Yes |

That's 5 hubs x 2 metrics x 3 windows = 30 features.

| Feature | How to Calculate | Knowable Day Before? |
|---------|-----------------|---------------------|
| `hub_max_lateaircraft_last1` | max(hub_0_lateaircraft_rate_last1, ..., hub_4_lateaircraft_rate_last1). Worst-hub signal. | Yes |

Total hub: 31 features.

#### Destination Baselines (4)

| Feature | How to Calculate | Knowable Day Before? |
|---------|-----------------|---------------------|
| `dest_depdelay_mean_last1` | Mean DepDelayMinutes at destination over previous 1 day. | Yes |
| `dest_lateaircraft_rate_last1` | Late-aircraft rate at destination, 1-day. | Yes |
| `dest_lateaircraft_rate_last7` | Same, 7-day. | Yes |
| `dest_lateaircraft_rate_last14` | Same, 14-day. | Yes |

#### Congestion (2)

| Feature | How to Calculate | Knowable Day Before? |
|---------|-----------------|---------------------|
| `origin_congestion_3h_total` | Count of all scheduled departures from origin within +/-1.5h of this flight's CRSDepTime. Computed from the day's full schedule. | Yes -- schedule is known |
| `origin_airline_congestion_3h_total` | Same, filtered to same airline only. | Yes |

#### Origin Weather -- Daily (4)

| Feature | How to Calculate | Knowable Day Before? |
|---------|-----------------|---------------------|
| `origin_temp_max_K` | Open-Meteo daily: temperature_2m_max at origin lat/lon, converted to Kelvin (C + 273.15). | Yes -- from day-before forecast |
| `origin_temp_min_K` | Same, temperature_2m_min. | Yes |
| `origin_daily_precip_sum_mm` | Open-Meteo daily: precipitation_sum at origin (mm). | Yes |
| `origin_daily_windgusts_max_kmh` | Open-Meteo daily: windgusts_10m_max at origin (km/h). | Yes |

#### Origin Weather -- Hourly at Departure (7)

| Feature | How to Calculate | Knowable Day Before? |
|---------|-----------------|---------------------|
| `origin_dep_temp_K` | Open-Meteo hourly: temperature_2m at origin for the UTC hour of CRSDepTime, converted to Kelvin. | Yes -- forecast |
| `origin_dep_precip_mm` | Hourly precipitation at origin at departure hour (mm). | Yes |
| `origin_dep_windgusts_kmh` | Hourly windgusts_10m at origin (km/h). | Yes |
| `origin_dep_visibility_m` | Hourly visibility at origin (m), clipped to 25000. | Yes |
| `origin_dep_cape_jkg` | Hourly CAPE (J/kg) at origin. Convective instability indicator. | Yes |
| `origin_dep_log1p_cape` | log1p(origin_dep_cape_jkg). Compresses heavy tail. | Yes |
| `origin_dep_cloudcover_pct` | Hourly cloud cover percentage at origin. | Yes |

#### Derived Weather (1)

| Feature | How to Calculate | Knowable Day Before? |
|---------|-----------------|---------------------|
| `wind_x_precip` | origin_dep_windgusts_kmh * origin_dep_precip_mm. Interaction capturing combined wind+rain severity. | Yes |

#### Strike (3)

| Feature | How to Calculate | Knowable Day Before? |
|---------|-----------------|---------------------|
| `strike_severity` | From strike cache: severity score (0-10) of nearest active/announced labor action affecting this carrier. | Yes |
| `days_to_strike` | Days until the nearest announced strike for this carrier (0-90 window). | Yes |
| `carrier_delay_rate_anomaly_7d` | Z-score: (carrier's 7-day delay rate - 60-day baseline) / 60-day std. Flags abnormal operational stress. | Yes |

#### Tail / Aircraft -- NEW (8)

| Feature | How to Calculate | Knowable Day Before? |
|---------|-----------------|---------------------|
| `has_recent_arrival_turn_5h` | **NEW API.** Binary: did this Tail_Number arrive at this Origin within the previous 5 hours? Computed from the tail's published schedule. 1 = quick turn, 0 = overnight/repositioned. | Yes -- tail schedule known |
| `turn_time_hours` | **NEW API.** Hours between this tail's previous scheduled arrival and current CRSDepTime. Null if first leg of day or no prior arrival. | Yes -- from tail schedule |
| `tail_leg_num_day` | **NEW API.** Ordinal: which leg of the day this is for the tail (1, 2, 3, ...). From the tail's daily rotation sequence sorted by CRSDepTime. | Yes -- from tail schedule |
| `tail_depdelay_mean_last1` | **NEW API.** Rolling 1-day mean DepDelayMinutes for this Tail_Number. GroupBy Tail_Number, rolling 1-day window, shift 1 day. | Yes -- uses yesterday's data |
| `tail_depdelay_mean_last14` | Same, 14-day rolling window. | Yes |
| `tail_lateaircraft_rate_last1` | Rolling 1-day fraction of this tail's flights with LateAircraftDelay > 0. | Yes |
| `tail_lateaircraft_rate_last14` | Same, 14-day. | Yes |
| `tail_n_legs_scheduled` | **BRAND NEW.** Count of total legs this Tail_Number is scheduled to fly today. From the day's complete tail schedule. | Yes -- from tail schedule |
| `tail_min_turn_time` | **BRAND NEW.** Minimum scheduled turn time (hours) across all turns in this tail's daily rotation. Tightest turn = cascade vulnerability. | Yes -- from tail schedule |
| `tail_has_tight_turn` | **BRAND NEW.** Binary: does this tail have any scheduled turn < 45 min today? Flag for cascade risk. | Yes -- from tail schedule |

### Departure v9 Total: 5 categorical + 80 numeric = 85 features

---

## Complete v9 Feature List: ARRIVAL

The arrival model uses all departure features above (merged on FlightDate + Airline + FlightNum + Origin + Dest + CRSDepTime) PLUS the following arrival-specific features.

### Categorical (5) -- same as departure

(Origin, Dest, DepTimeBlk, origin_dep_hour_weathercode, aircraft_type)

### Numeric: Departure features carried over (80)

All 80 departure numeric features above.

### Numeric: Arrival-specific (38)

#### Schedule (2)

| Feature | How to Calculate | Knowable Day Before? |
|---------|-----------------|---------------------|
| `CRSArrTime` | Scheduled arrival time in HHMM format. From schedule. | Yes |
| `sched_arr_hour` | Hour component of scheduled arrival time (0-23). From arr_dt_local. | Yes |

#### Route Arrival Delay History (10)

| Feature | How to Calculate | Knowable Day Before? |
|---------|-----------------|---------------------|
| `arrdelay_mean_7d_fn_od` | Mean ArrDelayMinutes for (Airline, FlightNum, Origin, Dest) over previous 7 days. | Yes |
| `arrdelay_n_7d_fn_od` | Count of flights in that 7-day window. | Yes |
| `arrdelay_mean_14d_fn_od` | Same, 14-day. | Yes |
| `arrdelay_n_14d_fn_od` | Count, 14-day. | Yes |
| `arrdelay_median_7d_fn_od` | Median ArrDelayMinutes, 7-day, flight+OD. | Yes |
| `arrdelay_mean_7d_car_od` | Mean ArrDelayMinutes for (Carrier, Origin, Dest) over previous 7 days. | Yes |
| `arrdelay_n_7d_car_od` | Count. | Yes |
| `arrdelay_mean_14d_car_od` | Same, 14-day. | Yes |
| `arrdelay_n_14d_car_od` | Count. | Yes |
| `arrdelay_median_7d_car_od` | Median, 7-day, carrier+OD. | Yes |

#### Air Time History (8)

| Feature | How to Calculate | Knowable Day Before? |
|---------|-----------------|---------------------|
| `airtime_mean_7d_fn_od` | Mean actual AirTime for (Airline, FlightNum, Origin, Dest) over 7 days. Captures en-route efficiency. | Yes |
| `airtime_n_7d_fn_od` | Count. | Yes |
| `airtime_mean_14d_fn_od` | Same, 14-day. | Yes |
| `airtime_n_14d_fn_od` | Count. | Yes |
| `airtime_mean_7d_car_od` | Mean AirTime for (Carrier, Origin, Dest) over 7 days. | Yes |
| `airtime_n_7d_car_od` | Count. | Yes |
| `airtime_mean_14d_car_od` | Same, 14-day. | Yes |
| `airtime_n_14d_car_od` | Count. | Yes |

#### Wheels-Off Slip History (8)

| Feature | How to Calculate | Knowable Day Before? |
|---------|-----------------|---------------------|
| `wo_slip_mean_7d_fn_od` | Mean minutes between CRSDepTime and actual WheelsOff for (Airline, FlightNum, Origin, Dest) over 7 days. Captures taxi-out delay patterns. | Yes |
| `wo_slip_n_7d_fn_od` | Count. | Yes |
| `wo_slip_mean_14d_fn_od` | Same, 14-day. | Yes |
| `wo_slip_n_14d_fn_od` | Count. | Yes |
| `wo_slip_mean_7d_origin_blk` | Mean WO slip for (Origin, DepTimeBlk) over 7 days. Airport+time pattern. | Yes |
| `wo_slip_n_7d_origin_blk` | Count. | Yes |
| `wo_slip_mean_14d_origin_blk` | Same, 14-day. | Yes |
| `wo_slip_n_14d_origin_blk` | Count. | Yes |

#### Destination Arrival Congestion (4)

| Feature | How to Calculate | Knowable Day Before? |
|---------|-----------------|---------------------|
| `dest_arrivals_pm60_sched` | Count of all scheduled arrivals at Dest within +/-60 min of this flight's CRSArrTime. From schedule. | Yes |
| `dest_airline_arrivals_pm60_sched` | Same, filtered to same airline. | Yes |
| `dest_arrivals_pm60_eta` | Count of arrivals within +/-60 min of ETA. ETA = CRSArrTime shifted by the flight's 7d or 14d mean arrival delay. | Yes |
| `dest_airline_arrivals_pm60_eta` | Same, filtered to same airline. | Yes |

#### Destination Weather at Arrival (7)

| Feature | How to Calculate | Knowable Day Before? |
|---------|-----------------|---------------------|
| `dest_arr_temperature_2m` | Open-Meteo hourly: temperature_2m at destination lat/lon at the UTC hour of CRSArrTime. | Yes -- forecast |
| `dest_arr_precipitation` | Hourly precipitation at destination at arrival hour (mm). | Yes |
| `dest_arr_windspeed_10m` | Hourly windspeed at destination (km/h). | Yes |
| `dest_arr_windgusts_10m` | Hourly wind gusts at destination (km/h). | Yes |
| `dest_arr_visibility` | Hourly visibility at destination (m). | Yes |
| `dest_arr_cape` | Hourly CAPE at destination (J/kg). | Yes |
| `dest_arr_cloudcover` | Hourly cloud cover at destination (%). | Yes |

### Arrival v9 Total: 5 categorical + 80 (dep) + 38 (arr-specific) = 123 features

---

## Summary of Changes: v8 to v9

| | v8 Dep | v9 Dep | v8 Arr | v9 Arr |
|---|--------|--------|--------|--------|
| Categorical | 4 | 5 (+aircraft_type) | 4 | 5 (+aircraft_type) |
| Numeric | 72 | 80 (+11 tail/cascade, -3 dropped last7) | 110 | 118 (same adds) |
| **Total** | **76** | **85** | **114** | **123** |

### New features requiring tail number (from new API):
- `aircraft_type` -- FAA registry lookup from Tail_Number
- `has_recent_arrival_turn_5h` -- binary from tail schedule
- `turn_time_hours` -- from tail schedule
- `tail_leg_num_day` -- from tail schedule sequence
- `tail_depdelay_mean_last1`, `tail_depdelay_mean_last14` -- rolling from BTS history
- `tail_lateaircraft_rate_last1`, `tail_lateaircraft_rate_last14` -- rolling from BTS history
- `tail_n_legs_scheduled` -- count from tail's daily schedule (NEW)
- `tail_min_turn_time` -- min turn in tail's rotation (NEW)
- `tail_has_tight_turn` -- binary cascade risk flag (NEW)

### Pipeline changes needed for v9:
1. Enable `tail_history` in blueprint (`enabled: true`)
2. Enable `add_aircraft_type: true`
3. Add new function for `tail_n_legs_scheduled`, `tail_min_turn_time`, `tail_has_tight_turn` in features_dep.py
4. Add `dest_arr_weathercode` to arrival training config categorical list
5. Drop `tail_depdelay_mean_last7`, `tail_lateaircraft_rate_last7` from training configs
6. Update training configs with new feature lists
