# v11 Feature Compatibility Guide

**Purpose:** for every v11 feature, explain in plain English what it means, how it
is computed from BTS history at training time, and how it should be obtained
from Aerodatabox (or another production source) at prediction time. Math is
shown for every derived feature so teammates can reproduce the computation
without reading the Python source.

**Sister documents:**
- `exploration/v11_feature_spec.md` — what changed from v10 to v11 and why.
- `feature_transferability/reports/2026-04-16-v11-aerodatabox-screening.md` — endpoint-by-endpoint Aerodatabox mapping with verdicts.
- `src/fetch_prune/features_dep.py`, `src/fetch_prune/features_arr.py` — authoritative implementations.

---

## 1. Core Rules

Every v11 feature satisfies **both** of these:

1. **Producible at prediction time without a tail number and without any actual gate / wheels-off / airtime observation for the target flight.** Tail numbers are not published by Aerodatabox for upcoming flights, and actual runway/taxi times are either unavailable or arrive too late to be usable.
2. **Train/serve consistent.** The feature value at prediction time must come from the same computation as at training time. For rolling aggregates this means the production system maintains a lookup table keyed on `(identifier, date)` that is refreshed from BTS as often as BTS publishes (monthly with a ~1-month lag).

Features that violated either rule were removed from v11 (see §8).

---

## 2. Data Sources at Prediction Time

| Source | What it supplies | Refresh cadence |
|--------|------------------|-----------------|
| **Aerodatabox (live)** | Schedule fields (`CRSDepTime`, `CRSArrTime`, `CRSElapsedTime`, `Origin`, `Dest`, `DepTimeBlk`, `Distance`, airline code, flight number), status fields (cancellation, diversion) | Real-time |
| **BTS-derived offline cache** | All rolling aggregates (delay means/medians, late-aircraft rates, OTP rate, cancel rate, divert rate, cancel-anomaly z-score, elapsed-time ratio). Stored as lookup tables keyed by `(identifier, FlightDate)`. | Monthly (on BTS publication) |
| **Open-Meteo (live)** | Daily and hourly forecast weather at origin and (for arrival model) destination | Day-before-flight forecast |
| **Strike cache (offline)** | Aviation labor-action calendar (`strike_severity`, `days_to_strike`, `carrier_delay_rate_anomaly_7d`) | Manual refresh when new strikes announced |
| **Static airport metadata** | Airport coordinates, timezone (for hour normalization) | Rarely (OurAirports) |

At prediction time, the inference service joins these four sources on `(Origin, Dest, Airline, FlightNumber, FlightDate)` — essentially the same keying used in training.

---

## 3. Notation

Throughout the guide:

- `H` = BTS history data frame with one row per historical flight.
- `t` = target flight's `FlightDate` (local date at origin).
- `D(t)` = the set of flights in `H` with `FlightDate = t`.
- `D(t; k=v)` = the subset of `D(t)` where key column `k` equals value `v`.
- `shift(1)` and "1-day forward shift" mean the same thing: the aggregate for target date `t` uses data up to and including `t − 1 day`. This prevents same-day leakage.
- `mean`, `median`, `count`, `std` refer to aggregation over individual flight rows unless otherwise noted.
- `rolling_W` denotes a rolling window of `W` days of per-day aggregates.

All rolling aggregates follow the v10/v11 leakage-prevention pattern:

```
1. Compute per-day aggregate over D(d) for each historical day d.
2. Apply shift(1) — the value indexed at date t uses data through t−1.
3. (Optional) Apply rolling(window=W, min_periods=1).mean() for multi-day windows.
4. Merge onto target flight by (key, FlightDate=t).
```

---

## 4. Schedule and Identity Features (4 categorical + 5 numeric)

These are **direct** from Aerodatabox at prediction time — no transformation needed.

### 4.1 `Origin` (categorical)
- **Meaning:** Origin airport IATA code, e.g. `DEN`, `ORD`.
- **BTS column:** `Origin`.
- **Aerodatabox field:** `departure.airport.iata`.

### 4.2 `Dest` (categorical)
- **Meaning:** Destination airport IATA code.
- **BTS column:** `Dest`.
- **Aerodatabox field:** `arrival.airport.iata`.

### 4.3 `DepTimeBlk` (categorical)
- **Meaning:** 1-hour bin of scheduled departure time, e.g. `"0600-0659"`, `"1500-1559"`.
- **BTS column:** `DepTimeBlk`.
- **Aerodatabox derivation:** From `departure.scheduledTime.local`, take the hour `H`, format as `"{H:02d}00-{H:02d}59"`.

### 4.4 `origin_dep_hour_weathercode` (categorical)
- **Meaning:** Origin airport + scheduled departure hour + forecast weather code, joined into a single string like `"DEN_14_2"` where `2` is the Open-Meteo WMO weather code for mainly clear.
- **BTS derivation:** Look up `CRSDepTime`'s hour at Origin for the FlightDate, then join `origin` + hour + weather code from Open-Meteo forecast.
- **Aerodatabox derivation:** Same — concatenate live airport code + scheduled hour + forecast weather code.
- **Math:** `origin_dep_hour_weathercode = f"{Origin}_{floor(CRSDepTime/100)}_{origin_dep_weathercode}"`.

### 4.5 `CRSDepTime` (numeric)
- **Meaning:** Scheduled departure time encoded as HHMM local, e.g. `1430` for 2:30 PM.
- **BTS column:** `CRSDepTime`.
- **Aerodatabox field:** `departure.scheduledTime.local` → format as `HHMM` integer.

### 4.6 `CRSElapsedTime` (numeric)
- **Meaning:** Scheduled block time (gate-to-gate) in minutes. Some BTS rows have rare negative values for overnight schedules that wrap midnight.
- **BTS column:** `CRSElapsedTime`.
- **Aerodatabox field:** `arrival.scheduledTime.utc − departure.scheduledTime.utc` in minutes.

### 4.7 `Distance` (numeric)
- **Meaning:** Great-circle distance between origin and destination, miles.
- **BTS column:** `Distance`.
- **Aerodatabox derivation:** Compute from airport coordinates (or from `distance` field if the endpoint includes it). Formula: great-circle distance in miles.

### 4.8 `dep_dow` (numeric, 0-6)
- **Meaning:** Day of week of scheduled departure (0=Monday).
- **Derivation:** From `FlightDate`.
- **Math:** `dep_dow = FlightDate.weekday()`.

### 4.9 `sched_dep_hour` (numeric, 0-23)
- **Meaning:** Hour portion of `CRSDepTime`.
- **Math:** `sched_dep_hour = floor(CRSDepTime / 100)`.

### 4.10 `is_peak_hour` (numeric 0/1)
- **Meaning:** Flag for departures scheduled in the afternoon/evening peak (hour 16-20 inclusive) when airport congestion typically causes cascading delays.
- **Math:** `is_peak_hour = 1 if 16 <= sched_dep_hour <= 20 else 0`.

---

## 5. Weather Features (Open-Meteo, 11 for departure + 7 for arrival)

All weather comes from Open-Meteo, queried the day before the flight. Aerodatabox is not involved. Units are standardized on conversion from Open-Meteo's defaults.

### 5.1 Origin daily weather (4 numeric)

| Feature | Meaning | Open-Meteo field | Transform |
|---------|---------|------------------|-----------|
| `origin_temp_max_K` | Max temperature on FlightDate at Origin, Kelvin | `temperature_2m_max` (°C) | + 273.15 |
| `origin_temp_min_K` | Min temperature on FlightDate at Origin, Kelvin | `temperature_2m_min` (°C) | + 273.15 |
| `origin_daily_precip_sum_mm` | Total precipitation on FlightDate at Origin, mm | `precipitation_sum` | identity |
| `origin_daily_windgusts_max_kmh` | Max wind-gust speed on FlightDate at Origin, km/h | `windgusts_10m_max` | identity |

### 5.2 Origin hourly weather at departure hour (7 numeric)

| Feature | Meaning | Open-Meteo field | Transform |
|---------|---------|------------------|-----------|
| `origin_dep_temp_K` | Temperature at Origin at scheduled hour, K | `temperature_2m` (°C) | + 273.15 |
| `origin_dep_precip_mm` | Precipitation at Origin at scheduled hour, mm | `precipitation` | identity |
| `origin_dep_windgusts_kmh` | Wind gust at Origin at scheduled hour, km/h | `windgusts_10m` | identity |
| `origin_dep_visibility_m` | Visibility at Origin at scheduled hour, meters | `visibility` | identity |
| `origin_dep_cape_jkg` | Convective Available Potential Energy at Origin, J/kg | `cape` | identity |
| `origin_dep_log1p_cape` | log(1 + CAPE) — squashes CAPE's long right tail. `log1p(cape) = ln(1 + cape)`. | derived | `log1p(origin_dep_cape_jkg)` |
| `origin_dep_cloudcover_pct` | Cloud cover %, 0-100 | `cloudcover` | identity |

### 5.3 Interaction feature (1 numeric)

- `wind_x_precip` = `origin_dep_windgusts_kmh * origin_dep_precip_mm`. Captures the combined effect of windy + rainy conditions, which together cause larger delay multipliers than either in isolation.

### 5.4 Destination weather at scheduled arrival (arrival model only, 7 numeric)

| Feature | Meaning | Source |
|---------|---------|--------|
| `dest_arr_temperature_2m` | °C at Dest at scheduled arrival hour | Open-Meteo |
| `dest_arr_precipitation` | mm at Dest at scheduled arrival hour | Open-Meteo |
| `dest_arr_windspeed_10m` | km/h | Open-Meteo |
| `dest_arr_windgusts_10m` | km/h | Open-Meteo |
| `dest_arr_visibility` | meters | Open-Meteo |
| `dest_arr_cape` | J/kg | Open-Meteo |
| `dest_arr_cloudcover` | % | Open-Meteo |

Production note: Open-Meteo's forecast API is free and does not require a key. Cache responses per `(airport, date, hour)` to avoid redundant calls.

---

## 6. Rolling Delay Baselines (BTS-offline lookup tables)

These are the bread and butter of v11. Every one is computed from BTS history
via the shared pattern in §3 and served at prediction time from a pre-computed
lookup table — not from a live Aerodatabox call.

### 6.1 Flight-number × OD baselines (6 numeric)

Groups flights by `(Reporting_Airline, Flight_Number_Reporting_Airline, Origin, Dest)` — the concrete recurring flight identity.

Let `G_{f} = {rows r in H : (r.Carrier, r.FlightNum, r.Origin, r.Dest) = f}`, and `G_{f,d} = G_{f} ∩ D(d)`.

| Feature | Meaning | Math |
|---------|---------|------|
| `flightnum_od_depdelay_mean_last7` | Mean dep delay of this flight identity over prior 7 days | ` rolling_7.mean(shift_1(daily_mean(DepDelayMinutes | G_{f,d}))) ` |
| `flightnum_od_depdelay_mean_last14` | Same, 14-day window | `rolling_14.mean(shift_1(daily_mean))` |
| `flightnum_od_depdelay_median_last1` | Median of dep delay across individual flights of this identity on the previous day | `median(r.DepDelayMinutes for r in G_{f, t-1})` |
| `flightnum_od_depdelay_median_last7` | Rolling 7-day mean of per-day medians | `rolling_7.mean(shift_1(daily_median))` |
| `flightnum_od_depdelay_median_last14` | Rolling 14-day mean of per-day medians | `rolling_14.mean(shift_1(daily_median))` |
| `flightnum_od_support_count_last14d` | How many flights this identity actually had in the prior 14 days. Used as a reliability-of-aggregate signal. | `sum(|G_{f,d}| for d in [t-14, t-1])` |

**Why median-last1?** A single 300-minute meltdown dominates the daily *mean*. Median is robust to outliers. 7d/14d windows stay as means because multi-day averaging already dampens outliers.

### 6.2 Carrier baselines (2 numeric)

Groups by `(Reporting_Airline)` only.

| Feature | Math |
|---------|------|
| `carrier_depdelay_median_last1` | `median(r.DepDelayMinutes for r in H with r.Carrier=c and r.FlightDate=t-1)` |
| `carrier_depdelay_mean_last7` | `rolling_7.mean(shift_1(daily_mean(DepDelayMinutes | carrier=c)))` |

### 6.3 Carrier × Origin baselines (3 numeric)

Groups by `(Reporting_Airline, Origin)` — catches airline-specific origin effects (e.g. WN at DEN differs from UA at DEN).

| Feature | Math |
|---------|------|
| `carrier_origin_depdelay_median_last1` | `median(r.DepDelayMinutes for r in H with (carrier=c, origin=o) and FlightDate=t-1)` |
| `carrier_origin_depdelay_mean_last7` | `rolling_7.mean(shift_1(daily_mean))` on the same group |
| `carrier_origin_depdelay_mean_last14` | `rolling_14.mean(shift_1(daily_mean))` |

### 6.4 Origin baselines (1 numeric)

Groups by `(Origin)` only — captures all-carrier congestion at an airport.

| Feature | Math |
|---------|------|
| `origin_depdelay_mean_last14` | `rolling_14.mean(shift_1(daily_mean(DepDelayMinutes | origin=o)))` |

### 6.5 Destination baselines (1 numeric)

`dest_depdelay_median_last1` — median dep delay of **flights originating at the destination airport** on the previous day.

- **Intuition:** captures spillover. If a destination airport was experiencing heavy delays yesterday, the aircraft inventory servicing today's inbound flights may be disrupted.
- **Math:** `median(r.DepDelayMinutes for r in H with r.Origin=d and r.FlightDate=t-1)` where `d = target_flight.Dest`.

### 6.6 Hub spillover (30 numeric)

For each of the airline's 5 top hubs (indexed 0-4), three stats across three windows:

```
AIRLINE_HUB_PRESETS = {
  "WN": ["DEN", "PHX", "BWI", "MDW", "BNA"],
  "UA": ["ORD", "EWR", "IAH", "DEN", "LAX"],
  "AA": ["DFW", "CLT", "ORD", "MIA", "PHX"],
  "DL": ["ATL", "DTW", "MSP", "LAX", "SLC"],
}
```

For hub `h ∈ hubs` and window `W ∈ {1, 7, 14}`:

| Feature | Math |
|---------|------|
| `hub_{i}_depdelay_median_last1` | 1d window only: `median(DepDelayMinutes | origin=h, FlightDate=t-1)` |
| `hub_{i}_depdelay_mean_last7`, `hub_{i}_depdelay_mean_last14` | `rolling_W.mean(shift_1(daily_mean))` |
| `hub_{i}_lateaircraft_rate_last{1,7,14}` | see §7 for the late-aircraft definition |

Plus one worst-hub signal:

- `hub_max_lateaircraft_last1 = max(hub_{0..4}_lateaircraft_rate_last1)`. Captures "today the worst of our hubs is very disrupted."

---

## 7. Late-Aircraft Rates (v11 uses ≥15-min threshold)

**v10 → v11 change:** BTS defines `LateAircraftDelay > 0` as "any prior-leg tardiness," but the industry standard (FAA, Aerodatabox, IATA OTP) uses 15+ minutes. v11 aligns with industry via the module constant:

```python
LATE_AIRCRAFT_THRESHOLD_MIN = 15
```

A flight row is "late from prior leg" when `LateAircraftDelay >= 15`.

For each grouping below, the rate is computed as:

```
Let I_i = 1 if H[i].LateAircraftDelay >= 15 else 0

late_rate(group, d) = mean(I_i for i in group on FlightDate d)
late_rate(group, t, W) = rolling_W.mean(shift_1(late_rate(group, d)))
```

Range: [0, 1].

Features produced for each grouping, for `W ∈ {1, 7, 14}`:

| Grouping | Features |
|----------|----------|
| Origin | `origin_lateaircraft_rate_last1`, `_last7` (14d not in dep production) |
| Carrier | `carrier_lateaircraft_rate_last1`, `_last7` |
| Destination (rates at destination airport, keyed by target's `Dest`) | `dest_lateaircraft_rate_last1/7/14` |
| Each hub (5 hubs × 3 windows = 15 features) | `hub_{i}_lateaircraft_rate_last1/7/14` |
| Max across hubs, 1d | `hub_max_lateaircraft_last1` |

**Aerodatabox serving:** the feature is served from the BTS-offline lookup table, **not** computed live from Aerodatabox. This is intentional because (a) Aerodatabox doesn't publish tail numbers for upcoming flights so we can't re-derive which prior leg was late, and (b) the aggregate rate is all we need — the individual tail mapping is irrelevant.

---

## 8. Congestion (2 numeric)

These count other flights scheduled in a 3-hour window around the target's scheduled departure, at the same origin.

| Feature | Meaning | Math |
|---------|---------|------|
| `origin_congestion_3h_total` | Total scheduled flights at Origin within ±90 min of target's CRSDepTime | count of rows in `D(t)` at `Origin` with scheduled departure in `[CRSDepTime − 90, CRSDepTime + 90]` |
| `origin_airline_congestion_3h_total` | Same, but restricted to target's airline | count restricted to `r.Carrier = target.Carrier` |

**Aerodatabox serving:** Derivable from the FIDS (Flight Information Display System) endpoint — the schedule endpoint returns all flights at an airport in a time window. At prediction time, count scheduled departures in the ±90-min window.

---

## 9. Strike Features (3 numeric)

Strike features come from a curated offline calendar of US aviation labor actions (`../flightrightdata/strike_cache/us_aviation_labor_actions.parquet`).

### 9.1 `strike_severity`
- **Meaning:** Max severity of any active labor action (strike, walkout, sickout) affecting the target flight's carrier on the FlightDate. Integer scale 0-5 (0 = none).
- **Source:** Curated cache; refreshed manually when new actions are announced.
- **Aerodatabox serving:** N/A — not derivable from Aerodatabox. Served from the same offline cache in production.

### 9.2 `days_to_strike`
- **Meaning:** Days until the nearest future announced labor action for the carrier. 0 = today.
- **Math:** `min((action.start_date - t).days for action in future_actions(carrier)) if any; else some high sentinel (e.g. 90)`.

### 9.3 `carrier_delay_rate_anomaly_7d`
- **Meaning:** Z-score of the carrier's 7-day dep-delay rate (fraction ≥ 15 min) vs its 60-day baseline. Catches carriers that are suddenly performing worse than typical — a symptom of brewing operational stress, including crew work-to-rule tactics that aren't formal strikes.
- **Math:**
  ```
  delay_rate(c, d) = mean(1 if r.DepDelayMinutes >= 15 else 0 for r in H with Carrier=c and FlightDate=d)
  cr_7  = shift_1(rolling_7.mean(delay_rate))
  cr_60 = shift_1(rolling_60.mean(delay_rate))
  sig_60 = shift_1(rolling_60.std(delay_rate))
  carrier_delay_rate_anomaly_7d = (cr_7 - cr_60) / sig_60
  ```
- **Typical range:** [-3, +3], registered bounds [-10, 10] for crisis events.

---

## 10. v10 AeroDataBox-Compatible BTS Aggregates (2 numeric; one dropped in v11)

### 10.1 `cancel_rate_origin_last1d`
- **Meaning:** Fraction of flights at Origin that were cancelled on the previous day.
- **Math:** `mean(1 if r.Cancelled else 0 for r in D(t-1) with r.Origin = target.Origin)`.
- **Aerodatabox serving:** Derivable live from the FIDS endpoint (count `status = "Cancelled"` at the origin for the prior day). Or served from the BTS lookup table — both work; the offline cache is preferred for train/serve consistency.

### 10.2 `divert_rate_origin_last14d`
- **Meaning:** 14-day rolling mean of daily diversion rates at Origin.
- **Math:**
  ```
  divert_rate(d, o) = mean(1 if r.Diverted else 0 for r in D(d) with origin=o)
  divert_rate_origin_last14d = rolling_14.mean(shift_1(divert_rate(d, o)))
  ```
- **Aerodatabox serving:** Diversions show in the flight-status endpoint. Live computation is possible but offline serving is simpler and consistent with training.

### 10.3 `origin_nasdelay_rate_last1d` — **DROPPED in v11**
- **Why dropped:** Aerodatabox's `/airports/delays/{icao}` endpoint returns only an aggregate delay index; it does not break delays down by cause (NAS / weather / carrier / late aircraft). Using the generic delay index as a proxy is an approximation, which the user has ruled out.
- **Current state:** The function `add_origin_nasdelay_rate_last1d` remains in `features_dep.py` for offline research. The blueprint flag `nas_rate_last1d.enabled` is set to `false` in all v11 configs; the feature name is removed from the `numeric_features` list in all v11 training configs.

---

## 11. v11 New Features

### 11.1 `flightnum_od_otp_rate_last14d`
- **Meaning:** Fraction of same-identity flights in the prior 14 days that departed within 15 minutes of schedule. Captures **how often** a given recurring flight is on time, orthogonal to *how much* it's delayed when late (which the median captures).
- **Math:**
  ```
  on_time(r) = 1 if r.DepDelayMinutes <= 15 else 0      # cancelled flights excluded
  daily(f, d) = mean(on_time(r) for r in G_{f, d})      # per-day fraction on time
  otp(f, t) = rolling_14.mean(shift_1(daily(f, d)))    # 14d rolling mean of dailies, 1d shift
  ```
- **Range:** [0, 1].
- **Missingness:** ~7% of target flights are new flight-identity instances with no 14-day history; CatBoost handles NaN natively.
- **Validated lift:** logistic log-loss drops +1.5% to +2.2% when added on top of `flightnum_od_depdelay_median_last14`. Partial Spearman with delay-severity labels grows with threshold: +0.099 at ge15, +0.389 at ge60, +0.491 at ge120 — the OTP rate captures tail-event variance the median misses.
- **Aerodatabox serving:** Served from the BTS-offline lookup table keyed `(Carrier, FlightNum, Origin, Dest, FlightDate)`. Live computation via the Aerodatabox Flight History endpoint (by flight number over a 14-day range) is possible but unnecessary for v11 — the offline table suffices.

### 11.2 `airline_cancel_rate_anomaly_7d`
- **Meaning:** Z-score of carrier's 7-day cancellation rate vs its 60-day baseline. Catches carriers in operational crisis (mass cancellations from weather, crew shortages, or system meltdowns). Operationally distinct from `carrier_delay_rate_anomaly_7d` which tracks *delays*; this tracks *cancellations*, which lag and signal different upstream pressures.
- **Math:**
  ```
  cancel_rate(c, d) = mean(1 if r.Cancelled else 0 for r in D(d) with Carrier=c)
  cr_7  = shift_1(rolling_7.mean(cancel_rate))
  cr_60 = shift_1(rolling_60.mean(cancel_rate))       # requires >=7 observations for validity
  sig_60 = shift_1(rolling_60.std(cancel_rate))
  anomaly = (cr_7 - cr_60) / sig_60
  ```
- **Typical range:** [-3, +3], registered bounds [-10, 10].
- **Enablement policy:** production blueprints have `cancel_anomaly_7d.enabled = true`; **smoke blueprints have it disabled** because 1-month windows lack the 60-day baseline. Production WN full-year analyses find Spearman ~0.08 vs y_dep_ge60 and ~0.16 vs the cancellation label.
- **Aerodatabox serving:** Served from the BTS-offline lookup table. Live derivation from Aerodatabox FIDS (count cancellations per carrier per day) is feasible but offline is simpler.

---

## 12. Arrival-Specific Features (arrival model only)

The arrival model inherits all departure numerics and categoricals, then adds:

### 12.1 Schedule (2 numeric)
- `CRSArrTime` — scheduled arrival time HHMM local. Direct from Aerodatabox.
- `sched_arr_hour = floor(CRSArrTime / 100)`.

### 12.2 Arrival-delay history, flight-number × OD (6 numeric)

For groupings `f = (Carrier, FlightNum, Origin, Dest)`:

| Feature | Math |
|---------|------|
| `arrdelay_mean_7d_fn_od` | `rolling_7.mean(shift_1(daily_mean(ArrDelayMinutes | G_{f,d})))` |
| `arrdelay_n_7d_fn_od` | support count in the 7-day window |
| `arrdelay_mean_14d_fn_od` | 14-day mean |
| `arrdelay_n_14d_fn_od` | 14-day support count |
| `arrdelay_median_7d_fn_od` | `rolling_7.mean(shift_1(daily_median))` |
| `arrdelay_median_14d_fn_od` | 14-day version |

### 12.3 Arrival-delay history, carrier × OD (6 numeric)

Same set but grouped by `(Carrier, Origin, Dest)` only — captures carrier-level OD effects across all flight numbers.

### 12.4 Destination arrival congestion (4 numeric)

| Feature | Meaning | Math |
|---------|---------|------|
| `dest_arrivals_pm60_sched` | # scheduled arrivals at Dest in the 2-hour window around target's scheduled arrival | count of rows in `D(t)` at Dest with scheduled arrival in `[CRSArrTime − 60, CRSArrTime + 60]` |
| `dest_airline_arrivals_pm60_sched` | Same, restricted to target's carrier | count restricted to `r.Carrier = target.Carrier` |
| `dest_arrivals_pm60_eta` | Count using **expected** arrival = scheduled + `arrdelay_mean_14d_car_od` (surviving fn_od means in v8, car_od in v10/v11). Captures how other flights' delays push their ETAs into the target's window. | shift schedules by expected delays, then count |
| `dest_airline_arrivals_pm60_eta` | Same, restricted to target's carrier | |

### 12.5 `elapsed_time_ratio_last14d`
- **Meaning:** 14-day rolling mean of `ActualElapsedTime / CRSElapsedTime` for the flight identity. Captures whether this flight consistently runs long (ratio >1.05) or short (ratio <0.95).
- **Math:** For each historical flight `r` in the identity group `f`:
  ```
  ratio(r) = r.ActualElapsedTime / r.CRSElapsedTime
  daily(f, d) = mean(ratio(r) for r in G_{f, d})
  elapsed_time_ratio_last14d(f, t) = rolling_14.mean(shift_1(daily(f, d)))
  ```
- **Typical range:** 0.8 to 1.3.
- **Aerodatabox serving:** `ActualElapsedTime` and `CRSElapsedTime` are both **gate-based** (not wheels-off), so they're safely available from BTS history offline. At prediction time, served from the BTS-offline lookup table.
- **Also emitted:** `elapsed_time_ratio_n_last14d` — support count for the 14-day window.

---

## 13. Features Explicitly NOT in v11

| Feature family | Reason excluded |
|----------------|-----------------|
| Tail-number features (aircraft_type, turn_time_hours, has_recent_arrival_turn_5h, tail_leg_num_day, tail_depdelay_mean_last*, tail_lateaircraft_rate_last*, tail_n_legs_scheduled, tail_min_turn_time, tail_has_tight_turn) | Aerodatabox does not publish tail numbers for upcoming flights. v9 production failure traced directly to these. |
| Wheels-off / airtime features (wo_slip_*, airtime_*_d_*) | Actual gate-out, wheels-off, wheels-on times are not available from Aerodatabox at prediction time in a usable form. |
| `origin_nasdelay_rate_last1d` | Aerodatabox airport-delays endpoint returns aggregate delay only, no cause breakdown. Using the generic index as a proxy is an approximation; user rule forbids approximations. |
| `airport_delay_index_origin/dest` (proposed v11 candidate, rejected) | Aerodatabox historical query cost for training-time backfill is prohibitive (~292k calls at BASIC tier). Deferred to v12 when a commercial tier is available, or dropped entirely in favor of BTS-derived proxies. |
| Numeric→categorical encoding (`dep_dow`, `sched_dep_hour`, `is_peak_hour` proposed as categoricals, rejected) | Empirical CatBoost tests on WN showed categorical encoding hurts log-loss by 0.1-0.3%. `DepTimeBlk` is already categorical and covers similar information. |
| `enroute_cape_max` (deferred to v12) | Requires extending Open-Meteo cache to sample along great-circle midpoints; architectural work not blocking v11. |
| IATA AHM 730 delay codes | Not publicly available for US carriers. Would require commercial OAG feed. |
| OPSNET facility-level causes | Requires FAA ASPM registration, no API, manual CSV download. |
| NOTAM runway closures | Requires custom parser from FAA NOTAM Search / FNS; no clean bulk API. |

---

## 14. Target Labels (for reference)

The v11 model trains 4 binary heads per airline per direction:

```
y_dep_ge15  = 1 if DepDelayMinutes >= 15  else 0
y_dep_ge30  = 1 if DepDelayMinutes >= 30  else 0
y_dep_ge60  = 1 if DepDelayMinutes >= 60  else 0
y_dep_ge120 = 1 if DepDelayMinutes >= 120 else 0
```

And analogously `y_arr_geN` for the arrival model. Inference combines the four calibrated probabilities into a 5-bin distribution (`<15, 15-30, 30-60, 60-120, >=120`) with midpoint weights `[7.5, 22.5, 45.0, 90.0, 180.0]` to compute expected delay minutes.

---

## 15. Production Serving Architecture (summary)

```
┌────────────────────────────────────────────────────────┐
│  Prediction-time request: (Carrier, FlightNum,          │
│                            Origin, Dest, FlightDate)    │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
 ┌──────────────────────────────────────────────────────┐
 │  Feature assembly service joins 4 sources:             │
 │                                                       │
 │  1. Aerodatabox (live)    → schedule fields,           │
 │     CRSDepTime, CRSArrTime, CRSElapsedTime, Distance,  │
 │     DepTimeBlk                                         │
 │                                                       │
 │  2. Open-Meteo (live)     → 18 weather features        │
 │     (origin+dest, daily+hourly)                        │
 │                                                       │
 │  3. BTS offline cache     → ~55 rolling aggregates     │
 │     served as lookup tables keyed by                   │
 │     (identifier, FlightDate)                           │
 │                                                       │
 │  4. Strike cache (offline) → 3 labor-action features    │
 └───────────────────────────┬──────────────────────────┘
                            │
                            ▼
 ┌──────────────────────────────────────────────────────┐
 │  CatBoost v11 bundle (per airline, per dep/arr)       │
 │  emits p_ge15, p_ge30, p_ge60, p_ge120 + expected      │
 │  delay in minutes + severity score                    │
 └──────────────────────────────────────────────────────┘
```

**Refresh schedules:**
- BTS offline cache: monthly, triggered by BTS's monthly release.
- Strike cache: manual, whenever new labor actions announced.
- Open-Meteo: real-time forecast, day-before prediction.
- Aerodatabox: real-time schedule lookup at prediction time.

---

## 16. Verification Commands

```bash
# (a) Smoke the v11 pipeline end-to-end (WN only)
.venv/bin/python src/fetch_prune/prepare_dataset.py data/blueprint_dep_WN_v11_smoke.json
.venv/bin/python src/fetch_prune/features_dep.py data/blueprint_dep_WN_v11_smoke.json
.venv/bin/python src/training/train_dep_bins_ordinal_catboost.py data/dep_train_WN_100_v11_smoke.json

# (b) Re-run the v11 validation suite
.venv/bin/python data_quality/2026-04-16-v11-smoke-validation.py

# (c) Confirm no v10 forbidden columns exist in v11 training configs
for f in data/*_train_*_v11.json; do
    grep -q 'origin_nasdelay_rate_last1d\|tail_\|wo_slip_\|airtime_mean_' "$f" && echo "LEAK in $f"
done

# (d) Confirm the late-aircraft threshold constant is set to 15
grep 'LATE_AIRCRAFT_THRESHOLD_MIN' src/fetch_prune/features_dep.py
```

---

## 17. Changelog of This Guide

- **2026-04-17** — Initial v11 compatibility guide. Covers 4 categorical + 78 numeric departure features, plus 22 arrival-specific features. Mathematical specifications for all 11 derived / computed features.
