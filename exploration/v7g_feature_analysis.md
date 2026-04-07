# v7g Feature Analysis: Departure & Arrival Models

## 1. Model Performance Overview

- **Departure model AUC**: 0.7626
- **Arrival model AUC**: 0.7488
- **Weather-only model AUC**: 0.6691 (baseline reference)

## 2. Feature Importance Rankings

### Departure Model -- Top 15 by SHAP

| Rank | Feature | SHAP |
|------|---------|------|
| 1 | CRSDepTime | 0.28728 |
| 2 | sched_dep_hour | 0.26803 |
| 3 | DepTimeBlk | 0.13633 |
| 4 | flightnum_od_depdelay_median_last14 | 0.13112 |
| 5 | dep_dow | 0.09645 |
| 6 | origin_daily_precip_sum_mm | 0.08990 |
| 7 | carrier_origin_depdelay_mean_last1 | 0.07409 |
| 8 | origin_congestion_3h_total | 0.06390 |
| 9 | origin_airline_congestion_3h_total | 0.05908 |
| 10 | hub_3_lateaircraft_rate_last1 | 0.05234 |
| 11 | flightnum_od_depdelay_median_last7 | 0.05178 |
| 12 | hub_0_lateaircraft_rate_last1 | 0.04217 |
| 13 | origin_dep_visibility_m | 0.03813 |
| 14 | Dest | 0.03756 |
| 15 | Origin | 0.03750 |

### Arrival Model -- Top 15 by SHAP

| Rank | Feature | SHAP |
|------|---------|------|
| 1 | CRSDepTime | 0.22524 |
| 2 | sched_dep_hour | 0.18006 |
| 3 | origin_daily_precip_sum_mm | 0.10771 |
| 4 | sched_arr_hour | 0.09336 |
| 5 | dep_dow | 0.08504 |
| 6 | arrdelay_median_7d_fn_od | 0.06244 |
| 7 | dest_arr_visibility | 0.06061 |
| 8 | arrdelay_median_7d_car_od | 0.05999 |
| 9 | wo_slip_mean_7d_origin_blk | 0.05979 |
| 10 | flightnum_od_depdelay_median_last14 | 0.05278 |
| 11 | hub_0_lateaircraft_rate_last1 | 0.05241 |
| 12 | origin_congestion_3h_total | 0.04747 |
| 13 | DepTimeBlk | 0.04458 |
| 14 | wo_slip_n_14d_fn_od | 0.04332 |
| 15 | hub_3_lateaircraft_rate_last1 | 0.04274 |

## 3. Redundancy Analysis

Found **112** feature pairs with |Spearman r| > 0.85.

### Highly Correlated Pairs (top 15)

| Feature A | Feature B | r |
|-----------|-----------|---|
| flightnum_od_depdelay_mean_last1 | flightnum_od_depdelay_median_last1 | 1.000 |
| flightnum_od_depdelay_mean_last7 | flightnum_od_depdelay_mean_last14 | 0.852 |
| carrier_depdelay_mean_last1 | carrier_lateaircraft_rate_last1 | 0.966 |
| carrier_depdelay_mean_last7 | carrier_depdelay_mean_last14 | 0.872 |
| carrier_depdelay_mean_last7 | carrier_lateaircraft_rate_last7 | 0.975 |
| carrier_depdelay_mean_last7 | carrier_lateaircraft_rate_last14 | 0.857 |
| carrier_depdelay_mean_last7 | hub_1_depdelay_mean_last7 | 0.858 |
| carrier_depdelay_mean_last7 | hub_1_lateaircraft_rate_last7 | 0.869 |
| carrier_depdelay_mean_last7 | hub_2_depdelay_mean_last7 | 0.862 |
| carrier_depdelay_mean_last7 | hub_2_lateaircraft_rate_last7 | 0.871 |
| carrier_depdelay_mean_last7 | hub_3_depdelay_mean_last7 | 0.866 |
| carrier_depdelay_mean_last7 | hub_3_lateaircraft_rate_last7 | 0.898 |
| carrier_depdelay_mean_last7 | hub_4_depdelay_mean_last7 | 0.895 |
| carrier_depdelay_mean_last7 | hub_4_lateaircraft_rate_last7 | 0.895 |
| carrier_depdelay_mean_last14 | carrier_lateaircraft_rate_last7 | 0.873 |

### Rolling Window Redundancy

Within each rolling window group (last1/7/14), features are highly correlated:

- **flightnum_od_depdelay**: min cross-correlation = 0.500
- **carrier_depdelay**: min cross-correlation = 0.549
- **carrier_origin**: min cross-correlation = 0.490
- **origin_depdelay**: min cross-correlation = 0.509
- **origin_lateaircraft**: min cross-correlation = 0.542

The last1 (1-day) window carries the most unique signal due to recency. 
The last14 (14-day) window is the most stable but least reactive to sudden changes. 
Consider keeping last1 and last14, dropping last7 where redundant.

## 4. Features Recommended to Drop

Of 85 total features, **49** are candidates for removal.

| Feature | Importance | SHAP | Reason |
|---------|-----------|------|--------|
| Reporting_Airline | 0.0 | 0.00000 | low_importance |
| is_holiday | 0.0 | 0.00029 | low_importance |
| dest_depdelay_mean_last7 | 0.2 | 0.00130 | low_importance |
| origin_daily_windspeed_max_kmh | 0.2 | 0.00212 | low_importance |
| origin_depdelay_mean_last7 | 0.1 | 0.00248 | low_importance |
| origin_dep_windspeed_kmh | 0.1 | 0.00291 | low_importance |
| flightnum_od_low_support_last14d | 0.1 | 0.00299 | low_importance |
| origin_depdelay_mean_last1 | 0.1 | 0.00357 | low_importance |
| origin_lateaircraft_rate_last14 | 0.1 | 0.00374 | low_importance |
| dest_depdelay_mean_last14 | 0.1 | 0.00406 | low_importance |
| origin_lateaircraft_rate_last7 | 0.1 | 0.00427 | low_importance |
| dest_depdelay_mean_last1 | 0.2 | 0.00429 | low_importance |
| hub_4_lateaircraft_rate_last7 | 0.3 | 0.00437 | low_importance, redundant_with_hub_4_depdelay_mean_last7(r=0.96) |
| carrier_lateaircraft_rate_last14 | 0.3 | 0.00466 | low_importance |
| origin_depdelay_mean_last14 | 0.1 | 0.00501 | low_importance |
| hub_3_depdelay_mean_last7 | 0.4 | 0.00509 | low_importance, redundant_with_hub_3_lateaircraft_rate_last7(r=0.95) |
| hub_0_lateaircraft_rate_last7 | 0.3 | 0.00525 | low_importance, redundant_with_hub_0_depdelay_mean_last7(r=0.97) |
| hub_4_lateaircraft_rate_last14 | 0.3 | 0.00565 | low_importance, redundant_with_hub_4_depdelay_mean_last14(r=0.97) |
| carrier_depdelay_mean_last14 | 0.3 | 0.00640 | low_importance, redundant_with_carrier_lateaircraft_rate_last14(r=0.98) |
| flightnum_od_depdelay_mean_last1 | 0.1 | 0.00657 | low_importance, redundant_with_flightnum_od_depdelay_median_last1(r=1.00), high_null(37.7%) |

## 5. Top Feature Interactions

CatBoost-detected pairwise interactions (internal splitting patterns):

| Rank | Feature A | Feature B | Strength |
|------|-----------|-----------|----------|
| 1 | CRSDepTime | origin_dep_visibility_m | 0.4 |
| 2 | Origin | DepTimeBlk | 0.3 |
| 3 | origin_airline_congestion_3h_total | CRSDepTime | 0.3 |
| 4 | Dest | origin_daily_windgusts_max_kmh | 0.3 |
| 5 | Dest | sched_dep_hour | 0.2 |
| 6 | sched_dep_hour | CRSDepTime | 0.2 |
| 7 | dep_dow | CRSDepTime | 0.2 |
| 8 | flightnum_od_depdelay_median_last14 | CRSDepTime | 0.2 |
| 9 | origin_congestion_3h_total | origin_daily_precip_sum_mm | 0.2 |
| 10 | hub_3_depdelay_mean_last1 | CRSDepTime | 0.2 |

## 6. Recommended New Features

Baseline full-model AUC: 0.7608

| Feature | Description | AUC Delta (bp) | Pearson r |
|---------|-------------|----------------|-----------|
| is_peak_hour | departure hour 16-20 flag | +2.9 | 0.1981 |
| hub_max_lateaircraft_last1 | max hub late-aircraft rate (last 1d) | +2.0 | 0.1241 |
| wind_x_precip | windspeed * precipitation | +1.8 | 0.0492 |
| is_convective | CAPE > 500 J/kg flag | -0.3 | 0.0764 |
| hub_max_delay_last1 | max hub delay across all hubs (last 1d) | -3.4 | 0.1066 |
| delay_trend_7d | carrier delay last1 - last7 (trend) | -9.7 | 0.0674 |
| gust_factor | windgusts / windspeed turbulence ratio | -11.2 | 0.0229 |

**Recommended to add** (positive AUC impact):

- **is_peak_hour**: departure hour 16-20 flag (+2.9 bp)
- **hub_max_lateaircraft_last1**: max hub late-aircraft rate (last 1d) (+2.0 bp)
- **wind_x_precip**: windspeed * precipitation (+1.8 bp)

## 7. New-API Feature Viability (Tail, Aircraft, Terminal)

With the new API providing tail number, aircraft type, and terminal 
assignments days ahead of departure, the following features become viable.

### Marginal AUC Impact (Departure Model)

| Feature Group | Description | AUC Delta (bp) | # Features |
|---------------|-------------|----------------|------------|
| ALL_tail_features | All tail/aircraft features combined | +55.1 | 11 |
| has_recent_arrival_turn_5h | Binary: tail arrived within last 5h (quick-turn flag) | +19.7 | 1 |
| aircraft_type | Aircraft model (737-7H4, 737-8, etc) | +9.4 | 1 |
| turn_time_hours | Hours since this tail's previous arrival (turnaround time) | +8.3 | 1 |
| tail_lateaircraft_history | Tail-specific late-aircraft rate (1/7/14d rolling) | -7.1 | 3 |
| tail_depdelay_history | Tail-specific mean dep delay (1/7/14d rolling) | -9.5 | 3 |
| tail_leg_num_day | Which leg of the day this tail is on (1st, 2nd, 3rd...) | -10.0 | 1 |
| Aircraft_Age_Bucket | Aircraft age bracket (0-5, 6-10, 11-15, 16+) | -11.7 | 1 |

**Combined tail/aircraft package**: +55.1 bp (11 features)

### Terminal Features

**Status**: NOT_IN_BTS_DATA

BTS On-Time data does not include terminal or gate fields. If the new API provides origin_terminal and dest_terminal, these would need to be engineered from scratch. Terminal-level congestion features (departures per terminal per hour) could be powerful at hub airports where different terminals serve different routes/airlines. Cannot evaluate from historical data alone.

**Recommendation**: WORTH TESTING when available. At major hubs (ATL, ORD, DFW, DEN), terminal assignment correlates with operational patterns. Suggested features: terminal_congestion_3h, terminal_delay_rate_last7d, is_concourse_change (connection risk at dest).

## 8. Tail-Based Historical Tracking

- Tail history marginal value beyond flight-number history: **-11.6 bp**
  - Base AUC (with flightnum): 0.76049
  - With tail history added: 0.75933

### Window-by-Window Tail Importance

| Window | AUC Delta (bp) |
|--------|----------------|
| last1 | -0.3 |
| last7 | -0.3 |
| last14 | -0.4 |

### Tail Delay Persistence: r = 0.011

High r means tail delay propensity is stable over time (individual aircraft have consistent delay patterns). Low r means tail history is noisy/transient.

### History Signal Correlation Comparison

| Source | last1 | last7 | last14 |
|--------|-------|-------|--------|
| tail_delay | 0.0473 | 0.0639 | 0.0667 |
| tail_lateaircraft | 0.0508 | 0.0742 | 0.0752 |
| flightnum_delay | 0.1522 | 0.1958 | 0.2210 |
| flightnum_median | 0.1522 | 0.1803 | 0.2008 |
| carrier_delay | 0.1326 | 0.1201 | 0.1078 |
| carrier_origin | 0.1360 | 0.1378 | 0.1342 |

## 9. Cascading Delay Risk

- **39,727** unique tail-days analyzed
- Average legs per tail-day: **5.0**
- Any delay: **53.7%** of tail-days
- Cascade (>1 leg delayed): **32.8%** of tail-days
- **P(cascade | first leg delayed) = 74.9%**

### Delay Propagation

- P(delay | previous leg delayed) = **0.740**
- P(delay | previous leg OK) = **0.161**
- **Lift factor: 4.61x**

### Delay Absorption by Turn Time

| Turn Time | P(next delayed | prev delayed) |
|-----------|-------------------------------|
| 30-45m | 0.809 |
| 45-60m | 0.596 |
| 4h+ | 0.719 |

### Cascade Prediction Model (schedule-only features)

Using only features knowable days ahead (# legs, min turn time, etc.):
- **AUC: 0.6433**, Average Precision: 0.5686
- Test cascade rate: 44.2%

### Proposed Cascade Risk Features

Features computable days ahead from tail number + schedule:

- **`tail_n_legs_scheduled`**: Number of legs this tail is scheduled for today. More legs = more cascade opportunity.
- **`tail_min_turn_time`**: Minimum scheduled turn time for this tail today. Tight turns have less recovery buffer.
- **`tail_mean_turn_time`**: Average scheduled turn time. Loose schedules absorb delays better.
- **`tail_has_tight_turn`**: Binary: any turn < 45 min in today's rotation. Single tight turn can cascade the rest of the day.
- **`tail_first_dep_hour`**: First scheduled departure hour. Early starts accumulate more legs.
- **`tail_route_complexity`**: Number of unique airports in today's rotation. More airports = more weather/congestion exposure.
- **`tail_historical_delay_propensity`**: Tail's 14-day rolling delay rate. Some aircraft consistently delay (maintenance issues, configuration problems).
- **`tail_schedule_buffer_ratio`**: Total scheduled block time / total actual block time (historical). Tight scheduling leaves less slack.

### Cascade Model Feature Importance

| Feature | Importance |
|---------|-----------|
| n_legs | 22.4 |
| max_sched_dep | 19.0 |
| mean_turn | 17.7 |
| mean_distance | 16.5 |
| min_sched_dep | 16.1 |
| n_airports | 8.3 |

## 10. Summary of Recommendations

### Add

- `is_peak_hour`: departure hour 16-20 flag
- `hub_max_lateaircraft_last1`: max hub late-aircraft rate (last 1d)
- `wind_x_precip`: windspeed * precipitation

### Drop (low importance or redundant)

- `Reporting_Airline`: low_importance
- `is_holiday`: low_importance
- `dest_depdelay_mean_last7`: low_importance
- `origin_daily_windspeed_max_kmh`: low_importance
- `origin_depdelay_mean_last7`: low_importance
- `origin_dep_windspeed_kmh`: low_importance
- `flightnum_od_low_support_last14d`: low_importance
- `origin_depdelay_mean_last1`: low_importance
- `origin_lateaircraft_rate_last14`: low_importance
- `dest_depdelay_mean_last14`: low_importance

### Add (Tail/Aircraft -- new API)

- `aircraft_type`: strongest single categorical addition
- `has_recent_arrival_turn_5h`: strong binary signal for quick-turn risk
- `turn_time_hours`: turnaround buffer signal
- `tail_leg_num_day`: later legs carry more cascade risk
- Full tail/aircraft package: best combined AUC gain

### Add (Cascade Risk -- new API)

- `tail_n_legs_scheduled`: more legs = more cascade opportunity
- `tail_min_turn_time`: tightest turn in rotation = weakest link
- `tail_has_tight_turn`: binary cascade vulnerability flag
- `tail_route_complexity`: # unique airports in rotation

### Keep Investigating

- Rolling window consolidation: test dropping last7 across all groups
- Hub spillover: test aggregating hub_0-4 into single max/mean features
- Arrival model: destination weather features need same deep-dive as origin weather
- Terminal features: evaluate when new API data available
- Tail history rolling windows: currently marginal individually, but valuable in the combined package

---
*Analysis run on WN (Southwest Airlines) data. See exploration/figures/ for all charts.*