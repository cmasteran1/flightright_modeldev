# v11 Bug-Fix Retrain Plan — Replace deployed v11 bundles with bug-fixed equivalents

## Context

The v11 stage-1 parity gate (10K Dec-2025 flights, run 20260505T140800Z) shows a **mean (airline × threshold) AUC gap of -0.0356** between BTS-feature scoring and ADB-feature scoring of the same v11 bundle. That gap is the cost of train/serve feature-pipeline drift introduced by a small set of code-level bugs that were diagnosed and fixed across this and prior sessions.

Per user direction, **this retrain stays at v11** — version numbers are reserved for actual feature changes. The deployed v11 bundles (joblibs in S3) are replaced in place with retrained equivalents that correct the bugs. The feature set is unchanged from the current v11 spec at [v11_feature_spec.md](v11_feature_spec.md).

## Bug fixes covered by this retrain (catalog)

### A. Already landed in modeldev (prior sessions)

| # | Bug | Symptom in deployed v11 | Code fix | Resolved by retrain because |
|---|---|---|---|---|
| **A1** | `flightnum_od_support_count_last14d` 2× double-count | `support_count` and `low_support` flag are 2× the true value, biasing the model away from low-support cells. Mean/median/std unaffected. | [features_dep.py:540](src/fetch_prune/features_dep.py:540) (landed 2026-04-25). Regression test: `tests/data_quality/test_flightnum_od_support_count_no_double_count.py`. | Re-extracting the training parquet uses the corrected aggregator and produces non-doubled support counts. Bundle gets retrained on a `support_count` distribution that matches what serving sees, eliminating the systematic bias. |

### B. Already landed in flightright (prior sessions, no retrain dependency)

These are serving-only fixes; they're listed for completeness so the retrain plan reader has the full picture but require no modeldev action:

- `_match_future_flight` UTC-vs-local-date leg selection (prior session, 2026-04-27)
- `_match_future_flight` block-plausibility tie-break for double-track routes (this session)
- `_parse_hhmm` accepts BTS HHMM format without colon (this session, [predict.py:229](../flightright/src/flightright/cli/predict.py:229))
- `_match_future_flight` 30-min local-time tolerance window (this session, [predict.py:316](../flightright/src/flightright/cli/predict.py:316))
- `dep_dow` from `flight_date.weekday()` not `sched_dep_utc.weekday()` ([schedule.py:101](../flightright/src/flightright/features/schedule.py:101), prior session)
- `flights_by_ident` 14-day single-call retry ([aerodatabox/client.py:222](../flightright/src/flightright/integrations/aerodatabox/client.py:222), prior session)
- `wind_x_precip` uses windspeed not gusts ([weather.py:204](../flightright/src/flightright/features/weather.py:204), prior session)
- `OpenMeteoClient` includes `historical-forecast-api` for visibility/cape on old dates (prior session)
- BTS-rollup seeder cancelled-flight asymmetry ([build_bts_rollups.py::aggregate](../flightright/scripts/build_bts_rollups.py)). Affects Step 4 (rebuild SQLite); does not require new modeldev code.

### C. Landed this session (2026-05-06)

| # | Bug | Root cause | Fix | Resolved by retrain because |
|---|---|---|---|---|
| **C1** | airports.csv timezones wrong for 7 airports | FL Panhandle: ECP, VPS, PNS misclassified as Eastern (year-round 60-min offset). AZ regional: AZA, FLG, PRC, YUM misclassified as Denver (60-min offset only during DST, March-November). `dep_dt_local` UTC instant 1 hour off → hourly weather features (`origin_dep_temperature_2m`, `_visibility_m`, `_windspeed_kmh`, `_windgusts_kmh`, `_precip_mm`) joined at the wrong UTC hour. `sched_dep_hour` itself is unaffected (it's `floor(CRSDepTime/100)`, tz-independent). | [airports.csv](../flightrightdata/data/meta/airports.csv) updated 2026-05-05: ECP/VPS/PNS → `America/Chicago`; AZA/FLG/PRC/YUM → `America/Phoenix`. PHX and TUS were already correct from a prior parity cleanup. | Re-extracting the training parquet recomputes `dep_dt_local` and re-joins weather at the correct UTC instants for these airports. Without retrain, the v11 model was trained on wrong-UTC weather for these airports while production now reads from corrected airports.csv → train/serve disagree by 1h on weather features at exactly these airports. |
| **C2** | `carrier_delay_rate_anomaly_7d` serving feature was constant | Serving's `delay_rate = dep_delay_count / dep_flight_count` was always exactly 1.0 — `dep_delay_count` is misnamed; it counts non-cancelled flights with a recorded `DepDelayMinutes` value (data completeness), not delayed flights. So the serving z-score was uniformly ~0 while training computed a real `(CarrierDelay > 0)` rate z-score. Surfaced in the v11 stage-1 gate scorecard as avg_z=1.11 vs BTS. | Added `carrier_delay_count` column to the BTS rollup ([build_bts_rollups.py:108-126](../flightright/scripts/build_bts_rollups.py)), schema migration in [sqlite.py](../flightright/src/flightright/storage/sqlite.py), new getter `get_carrier_daily_carrier_delay_rates` in [rollups.py](../flightright/src/flightright/storage/rollups.py), and switched [strike.py::compute_carrier_delay_anomaly](../flightright/src/flightright/features/strike.py) to use it. Sample-stddev (ddof=1) now matches pandas' default. Deploy-order guard: if every rate in the 67-day window is 0.0 the function returns None (signals "SQLite not yet rebuilt"). | Training side was already correct — no training code change needed. The retrain is required because the deployed v11 model learned the feature's distribution from a real signal but production was serving constant zeros. After Step 4 (SQLite rebuild) populates the column AND the bundles are retrained on the same date range, train and serve agree on the real signal. |
| **C3** | `flightnum_od_otp_rate_last14d` aggregation mismatch | Training computed mean-of-daily-rates (each day weighted equally regardless of how many flights that identity flew that day); serving computed per-flight mean. Diverged on double-track / multiple-daily-leg flight numbers — exactly the cases where OTP signal is most informative. NOT the leg-population mismatch flagged in the parity memo as irreducible — that note was about `flightnum_od_*` mean/median; the OTP rate aggregation bug is independent and fully closeable. Avg z-score 0.565 (max 5.67) in the gate scorecard. | Rewrote [features_dep.py::add_flightnum_od_otp_rate_last14d](src/fetch_prune/features_dep.py) (this session) to use per-flight rolling aggregation matching serving's [build_flightnum_od_history](../flightright/src/flightright/features/flightnumber_history.py) and the convention of `flightnum_od_depdelay_mean_last14`. Verified on a synthetic double-track case: per-flight rate=0.6667; old daily-mean would have produced 0.5. | Re-extracting the training parquet uses the corrected aggregator. Bundle is retrained on per-flight OTP rates that match serving's calculation. |

### D. Structurally not closeable by retrain (out of scope)

- **Airport-pool filter mismatch** (~±0.005-0.03 min/day per rolling feature). Training filters per-airline top-100; SQLite is one shared store filtered to v7g UNION (141 airports). Closing requires per-airline rollup DBs.
- **AA structural weakness at high thresholds** (training AUC at ≥120 = 0.726). Upstream model-architecture issue, not a parity issue.

## Implementation plan — modeldev

### Step 1 — Verify airports.csv fix has propagated

Both modeldev and flightright read [flightrightdata/data/meta/airports.csv](../flightrightdata/data/meta/airports.csv).

```sh
grep -nE '^(ECP|VPS|PNS|AZA|FLG|PRC|YUM),' /Users/connermasteran/software/mpqc/total_flightproject/flightrightdata/data/meta/airports.csv
```
Expect: ECP/VPS/PNS → `America/Chicago`; AZA/FLG/PRC/YUM → `America/Phoenix`.

### Step 2 — Rebuild training parquets (2023-2025)

Per user direction, the bug-fix retrain narrows the date range from the original v11's 2022-2025 to **2023-2025** (3 years). Configs to use are in [data/](../data) under the `_v11_retrain` suffix (generated this session).

```sh
cd /Users/connermasteran/software/mpqc/total_flightproject/flightright_modeldev
# Per airline (parallelizable across airlines):
.venv/bin/python src/fetch_prune/prepare_dataset.py data/blueprint_dep_AA_v11_retrain.json
.venv/bin/python src/fetch_prune/prepare_dataset.py data/blueprint_dep_DL_v11_retrain.json
.venv/bin/python src/fetch_prune/prepare_dataset.py data/blueprint_dep_UA_v11_retrain.json
.venv/bin/python src/fetch_prune/prepare_dataset.py data/blueprint_dep_WN_v11_retrain.json
# Then features_dep.py (ditto):
.venv/bin/python src/fetch_prune/features_dep.py data/blueprint_dep_AA_v11_retrain.json
.venv/bin/python src/fetch_prune/features_dep.py data/blueprint_dep_DL_v11_retrain.json
.venv/bin/python src/fetch_prune/features_dep.py data/blueprint_dep_UA_v11_retrain.json
.venv/bin/python src/fetch_prune/features_dep.py data/blueprint_dep_WN_v11_retrain.json
# And same 8 commands for the arr blueprints.
```

This re-runs `_mk_local_dt` with corrected tz strings (Step C1), the `support_count` aggregator (Step A1), and the `flightnum_od_otp_rate_last14d` per-flight aggregator (Step C3). All three fixes land in one parquet rebuild.

**Validation gate:** for ECP/VPS/PNS, spot-check 5 rows that `origin_dep_temperature_2m` and `origin_dep_visibility_m` differ from the prior v11 parquet. For AZA/FLG/PRC/YUM, confirm summer 2024 rows differ but winter 2024 rows do not (DST-conditional).

### Step 3 — Rebuild rolling-stats SQLite

The rolling-stats DB at `~/.flightright/rolling_stats_bts_dec2025_v7g_v2.db` needs a rebuild for two reasons: (a) it carries the cancelled-flight asymmetry from the prior session's seeder fix, (b) it lacks the new `carrier_delay_count` column populated (currently 0 by default, which is why the deploy-order guard in `compute_carrier_delay_anomaly` returns None).

```sh
cd /Users/connermasteran/software/mpqc/total_flightproject/flightright/scripts
.venv/bin/python build_bts_rollups.py \
  --months 2023-01,2023-02,2023-03,2023-04,2023-05,2023-06,2023-07,2023-08,2023-09,2023-10,2023-11,2023-12,\
2024-01,2024-02,2024-03,2024-04,2024-05,2024-06,2024-07,2024-08,2024-09,2024-10,2024-11,2024-12,\
2025-01,2025-02,2025-03,2025-04,2025-05,2025-06,2025-07,2025-08,2025-09,2025-10,2025-11,2025-12 \
  --airport-filter-csv /tmp/v7g_seeder_format.csv \
  --out ~/.flightright/rolling_stats_bts_2023_2025_v11_retrain.db
```

After build, point both [data_quality/parity/scripts/measure_auc.py](data_quality/parity/scripts/measure_auc.py) and the production `ServiceConfig.rolling_sqlite_path` at the new DB. Production switch is reversible (just change the path back to roll back).

### Step 4 — Retrain v11 bundles

Per-airline (AA, DL, UA, WN), per-direction (dep, arr). Bundle output names are unchanged from the deployed v11 (`dep_delay_bins_bundle_<AIRLINE>_100_v11.joblib`) so an S3 upload overwrites the deployed bundles.

```sh
cd /Users/connermasteran/software/mpqc/total_flightproject/flightright_modeldev
.venv/bin/python src/training/train_dep_bins_ordinal_catboost.py data/dep_train_AA_100_v11_retrain.json
.venv/bin/python src/training/train_dep_bins_ordinal_catboost.py data/dep_train_DL_100_v11_retrain.json
.venv/bin/python src/training/train_dep_bins_ordinal_catboost.py data/dep_train_UA_100_v11_retrain.json
.venv/bin/python src/training/train_dep_bins_ordinal_catboost.py data/dep_train_WN_100_v11_retrain.json
# arr direction uses the equivalent script: train_arr_bins_ordinal_catboost.py (or whatever the arr script is named)
.venv/bin/python src/training/train_arr_bins_ordinal_catboost.py data/arr_train_AA_100_v11_retrain.json
.venv/bin/python src/training/train_arr_bins_ordinal_catboost.py data/arr_train_DL_100_v11_retrain.json
.venv/bin/python src/training/train_arr_bins_ordinal_catboost.py data/arr_train_UA_100_v11_retrain.json
.venv/bin/python src/training/train_arr_bins_ordinal_catboost.py data/arr_train_WN_100_v11_retrain.json
```

Configuration unchanged from the live v11 spec other than:
- Date range narrowed from 2022-2025 → 2023-2025.
- Output parquet paths use `_v11_retrain_unbalanced.parquet` to avoid clobbering the in-place v11 parquets during validation, but the final bundle joblib name is the same as deployed v11.
- `feature_balance.eval.eval_frac = 0.15` (already at the post-fix value in the existing main v11 blueprints; preserved).

This is a multi-hour campaign — best launched overnight or via the model-runtime-manager skill.

### Step 5 — Validate against the parity gate

Before pushing bundles, run the parity gate against the new bundles in a local-only run:

```sh
cd /Users/connermasteran/software/mpqc/total_flightproject
PYTHONPATH=verification/src:flightright/src \
  flightright_modeldev/.venv/bin/python -m verification gate \
    --model-version v11 --only stage1 --allow-network
```

**Pass criteria:**
- MUST_MATCH features all 100% (CRSDepTime, DepTimeBlk, sched_dep_hour, dep_dow, Origin, Dest, is_peak_hour, strike_severity).
- KNOWN_GAP fail list empty (`carrier_delay_rate_anomaly_7d` and `flightnum_od_otp_rate_last14d` both pass).
- Mean (airline × threshold) AUC gap **≤ -0.020** (vs current v11's -0.0356 — half the gap closed).
- ADB-side mean AUC gap vs training **≤ -0.005** (vs current v11's -0.0143).

### Step 6 — Deploy

```sh
# Upload retrained bundles to S3 — overwrites deployed v11
.venv/bin/python scripts/upload_models_to_s3.py --version v11 --airlines AA,DL,UA,WN

# Restart the API service to pick up the new bundles (existing v11 entries in
# config/models.toml are unchanged; the service refetches bundles on startup).
flyctl deploy -a flightright --config fly.toml --dockerfile dockerfile --strategy immediate
```

After deploy, run a small live-traffic comparison using `flightright/src/flightright/cli/compare_models.py --backtest --date <today>` to spot-check: predictions for ECP/VPS/PNS flights should now reflect correct local time; `carrier_delay_rate_anomaly_7d` should be a real, varying signal; double-track flight numbers' OTP rates should match BTS-side spot checks.

**Rollback path:** keep a copy of the prior v11 bundles before overwriting. If the new bundles regress, restore from the backup and switch the rolling-stats SQLite path back.

## Sequencing (recommended, ~2 days)

1. **Day 1 morning** — Step 1 (5 min) + Step 2 parquet rebuild (~1-2 hr per airline-direction, parallelizable across 4 cores ≈ 1-2 hr wall).
2. **Day 1 afternoon** — Step 3 SQLite rebuild over 36 months (~30-60 min depending on machine + S3 throughput).
3. **Day 1 overnight** — Step 4 retrain campaign (8 bundles × 4 thresholds = 32 model fits; CatBoost on a single machine takes ~6-12 hr unattended).
4. **Day 2 morning** — Step 5 parity gate validation (~1 hr).
5. **Day 2 afternoon** — Step 6 deploy + smoke test.

## Acceptance criteria (success looks like)

- Retrained v11 bundles uploaded to S3 with the same filenames as the deployed bundles, overwriting them in place.
- New gate report at `data_quality/parity/reports/gate/<retrain-run-ts>/REPORT.md` showing mean ΔAUC ≤ -0.020.
- ECP/VPS/PNS year-round and AZA/FLG/PRC/YUM summer flights have correct `origin_dep_*` weather features in both training and serving.
- `flightnum_od_support_count_last14d` no longer 2× the true value in deployed bundles.
- `carrier_delay_rate_anomaly_7d` is a real cause-attributed delay z-score in both training and serving (no longer constant 0 at serve time).
- `flightnum_od_otp_rate_last14d` agrees per-flight between training and serving for double-track routes.
- The standing memory `project_v11_double_count_bug.md` can be archived once the retrain ships.
