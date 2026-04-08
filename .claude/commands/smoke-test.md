---
description: Run a fast end-to-end pipeline smoke test for a v10 model to verify nothing is broken.
argument-hint: "[airline] [model] (defaults: WN dep)"
allowed-tools: Bash, Read, Glob, Grep, Edit, Write
---

# Smoke test

Run a fast, lightweight end-to-end test of the v10 feature pipeline to verify the
plumbing is intact. A smoke test is **not** a correctness test — it only checks that
the pipeline runs without crashing, that the new feature columns are populated, that
no forbidden columns (tail/wheels-off/aircraft) leak through, and that CatBoost
accepts the resulting feature list.

## Arguments

`$ARGUMENTS` should be one of:

- *(empty)* → `WN dep` (default)
- `WN` / `UA` / `AA` / `DL` → that airline, departure model
- `WN dep` / `UA arr` / `AA cancel` etc. → airline + model

If the user passes an unknown airline or an unknown model, stop and ask which they meant.

## What "smoke test" means here

A trimmed-down run that finishes in a few minutes:

- ~1 month of BTS data instead of ~2 years
- Top 20 airports instead of top 100
- Strike features disabled (no cache in dev)
- 200 CatBoost iterations instead of 6,000
- Single delay threshold (`ge15`) instead of all four
- Date range chosen to fall inside the cached weather window

## Steps

1. **Pick a date range that fits the cached weather.** Run:

   ```
   ls ../flightrightdata/weather_cache/ | head -3
   ```

   Choose a 1-month input window whose end date is inside every cached parquet's
   range. As of the last working date, June 2025 is safe (cached weather extends
   to 2025-06-30 daily / 2025-07-02 hourly).

2. **Confirm the smoke configs exist** for the requested airline+model. The
   reference set lives at:

   - `data/blueprint_dep_{AIRLINE}_v10_smoke.json`
   - `data/blueprint_arr_{AIRLINE}_v10_smoke.json`
   - `data/dep_train_{AIRLINE}_100_v10_smoke.json`
   - `data/arr_train_{AIRLINE}_100_v10_smoke.json` (if it exists)

   If a smoke config is missing for the requested airline, derive it from the
   matching WN smoke config by copying and rewriting the airport filter, hub
   list, and output paths. Do **not** modify the production v10 configs.

3. **Run the pipeline stages in order.** Always invoke Python via `.venv/bin/python`.
   Each command writes its output parquet to `../flightrightdata/data/...`.

   For `dep`:

   ```
   .venv/bin/python src/fetch_prune/prepare_dataset.py data/blueprint_dep_{AIRLINE}_v10_smoke.json
   .venv/bin/python src/fetch_prune/features_dep.py    data/blueprint_dep_{AIRLINE}_v10_smoke.json
   .venv/bin/python src/training/train_dep_bins_ordinal_catboost.py data/dep_train_{AIRLINE}_100_v10_smoke.json
   ```

   For `arr` (depends on the dep enriched parquet, so run dep first if missing):

   ```
   .venv/bin/python src/fetch_prune/features_arr.py data/blueprint_arr_{AIRLINE}_v10_smoke.json
   ```

   For `cancel`:

   ```
   .venv/bin/python src/fetch_prune/prepare_cancel_dataset.py data/blueprint_cancel_v10_smoke.json
   .venv/bin/python src/training/train_cancellation_catboost.py data/cancel_train_v10_smoke.json
   ```

   The training script may exit non-zero with `ValueError: This script currently
   expects exactly 4 thresholds.` after the per-threshold training step finishes.
   That is **expected** for the smoke run (we only train one threshold) and is
   **not** a failure. Confirm CatBoost reported a final AUC for the threshold and
   wrote the model + reliability plot under `data/models/`.

4. **Run the column audit on the produced parquet** to confirm zero forbidden
   columns leaked through and that the new v10 columns are populated:

   ```
   .venv/bin/python - <<'PY'
   import pandas as pd, sys
   p = "../flightrightdata/data/processed/features_dep_{AIRLINE}_v10_smoke_unbalanced.parquet"
   df = pd.read_parquet(p)
   FORBIDDEN = ("tail_","aircraft_type","has_recent_arrival_turn",
                "turn_time_hours","wo_slip_","airtime_mean_","airtime_n_")
   hits = [c for c in df.columns if any(c.startswith(f) for f in FORBIDDEN)]
   print("rows:", len(df), "cols:", len(df.columns), "forbidden:", hits)
   for c in ["carrier_depdelay_median_last1","dest_depdelay_median_last1",
             "origin_nasdelay_rate_last1d","cancel_rate_origin_last1d",
             "divert_rate_origin_last14d"]:
       if c in df.columns:
           s = df[c]
           print(f"  {c}: cov={s.notna().mean():.3f} mean={s.mean():.4f}")
       else:
           print(f"  MISSING: {c}")
   PY
   ```

   For `arr`, swap the parquet path to `features_arr_{AIRLINE}_v10_smoke_unbalanced.parquet`
   and check `arrdelay_median_14d_fn_od`, `arrdelay_median_14d_car_od`,
   `elapsed_time_ratio_last14d` instead.

5. **Report the result.** A clean smoke test reports:

   - Row counts at every stage
   - Coverage and mean of every new v10 feature
   - "Forbidden columns: 0"
   - CatBoost final AUC for the trained threshold

   If anything fails, surface the actual error (do not summarize it away) and
   stop — do **not** start "fixing" things until the user confirms what to do.

## Out of scope

A smoke test never:

- trains all four thresholds
- runs multi-airline batches
- modifies production v10 configs (`blueprint_*_v10.json`, `*_train_*_v10.json`)
- pushes commits
- counts as evidence the model is good
