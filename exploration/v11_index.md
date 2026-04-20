# v11 Index — Models, Mappings, Session Findings

One-page pointer to everything v11. Most-recent updates at the top.

## Trained v11 production bundles (2022–2025, on remote machine)

Location: `../flightrightdata/data/models/`

| Model | Path | Deployable joblib |
|---|---|---|
| WN dep | `dep_WN_100_v11/` | `dep_delay_bins_bundle_WN_100_v11.joblib` (31 MB) |
| AA dep | `dep_AA_100_v11/` | `dep_delay_bins_bundle_AA_100_v11.joblib` (18 MB) |
| DL dep | `dep_DL_100_v11/` | `dep_delay_bins_bundle_DL_100_v11.joblib` (16 MB) |
| UA dep | `dep_UA_100_v11/` | `dep_delay_bins_bundle_UA_100_v11.joblib` (22 MB) |
| WN arr | `arr_WN_100_v11/` | `arr_delay_bins_bundle_WN_100_v11.joblib` (48 MB) |
| AA arr | `arr_AA_100_v11/` | `arr_delay_bins_bundle_AA_100_v11.joblib` (32 MB) |
| DL arr | `arr_DL_100_v11/` | `arr_delay_bins_bundle_DL_100_v11.joblib` (33 MB) |
| UA arr | `arr_UA_100_v11/` | `arr_delay_bins_bundle_UA_100_v11.joblib` (36 MB) |

Each model directory contains per-threshold `.cbm`, `_calibrator.joblib`, `meta.json`, and a `bins_meta.json` (dep) or `arr_train_metrics_*.json` (arr). As of 2026-04-20, the **dep trainer also writes a consolidated `dep_train_metrics_*.json`** (newly added — parallels arr for easy AUC extraction).

Retrained 2025-only v10-regime variants (eval_frac=0.15, pos_frac=0.40) live alongside as `{dep,arr}_{AIR}_100_v11_2025_v10regime_pf040/`. These recover the v10 AUC advantage and are the de-facto reference for "what v11 can do" on same-distribution eval.

## Serve-time feature sourcing (Aerodatabox mapping)

Primary reference — start here for any feature:
- **[v11 Feature Compatibility Guide](v11_feature_compatibility_guide.md)** — per-feature plain-English definition + math + Aerodatabox path at prediction time. 521 lines. The source of truth.

Deeper / more recent:
- **[v11 Full Feature Transferability](../feature_transferability/reports/2026-04-17-v11-full-feature-transferability.md)** — endpoint-by-endpoint verdicts (Adopt / Approximate / Train-only / Drop) for every v11 field.
- **[v11 Aerodatabox Screening](../feature_transferability/reports/2026-04-16-v11-aerodatabox-screening.md)** — original screen.
- **[2026-04-19 origin_nas_rate Aerodatabox parity](../feature_transferability/reports/2026-04-19-origin-nas-rate-aerodatabox-parity.md)** — current verdict: **Approximate only** via FIDS total-delay-rate. Empirical parity test blocked on API key.

## Session headline findings (2026-04-20)

Two findings dominated everything else we learned this session:

### 1. `feature_balance.eval.eval_frac` was flipped 0.15 → 0.85 in v11. **This is the main regression cause.**

- v10 ran with 85% train_pool / 15% eval (eval_frac = 0.15).
- v11 runs with 15% train_pool / 85% eval (eval_frac = 0.85).
- Net: 5.7× fewer training rows in v11, severe-threshold positives especially starved.
- **Fix**: revert to 0.15 across all 8 v11 blueprints. Empirical result on 2025-only: +0.012 to +0.044 AUC lift per (airline × direction × threshold), mean **+0.022 AUC across 32 cells**.
- Details: [2026-04-20-v10-vs-v11-on-v10regime-eval.csv](reports/2026-04-20-v10-vs-v11-on-v10regime-eval.csv)

### 2. Dropping `origin_nasdelay_rate_last1d` from training configs is a minor loss (AUC ~−0.002 on dep, ~−0.011 on arr).

- v11 still COMPUTES the underlying feature (as `origin_nas_rate_last1` in every parquet) but v11 training configs don't list it.
- Ablation result: re-adding it gives essentially zero lift on dep (within ±0.001 AUC).
- On arr, the loss is larger (-0.010 to -0.017 in some cells) because v11 arr training configs dropped NAS WITHOUT adding any v11-new feature (unlike dep which added `flightnum_od_otp_rate_last14d`).
- **Fix (minor)**: add `origin_nas_rate_last{1,7,14}` to all 4 arr training configs — they're already columns in the parquets.

### Everything else we tested was a null result:

- LA threshold change (`> 0` → `>= 15`): <0.002 AUC impact (verified via univariate A/B at multiple thresholds).
- `airline_cancel_rate_anomaly_7d` addition: 0.54 AUC (noise), but dropping doesn't move model-level AUC.
- Categorical / numeric type flips: none (identical 4 categoricals v10 ↔ v11).
- NOAA-ISD visibility swap for Open-Meteo: 0 model lift despite univariate lift (redundant with CAPE / precip / LA features). Kept as dormant code.
- HPO tuning: +0.001 to +0.003 per-threshold; below the +0.01 worth-it threshold.

## Post-session followups (for reference when work resumes)

1. Update all 8 v11 blueprints to set `feature_balance.eval.eval_frac = 0.15`. One-line edit × 8 files.
2. Add `origin_nas_rate_last{1,7,14}` to all 4 arr training configs. One-line edit × 4 files.
3. Drop `airline_cancel_rate_anomaly_7d` from all 8 training configs. Clean up.
4. AA remains structurally weakest — feature-level gap (37% CarrierDelay share not captured by v11). Research/feature-implementation track.

## Where's what — quick navigation

| Need | Location |
|---|---|
| A live model bundle | `../flightrightdata/data/models/{dep,arr}_{AIR}_100_v11/` |
| Per-feature definition + serve-time source | `exploration/v11_feature_compatibility_guide.md` |
| Aerodatabox endpoint verdict per feature | `feature_transferability/reports/2026-04-17-v11-full-feature-transferability.md` |
| v10 vs v11 AUC on same eval | `exploration/reports/2026-04-20-v10-vs-v11-on-v10regime-eval.csv` |
| This session's correlations finding (cross-airline AUC gap) | `exploration/reports/2026-04-19-non-wn-airline-delay-signal.md` |
| Session retrain logs | `/tmp/v11_2025_logs/` (ephemeral) |
| Blueprint generators | `scripts/smoke/generate_v11_2025_configs.py`, `rebalance_v11.py` |
| Trainer code | `src/training/train_{dep,arr}_bins_ordinal_catboost.py` |
| Feature-engineering code | `src/fetch_prune/features_{dep,arr}.py` |
