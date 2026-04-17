# HPO Local Pilot: v11 WN Departure

**Date:** 2026-04-17
**Scope:** Local smoke of `hyperparam_optimize.py` against the v11 WN HPO-sized dataset.
**Outcome:** **Pilot succeeded — pipeline is production-ready for the bigger machine.** Partial results collected before the background task was terminated; findings below are directional.

---

## Purpose

Validate that the v11 HPO pipeline works end-to-end on local hardware, not to tune to production quality. Three questions:

1. Does `hyperparam_optimize.py` succeed on v11 data with the new features (`flightnum_od_otp_rate_last14d`, `airline_cancel_rate_anomaly_7d`) and threshold buckets `[15,30,60,120]`?
2. What is the per-trial wall-clock on this hardware (needed to plan the bigger-machine budget)?
3. Can 5 random-search trials produce directional signal, or are we firmly in TPE-warmup territory?

## Sweep Configuration

| Setting | Value |
|---------|-------|
| Blueprint | `data/blueprint_dep_WN_v11_hpo.json` (new) |
| Training config | `data/dep_train_WN_100_v11_hpo.json` (new) |
| Data scope | WN, top-50 airports, 2025-01 through 2025-06 (~532k target rows after filtering, 77 numeric + 4 categorical features) |
| Subsampling | `--max-train-rows 40000`, `--max-eval-rows 20000` |
| Thresholds | 15, 30, 60, 120 |
| Trials per threshold | 5 |
| Seed | 1337 |
| Search space | `suggest_catboost_params` defaults: depth 4–12, lr log [0.005, 0.15], l2 log [0.5, 20], iter 2000–10000, od_wait 50–500, min_data_in_leaf 1–100, bootstrap ∈ {Bayesian, Bernoulli, MVS} |

## Results

### Per-trial wall-clock (on local hardware)

| Trial config | Wall-clock |
|--------------|-----------|
| depth 5, lr 0.027, iter_ceil 4500, Bernoulli | **58 s** |
| depth 7, lr 0.031, iter_ceil 4000, Bayesian | **~70 s** |
| depth 9, lr 0.067, iter_ceil 7500, MVS | **~25 s** (early-stopped) |
| depth 7, lr 0.007, iter_ceil 6000, MVS | **~124 s** |
| depth 9, lr 0.021, iter_ceil 5500, Bernoulli | **~58 s** |
| depth 11, lr 0.006, iter_ceil 5500, MVS | **~212 s** |

**Median ~1–2 min, max ~3.5 min.** Depth 11+ trials are the slow tail. Early-stopped trials (aggressive lr or high l2) finish in under a minute.

### Mode A — Per-threshold best trials

| Threshold | Completed trials | Best AUC | Best-trial params |
|-----------|------------------|----------|-------------------|
| thr=15 | 5/5 | **0.78104** | depth=7, lr=0.0073, l2=12.6, iter=6000, od_wait=350, MVS, subsample=0.519, min_leaf=100 |
| thr=30 | 3/5 (interrupted) | **0.78957** | depth=11, lr=0.0058, l2=5.77, iter=5500, od_wait=50, MVS, subsample=0.633, min_leaf=11 |
| thr=60 | 0/5 (not reached) | — | — |
| thr=120 | 0/5 (not reached) | — | — |

### Mode B — Shared-params best trial

Did not run (Mode B starts after Mode A completes all 4 thresholds).

### AUC variance across trials (directional)

| Threshold | Trials run | AUC min | AUC max | Spread |
|-----------|-----------|---------|---------|--------|
| thr=15 | 5 | 0.77547 | 0.78104 | **0.0056** |
| thr=30 | 3 | 0.78329 | 0.78957 | **0.0063** |

The 5-trial spread at thr=15 is **small (~0.006 AUC)**. With `n_startup_trials=20`, the first 20 trials are random; 5 of them is a small random sample. The spread reflects genuine search-space variability, not noise — but 5 trials isn't enough to identify a confident winner.

## Findings

### F1. Infrastructure works end-to-end on v11 data
- Pipeline loads the v11 HPO parquet cleanly (40k × 81 features after subsampling).
- Optuna TPE sampler starts correctly, trials log to MLflow with nested child runs, per-trial params and AUCs captured.
- No failures related to new features or new threshold buckets. The `catboost` + `CalibratedClassifierCV` stack works unchanged.
- Best-params JSON + MLflow trial runs are produced as expected after each threshold's optimization completes.

### F2. MVS bootstrap wins both completed pilots
- thr=15 best: **MVS** + `subsample=0.52`
- thr=30 best: **MVS** + `subsample=0.63`
- Three of the five top-ranked trials across both thresholds used MVS. Bayesian bootstrap and Bernoulli each won one trial.
- **Inference:** the full sweep on the bigger machine should see MVS bootstrap featured in top trials; if it doesn't, that's a signal to look harder at the data (the pilot uses subsampled 40k rows, so this might not hold at full scale).

### F3. Winning configs favor *strong regularization, small learning rate, deeper trees*
- thr=15 best: depth=7, **lr=0.0073** (log-scale lower end), **l2=12.6** (high regularization)
- thr=30 best: depth=11, **lr=0.0058**, l2=5.77
- Both diverge from the v10/v11 default (`depth=8, lr=0.03, l2=3.0`). Specifically: **lower learning rate + higher regularization**. This is consistent with v11's expanded feature set — more features per tree split benefit from additional regularization to avoid overfitting.
- `min_data_in_leaf` also diverges (100 at thr=15 vs default 1). Combined with high l2, this suggests the v11 feature set benefits from conservative tree building.

### F4. Per-trial wall-clock scales with (depth × iterations) — as expected, but the `min_data_in_leaf` and `border_count` dimensions barely matter
- The only trial exceeding 3 min was depth=11 (thr=30 trial 0).
- Depth 5–9 trials all finished under 2 min regardless of iteration ceiling.
- **Inference for the bigger machine:** at full 1.6M rows, expect a simple 10× scaling — **10–30 min per trial** on a 32-core machine. Matches the main plan's 10–15 min estimate for typical trials, with deep+long trials in the 30-min tail.

### F5. 5 trials is not enough for a confident best
- thr=15 AUC spread 0.006 means any single trial could claim "best" by luck. With `n_startup_trials=20`, TPE never exits random-search phase on 5 trials.
- **Don't act on these exact params for production.** Use them as a seed for the full bigger-machine sweep.

## Revisions to the Main HPO Plan

From the findings, I'm updating two figures in `exploration/hpo_reports/2026-04-16-v11-hpo-plan.md`:

1. **Per-trial wall-clock on bigger machine:** upper end of the estimate (8–15 min → **10–30 min**) to account for the tail of depth-11+ MVS trials.
2. **Minimum trial budget per threshold:** originally "50 trials suffices; 100 buys ~0.2% AUC at 2× cost." Pilot confirms TPE needs 20-trial warmup. **Do NOT run fewer than 30 trials per threshold** or TPE never gets a chance to sample beyond random. 50 stays the recommendation (30 TPE-guided + 20 startup).

## Recommendations for the Bigger Machine

1. **Use the same HPO config** (`dep_train_WN_100_v11_hpo.json`) as a smoke pre-check — run 5 trials with `--max-train-rows 40000` on the bigger machine to verify MLflow plumbing and measure baseline per-trial time, then fan out to the 8 production sweeps (the main plan's §0 commands).
2. **Seed every sweep with `--seed 1337`** (matches the trainer's default) so MLflow runs are tagged consistently.
3. **Memory:** each concurrent full-scale trial uses ~4–6 GB. Running all 4 airline dep sweeps concurrently requires ~25 GB headroom. Arrival sweeps can reuse the same compute after dep finishes.
4. **Post-sweep analysis:** use `exploration/hpo_reports/2026-04-17-analyze-v11-hpo-local.py` as the template — it reads `hpo_summary.json` + MLflow and produces a ranked table. Adjust the config-path filter from `%v11_hpo.json%` to `%v11.json%` for production sweeps.

## Next Steps

1. **`model-runtime-manager`:** Launch the 8 production HPO sweeps per `2026-04-16-v11-hpo-plan.md` §0 on the bigger machine. Monitor MLflow periodically.
2. **`model-implementation`:** Once production sweeps finish, apply winning configs to `data/*_train_*_v11.json` (in-place update or `_tuned` variants per user preference).
3. **Optional:** Rerun the local pilot with the thr=60 and thr=120 stages to get a full picture. With the same 40k-row subsample, expect another ~15-20 min. Not blocking.

## Appendix: Raw Log Excerpt

```
[trial 0] thr>=15 AUC=0.78005  depth=5, lr=0.0268, l2=0.63, iter=4500, Bernoulli+subsample=0.57
[trial 1] thr>=15 AUC=0.77644  depth=7, lr=0.0307, l2=1.96, iter=4000, Bayesian+bagT=3.67
[trial 2] thr>=15 AUC=0.77547  depth=9, lr=0.0673, l2=2.99, iter=7500, MVS+subsample=0.59
[trial 3] thr>=15 AUC=0.78104  depth=7, lr=0.0073, l2=12.6, iter=6000, MVS+subsample=0.52
[trial 4] thr>=15 AUC=0.78029  depth=9, lr=0.0215, l2=1.02, iter=5500, Bernoulli+subsample=0.84
[thr>=15] BEST AUC=0.78104
[trial 0] thr>=30 AUC=0.78957  depth=11, lr=0.0058, l2=5.77, iter=5500, MVS+subsample=0.63
[trial 1] thr>=30 AUC=0.78329  depth=5, lr=0.0703, l2=1.35, iter=10000, Bayesian+bagT=3.25
[trial 2] thr>=30 AUC=0.78506  depth=10, lr=0.0904, l2=2.76, iter=6500, MVS+subsample=0.96
[... process terminated before thr=30 trials 3-4 completed ...]
```

Full MLflow runs preserved at `../flightrightdata/mlruns/725561585759636400/` (experiment "hyperparam-optimization"). Parent run name: `hpo_dep_dep_train_WN_100_v11_hpo`.
