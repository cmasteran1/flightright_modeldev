---
name: ml-hyperparameter-optimization
description: >
  This skill should be used when the user asks to "tune hyperparameters",
  "optimize the model", "improve training", "what loss should I use",
  "how should I split the data", "pick the best model architecture",
  "set up cross-validation", "calibrate predictions", "do a sweep",
  or otherwise wants machine learning expertise about model
  implementation details and hyperparameter optimization for the
  flight delay prediction project. Use this skill to reason about
  best practices and to design and run HPO sweeps — NOT for picking
  candidate features (use `research` /
  `correlations-and-interactions`) and NOT for wiring feature code
  into `src/` (use `model-implementation`).
metadata:
  version: "0.1.0"
  domain: "flight-delay-ml"
---

# ML Hyperparameter Optimization

Act as the project's machine learning expert. Given a concrete
modeling question or an existing feature set, determine the best
method and hyperparameter configuration to answer the question and
squeeze the most value from the data.

## When To Use This Skill

- Choosing a model family (GBDT, linear, neural, survival, quantile
  regression, etc.) for departure or arrival delay prediction.
- Selecting loss functions suited to heavy-tailed delay
  distributions (log-cosh, Huber, quantile, Tweedie, focal loss for
  thresholded binary labels, etc.).
- Designing a cross-validation scheme that respects temporal
  structure and avoids leakage.
- Calibrating predicted probabilities for delay risk.
- Setting up and running a hyperparameter sweep
  (random, Bayesian, Hyperband, successive halving).
- Diagnosing why a training run underperforms (bias vs. variance,
  leakage, data shift, label noise, class imbalance).
- Optimizing data extraction parameters (lookback windows, rolling
  aggregation widths, bucket sizes) alongside model params.

## When NOT To Use This Skill

- Brainstorming new features or reading papers → `research`.
- Measuring signal strength of a single feature →
  `correlations-and-interactions`.
- Verifying that data values are correct → `data-quality-analysis`.
- Writing the actual pipeline code change into `src/` →
  `model-implementation`.
- Launching and babysitting long training runs →
  `model-runtime-manager`.

## Decision Framework

Before touching hyperparameters, nail down:

1. **The target.** Regression on delay minutes? Binary "delayed more
   than T minutes"? Quantile bands? Zero-inflated delay? The choice
   of target changes everything downstream.
2. **The evaluation metric.** What does the stakeholder actually
   care about: calibrated probability, MAE on delay minutes, quantile
   coverage, top-decile precision? Optimize for *that*, not for a
   metric that is merely convenient.
3. **The split.** Always time-based for this project. Define train,
   validation, and test windows up front and never cross them.
4. **The baseline.** A strong baseline (carrier × origin × hour mean,
   or a shallow LightGBM on core features) is the number any
   hyperparameter sweep must beat.

Only after these four are settled should hyperparameter optimization
begin.

## Recommended Defaults

Use these as starting points, not dogma. Justify any deviation.

- **Model family default:** LightGBM. Fast, handles missing values,
  tolerates categoricals, strong on tabular flight data. Switch to
  XGBoost or CatBoost if there is a specific reason.
- **Loss for delay minutes regression:** Huber or quantile loss
  (0.5) to resist long-tail blowups. Report metrics in both minutes
  and log-minutes.
- **Loss for binary "delayed":** Log-loss with class weighting or
  focal loss when the positive rate is below ~15%.
- **Cross-validation:** Expanding-window time split with at least 3
  folds; never shuffled k-fold.
- **Early stopping:** Always on. Validate every N iterations on a
  held-out window.
- **Categorical handling:** Native categorical support first;
  target encoding only with out-of-fold protection.
- **Imbalance handling:** Prefer class weights over resampling for
  calibration reasons; follow up with Platt or isotonic calibration.

## Hyperparameter Sweeps

Prefer in this order:

1. **Manual coarse pass.** A handful of runs to sanity-check
   ranges and confirm the pipeline is healthy end to end.
2. **Random search** on log-uniform ranges for 20-50 trials — cheap
   and surprisingly strong.
3. **Bayesian optimization** (Optuna / scikit-optimize) once the
   ranges are trustworthy and each trial is non-trivial.
4. **Successive halving / Hyperband** when trials are expensive and
   partial progress is a reliable signal.

Always include these ranges for GBDT by default:

- `learning_rate`: log-uniform [1e-3, 3e-1]
- `num_leaves` or `max_depth`: integer, model-dependent
- `min_data_in_leaf`: log-uniform [5, 500]
- `feature_fraction`: uniform [0.5, 1.0]
- `bagging_fraction`: uniform [0.5, 1.0]
- `lambda_l1`, `lambda_l2`: log-uniform [1e-8, 10]

Do not tune `n_estimators` directly — use early stopping and record
the selected round.

## Data Extraction Hyperparameters

Treat feature extraction windows as tunable:

- Rolling carrier on-time rate window (1, 7, 14, 30 days).
- Airport congestion aggregation window (15, 30, 60 minutes).
- Number of lagged scheduled arrivals/departures to include.
- Bucket widths for time-of-day discretization.

Sweep these alongside the model when feasible, but always log the
extraction config with the trial results so runs are reproducible.

## Leakage and Pitfalls

- Rolling aggregations must use only data **before** the target
  flight's `scheduled_departure`.
- Encoders must be fit on the training fold only.
- Do not normalize on statistics computed over the full dataset.
- Weather "nowcast" features are fine; weather "actual observed at
  time of landing" is leakage for an arrival-delay model whose
  prediction is made hours earlier. Call this out to the user
  explicitly.

## Deliverable

When a sweep or tuning exercise completes, produce a short markdown
report:

```markdown
# HPO Report: <experiment name>

**Date:** <YYYY-MM-DD>
**Target:** <label and horizon>
**Metric:** <primary metric>
**Baseline:** <value>
**Best trial:** <value and delta vs. baseline>

## Search Space
Table of hyperparameters and ranges.

## Top Trials
Small table of the best 5 trials with their configs and metrics.

## Recommendations
- Adopt these settings? Yes/no, with reasoning.
- Next experiment to queue in `model-runtime-manager`.
- Anything that looks suspicious and should be checked by
  `data-quality-analysis`.
```

## Handoffs

- Code changes required → `model-implementation`.
- Need to actually launch long-running training → `model-runtime-manager`.
- Found a suspicious data pattern → `data-quality-analysis`.
- Found a promising new direction that needs literature support →
  `research`.
