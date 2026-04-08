---
name: correlations-and-interactions
description: >
  This skill should be used when the user asks to "run correlations",
  "check feature importance", "see if this feature matters",
  "test interactions between features", "measure statistical relevance",
  "compute mutual information", "run SHAP on a feature", or otherwise
  wants empirical validation of whether a candidate feature carries
  signal for departure or arrival delay prediction. Use this skill for
  statistical analysis on the real dataset — NOT for literature
  research (use `research`) and NOT for running full training cycles
  (use `ml-hyperparameter-optimization` or `model-runtime-manager`).
metadata:
  version: "0.1.0"
  domain: "flight-delay-ml"
---

# Correlations and Interactions

Empirically measure the statistical importance of candidate features
for the flight delay prediction model using the real dataset. Produce
actionable insights so the user can decide whether a feature is worth
implementing in the pipeline.

## When To Use This Skill

- A feature has been proposed (by the user or by a research report)
  and needs empirical validation before implementation.
- The user wants to quantify how strong a relationship is between a
  feature and the delay label.
- The user wants to detect interaction effects between two or more
  features.
- The user wants to prioritize a backlog of candidate features by
  statistical signal strength.

## When NOT To Use This Skill

- Do not run end-to-end training cycles here — that is
  `ml-hyperparameter-optimization` and `model-runtime-manager`.
- Do not read papers or browse the web for ideas — that is `research`.
- Do not validate that the raw data values are correct, well-formed,
  or match their field names — that is `data-quality-analysis`.
- Do not modify `src/` pipeline code to permanently add a feature —
  that is `model-implementation`.

## Exploration Inventory

Before writing anything new, check the `exploration/` directory in
the workspace. It contains prior analyses and reusable techniques.
Reuse and extend those scripts wherever possible rather than
reinventing. If a similar analysis already exists, adapt it and note
the lineage in the report.

## Analysis Techniques

Pick the right tool for the feature type and the question. Common
techniques and when to use each:

- **Pearson correlation** — linear relationship, numeric feature vs.
  continuous delay minutes. Limited because delay distributions are
  heavy-tailed.
- **Spearman rank correlation** — monotonic relationship, robust to
  outliers. Usually a better first pass than Pearson for delay data.
- **Point-biserial / logistic univariate** — numeric feature vs.
  binary "delayed > threshold" label.
- **Chi-squared / Cramér's V** — categorical feature (airline, origin,
  aircraft type) vs. binary delay label.
- **Mutual information** — nonlinear dependence for any feature type.
  Good default when no functional form is assumed.
- **ANOVA F-test** — differences in mean delay across groups.
- **Grouped delay rate plots** — visualize mean or quantile delay per
  bucket of the candidate feature. Watch for monotonicity.
- **Partial dependence / ALE plots** from a quick tree model — captures
  nonlinear marginal effects without committing to a full training
  cycle.
- **SHAP on an interim model** — when a fast baseline already exists,
  pull per-feature SHAP values. Use only an existing or lightweight
  baseline; do not launch full training runs.
- **Interaction tests** — H-statistic, pairwise SHAP interaction
  values, or stratified correlations (correlation of A vs. label
  within buckets of B).

Default workflow for a single new feature:

1. Descriptive stats and missingness summary for the feature.
2. Univariate signal: Spearman for numeric, Cramér's V for
   categorical, plus mutual information against the label.
3. Grouped delay rate or mean-delay plot.
4. Quick lightweight tree model (e.g., LightGBM with default params
   on a sample) trained with and without the feature to measure
   delta in log-loss / MAE / quantile loss. This is a diagnostic,
   not a production training run.
5. Interaction check against the top three existing features.
6. Written verdict with recommended action.

## Implementation Guidance

- Work inside a notebook or scratch script under
  `exploration/<YYYY-MM-DD>-<feature-or-question>.{ipynb,py}`.
- Load data via the project's existing data-loading utilities where
  possible. If those do not exist, ask before reading raw files from
  unusual paths.
- Sample aggressively when iterating. Only run on the full dataset
  after the analysis plan is clear.
- Control for confounders where relevant. A feature that correlates
  with delay only because it correlates with carrier should be
  flagged, not celebrated.
- Split by time, not randomly, when approximating generalization.
  Delay patterns shift seasonally and after schedule updates.
- Never peek at the test split.

## Report Format

Produce a short markdown report at
`exploration/reports/<YYYY-MM-DD>-<feature>.md` with these sections:

```markdown
# Feature Signal: <feature name>

**Date:** <YYYY-MM-DD>
**Label:** <departure_delay | arrival_delay | delayed_15min | ...>
**Data slice:** <time range, rows, sampling strategy>

## Summary
One-paragraph verdict. Recommend one of: **Adopt**, **Explore
further**, **Reject**.

## Univariate Signal
Table of metrics (Spearman, Cramér's V, mutual info, univariate
log-loss delta) with interpretation.

## Grouped Behavior
Chart(s) of mean or quantile delay across feature buckets.

## Interaction Effects
Notable interactions with existing features.

## Confounders and Caveats
Things that could explain away the signal.

## Recommended Next Action
- Hand off to `feature-transferability` to confirm sourcing? Yes/no.
- Hand off to `model-implementation` to wire into the pipeline?
  Yes/no, with priority.
- Any follow-up analyses to queue up?
```

## Guardrails

- Never claim causation from correlation.
- Never report a single point estimate without noting the sample
  size and a basic uncertainty band.
- If the feature has more than ~30% missingness on the relevant
  slice, call it out before interpreting any signal.
- If two analyses disagree, report both and propose a tiebreaker —
  do not hide the disagreement.
- When in doubt, recommend `data-quality-analysis` before trusting
  the numbers.
