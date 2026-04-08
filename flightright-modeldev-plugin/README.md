# flightright-modeldev

A Cowork plugin for the data scientist building machine learning
models that predict departure and arrival delay probabilities for
flights. It bundles seven specialized skills that cover the full
iteration loop — from literature research to feature sourcing, data
quality, statistical validation, code implementation, hyperparameter
optimization, and actually running training jobs.

## What it does

The plugin turns Claude into a project-aware ML collaborator for the
flight delay prediction pipeline. Each skill owns one stage of the
workflow and hands off cleanly to the others, so investigations,
experiments, and code changes all produce artifacts that the next
stage can pick up.

## Skills

| Skill | Purpose |
|-------|---------|
| `research` | Investigate candidate features, modeling strategies, and papers. Produces structured markdown reports with sourced findings and recommended next actions. Does not run statistics on real data. |
| `correlations-and-interactions` | Run statistical analyses (Spearman, mutual information, Cramér's V, quick GBDT deltas, interaction tests) on the real dataset to measure whether a candidate feature carries signal. Does not run full training cycles. |
| `data-quality-analysis` | Verify that computed features are complete, accurate, and semantically faithful to their names. Writes reusable validation tests and hunts for new ones. |
| `ml-hyperparameter-optimization` | Act as the ML expert. Choose targets, metrics, losses, splits, and hyperparameter search spaces. Design sweeps. Report best configurations. |
| `model-implementation` | Translate recommended changes into real edits in `src/`, run a small-slice smoke test, and hand off to `data-quality-analysis` for verification. |
| `feature-transferability` | Map BTS fields to Aerodatabox or trusted free alternatives, and — when an API key is available — run empirical parity tests between the two sources. |
| `model-runtime-manager` | Launch, queue, monitor, and manage training jobs. Handles concurrency, resources, scheduling, and graceful shutdown. |

## How the skills fit together

A typical loop looks like this:

1. `research` proposes a candidate feature backed by literature.
2. `feature-transferability` confirms whether it can be served from
   Aerodatabox or a trusted free source.
3. `correlations-and-interactions` measures whether it carries
   signal against the real data.
4. `model-implementation` wires it into `src/`, runs a smoke test,
   and asks `data-quality-analysis` to sanity-check the values.
5. `ml-hyperparameter-optimization` designs a sweep to retune the
   model with the new feature in place.
6. `model-runtime-manager` actually launches the sweep, monitors the
   runs, and reports results back.

Each skill explicitly names which other skill to hand off to for
follow-up work.

## Assumptions about the workspace

- Pipeline source code lives at `src/` at the workspace root.
- Tests live at `tests/` at the workspace root.
- Prior ad-hoc analyses live at `exploration/` — the
  `correlations-and-interactions` skill reuses and extends them.

If any of these assumptions need to change for your repo layout,
tell Claude when you invoke a skill and it will adapt.

## Optional: Aerodatabox API key

Only the `feature-transferability` skill needs an API key, and only
when running a parity test. Provide the key via an environment
variable (recommended) such as `AERODATABOX_API_KEY`. The skill will
never write the key to disk or include it in reports.

## Setup

This plugin is shipped as an unzipped directory inside the
`flightright_modeldev` repo and enabled automatically via the
committed `.claude/settings.json`, which registers the repo root as
a local marketplace (`flightright-modeldev-local`) and marks this
plugin as enabled. Teammates who clone the repo and open it in
Claude Code will see the `flightright-modeldev:*` skills with no
manual install step — but note that the `extraKnownMarketplaces` /
`enabledPlugins` auto-install flow requires a reasonably recent
Claude Code CLI. If skills don't appear, run `claude update` first.

Once enabled, invoke any skill by describing the task in plain
language — for example "research rotation-based features for
departure delay," "run correlations on origin congestion," or
"launch the HPO sweep from yesterday's report."

To edit a skill, open the corresponding `skills/<name>/SKILL.md`
file directly and commit the change. The next Claude Code session
will pick up the new content.

## Version

`0.1.0` — initial release.
