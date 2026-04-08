flightright_modeldev
=====================

A research and modeling repo that trains machine-learning models to
predict commercial-flight departure delays, arrival delays, and
cancellations from BTS historical data. The trained models are
intended to be consumed by downstream software so that travelers
get more accurate, probabilistic delay estimates than a generic
on-time rate.

This is a pipeline repo, not a packaged library. It is organized
around a few long-lived ingest/feature/training stages driven by
versioned JSON blueprints and training configs under data/.


Current scope
-------------

Airlines covered: WN, UA, AA, DL (the four large US domestic
carriers, ranked by BTS share).

Models (all calibrated CatBoost classifiers, per airline unless
otherwise noted):

  - Departure delay  — one ordinal model per airline, with binary
    heads at thresholds 15 / 30 / 45 / 60 minutes, isotonically
    calibrated and combined into a delay-severity distribution.
    Trainer: src/training/train_dep_bins_ordinal_catboost.py

  - Arrival delay    — same shape as departure, one per airline.
    Trainer: src/training/train_arr_bins_ordinal_catboost.py

  - Cancellation     — single cross-airline binary model producing
    a calibrated cancellation probability + recommended decision
    threshold.
    Trainer: src/training/train_cancellation_catboost.py

Hyperparameter sweeps live in src/training/hyperparam_optimize.py
and run over the same feature set via CatBoost + Bayesian search.

The feature set evolves continuously. Each iteration bumps a
version suffix in the blueprint and training-config filenames
(see the data/ layout below), and the authoritative spec for the
current version lives in exploration/ as a markdown design doc.
New models are designed, implemented, and iterated on primarily
through the flightright-modeldev Claude Code plugin (see below).


End-to-end pipeline
-------------------

All raw inputs and generated artifacts live OUTSIDE this repo, in
the sibling directory ../flightrightdata/. This repo only carries
source code, configs, blueprints, reports, and exploration
notebooks.

Stages, in order:

  1. Cache external signals (weather, strike calendar)
       src/fetch_prune/collect_weather_cache.py
       src/fetch_prune/collect_strike_cache.py
     Writes cached parquets into ../flightrightdata/weather_cache*
     and ../flightrightdata/strike_cache/.

  2. Prepare the flight-level dataset from BTS history
       src/fetch_prune/prepare_dataset.py  blueprint_*.json
     Reads raw BTS parquets from
       ../flightrightdata/data/raw_bts/Year=*/Month=*/
     and writes enriched, filtered flight records into
       ../flightrightdata/data/intermediate/.

  3. Compute features (rolling history, weather joins, congestion,
     hub spillover, holiday flags, strike severity, etc.)
       src/fetch_prune/features_dep.py          blueprint_dep_*.json
       src/fetch_prune/features_arr.py          blueprint_arr_*.json
       src/fetch_prune/prepare_cancel_dataset.py blueprint_cancel_*.json
     Writes balanced (per-threshold) and unbalanced feature parquets
     to ../flightrightdata/data/processed/.

  4. Train the models
       src/training/train_dep_bins_ordinal_catboost.py   dep_train_*.json
       src/training/train_arr_bins_ordinal_catboost.py   arr_train_*.json
       src/training/train_cancellation_catboost.py       cancel_train_*.json
     Writes per-threshold .cbm models, isotonic calibrators (.joblib),
     a registry.json, and a deployable bundle into
       ../flightrightdata/data/models/.

  5. Validate feature health + bundle integrity
       src/health/feature_report.py  <features.parquet>
       src/health/health_checks.py
     Standalone scripts that check schema, null rates, bounds, and
     bundle structure. Tests live at src/health/test_health.py.

For a fast end-to-end sanity check over one airline, use the
committed Claude Code slash command /smoke-test (see
.claude/commands/smoke-test.md), or run the current smoke
blueprints directly from data/ (look for filenames with a
_smoke suffix).


Repository layout
-----------------

  src/
    fetch_prune/   BTS ingest + feature engineering + strike/weather
                   caches. Entry points for stages 1–3 above.
    training/     CatBoost trainers (dep, arr, cancel) + HPO sweeper.
    health/       Feature and model-bundle validation.
    analysis/     Ad-hoc cross-feature analyses (e.g. strike correlations).

  data/
    blueprint_{dep,arr}_{WN,UA,AA,DL}_<ver>.json
                   Feature pipeline specs, one per airline per direction.
    blueprint_cancel_<ver>.json
                   Cross-airline cancellation pipeline spec.
    {dep,arr}_train_{WN,UA,AA,DL}_100_<ver>.json
                   Per-airline training configs (features list, splits,
                   CatBoost params).
    cancel_train_<ver>.json
                   Cancellation training config.
    *_smoke.json
                   Tiny-slice configs for the smoke test.

    New feature-set iterations bump <ver> and are added alongside
    the previous versions so prior runs stay reproducible.

  exploration/
    Numbered ad-hoc investigation scripts (univariate summaries,
    feature importance, cascading-delay risk, weather EDA, etc.),
    plus a per-version feature_spec markdown document describing
    the feature families included in that iteration of the model.
    utils.py holds shared plotting/stat helpers.

  docs/
    strike_feature_guide.txt   Notes on the strike-cache feature family.

  flightright-modeldev-plugin/
    Claude Code plugin with 7 data-science skills (research,
    correlations-and-interactions, data-quality-analysis,
    ml-hyperparameter-optimization, model-implementation,
    feature-transferability, model-runtime-manager). Auto-enabled
    via .claude/settings.json — teammates who clone the repo get
    these skills in Claude Code with no setup. See the plugin's
    own README.md for details on each skill.

  .claude/
    settings.json        Committed — registers the plugin marketplace
                         and enables the flightright-modeldev plugin.
    settings.local.json  Personal permissions (gitignored).
    commands/
      smoke-test.md      /smoke-test slash command for fast e2e check.

  requirements.txt       Python dependencies.
  pytest.ini             Test configuration.


Where the data lives
--------------------

This repo holds only code, configs, and reports. All large data
artifacts live in a sibling directory:

  ../flightrightdata/
    data/raw_bts/          Raw BTS flight history, partitioned by
                           Year=YYYY/Month=MM.
    data/intermediate/     Enriched / filtered flight records from
                           stage 2 (prepare_dataset.py).
    data/processed/        Per-threshold balanced and unbalanced
                           feature parquets from stage 3.
    data/models/           Trained model bundles (.cbm + .joblib +
                           registry.json) from stage 4.
    weather_cache/         Daily Open-Meteo cache.
    weather_cache_hourly/  Hourly Open-Meteo cache.
    strike_cache/          US aviation labor-action calendar.

Scripts that read or write these paths resolve them through a
helper (_as_data_path) so that the repo root stays portable.


Getting started
---------------

Requirements: Python 3.10+ (developed and tested on 3.12). The
repo ships with a local .venv/ for Mac/Linux.

  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt

Key runtime dependencies: numpy, pandas, pyarrow, scikit-learn,
catboost>=1.2.5, joblib, requests. Apple Silicon is supported
natively via catboost's ARM build. CatBoost GPU training is
optional — set task_type="GPU" in a training config if desired.

To run the smoke test (fastest way to verify nothing is broken
end-to-end), open the repo in Claude Code and run /smoke-test.
The slash command drives the current smoke blueprint, training
config, and feature-health report in sequence. See
.claude/commands/smoke-test.md for the exact steps it runs if
you'd rather invoke them by hand.


Experiment tracking
-------------------

The trainers and the HPO sweeper log every run to a local MLflow
store at ../flightrightdata/mlruns/. Each training invocation
becomes one MLflow run that records:

  - Parameters: the full training config (flattened to dotted
    keys) plus tags for model type, config path, git commit, and
    FAST_TRAIN mode
  - Metrics:     per-threshold AUCs, multibin log-loss/accuracy,
    row counts, positive rates, and Youden thresholds
  - Artifacts:   metrics JSON, resolved feature list, prediction
    samples, and the deployable bundle.joblib

hyperparam_optimize.py additionally writes one NESTED MLflow run
per Optuna trial, so a sweep produces a sortable/filterable
trial table in the UI alongside the final baseline-vs-optimized
comparison on the parent run.

MLflow experiments used:

  departure-delay          — train_dep_bins_ordinal_catboost.py
  arrival-delay            — train_arr_bins_ordinal_catboost.py
  cancellation             — train_cancellation_catboost.py
  hyperparam-optimization  — hyperparam_optimize.py

To browse runs:

  .venv/bin/mlflow ui --backend-store-uri \
      file://../flightrightdata/mlruns

then open http://127.0.0.1:5000 in a browser.

Larger per-airline artifacts still live in
../flightrightdata/data/models/<airline>_<version>/ and are
referenced from MLflow as logged artifacts. Ad-hoc analyses
outside the training pipeline live in exploration/ as numbered
Python scripts with co-located markdown reports, and are NOT
logged to MLflow. Feature health is checked offline via
src/health/.

CatBoost writes per-run log directories named catboost_info/
wherever the trainer is launched from. These are gitignored at
any depth and can be deleted safely.
