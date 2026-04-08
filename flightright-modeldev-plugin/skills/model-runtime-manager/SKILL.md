---
name: model-runtime-manager
description: >
  This skill should be used when the user asks to "run a training
  job", "launch training", "schedule this experiment", "queue these
  runs", "monitor my training", "manage training resources", "run
  these experiments in parallel", "kick off an overnight sweep",
  "kill that run", or otherwise wants to start, schedule, monitor,
  or manage one or more machine learning training sessions for the
  flight delay prediction project. Use this skill to actually launch
  and supervise training runs. It does NOT design experiments or
  pick hyperparameters (use `ml-hyperparameter-optimization`) and
  does NOT modify pipeline code (use `model-implementation`).
metadata:
  version: "0.1.0"
  domain: "flight-delay-ml"
---

# Model Runtime Manager

Actually start, schedule, monitor, and tear down machine learning
training sessions for the flight delay prediction pipeline. Manage
memory, compute, and concurrency so that the user can run one job
or a queue of many without stepping on themselves.

## When To Use This Skill

- The user wants to launch a single training run with a specific
  configuration.
- The user wants to queue a batch of runs — for example, an HPO
  sweep designed by `ml-hyperparameter-optimization`.
- A training run is misbehaving (OOM, hang, runaway GPU usage) and
  needs triage.
- The user wants a status report across active and queued runs.
- The user wants to schedule a training job for later (e.g.,
  overnight).

## When NOT To Use This Skill

- Designing the experiment or choosing hyperparameters →
  `ml-hyperparameter-optimization`.
- Changing model or feature code → `model-implementation`.
- Debugging whether the input data is correct →
  `data-quality-analysis`.
- Doing statistical feature analysis →
  `correlations-and-interactions`.

## Capabilities

This skill is authorized to:

- Launch training jobs via the project's existing entry points
  (Python scripts, Makefile targets, or a project CLI) using Bash.
- Write and manage a simple local run queue (JSON or a directory of
  config files) under `runs/queue/` in the workspace.
- Monitor running jobs via process inspection, log tailing, and
  (if available) GPU utilization commands like `nvidia-smi`.
- Stop / kill runaway jobs on the user's explicit confirmation.
- Schedule a future launch using the `schedule` skill, when the
  request is deferred rather than immediate.

This skill does **not**:

- Change any code under `src/`.
- Commit, push, or publish any artifacts.
- Make production model promotions.

## Before Launching Anything

Confirm all of the following, and ask the user if any are unclear:

1. **Entry point.** What command actually starts the training run?
   Prefer a project-provided CLI or Makefile target over ad-hoc
   scripts.
2. **Config source.** Where do hyperparameters come from — an
   inline override, a config file in `configs/`, or an HPO report?
3. **Data slice.** Which training window and which label.
4. **Compute budget.** CPU-only? Single GPU? How much memory is
   safe to request? What is the expected wall-clock time?
5. **Output location.** Where do logs, checkpoints, metrics, and
   final artifacts land? Reuse the project's convention.
6. **Concurrency.** Is this run allowed to start while other runs
   are active? What is the max concurrent job count the host can
   safely sustain?

Never improvise these. If the project has conventions, follow them.
If it does not, set sensible defaults and document them in the run
log.

## Launching a Single Run

1. Resolve the full command and working directory.
2. Create a run id: `<YYYYMMDD-HHMMSS>-<shortslug>`.
3. Create a run directory `runs/<run_id>/` containing the config,
   a copy of the resolved command, and empty `stdout.log` and
   `stderr.log` files.
4. Start the process in the background via Bash, redirecting
   stdout and stderr into the run directory. Capture and record
   the PID.
5. Append a row to `runs/registry.jsonl` with: run_id, pid, start
   time, command, config path, status = "running".
6. Report the run id and PID to the user.

Do not block on the run — training runs are long. Return control
to the user and let them ask for status.

## Queueing Multiple Runs

When the user supplies a batch (e.g., an HPO report) or asks to
"run these five configs":

1. Turn each config into a row in `runs/queue/queue.jsonl` with a
   pending status.
2. Honor the concurrency limit. Launch up to N jobs at a time; as
   each finishes, pull the next pending row from the queue.
3. Poll at a reasonable interval — do not tight-loop. Use
   file-modified timestamps and process existence checks, not
   constant log reads.
4. On failure, mark the run as failed in the registry, capture the
   last ~200 lines of stderr into the run directory, and continue
   the queue unless the user said "abort on first failure."

## Monitoring

When asked for status, produce a compact table:

```markdown
| Run id | Status | PID | Started | Elapsed | Last metric | MLflow | Notes |
|--------|--------|-----|---------|---------|-------------|--------|-------|
| 20260408-143210-gbdt-a | running | 48211 | 14:32 | 1h12m | auc_eval_ge15=0.79 | dep_train_WN_v11 | fold 2/3 |
| 20260408-150044-gbdt-b | queued  | -     | -     | -     | -           | -      | waiting for slot |
| 20260408-121005-gbdt-c | done    | -     | 12:10 | 2h03m | multibin_logloss=1.52 | dep_train_WN_v11 | best so far |
```

Pull the "last metric" line from MLflow (see below) rather than
reinterpreting the whole stdout log — MLflow is the source of
truth for per-threshold and multibin metrics once a trainer
reaches the logging step. Only fall back to tailing
`stdout.log` if MLflow has no active run yet (e.g., the trainer
crashed before `init_mlflow`).

## MLflow: The Source of Truth for Run Results

The trainers in `src/training/` each log one MLflow run per
invocation to a local file store at
`../flightrightdata/mlruns/`. Experiments by script:

- `departure-delay`         — `train_dep_bins_ordinal_catboost.py`
- `arrival-delay`           — `train_arr_bins_ordinal_catboost.py`
- `cancellation`            — `train_cancellation_catboost.py`
- `hyperparam-optimization` — `hyperparam_optimize.py` (writes a
  parent run per invocation plus one nested child run per Optuna
  trial)

Each run records the training config as params, per-threshold
AUCs and multibin logloss/accuracy as metrics, and the deploy
bundle + registry.json + metrics JSON as artifacts. Tags include
`model_type`, `config_path`, `fast_mode`, and `git_commit`.

The MLflow run name defaults to the training config file stem
(e.g. `dep_train_WN_100_v10`), which you should record in
`runs/registry.jsonl` alongside the local run_id so a teammate
can click through from your status table to the full run detail.

### Report Results Back Through MLflow

When a training run finishes, read its final state from MLflow
rather than from stdout. Example:

```python
import mlflow
from pathlib import Path
mlflow.set_tracking_uri(
    Path("../flightrightdata/mlruns").resolve().as_uri()
)
client = mlflow.MlflowClient()
exp = client.get_experiment_by_name("departure-delay")
run = client.search_runs(
    [exp.experiment_id],
    filter_string="tags.config_path LIKE '%dep_train_WN_100_v10.json'",
    order_by=["start_time DESC"],
    max_results=1,
)[0]
print(run.info.run_name, run.info.status)
for k in sorted(run.data.metrics):
    print(f"  {k}: {run.data.metrics[k]:.4f}")
```

When reporting a finished run to the user, always include:
- The MLflow run name and run_id (first 12 chars is enough)
- The key metrics for that model family:
  - dep/arr: `auc_eval_unbal_ge15/30/45/60`,
    `multibin_log_loss`, `multibin_accuracy`
  - cancellation: `auc`, `average_precision`, `log_loss`, `brier`
  - HPO parent: `baseline_auc_ge*`, `per_thr_opt_auc_ge*`,
    `shared_opt_auc_ge*`
- The path to the deploy bundle (the `.joblib` logged as an
  artifact on the run)
- A one-line diff vs. the previous run on the same experiment
  and same config path, if one exists — this makes regressions
  obvious instead of hiding them in absolute numbers.

### Launching a Sweep Batch

When queueing an HPO-designed sweep from
`ml-hyperparameter-optimization`, the cleanest launch is usually
a single invocation of `hyperparam_optimize.py` with a trial
budget (e.g. `--n-trials 100`) rather than launching N separate
trainer jobs. That way MLflow records all trials under one
parent run and the HPO skill can query them as a single study.
Only split into per-config trainer jobs when the user
specifically needs separate full-quality training runs (not
sweep trials) for a handful of candidate configs.

### Browsing

To hand the user a clickable view:

```
.venv/bin/mlflow ui --backend-store-uri ../flightrightdata/mlruns
```

Then open http://127.0.0.1:5000. Do NOT prefix the path with
`file://` — a relative URI like `file://../path` is malformed
and MLflow will crash trying to create a directory at the
filesystem root.

## Resource Management

- Before launching, check current load: CPU, memory, and GPU. Use
  `ps`, `free`, and `nvidia-smi` where available.
- Refuse to launch a new run that would push memory above a safe
  threshold (e.g., 85% of total) unless the user overrides.
- If `nvidia-smi` shows a stuck process consuming GPU memory with
  no running job in the registry, surface it to the user before
  launching anything new.
- When killing a job, send SIGTERM first, wait briefly, then
  SIGKILL only if needed. Record the reason in the registry.

## Scheduling Future Runs

When the user asks to "launch this at 2am" or "run this after the
current job finishes":

- For clock-time scheduling, defer to the `schedule` skill and
  create a scheduled task whose prompt re-invokes this skill with
  the prepared run config.
- For dependency scheduling ("after run X finishes"), add the new
  run to the queue with a `depends_on: <run_id>` field and only
  pull it once the dependency is marked done.

## Safety and Guardrails

- Never launch a run that writes to a production artifact path.
  Training runs go under `runs/`.
- Never delete run directories or the registry file without
  explicit user confirmation.
- Never kill a process the user did not ask to kill.
- Never run with `--force` style flags unless the user explicitly
  asked.
- If a run hangs, do not automatically restart it. Surface the
  issue and wait for instructions.

## Handoffs

- The run produced interesting results but needs deeper analysis →
  `correlations-and-interactions` or `ml-hyperparameter-optimization`.
- The run failed because of data issues → `data-quality-analysis`.
- The run failed because of a code bug → `model-implementation`.
