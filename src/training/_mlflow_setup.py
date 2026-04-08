"""MLflow tracking setup shared by the flight-delay trainers and HPO sweeper.

Trainers call init_mlflow("experiment-name") once at the start of main(),
then use mlflow.* APIs directly (start_run, log_params, log_metric,
log_artifact, end_run) to record each training invocation as an MLflow run.

Runs are stored locally in ../flightrightdata/mlruns/ so they sit alongside
the existing model artifacts and never leave the machine. Browse with:

    mlflow ui --backend-store-uri file://../flightrightdata/mlruns

This helper lives next to the trainers (rather than in a src-level package)
so that the trainers can import it with a bare `from _mlflow_setup import ...`
when run directly as scripts (Python puts the script's directory on sys.path).
"""
from __future__ import annotations

import subprocess
import warnings
from pathlib import Path
from typing import Any, Mapping

import mlflow

_HELPER_DIR = Path(__file__).resolve().parent
REPO_ROOT = _HELPER_DIR.parents[1]
TRACKING_DIR = (REPO_ROOT.parent / "flightrightdata" / "mlruns").resolve()


def init_mlflow(experiment: str) -> None:
    """Point MLflow at ../flightrightdata/mlruns/ and select the experiment.

    We use the filesystem tracking backend, which MLflow has deprecated in
    favor of a database (e.g. SQLite) backend. Migration path when this
    eventually stops working: set tracking_uri to
    f"sqlite:///{TRACKING_DIR}/mlflow.db" and pass artifact_location to
    set_experiment. For now, file store is still fully functional; the
    deprecation warning is silenced to keep trainer output clean.
    """
    TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"The filesystem tracking backend .* is deprecated.*",
            category=FutureWarning,
        )
        mlflow.set_tracking_uri(TRACKING_DIR.as_uri())
        mlflow.set_experiment(experiment)


def git_commit() -> str | None:
    """Return the current git HEAD SHA, or None if not in a git repo."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None


def loggable_params(cfg: Mapping[str, Any]) -> dict[str, str]:
    """Flatten a config dict to a {str: str} mapping suitable for mlflow.log_params.

    - Nested dicts become dotted keys.
    - Lists/tuples of scalars become comma-joined strings.
    - Non-scalar values, overly long strings, and keys/values that exceed
      MLflow's length limits are silently dropped.
    """
    out: dict[str, str] = {}

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, Mapping):
            for k, v in value.items():
                key = f"{prefix}.{k}" if prefix else str(k)
                walk(key, v)
        elif isinstance(value, (list, tuple)):
            if all(isinstance(x, (str, int, float, bool)) for x in value):
                joined = ",".join(str(x) for x in value)
                if prefix and len(prefix) <= 250 and len(joined) <= 500:
                    out[prefix] = joined
        elif isinstance(value, (str, int, float, bool)) or value is None:
            s = "" if value is None else str(value)
            if prefix and len(prefix) <= 250 and len(s) <= 500:
                out[prefix] = s

    walk("", cfg)
    return out
