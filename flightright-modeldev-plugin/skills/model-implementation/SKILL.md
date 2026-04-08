---
name: model-implementation
description: >
  This skill should be used when the user asks to "implement this
  feature", "add this feature to the pipeline", "apply the
  recommendations from the report", "wire this into src", "update the
  training code", "implement the changes proposed in this document",
  or otherwise wants recommended feature changes translated into real
  code edits in the `src/` pipeline. Use this skill to modify code
  under `src/`, then run a smoke test on a small data slice, then
  invoke `data-quality-analysis` to verify correctness. This skill
  does NOT design features (use `research` /
  `correlations-and-interactions`) and does NOT run full training
  campaigns (use `model-runtime-manager`).
metadata:
  version: "0.1.0"
  domain: "flight-delay-ml"
---

# Model Implementation

Translate recommended feature or model changes — from the user
directly, a research report, or an HPO report — into actual code
edits in the `src/` tree of the flight delay prediction pipeline.
Every implementation ends with a small-slice smoke test and a
handoff to `data-quality-analysis`.

## When To Use This Skill

- The user says "go implement X" where X is already specified (name,
  source, transformation, join keys).
- A report from `research`, `correlations-and-interactions`, or
  `ml-hyperparameter-optimization` lists concrete "Recommended next
  actions" that require code changes.
- A bug in the pipeline has been diagnosed and the fix is known.

## When NOT To Use This Skill

- The feature is still being debated → `research` or
  `correlations-and-interactions`.
- The goal is a tuning sweep rather than a code change →
  `ml-hyperparameter-optimization`.
- You're trying to figure out whether the data is correct →
  `data-quality-analysis`.
- You're scheduling a long-running training job →
  `model-runtime-manager`.

## Assumptions

- The pipeline source lives at `src/` at the workspace root.
- Tests live at `tests/` at the workspace root.
- The project uses a standard Python packaging layout with a
  reproducible dev environment.

If any of these assumptions turn out to be wrong, stop and confirm
with the user before editing files.

## Implementation Workflow

Follow every step in order.

### 1. Restate the change

Write a one-paragraph summary of the change to implement: the
feature or fix name, the source of the requirement (which document,
which report), and the acceptance criteria. This becomes the commit
message later.

### 2. Read before you write

- Open the relevant modules under `src/`. Understand how existing
  features are defined, named, joined, and tested.
- Match existing patterns. Do not introduce a new abstraction for a
  single feature when the existing template fits.
- Identify the blast radius: which modules, configs, feature lists,
  schemas, and tests will need to change.

### 3. Plan the edits

List the files to touch and the nature of each change. If the plan
spans more than ~6 files or touches cross-cutting concerns (the
schema, the training entry point, and the inference path), summarize
the plan to the user and confirm before editing.

### 4. Make the edits

- Edit code in `src/` directly using the file tools.
- Keep changes minimal and localized. Avoid drive-by refactors.
- Add the new feature name to every registry / config / schema /
  feature list that gates what the model sees. Missing one of these
  is the single most common defect in this workflow — double-check.
- Keep formatting consistent with existing code. Do not run
  project-wide formatters unless explicitly asked.
- Add or update unit tests next to the code change. A new feature
  without a unit test is not done.

### 5. Smoke test on a small slice

Run the pipeline on the smallest viable data slice to prove the
change is wired up. Goal: confirm that the new feature column
appears in the training matrix with non-trivial values, that the
training loop runs for at least one iteration, and that no
exception is raised. This is **not** a model evaluation; it is a
wiring check.

Prefer the project's existing "tiny run" / "dev sample" entry point
if one exists. If it does not, create a clearly named throwaway
script under `scripts/smoke/` that loads a few days of data, runs
the extraction, and prints shape and basic stats.

### 6. Verify correctness

Immediately after the smoke test, invoke the
`data-quality-analysis` skill to check the freshly added or
modified feature. Do not declare the implementation done until
data-quality-analysis has passed or has produced an explicit list
of accepted caveats.

### 7. Summarize the change

Write a short changelog entry describing:

- What changed and why.
- Files touched.
- Smoke test result.
- Data quality verdict.
- Any follow-up tasks (e.g., "rerun HPO in
  `ml-hyperparameter-optimization`").

## Code Quality Rules

- Never commit or push code on behalf of the user.
- Never leave a feature half-wired: if it's added to extraction it
  must also appear in the model's feature list and in any
  serialization artifact (e.g., feature name manifest).
- Never silently change defaults for existing features.
- Never delete a feature without confirmation from the user, even if
  a report recommended it.
- Prefer pure functions for new transformations so they are easy to
  unit test.
- Guard against division by zero, empty groups, and timezone
  confusion — all three are recurring bug sources in flight data.

## Failure Handling

If the smoke test fails:

1. Capture the full error and the minimum reproducer.
2. Decide whether the bug is in the new code, in an existing
   assumption that the new code exposed, or in the data.
3. If the data is suspect, hand off to `data-quality-analysis`
   before attempting another fix.
4. If the code is the problem, fix it and re-run the smoke test.
   Do not claim success until the smoke test actually passes.

## Guardrails

- Never skip the smoke test, even for "trivial" changes.
- Never skip the data-quality-analysis handoff.
- Never edit code outside `src/`, `tests/`, `scripts/smoke/`, or
  the specific config files the change requires without first
  telling the user.
- Never store credentials or API keys in the repo.
