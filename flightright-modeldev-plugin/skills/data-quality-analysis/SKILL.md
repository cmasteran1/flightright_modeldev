---
name: data-quality-analysis
description: >
  This skill should be used when the user asks to "check data quality",
  "validate a feature", "does this feature actually mean what it says",
  "write a validation test for X", "are these values sane", "audit a
  column", "why does feature Y look wrong", or otherwise wants to
  ensure that computed features are complete, accurate, and semantically
  match their name and intent. Use this skill for verifying data
  correctness and writing reusable validation tests — NOT for measuring
  statistical signal (use `correlations-and-interactions`) and NOT for
  modifying pipeline code (use `model-implementation`).
metadata:
  version: "0.1.0"
  domain: "flight-delay-ml"
---

# Data Quality Analysis

Ensure that features computed by the flight delay prediction pipeline
are exactly what the user expects: complete, accurate, and
semantically faithful to their field names. Write reusable validation
tests and hunt for new ones as the dataset evolves.

## When To Use This Skill

- A new feature has just been added and needs a sanity pass before
  anyone trusts it.
- A feature "works in training" but the user suspects its values are
  not what the name implies (e.g., `wheels_off_local_hour` is
  actually UTC; `origin_airport_congestion` counts all ops, not just
  concurrent departures).
- Null / missing rates jumped after a pipeline change.
- Monotonic invariants are suspicious (a timestamp went backwards, a
  duration is negative, a fraction exceeds 1).
- The user wants to add unit tests or validation tests for the
  feature extraction layer.

## When NOT To Use This Skill

- Measuring statistical relevance / feature importance →
  `correlations-and-interactions`.
- Brainstorming new features or reading literature → `research`.
- Changing pipeline code permanently → `model-implementation`.
- Sourcing features from Aerodatabox vs. BTS →
  `feature-transferability`.

## Core Principle

"Present" is not the same as "correct." A column can be 100% populated
and still be wrong. Every quality check must answer two questions:

1. Is the value **there**?
2. Is the value **the thing the name says it is**?

Both questions are required. If you only answer the first, stop and
answer the second.

## Validation Layers

Run checks in this order of increasing strictness:

1. **Schema layer.** Column exists, dtype matches contract, null rate
   is within expected bounds.
2. **Range layer.** Values fall within physically possible bounds
   (latitude in [-90, 90], delay minutes rarely exceed some cap,
   probabilities in [0, 1], hours in [0, 23], etc.).
3. **Invariant layer.** Monotonic or conservation rules hold
   (`actual_departure >= scheduled_departure - max_early`,
   `wheels_on > wheels_off`, `arrival_delay = wheels_on -
   scheduled_arrival` up to timezone handling).
4. **Semantic layer.** The value means what the name says. This is
   the hardest layer and requires cross-checking with a trusted
   reference. Examples:
   - Spot-check a sample of rows against the raw source (Aerodatabox
     response or BTS record).
   - Re-derive the value from primary fields and compare.
   - Compare against a known external fact (e.g., "JFK has how many
     gates? Does our congestion denominator reflect that?").
5. **Distribution layer.** Distribution has not shifted unexpectedly
   vs. a reference slice.

## Writing Reusable Tests

Store validation tests in `tests/data_quality/`. Use the project's
existing test framework (pytest unless told otherwise). Each test
should:

- Name the feature and the invariant in the test name.
- Load the smallest dataset slice that can exercise the invariant.
- Assert the invariant with a clear failure message that points at
  the offending row and the expected vs. observed values.
- Be idempotent and deterministic.

Group related tests into parameterized suites so adding a new
feature means adding a row to a table, not a new file.

Template:

```python
import pytest

EXPECTED_RANGES = {
    "scheduled_departure_hour_local": (0, 23),
    "taxi_out_minutes": (0, 240),
    "arrival_delay_minutes": (-60, 720),
}

@pytest.mark.parametrize("feature,bounds", EXPECTED_RANGES.items())
def test_feature_in_range(feature, bounds, sample_df):
    lo, hi = bounds
    bad = sample_df[(sample_df[feature] < lo) | (sample_df[feature] > hi)]
    assert bad.empty, (
        f"{feature} has {len(bad)} rows outside [{lo}, {hi}]; "
        f"first offender: {bad.iloc[0].to_dict()}"
    )
```

## Hunting For New Tests

Periodically ask: what invariants are we *not* testing that a
reasonable reviewer would expect to hold? Candidates to look for:

- Time-zone consistency across timestamp columns.
- Aggregations that should sum to known totals.
- Features that should be 0 or null when a flight was cancelled.
- Features that should be symmetric (origin vs. destination pairs).
- Features derived from joins that should never orphan a row.

Turn each new idea into a concrete parameterized test.

## Investigation Workflow

When the user says "this feature looks wrong":

1. Reproduce the symptom on a small slice.
2. Locate the code path that computes the feature.
3. Re-derive the value manually for 3-5 rows.
4. Classify the defect: missing data, wrong source, wrong formula,
   wrong timezone, wrong join, wrong name.
5. Propose the fix, but do not apply code changes to `src/` here —
   hand off to `model-implementation`.
6. Write or update the validation test that would have caught this
   defect. That test must fail on the current (buggy) data and pass
   once the fix lands.

## Report Format

Write findings to `data_quality/reports/<YYYY-MM-DD>-<feature>.md`:

```markdown
# Data Quality: <feature>

**Date:** <YYYY-MM-DD>
**Verdict:** Pass | Conditional pass | Fail

## What Was Checked
Bullet list of layers run: schema, range, invariant, semantic,
distribution.

## Findings
Ordered by severity. Each finding: description, affected rows,
example row, suspected cause.

## New Tests Added
Path + brief description.

## Recommended Fix
What should change and who should do it (`model-implementation`,
`feature-transferability`, or outside the plugin).
```

## Guardrails

- Never claim a feature is "fine" based on null rate alone.
- Never silently drop bad rows to make a test pass.
- Never weaken an assertion to paper over a real defect. If a bound
  really needs to be widened, document *why* in the test.
- When you are unsure whether a value is correct, say so — do not
  guess.
