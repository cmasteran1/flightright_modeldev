---
name: feature-transferability
description: >
  This skill should be used when the user asks to "can we get this
  feature from Aerodatabox", "is this BTS feature available in our
  API", "how do we source feature X in production", "compare
  Aerodatabox to BTS for feature Y", "find a free alternative data
  source", "check if this field is transferable", or otherwise wants
  to determine how a feature present in the BTS dataset can be
  obtained at inference time from Aerodatabox or a trusted free
  alternative. Use this skill for sourcing and parity testing — NOT
  for deciding whether a feature is statistically useful (use
  `correlations-and-interactions`) and NOT for implementing the
  feature in the pipeline (use `model-implementation`).
metadata:
  version: "0.1.0"
  domain: "flight-delay-ml"
---

# Feature Transferability Assessment

For every feature the model uses or considers, determine how it can
be obtained at inference time from Aerodatabox or a trusted free
alternative, and — when an Aerodatabox API key is available —
empirically compare the values to the BTS reference values.

This skill closes the gap between "this feature helps on historical
BTS data" and "we can actually get this feature in production."

## When To Use This Skill

- A feature has been shown to carry signal against BTS labels and
  the user needs to know whether it can be served at prediction
  time.
- The user wants a mapping document that lists every BTS field and
  its production source.
- A research report flagged a candidate feature as "availability
  uncertain" — resolve the uncertainty here.
- The user wants to swap the current data source for a free
  alternative and needs a gap analysis.
- The user provides an Aerodatabox API key and wants to measure
  field-level parity between Aerodatabox and BTS for a sample.

## When NOT To Use This Skill

- Measuring statistical signal of a feature →
  `correlations-and-interactions`.
- Brainstorming new feature ideas → `research`.
- Checking whether existing computed values are internally consistent
  → `data-quality-analysis`.
- Wiring a sourced feature into the pipeline code →
  `model-implementation`.

## Sources In Scope

Primary target: **Aerodatabox**. It is the current production API.

Trusted free alternatives (include when Aerodatabox cannot supply a
field, or as a secondary check):

- FAA ASWS / ASPM public feeds (US).
- NOAA ADDS / aviationweather.gov (METAR, TAF, SIGMET).
- OpenSky Network (flight states, historical tracks; subject to
  their fair-use policy).
- OurAirports static airport metadata.
- IATA / ICAO airline and airport code tables.

Reject any source that is paid, unofficial, scraped from a site that
forbids it, or unstable. Note the rejection in the report.

## Workflow

### 1. Enumerate the target BTS fields

Start from the BTS field list relevant to the model (or the subset
the user names). For each field record the canonical name, dtype,
unit, and typical range in BTS.

### 2. Draft a mapping

For each BTS field, propose a production source:

- **Direct:** Aerodatabox returns the same concept under a specific
  endpoint and field name.
- **Derivable:** Aerodatabox (or a trusted alternative) returns
  inputs from which the field can be computed. Document the
  derivation.
- **Approximate:** A close-but-not-identical proxy exists. Describe
  the expected bias.
- **Unavailable:** No trustworthy source. Recommend either dropping
  the feature, or flagging it as "train-only" (used for offline
  research, never served in production).

Capture everything in the mapping table (below). Do not commit to a
specific endpoint name that has not been checked against current
Aerodatabox documentation.

### 3. Parity test (when a key is provided)

If — and only if — the user has explicitly provided an Aerodatabox
API key for testing, run a parity check:

- Pick a small sample of historical flights for which BTS ground
  truth is known.
- Fetch the equivalent Aerodatabox records. Respect rate limits.
  Never issue broad historical sweeps.
- For each mapped field, compute agreement metrics: exact match
  rate for categoricals, MAE and bias for numerics, coverage rate
  for fields that may be missing.
- Summarize discrepancies, flag systematic biases (timezone, units,
  rounding, definitional drift like "wheels-off" vs. "off-block").

Store the raw comparison output under
`feature_transferability/parity/<YYYY-MM-DD>-<batch>.csv` and the
summary under `feature_transferability/reports/`.

### 4. Credential handling

- Never write the API key to disk.
- Never include the key in report files, logs, or commit messages.
- Read the key from an environment variable the user sets (e.g.,
  `AERODATABOX_API_KEY`). If the user pastes a key in chat, read it
  once, use it in-memory, and remind them not to commit it.
- If the user has not provided a key, skip the parity test and say
  so in the report. Do not fabricate numbers.

## Mapping Table Format

```markdown
| BTS field | Meaning | Source tier | Endpoint / field | Transform | Parity (n, metric) | Verdict |
|-----------|---------|-------------|------------------|-----------|--------------------|---------|
| DEP_DELAY | Departure delay minutes | Direct | Aerodatabox flights/{fr24Id} departure.actualTime vs scheduledTime | actual - scheduled, minutes | 200 samples, MAE 1.8 min | Adopt |
| TAXI_OUT | Taxi-out minutes | Derivable | wheelsOffTime - actualOffBlockTime | subtract, cap at 0 | 200 samples, MAE 2.4 min | Adopt with caveat |
| WHEELS_OFF | Wheels-off UTC | Direct | departure.runway.actualTime | parse as UTC | n/a | Adopt |
| ... | ... | ... | ... | ... | ... | ... |
```

`Verdict` is one of: **Adopt**, **Adopt with caveat**, **Approximate
only**, **Train-only**, **Drop**.

## Report Format

Write to
`feature_transferability/reports/<YYYY-MM-DD>-<scope>.md`:

```markdown
# Feature Transferability: <scope>

**Date:** <YYYY-MM-DD>
**Aerodatabox key used:** Yes / No
**Sample size (parity test):** <n> flights, <date range>

## Summary
Short verdict across the scope. How many fields are adoptable,
how many need caveats, how many are unavailable.

## Mapping Table
(as above)

## Parity Findings
Per-field detail for any field with non-trivial disagreement.

## Unavailable Features
What they are and options for each: drop, proxy, or keep as
train-only.

## Recommended Next Actions
- Features ready for `model-implementation`.
- Features that need another pass from `research`.
- Features that need more data before a verdict
  (`correlations-and-interactions` or larger parity sample).
```

## Guardrails

- Do not commit to endpoint names you haven't verified against
  current documentation. If unsure, mark the row "needs
  verification."
- Do not exhaust rate limits during parity testing. Keep samples
  small and stratified.
- Do not conflate "present in Aerodatabox" with "identical to
  BTS." Measure, don't assume.
- Units and timezones are the top two sources of silent
  disagreement. Always check them first.
