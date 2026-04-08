---
name: research
description: >
  This skill should be used when the user asks to "research a feature",
  "investigate a potential feature", "look into a new ML strategy",
  "read this paper", "digest a paper", "find better modeling approaches",
  "what features might help predict delays", or otherwise requests a
  literature/technical investigation for the flight delay prediction
  project. Use this skill for gathering and synthesizing information —
  NOT for running statistics, correlation tests, or data experiments
  against the actual dataset (use the correlations-and-interactions
  skill for that).
metadata:
  version: "0.1.0"
  domain: "flight-delay-ml"
---

# Research

Investigate potential features, modeling strategies, and relevant
literature for the flight departure and arrival delay prediction
project. Produce structured, well-sourced reports that downstream
skills (correlations-and-interactions, feature-transferability,
model-implementation) can act on.

## When To Use This Skill

Use this skill when the user wants to:

- Brainstorm or evaluate candidate features (weather, airport
  congestion, aircraft rotation, crew scheduling signals, historical
  on-time performance, etc.)
- Determine whether a proposed feature is even *possible* to obtain
  given the Aerodatabox API and trusted free alternatives (high-level
  feasibility check only — deeper sourcing belongs to the
  feature-transferability skill)
- Investigate better machine learning strategies, architectures, loss
  functions, calibration techniques, or uncertainty quantification
  methods for delay prediction
- Read and digest scientific or technical papers and return a concise,
  actionable summary
- Compare industry or academic approaches to flight delay prediction

## When NOT To Use This Skill

Do not use Research for:

- Running statistical tests, correlations, mutual information, SHAP,
  or interaction analyses on real data → use
  `correlations-and-interactions`
- Checking whether a concrete data field is present, complete, or
  sensible → use `data-quality-analysis`
- Picking hyperparameters or tuning a model → use
  `ml-hyperparameter-optimization`
- Writing code into `src/` → use `model-implementation`
- Verifying Aerodatabox vs. BTS field parity → use
  `feature-transferability`

If the user's request needs one of those, stop and hand off.

## Research Process

Follow this sequence. Do not skip steps.

1. **Clarify the question.** Restate the research question in one or
   two sentences. If the user gave a vague prompt ("find better
   features"), narrow it down by asking: target label (departure vs.
   arrival delay? binary vs. regression vs. quantile?), horizon
   (forecast window), and any constraints (must be free, must be
   available at prediction time, etc.).

2. **Scope the sources.** Decide which of the following are in scope:
   peer-reviewed papers, preprints, industry blog posts, Kaggle
   writeups, airline/FAA reports, Aerodatabox documentation, BTS
   documentation, weather API documentation. Prefer primary sources.

3. **Gather.** Use WebSearch and WebFetch to pull the most relevant
   material. For each source, capture: title, author/publisher, year,
   URL, and a one-sentence relevance note. Skim before deep-reading.

4. **Extract claims.** For each candidate feature or technique, pull
   out: (a) what it is, (b) why it's expected to help predict delays,
   (c) the empirical evidence cited, (d) known failure modes or
   limitations, (e) high-level availability signal (can this plausibly
   be fetched from Aerodatabox or a trusted free source? — do not
   commit to a specific endpoint, leave that to the
   feature-transferability skill).

5. **Check for leakage.** Flag any feature whose value is only known
   *after* the flight has already departed/landed. Leakage-prone
   candidates must be called out explicitly.

6. **Prioritize.** Rank candidates by expected impact × availability ×
   implementation cost. Use a simple High / Medium / Low rubric; never
   invent numeric probabilities you did not measure.

7. **Write the report.** Use the report template below.

## Report Template

Write findings to a markdown file in the workspace so other skills can
reference them. Default location: `research/<YYYY-MM-DD>-<topic>.md`.
Create the `research/` directory if it does not exist.

```markdown
# Research: <topic>

**Date:** <YYYY-MM-DD>
**Question:** <the one-sentence research question>
**Scope:** <features | modeling strategy | paper digest | other>

## Executive Summary

Three to five bullet points capturing the most important findings and
recommendations. Each bullet should be actionable.

## Candidate Features

For each feature:

### <feature name>

- **Description:** What the feature represents.
- **Rationale:** Why it should help predict departure or arrival
  delay.
- **Evidence:** Citations or links to supporting material.
- **Availability signal:** Plausible / uncertain / likely unavailable
  from Aerodatabox or trusted free sources. (Defer definitive sourcing
  to `feature-transferability`.)
- **Leakage risk:** None / Low / Medium / High, with explanation.
- **Priority:** High / Medium / Low.
- **Next step:** What skill should act on this next
  (correlations-and-interactions, feature-transferability, etc.).

## Modeling Strategy Notes

Only include this section if the research question was about methods.
Summarize approaches, trade-offs, and any benchmark numbers reported
by the source.

## Paper Digests

For each paper read:

- **Citation:** Author (Year). *Title*. Venue. URL.
- **Problem:** What the paper is solving.
- **Method:** One-paragraph summary.
- **Results:** Key numbers as reported. Do not extrapolate.
- **Relevance to flightright-modeldev:** Bullet list.

## Open Questions

Things that need empirical validation (hand off to
`correlations-and-interactions`) or sourcing confirmation (hand off to
`feature-transferability`).

## Sources

Numbered list of every URL used, with titles.
```

## Writing Rules

- Cite every non-obvious claim with a source link. No unsourced
  statistics.
- Never fabricate paper titles, authors, or numbers. If a search turns
  up nothing useful, say so.
- Quote sparingly and always in quotation marks. Prefer paraphrase.
- Keep language precise. Distinguish "the paper reports X" from "X is
  true in general."
- When a user proposes a feature that is likely to leak, push back
  clearly before filing it under "candidates."

## Handoffs

End every research report with a short "Recommended next actions"
section that names the specific skills to invoke and what to test.
For example:

> Recommended next actions:
> 1. `correlations-and-interactions`: test `origin_airport_congestion`
>    against historical BTS delay labels.
> 2. `feature-transferability`: confirm whether Aerodatabox exposes
>    METAR-derived ceiling and visibility.
