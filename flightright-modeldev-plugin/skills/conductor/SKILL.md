---
name: conductor
description: >
  This skill should be used when the user asks to "plan a version bump",
  "coordinate the agents", "run the full pipeline", "orchestrate v11",
  "manage the development process", "what should we do next", "launch
  the agents", or otherwise wants to coordinate the deployment and data
  flow across the other seven skills in the flightright-modeldev plugin.
  Use this skill as the top-level orchestrator — it does NOT do research,
  statistics, code, or training itself. It reads production constraints
  and findings from other skills, decides what to launch next, and
  tracks the state of the version development process.
metadata:
  version: "0.1.0"
  domain: "flight-delay-ml"
---

# Conductor

Orchestrate the full model development lifecycle by coordinating the
seven specialized skills. The conductor owns the *process*, not the
*content* — it reads constraints, interprets findings from other
agents, decides what to launch next, and maintains a coherent state
of the version development effort.

## When To Use This Skill

- The user wants to plan or execute a version bump (e.g., "build v11").
- Multiple skills need to run in a specific order with data flowing
  between them.
- The user asks "what should we do next?" and the answer depends on
  the current state of research, transferability, and correlation
  reports.
- The user provides new constraints (e.g., "change the delay
  thresholds", "drop NAS features") and the conductor must propagate
  those constraints to the right downstream skills.
- A skill has completed and its report needs to be read, interpreted,
  and turned into launch instructions for the next skill.

## When NOT To Use This Skill

- The user wants a single skill invoked directly (e.g., "run
  correlations on feature X") — just invoke that skill.
- The user wants to understand a specific analysis result — read the
  report directly.
- The task is a one-off question that doesn't require multi-skill
  coordination.

## The Seven Skills Under Coordination

| Skill | Owns | Produces | Consumes |
|-------|------|----------|----------|
| `research` | Literature review, feature brainstorming | `research/<date>-<topic>.md` | User questions, prior research |
| `feature-transferability` | Aerodatabox/source mapping, parity tests | `feature_transferability/reports/<date>-<scope>.md` | Research candidates, API key |
| `correlations-and-interactions` | Statistical signal testing | `exploration/reports/<date>-<feature>.md` | Research candidates, feature parquets |
| `data-quality-analysis` | Feature correctness validation | `data_quality/reports/<date>-<feature>.md` | Implemented features, parquets |
| `model-implementation` | Code changes in `src/`, configs, smoke tests | Modified source files, v{N} configs | Validated features, transferability verdicts |
| `ml-hyperparameter-optimization` | HPO sweep design, metric analysis | HPO reports, sweep configs | Implemented features, training configs |
| `model-runtime-manager` | Training job execution, monitoring | MLflow runs, model bundles | Training configs, HPO configs |

## Orchestration Workflow

The canonical flow for a version bump has six phases. The conductor
manages transitions between phases and decides when to advance.

### Phase 1: Constraints and Research (parallel)

**Inputs:** User-provided constraints (threshold changes, data source
rules, compatibility requirements), prior version reports.

**Launch:**
- `research` — brainstorm new feature candidates within constraints
- `feature-transferability` — screen mandatory changes against
  Aerodatabox (e.g., threshold alignment, field availability)

**Gate to Phase 2:** Both reports are complete. The conductor reads
them and extracts:
- Which features are confirmed available (Adopt / Adopt with caveat)
- Which features are unavailable (Drop)
- Which features need empirical validation (hand to Phase 2)
- Any new constraints discovered (e.g., API rate limits, historical
  depth limitations)

### Phase 2: Signal Validation (parallel where possible)

**Inputs:** Candidate features from Phase 1, transferability verdicts.

**Launch:**
- `correlations-and-interactions` — test signal strength of Tier 1
  and Tier 2 candidates against delay labels
- Additional `feature-transferability` checks if Phase 1 flagged
  uncertain availability

**Gate to Phase 3:** Correlation reports are complete. The conductor
reads them and produces a **final feature manifest** — the definitive
list of features for the new version, with:
- Features carried forward from the prior version (unchanged)
- Features modified (e.g., threshold change)
- Features added (validated signal + confirmed availability)
- Features removed (no Aerodatabox source, or signal too weak)

### Phase 3: Implementation

**Inputs:** Final feature manifest, prior version configs.

**Launch:**
- `model-implementation` — code changes, new configs, smoke test

The conductor provides the implementation skill with:
- Exact code change locations (file, line, old value, new value)
- Config template (which fields change from v{N} to v{N+1})
- Smoke test acceptance criteria

**Gate to Phase 4:** Smoke test passes, code changes are reviewed.

### Phase 4: Quality Validation

**Inputs:** Smoke test output parquets.

**Launch:**
- `data-quality-analysis` — validate all new/modified features

**Gate to Phase 5:** All quality checks pass or have documented
accepted caveats.

### Phase 5: Hyperparameter Optimization (optional)

**Inputs:** Implemented features, training configs.

**Launch:**
- `ml-hyperparameter-optimization` — design sweep for new version

**Gate to Phase 6:** HPO report with recommended configs.

### Phase 6: Production Training

**Inputs:** Final configs, HPO recommendations.

**Launch:**
- `model-runtime-manager` — execute full training runs

**Completion:** All models trained, MLflow runs logged, deploy
bundles produced.

## State Tracking

The conductor maintains state in the plan file and todo list. For
each phase, track:

- **Status:** not started / in progress / blocked / complete
- **Blocking issues:** what is preventing advancement
- **Key findings:** extracted from skill reports (1-2 sentences each)
- **Next actions:** which skills to launch with what inputs

## Decision Framework

When a skill report contains ambiguous or conflicting findings, the
conductor applies these rules:

1. **User constraints are absolute.** If the user said "no
   approximations," a feature with verdict "Approximate only" is
   dropped, period.
2. **Train/serve consistency beats marginal signal.** A weaker
   feature that can be served identically in training and production
   is preferred over a stronger feature with train/serve skew.
3. **Empirical evidence trumps literature claims.** If correlations
   show weak signal despite strong literature support, trust the
   data.
4. **Implementation cost is a tiebreaker, not a gate.** Don't reject
   a high-signal feature because it's complex to implement. Do
   defer it to the next version if simpler alternatives exist with
   comparable signal.
5. **When in doubt, test.** Launch `correlations-and-interactions`
   before committing to implementation.

## Backward Flow: Feedback Loops

The development process is not strictly linear. When a downstream
skill produces findings that invalidate or weaken an upstream skill's
recommendations, the conductor must propagate information backward
and re-launch the upstream skill with updated context.

### Common Backward-Flow Patterns

#### Research <-> Correlations Loop

When `correlations-and-interactions` finds that a research-recommended
feature has weak or no signal:

1. Read the correlation report to understand *why* the signal failed
   (redundancy with existing features? confounded by carrier? wrong
   aggregation window? too much missingness?).
2. Re-launch `research` with a **refined prompt** that includes:
   - The specific feature that failed and the failure mode
   - What the correlation analysis revealed about the data structure
   - A request for alternative features that address the same
     underlying signal through a different mechanism
   - Example: "The simple upstream_airport_delay_yesterday_mean was
     mostly captured by carrier_depdelay_median_last1 (partial r
     dropped from 0.095 to 0.042). Research alternative approaches
     to capture delay propagation that are orthogonal to carrier-level
     baselines — e.g., weighted graph scores, non-hub feeder airport
     signals, or cross-airline propagation."
3. Track the iteration count to avoid infinite loops. After 2 rounds
   of research -> correlations -> rejection on the same concept, mark
   it as "explored, no viable feature found" and move on.

#### Transferability -> Research Loop

When `feature-transferability` finds that a candidate feature has no
viable Aerodatabox source:

1. Re-launch `research` to find alternative features that capture
   the same underlying signal but use different data sources.
2. Example: "NASDelay has no direct Aerodatabox equivalent. Research
   alternative signals for ATC-driven delay that can be sourced from
   the Aerodatabox airport delays endpoint, FAA NAS status feeds, or
   other free sources."

#### Correlations -> Implementation -> Data Quality Loop

When `data-quality-analysis` finds that an implemented feature has
incorrect values:

1. Diagnose whether the issue is in the implementation or the data.
2. If implementation bug: re-launch `model-implementation` with the
   specific bug description.
3. If data issue: re-launch `data-quality-analysis` to characterize
   the extent, then decide whether to fix or drop the feature.

#### Implementation -> Correlations Loop

When `model-implementation` smoke test reveals that a feature has
unexpected distributions (e.g., 90% null, or all-zero):

1. Re-launch `correlations-and-interactions` on the raw data to
   verify the feature actually has variance before debugging the
   implementation.

### Feedback State Tracking

For each feature under development, track its state:

```
Feature: <name>
Status: researched | tested | failed_signal | failed_source | implemented | validated
Iterations: <count>
Last failure: <description of what went wrong>
Next action: <which skill, with what refined prompt>
```

When a feature has `failed_signal` or `failed_source`, the conductor
must either:
- Launch a backward-flow action (re-research with refined context)
- Mark it as "explored, no viable feature" and remove from the
  manifest
- Escalate to the user if the failed feature was a mandatory
  constraint

## Constraint Propagation

When the user provides a new constraint mid-process, the conductor:

1. Identifies which phases and skills are affected.
2. Updates the plan file with the new constraint.
3. Re-launches affected skills with updated inputs if they haven't
   started, or flags findings that may be invalidated if they have.
4. Does NOT silently absorb constraints — always confirms
   understanding with the user before propagating.

## Report Reading Protocol

When reading a skill report to decide next steps:

1. Read the **Summary / Executive Summary** first.
2. Check for **Unavailable Features** or **Drop** verdicts — these
   are hard constraints.
3. Check for **Adopt with caveat** — these need the conductor to
   decide if the caveat is acceptable given user constraints.
4. Read **Recommended Next Actions** — these are the skill's own
   handoff suggestions.
5. Cross-reference with the current phase gate criteria to decide
   whether to advance.

## Version Manifest Template

When Phase 2 completes, produce a version manifest:

```markdown
# v{N+1} Feature Manifest

**Base version:** v{N}
**Date:** <YYYY-MM-DD>
**Delay thresholds:** [t1, t2, t3, t4]

## Mandatory Changes (from user constraints)
- [ ] Change X: description, affected files
- [ ] Change Y: description, affected files

## New Features (validated signal + confirmed availability)
- [ ] Feature A: signal strength, source, implementation notes
- [ ] Feature B: signal strength, source, implementation notes

## Removed Features
- [ ] Feature C: reason (no Aerodatabox source / weak signal / redundant)

## Unchanged Features
(carried forward from v{N} — list count, not individual features)

## Config Changes
- Thresholds: [old] -> [new]
- Bin weights: [old] -> [new]
- New blueprint flags: list
- Removed blueprint flags: list

## Verification Criteria
- Smoke test: what to check
- Data quality: which features to validate
- Forbidden columns: list
```

## Guardrails

- Never run statistics, write code, or launch training directly.
  Always delegate to the appropriate skill.
- Never fabricate findings. If a skill report hasn't been produced
  yet, say so — don't guess what it will find.
- Never skip a phase gate. If the gate criteria aren't met, either
  re-launch the blocking skill or escalate to the user.
- Never propagate a constraint without confirming understanding
  with the user first.
- Always read skill reports before deciding next steps — don't
  rely on assumptions about what they found.
