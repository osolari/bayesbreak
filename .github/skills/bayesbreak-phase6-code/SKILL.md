---
name: bayesbreak-phase6-code
description: 'Converge the existing BayesBreak Python package with the Phase 6 coding handoff and incomplete repository skeleton. Use to implement CODE-BB-001 through CODE-BB-016 after report adoption and explicit package-phase approval.'
argument-hint: '[plan|execute <CODE-BB-ID>|status]'
user-invocable: true
disable-model-invocation: true
---

# BayesBreak Phase 6 Code Adoption

Implement the canonical coding handoff in the existing package. Preserve working public APIs
unless a documented Phase 6 requirement needs a compatible extension or approved migration.

## Preconditions

1. Read [the adoption orchestrator](../bayesbreak-phase6-adoption/SKILL.md).
2. Require Gate B approval and explicit approval for the package phase or named task.
3. Load `shared/handoffs/coding_agent_handoff.json` as the canonical task specification.
4. Confirm report revisions satisfying each task's `REV-BB-*` dependencies are adopted.
5. Baseline focused and full tests before changing code; record failures and unresolved tests.
6. Compare skeleton files with current modules. Never copy skeleton `NotImplementedError`
   bodies over functioning code.

## Merge Rules

- Existing behavior plus Phase 6 tests controls implementation; the skeleton controls missing
  interfaces, schemas, registries, and explicit incomplete states.
- Merge types and contracts deliberately. Do not create duplicate modules solely to match the
  skeleton when an existing module owns the behavior.
- Keep sum-product and max-sum implementations and outputs distinct.
- Separate exposure/trial structure from likelihood-power weighting.
- Reject nonfinite or dimensionally inconsistent segment scores before partition inference.
- Preserve deterministic tie handling and log-domain numerical stability.
- Label numerical integration outputs approximate and retain routine-specific failure states.
- Implement one task at a time; after its first substantive edit, run its cheapest focused
  test before reading or editing another task area.

## Implementation Waves

### Wave A: Contracts, Prior, and Exact Inference

**CODE-BB-010: result and provenance schemas**

- Adopt and version result sidecar, approximation-error, and boundary-metric schemas.
- Add readers/migration paths for archived records without rewriting populated values.
- Require result ID, optional/required parent ID by status, data/config/code/environment
  hashes, execution status, interpretation status, coordinate/reference metadata, and paths
  relative to the repository.
- Reject corrected releases that lack lineage or hashes (`FAIL-BB-006`).

**CODE-BB-001: cohesion and boundary-hazard prior**

- Reconcile `base.py`, `dp.py`, and existing design-prior behavior with the skeleton's
  `priors.py` and `design_prior.py` contracts.
- Represent segment cohesion and interior-boundary hazard as distinct factors on structural
  partition support.
- Implement fixed-count Poisson interval occupancy with local odds `exp(Lambda_j) - 1`.
- Test small finite enumerations, irregular grids, zero support, and sensitivity fixtures.

**CODE-BB-002: repaired prior in exact DP**

- Integrate prior factors with sum-product posterior recursions and a separate max-sum
  backtracking path.
- Reconcile missing `map.py` and `posterior.py` interfaces with current `dp.py` ownership.
- Test randomized finite cases against exhaustive enumeration, including ties and impossible
  blocks.

### Wave B: Hierarchies and Latent Groups

**CODE-BB-004: grouped and replicate aggregation**

- Harden `replicates.py` and `groups.py` log-domain aggregation.
- Test adversarial score magnitudes, unequal information, common partitions, known groups,
  and ranking invariance under stable aggregation.

**CODE-BB-003: latent-group objective and restarts**

- Repair `mixture.py` so every successful exit returns the objective of the returned final
  state and the final trace entry agrees within declared tolerance.
- Include the implementation-scale `n_g log gamma_u(i,j)` group weight.
- Mark a restart invalid when its objective is nonmonotone beyond tolerance or stale.
- Exclude invalid restarts from selection; make ties, collapsed groups, label ordering, and
  seeded restart ordering deterministic.
- Add regression tests for every exit path before accepting the fix.

### Wave C: Approximation, Prediction, and Metrics

**CODE-BB-005: nonconjugate error records**

- Implement structured approximation error records and conditional propagation from a
  uniform admissible-segment log-score bound.
- Do not invent routine-wide convergence rates. Return explicit unverifiable/failure states
  when assumptions cannot be checked.

**CODE-BB-006: Beta-observation posterior prediction**

- Implement the observation-family predictive distribution for `BayesBreakBetaObs`.
- Route through family-owned block predictive methods rather than a Gaussian fallback.
- Test analytic or high-accuracy numerical references, normalization where applicable,
  precision/weight handling, endpoints, and finite edge cases.

**CODE-BB-007: explicit support/extrapolation policy**

- Require named policies for coordinates outside fitted support; default to an explicit error.
- Record the selected policy in outputs and provenance.
- Test left/right boundaries, exact endpoints, unsorted inputs, and every supported policy.

**CODE-BB-008: canonical boundary matching and metrics**

- Implement maximum-cardinality, minimum-total-distance, one-to-one matching.
- Return `NA` matched-distance summaries when no pair matches.
- Separate matched-k agreement, predictive scoring, and external-truth accuracy.
- Test hand-computed cases, ties, empty sets, symmetry, tolerance boundaries, and duplicates.

**CODE-BB-009: comparator axes and tuning budgets**

- Validate matrix dimensions, coordinate axes, reference type, and tuning budget before any
  metric computation.
- Deterministically reject the historical cached single-trace CGH comparator route
  (`FAIL-BB-002`).
- Cover the valid raw matrix route and common joint comparison targets.

### Wave D: Data, Outputs, and Reproducibility

**CODE-BB-011:** synchronize well-log, CGH, SPX, and methylKit loaders, data cards, captions,
hashes, coordinate conventions, family declarations, and generated sidecars.

**CODE-BB-015:** regenerate figures and tables from versioned inputs with source hashes,
interpretation status, caption anchors, semantic checks, and visual QA. Never overwrite
archived numerical assets in place.

**CODE-BB-012:** profile and run the full test suite. A bounded timeout is unresolved with
environment, seed, profile, owner/reason, and impact; it is not a pass.

**CODE-BB-013:** reconcile package version, generated `_version.py`, CI, changelog, citation,
preprint lineage, supported Python versions, and one canonical release target.

**CODE-BB-014:** render Markdown and TeX handoffs and task registries from canonical JSON;
verify IDs and semantics rather than hand-maintaining synchronized copies.

**CODE-BB-016:** automate bibliography annotation completeness and reject missing, duplicate,
orphaned, or relationship-free entries.

## Test Matrix

For each task, add the narrowest relevant layers:

- unit: family calculations, prior factors, coordinate assignment, schemas, and metric edges;
- property: exact recursions versus exhaustive finite enumeration;
- numerical: stable aggregation, reference integration, tolerance tightening, and failures;
- integration: package APIs, loaders, prediction, comparators, and artifact generation;
- statistical: repeated calibration/recovery with predeclared uncertainty and null outcomes;
- regression: archived read-only behavior, lineage, exclusions, captions, and semantics;
- reproducibility: pinned environment, seeds, hashes, commands, and clean regeneration.

## Task Completion Rule

A `CODE-BB-*` task is complete only when its focused tests pass, the full relevant test slice
passes, public/API documentation is synchronized, schemas or sidecars validate, and its
failure behavior is tested. Update the implementation ledger after each task. Do not set the
skeleton's scientific implementation status true until all required tasks and integrated
checks are complete.

## Gate C Evidence

Provide completed and blocked task IDs, module/API changes, focused and full test outputs,
schema migrations, compatibility notes, resource profiles, and explicit incomplete states.
Obtain separate approval before any scientific rerun.
