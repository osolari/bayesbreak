---
name: bayesbreak-phase6-reruns
description: 'Plan or execute the three controlled BayesBreak Phase 6 corrected experiments: latent-group, array-CGH comparator, and methylation posterior prediction. Use only after required code tasks pass and the user explicitly approves each rerun.'
argument-hint: '[plan|latent-group|array-cgh|methylation|status]'
user-invocable: true
disable-model-invocation: true
---

# BayesBreak Phase 6 Corrected Reruns

Run only the minimum clean-submission set unless the user separately approves expanded
experiments. Historical computations remain immutable and keep their interpretation status.

## Authorization and Preconditions

1. Read [the adoption orchestrator](../bayesbreak-phase6-adoption/SKILL.md).
2. Require Gate C evidence and explicit approval for each named experiment.
3. Resolve the experiment and parent result from canonical registries before execution.
4. Allocate a new unused result ID before writing output. Never reuse the parent ID.
5. Freeze and hash data, configuration, code revision/diff, environment, and random seeds.
6. Validate output directories and sidecars without touching archived populated assets.
7. Run the experiment's focused unit/integration tests immediately before the scientific run.
8. Estimate wall time, CPU, memory, storage, and repetitions from a pilot before a large run.

Every new record must distinguish `executed`, `valid for stated interpretation`, `excluded
from named analysis`, and `planned` rather than collapsing these into one status.

## Rerun 1: Latent Groups

Canonical protocol: `EPR-BB-005`. Resolve its archived parent ID from
`result_interpretation.json`; do not infer it from display order.

Required code evidence:

- `CODE-BB-003` focused tests pass;
- returned objective equals the final trace value within the declared tolerance;
- objective traces are monotone within tolerance;
- stale/nonmonotone restarts are invalid and cannot win selection;
- ties, collapsed groups, label order, exit paths, and seeded restart order are covered;
- full required package tests satisfy `CODE-BB-012` or retain explicit unresolved statuses.

Execution plan:

1. Reproduce the archived design and seeds before adding stress cases.
2. Record every restart's seed, validity, trace, final state hash, and selection reason.
3. Run declared separation, imbalance, initialization, collapse, and tie cases.
4. Report recovery and objective behavior with repetitions and uncertainty.
5. Treat the historical result as usable with its existing limitations until the corrected
   result independently passes validation; do not silently replace it.

## Rerun 2: Array-CGH Comparator

Canonical protocols: `EPR-BB-010` and the applicable stratum of `EPR-BB-013`.
Historical parent: `RES-BB-CMP-002`, which remains excluded from comparator conclusions.

Required code evidence:

- `CODE-BB-004`, `CODE-BB-008`, `CODE-BB-009`, `CODE-BB-010`, `CODE-BB-011`, and
  `CODE-BB-015` pass their acceptance checks;
- the incompatible cached single-trace route is rejected before metric computation;
- the raw array-CGH matrix orientation, expected probe and subject dimensions, and probe
  coordinate axis are validated from the hashed source;
- matching rule, reference type, common target, and tuning budget are declared in advance.

Execution plan:

1. Load the exact hashed raw matrix and validate probes-by-subjects orientation.
2. Fit shared and independent models according to the registered protocol.
3. Run comparators on compatible raw matrix inputs and score only on a common coordinate
   axis and declared target.
4. Keep matched-k agreement separate from independently tuned prediction and external-truth
   accuracy; no unavailable external annotation may be implied.
5. Record runtime and tuning cost, uncertainty method, axis metadata, dimensions, and hashes.
6. Create a new sidecar and corrected table/figure linked to `RES-BB-CMP-002`; never edit the
   parent's values or reclassify its historical execution as valid evidence.

Abort as a diagnostic record if axes or dimensions differ (`FAIL-BB-002`).

## Rerun 3: Methylation Posterior Prediction

Canonical protocol: `EPR-BB-012`. Historical parent: `RES-BB-RD-007Q`, which remains
excluded from posterior-predictive conclusions.

Required code evidence:

- `CODE-BB-006`, `CODE-BB-007`, `CODE-BB-010`, and `CODE-BB-011` pass;
- `BayesBreakBetaObs` uses a family-correct predictive calculation;
- no Gaussian fallback is reachable for Beta observations;
- coordinates outside fitted support error by default unless a named policy is selected and
  recorded;
- exact support endpoints and left/right out-of-range tests pass.

Execution plan:

1. Use the exact hashed methylKit `test1.myCpG` chromosome 21 source and record precision or
   weights, preprocessing, coordinate axis, and observation family.
2. Predeclare an in-support, scientifically justified block-aware split; do not use implicit
   endpoint assignment or clipping.
3. Run repeated splits with declared seeds and uncertainty intervals.
4. Report total and mean log predictive scores with denominator, calibration where defined,
   MAP segment count, boundary stability, and runtime.
5. If extrapolation is scientifically required, name, configure, test, and record the policy
   before execution.
6. Create a new sidecar and corrected table/figure linked to `RES-BB-RD-007Q`; leave the
   parent excluded and byte-identical.

Abort unsupported family or coordinate requests explicitly (`FAIL-BB-003`).

## Post-Run Validation

For every rerun:

1. Validate the sidecar schema and parent linkage.
2. Recompute data, configuration, code, environment, and output hashes.
3. Re-run focused regression tests and relevant artifact generators.
4. Check that historical asset hashes remain unchanged.
5. Independently review metric semantics and claimed interpretation before marking usable.
6. Generate figures/tables to new versioned paths with source hashes and caption anchors.
7. Update registries and generated views from canonical data, preserving both parent and
   corrected records.
8. Record null, reversed, failed, and timed-out outcomes rather than filtering them out.

## Gate D Report

Report new result IDs, parent IDs, commands, seeds, hashes, resource use, statistical
summaries, schema validation, artifact paths, interpretation decisions, and any abort or
unresolved state. Obtain explicit approval before integrated release validation.
