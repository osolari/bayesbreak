# RES-BB-SYN-006 EPR-BB-015 resource pilot

This file preserves the sequence of pilot decisions. The corrected 400-run suite was
subsequently executed and finalized; current full-run evidence is in `SUMMARY.md`,
`failure_summary.csv`, `failure_map.png`, and `result_sidecar.json`.

The authorized one-dataset-per-cell pilot executed from code commit
`88eb9daddc81249379635f031327fc2e39fb22d6`.

- Eight predeclared cells ran in 77.02 seconds with 157 MB peak RSS.
- The projected 50-repetition full suite is 3,851 seconds (64.2 minutes), above the
  predeclared 30-minute renewed-approval threshold.
- The null Gaussian cell selected four segments, producing three false boundaries.
- Heavy-tailed Gaussian data recovered both true boundaries with F1@3 = 1.00.
- Zero-inflated Poisson data saturated at `k=8` and produced F1@3 = 0.444.
- Dense Gaussian changes selected `k=15` with F1@3 = 0.880.
- The four-observation short segment selected `k=4` with F1@3 = 0.800.
- The prior-conflict cell returned an explicit failure (`No valid segment counts produced
  finite evidence`) because its minimum-length prior excluded every feasible truth-compatible
  partition. The failure remains in the record.
- Shared-boundary heterogeneity produced shared F1@3 = 0.444 versus mean independent F1@3 =
  0.628 and selected the subject-specific boundary near index 60 as shared.
- The logistic-normal reference selected `k=3`. EP consumed 75.73 seconds, had maximum reachable
  block error 6.575 log units, and empirical posterior TV 0.505. The lower-order quadrature and
  Laplace comparisons were fast but still had large maximum block errors.

At this first-pilot stage, the full suite had not been executed and required renewed approval of
the projected resource cost and a predeclared EP budget.

## Bounded EP re-pilot

The approved re-pilot executed from commit
`236ec8c509224688cc6e29d855a2df182a869ca7` with a predeclared 20-second EP
subprocess timeout.

- All eight cells completed in 21.28 seconds with 154 MB peak RSS.
- The projected 50-repetition full suite is 1,064 seconds (17.7 minutes), below the
  predeclared 30-minute renewed-approval threshold.
- The seven non-EP scientific outcomes and the quadrature/Laplace logistic diagnostics are
  byte-for-byte or field-for-field unchanged from the first pilot.
- EP timed out at 20.01 seconds and is retained as a timed-out scientific outcome; no EP error
  or posterior TV value is imputed for that run.
- The prior-conflict cell remains an explicit failed outcome.

At this bounded re-pilot stage, the full suite remained unexecuted and required separate approval.

## Semantic redesign and valid bounded pilot

The approved final pilot executed from commit
`d03a06323d0e4ca5afb9409dc18c716e5a5c5c56` after the semantic redesign. Its
immutable artifact is `pilot-semantic-corrected-v2.json` with SHA-256
`3cd660ccbc660a8a2300405fa3dd0af4cde537a50cf35392ceacb2cdeab117e0`.

- All eight cells executed in 22.13 seconds with 160.17 MB peak RSS. The projected
  400-run suite is 1,106.55 seconds (18.44 minutes), below both renewed-approval
  thresholds.
- Every record carries complete data, truth, and effective-configuration hashes.
- Binary rates use Wilson score intervals. Exact recovery requires no missed or extra
  boundaries.
- The null cell produced three false boundaries; its one-dataset false-positive rate is
  1.0 with Wilson 95% interval [0.207, 1.0].
- Zero-inflated Poisson and dense Gaussian both saturated their declared segment budgets.
  Dense Gaussian found every true boundary within tolerance but added three boundaries,
  so exact recovery is false.
- The short-segment cell found both true boundaries but added one boundary, so exact
  recovery is false despite a zero missed-change rate.
- The prior-conflict cell executed with `k=2`; unsupported counts have zero posterior
  mass instead of poisoning feasible counts.
- Shared and independent methods were scored against the same per-subject truths. Mean
  F1@3 was 0.483 for the shared model and 0.628 independently; the shared model selected
  the subject-specific boundary near 60.
- EP timed out after 20.01 seconds of fit time (20.75 seconds total worker lifecycle),
  with child RSS retained. No EP diagnostics were imputed.

The intermediate `pilot-semantic-corrected.json` remains excluded because its audit found
non-exact recovery semantics, degenerate rate intervals, and a subprocess deadline that
included worker startup. The valid pilot is implementation-verification evidence only.
At this final-pilot stage, the full suite remained machine-blocked. It was later approved and
executed from commit `96464039e12b43207735835b004b0a59a9966b57`; scientific interpretation
remains pending independent review.
