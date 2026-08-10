# Phase 6 independent scientific verification

## Decision

The final document set preserves the author-approved title and the original scientific direction:

> **Generalized Hierarchical Bayesian Segmentation with Irregular Designs, Multi-Sequence Hierarchies, and Grouped/Latent-Group Designs**

BayesBreak is presented as a generalized hierarchical Bayesian method for multiple-changepoint segmentation. Segment marginal likelihoods and posterior segment summaries are supplied by the chosen observation model; the partition recursions then operate over contiguous segments. Irregular designs, multiple related sequences, known groups, and finite latent groups remain central components rather than secondary software features.

## Mathematical corrections made in Phase 6

- **partition support and evidence.** Separated admissible-partition support from observed segment marginal likelihoods; zero likelihood no longer changes the prior normalizing set. Clarifies the probability model without changing the estimator or any archived result.
- **reported segment functional.** Defined the reported quantity as m_star(theta)=m(theta;d_star) when the conditional mean depends on exposure or trial structure. Distinguishes a scientific reporting scale from observation-specific exposure.
- **Beta-observation segment integration.** Retained the one-dimensional integral and removed unsupported universal monotonicity and routine-wide convergence-rate language. Exactness and approximation statements now match the available derivation and numerical checks.
- **sum-product and MAP inference.** Separated the sum-product result from the max-sum/backtracking result and corrected storage/output complexity statements. Prevents marginal posterior summaries from being conflated with the joint MAP partition.
- **decision rules.** Stated the conditions under which coordinatewise posterior decisions minimize additive or Hamming loss. Makes the loss-function assumptions explicit.
- **irregular-design prior.** Corrected the fixed-count Poisson interval-occupancy factor to local odds exp(Lambda_j)-1. Aligns the prior formula with conditioning on the number of occupied intervals.
- **latent-group optimization.** Aligned the Jensen-minorized criterion with the implementation-scale group weight, including n_g log gamma_u(i,j). Matches the stated coordinate-ascent objective to the implemented scoring convention.
- **nonconjugate approximation.** Tightened assumptions, removed unsupported method-specific rates, and bounded normalized-distribution error by min(1, exp(2 eta)-1). Limits the theorem to consequences implied by a verified uniform segment log-score bound.
- **changepoint matching.** Specified maximum-cardinality, minimum-total-distance one-to-one matching and NA when no match exists. Defines the evaluation metric unambiguously.

## Independent checks

- All **13** finite-case or numerical checks passed: four conjugate segment marginal likelihoods, Beta-observation quadrature, fixed-count partition evidence, joint MAP backtracking, ordered-boundary and event marginals, posterior mean and second-moment curves, shared-boundary pooling, segment-score error propagation, Poisson interval odds, and the latent-group Jensen decomposition.
- The technical book contains 28 established theorem-like statements with 28 immediately following proofs and one proof obligation. The journal paper contains 12 established statements with 12 immediately following proofs and one proof obligation.
- The bibliography contains 38 unique keys and 38 one-to-one project-specific annotations. The Phase 6 metadata corrections are recorded in `BIBLIOGRAPHY_VERIFICATION.md`.
- All 53 archived result assets match the corrective Phase 4R SHA-256 baseline. No research experiment was rerun and no populated archived value was changed.
- The Phase 6 bounded implementation-test rerun collected 179 tests: 173 passed, five were skipped for optional dependencies, one EP logistic-normal test did not finish under a 20-second per-test cap, and none failed. The historical `RES-BB-QA-002` state remains unchanged in the result record.
- All 257 compiled pages were rendered at 150 dpi and inspected through 22 contact sheets, with additional full-page checks of the principal corrected formulas and proofs.

## Interpretation of real results

`RES-BB-CMP-002` remains a real historical execution but is excluded from comparator conclusions because the compared outputs use incompatible coordinate axes. `RES-BB-RD-007Q` remains a real historical execution but is excluded from posterior-predictive conclusions because the computation used an inappropriate observation-family predictive routine and an implicit endpoint rule. Neither result was deleted or replaced.

## Explicitly incomplete items

- One routine-specific nonconjugate approximation-rate statement remains a proof obligation and is not used as an established rate result.
- The EP logistic-normal comparison did not complete under the 20-second per-test cap; it is unresolved rather than failed.
- The previously identified latent-group returned-objective/restart-selection defect was not repaired because package-code changes were outside this manuscript phase.
- Corrected array-CGH comparator and methylation posterior-predictive computations were not run.
- Independent external changepoint annotations are not available for the applied studies.
- The final journal venue, permanent software repository, and data-release locations remain unset.
