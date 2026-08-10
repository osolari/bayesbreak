# Mathematical and empirical status ledger

## Mathematical statements

| ID | Statement | Status |
|---|---|---|
| CLM-BB-001 | Common partition recursions operate on any finite segment-marginal-likelihood array with the required factorized partition prior. | Established by derivation; finite exact-regime checks support the algorithms. |
| CLM-BB-002 | Supported regular exponential-family segment models with proper conjugate priors yield exact segment marginal likelihoods and posterior moments from sufficient statistics. | Exactly derived under the stated regularity and integrability conditions. |
| CLM-BB-003 | Sum-product dynamic programming returns fixed-count marginal likelihoods, the segment-count posterior, changepoint marginals, and partition-averaged signal moments. | Established. |
| CLM-BB-004 | Max-sum dynamic programming with backtracking returns the joint MAP partition. | Established; marginal changepoint modes are not equivalent in general. |
| CLM-BB-005 | Irregular coordinates may enter through distinct segment-cohesion and interior-boundary-hazard factors while preserving factorization. | Model and recursion established; package implementation pending. |
| CLM-BB-006 | Aligned conditionally independent sequences with a common partition are pooled by multiplying sequence-specific segment marginal likelihoods. | Established algebraically. |
| CLM-BB-007 | Finite-candidate common-partition posterior concentration holds under a positive average expected log-score gap and suitable law-of-large-numbers conditions. | Established under assumptions. |
| CLM-BB-008 | The finite latent-group criterion admits Jensen minorization and responsibility-weighted max-sum template updates. | Derivation established; one final-objective implementation path requires repair. |
| CLM-BB-009 | The current latent-group criterion is a normalized identifiable finite-mixture sampling model. | Not established and not part of the corrected manuscript. |
| CLM-BB-010 | A uniform reachable-segment log-marginal-likelihood error gives bounds on partition marginal likelihoods and posterior odds. | Established conditionally on the segmentwise error assumption. |
| CLM-BB-011 | Specific nonconjugate routines satisfy uniform rates over all reachable segments. | Proof obligation; do not present as established. |
| CLM-BB-012 | Posterior prediction must use the fitted observation family and an explicit coordinate-support rule. | Statistical requirement established; implementation incomplete for Beta observations and extrapolation. |

## Selected executed results

| ID | Real executed output | Permitted statement |
|---|---|---|
| RES-BB-SYN-001 | ECE approximately 0.010; Brier approximately 0.011 in the archived Gaussian calibration design. | Calibration result for that design; broader uncertainty reporting remains planned. |
| RES-BB-SYN-002 | 96% hard allocation accuracy at sigma=1.0 with 24 sequences. | Result for the stated finite latent-group criterion and design; not mixture identifiability. |
| RES-BB-SYN-003 | Timing study over n=50 to 800; empirical slopes about 1.07 and 1.14 for two k_max settings. | Finite-range empirical timing; not asymptotic complexity. |
| RES-BB-RD-001/002 | Well-log fits with k-hat 23 and 25 under two prior specifications. | Descriptive segmentation and prior sensitivity; no external-boundary accuracy. |
| RES-BB-RD-003/004 | Shared and independent array-CGH fits. | Descriptive outputs from different factorizations; not a direct Bayes factor. |
| RES-BB-RD-005/006 | SPX Gaussian and Bernoulli fits. | Two response-model analyses; contextual alignment only. |
| RES-BB-RD-007 | Methylation segmentation with k-hat 15, log marginal likelihood -9518.6675, n=1904. | Descriptive segmentation of the archived methylKit chromosome-21 example. |
| RES-BB-CMP-002 | Comparator calculation on incompatible CGH axes. | Historical diagnostic only; excluded from comparator conclusions. |
| RES-BB-RD-007Q | Held-out value -387.50040013308154 on m=381 with the wrong predictive family and endpoint assignment. | Historical diagnostic only; excluded from posterior-predictive conclusions. |
| RES-BB-QA-002 | 179 collected, 157 passed, five skipped, 17 unresolved. | Historical Phase 1 bounded verification; retained unchanged. |
| RES-BB-QA-003 | 179 collected, 173 passed, five skipped, one timeout, zero failed. | Current Phase 6 bounded verification; the remaining timeout is neither a pass nor a failure. |

Any corrected computation must use a new result ID and an explicit parent-result link. Do not overwrite these archived values.
