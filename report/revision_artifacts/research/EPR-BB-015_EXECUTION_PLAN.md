# EPR-BB-015 Misspecification and Negative-Control Execution Plan

## Purpose

`EPR-BB-015` is the next bounded research extension after the corrected Gate D results and
family-specific MAP predictive certification. It is a failure-map experiment, not a search for a
single favorable headline. The planned result ID is `RES-BB-SYN-006`; it is an original result and
has no parent.

The machine-readable source of this plan is
`provenance/epr-bb-015-plan.json`.

## Predeclared Cells

| Cell | Model fitted | Failure boundary |
|---|---|---|
| `null-gaussian` | Gaussian | False boundaries under no change |
| `heavy-tail-gaussian` | Gaussian | Outlier-driven oversegmentation under scaled Student-t noise |
| `zero-inflated-poisson` | Poisson | Count-likelihood misspecification with 35% structural zeros |
| `dense-gaussian` | Gaussian | Undersegmentation and count saturation with changes every 10 observations |
| `short-segment-gaussian` | Gaussian | Recovery of a four-observation high-amplitude segment |
| `prior-conflict-gaussian` | Gaussian | Truth excluded by a minimum segment length of 50 |
| `shared-boundary-heterogeneity` | Shared-boundary Gaussian | Forced-sharing bias with unequal information and subject-specific deviations |
| `logistic-approximation-failure` | Logistic-normal | Reachable-block error and convergence under near separation |

## Execution Budget

The pilot runs one seeded dataset per cell. The full design runs 50 datasets per cell with shared
seeds for paired contrasts. The seed base is `261501`; the boundary-matching tolerance is three
indices. Each EP fit runs in a subprocess with a predeclared 20-second timeout; timeout frequency is
a scientific failure metric and timed-out runs remain in the result. Full execution requires a
reviewed pilot and explicit approval.

The pilot must report projected wall time, peak RSS, output size, and per-cell runtime. A projection
over 30 minutes or 4 GiB peak RSS requires renewed resource approval.

## Metrics and Interpretation

Every truth-bearing cell reports canonical one-to-one boundary precision, recall, F1, and matched
MAE, plus segment-count error and posterior entropy. The null cell reports false-discovery rate.
Dense and short-segment cells report missed changes. The shared-boundary cell reports pooled and
independent behavior without treating either fitted result as truth. The logistic cell compares
reachable block support against a high-accuracy reference and reports maximum block error, empirical
posterior TV, its conditional bound, convergence state, and MAP overlap.

Cell-wise summaries use generated datasets as the uncertainty unit and report means or rates,
standard errors, and 95% t intervals. No global superiority test is planned. Reversed, null,
failed, nonconverged, and timed-out outcomes remain in the result.

## Abort Rules

- Do not overwrite any populated result directory.
- Execute only from a committed, hashable code/configuration revision.
- Reject NaN or positive-infinite required scores; record the cell as failed rather than filtering it.
- Preserve every warning, exception, nonconvergence state, and timeout.
- Do not call agreement with a fitted partition external truth.
- Do not run the full suite until the pilot record is reviewed and explicitly approved.

## Remaining Approval Boundary

The one-dataset-per-cell pilot executed from commit
`88eb9daddc81249379635f031327fc2e39fb22d6` in 77.02 seconds at 157 MB peak RSS.
It projects 64.2 minutes for the 50-repetition suite, exceeding the predeclared 30-minute renewed
approval threshold. EP consumed 75.73 seconds of the pilot and returned empirical posterior TV
0.505; the prior-conflict cell failed explicitly because no feasible segment count had finite
evidence. A 20-second EP timeout is now predeclared and approved for a second pilot only. Full
execution remains unapproved pending the bounded re-pilot's resource and semantic review.
