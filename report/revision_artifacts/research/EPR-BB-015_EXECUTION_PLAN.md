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
indices. Each EP fit runs in a subprocess with a predeclared 20-second fit-only timeout that starts
after worker setup; timeout frequency is a scientific failure metric and timed-out runs remain in
the result. Full execution requires a reviewed pilot and explicit approval.

The pilot must report projected wall time, peak RSS, output size, and per-cell runtime. A projection
over 30 minutes or 4 GiB peak RSS requires renewed resource approval.

## Metrics and Interpretation

Every truth-bearing cell reports canonical one-to-one boundary precision, recall, F1, and matched
MAE, plus segment-count error and posterior entropy. Complete recovery requires an exact one-to-one
match with no missed or extra boundaries. The null cell reports false-positive dataset rate and
false-boundary count. Dense and short-segment cells report missed-change counts and rates. The shared-boundary cell reports pooled and
independent behavior without treating either fitted result as truth. The logistic cell compares
reachable block support against a high-accuracy reference and reports maximum block error, empirical
posterior TV, its conditional bound, convergence state, and MAP overlap.

Cell-wise summaries use generated datasets as the uncertainty unit. Continuous summaries report
means, standard errors, and 95% t intervals; binary rates report 95% Wilson score intervals. No
global superiority test is planned. Reversed, null, failed, nonconverged, and timed-out outcomes
remain in the result.

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

The bounded re-pilot executed from commit `236ec8c509224688cc6e29d855a2df182a869ca7`
in 21.28 seconds at 154 MB peak RSS. EP timed out at 20.01 seconds and the timeout is retained.
All non-EP scientific outputs matched the first pilot. The revised full-suite projection is
17.7 minutes, below the 30-minute resource threshold. Full execution remains a separate explicit
approval boundary. The 400-run full suite was explicitly approved after review of the bounded
re-pilot; execution remains pending until the approved-status revision is committed.

The approved full attempt was stopped before writing `results.json` after an independent semantic
audit identified three invalidating defects: unsupported prior counts poisoned feasible posterior
counts through `-inf - -inf`, shared and independent methods were compared against different truth
targets, and the EP timeout included reference fitting rather than only the EP fit. No full result
was retained. Approval is revoked until the corrected DP, metric semantics, provenance fields, and
EP-fit-only timeout pass a new bounded pilot.

An intermediate pilot from commit `5bcb82eb44054042b33f9b140caa20205bbfa762` is retained as
`pilot-semantic-corrected.json` but excluded from scientific interpretation. Its audit showed that
complete recovery did not penalize extra boundaries, binary rates used degenerate t intervals, and
the EP subprocess deadline still included worker startup. The final redesign requires exact
one-to-one recovery, Wilson score intervals for binary rates, and a worker-ready/start handshake
that starts the deadline only after setup.

The valid bounded pilot executed from commit `d03a06323d0e4ca5afb9409dc18c716e5a5c5c56`
in 22.13 seconds at 160.17 MB peak RSS. Its projected 400-run cost is 18.44 minutes, below
the 30-minute and 4-GiB renewed-approval thresholds. Machine audit certified complete input
hashes, same-subject truth pairing, Wilson intervals, exact-recovery semantics, and a retained
20.01-second EP fit-only timeout. The corrected 400-run suite is now explicitly approved; execution
remains pending until this machine-readable approval revision is committed.
