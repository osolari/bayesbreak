# Build and QA report

## Phase 6 final document state

| Target | Pages | Page size | SHA-256 |
|---|---:|---|---|
| Technical book | 168 | A4 | `b1736341ef8a8400c491dda9db15649a0bdc257c234208e13f4670e90adb97ec` |
| Main paper, two-column | 35 | US letter | `de4205f1f0dcda438b834f883beef6cc8eea13e1ecc7b36813a62e23218707e2` |
| Main paper, single-column | 42 | US letter | `b978f68c1b199799659befc6b15b306203febad202ede4fd614307c1f60b8d8b` |
| Executive summary | 12 | US letter | `9db1afb736f91969562cb2308f7162e597ca020f8f2a0b3f60a8831d1a8aefd2` |

## Mechanical and formal checks

All four targets use the exact title **Generalized Hierarchical Bayesian Segmentation with Irregular Designs, Multi-Sequence Hierarchies, and Grouped/Latent-Group Designs**, the SAIM template, embedded fonts, and the required page size. The final logs contain zero undefined references or citations, duplicate labels, missing graphics, missing characters, overfull boxes, or fatal errors. Extracted PDF text contains no unresolved `??` marker or rejected scientific branding checked by the release validator.

The technical book contains 17 chapter source files, five required appendices, 28 established theorem-like statements with 28 immediately following proofs, and one proof obligation. The paper contains 12 established statements with 12 immediately following proofs and one proof obligation.

## Scientific and numerical checks

Thirteen independent finite-case or numerical checks passed. They cover conjugate segment marginal likelihoods, Beta-observation numerical integration, sum-product evidence, max-sum backtracking, ordered-boundary and event marginals, posterior moment curves, shared-boundary pooling, score-error propagation, fixed-count Poisson interval odds, and the latent-group Jensen decomposition.

The bibliography contains 38 unique keys and 38 one-to-one annotations. The synchronized implementation specification contains 16 coding tasks, 15 experiment protocols, 14 claims, and 17 result records. All 53 archived result assets match the Phase 4R hashes.

## Implementation tests

The historical `RES-BB-QA-002` state remains 179 collected, 157 passed, five skipped, and 17 unresolved. The independent bounded rerun `RES-BB-QA-003` records 179 collected, 173 passed, five skipped, one unresolved EP logistic-normal timeout under a 20-second per-test cap, and zero failed.

## Visual inspection

All 257 pages were rendered at 150 dpi and inspected through 22 contact sheets: 168 book pages, 35 two-column paper pages, 42 single-column paper pages, and 12 executive-summary pages. Targeted full-page checks covered the principal corrected formulas and proofs. No clipping, overlap, blank page, broken glyph, unresolved figure frame, or unreadable table was observed.

## Numerical-result integrity

No research experiment was rerun and no populated archived value was changed. `RES-BB-CMP-002` and `RES-BB-RD-007Q` remain recorded as real executions but are excluded from the specific conclusions described in the result registry. The original scientific package implementation was not modified.

## Clean-source rebuild

The clean-source rebuild and rendered-page comparison are recorded in `revision_artifacts/phase6/SOURCE_REBUILD_REGRESSION.md` and are included in the final package.
