# BayesBreak Phase 6 artifact summary

## Final scientific identity

**Generalized Hierarchical Bayesian Segmentation with Irregular Designs, Multi-Sequence Hierarchies, and Grouped/Latent-Group Designs**

The final release preserves the original research direction: BayesBreak is a generalized hierarchical Bayesian method for multiple-changepoint segmentation, with exponential-family segment models, irregular designs, multiple related sequences, known groups, and finite latent groups treated as central parts of the method.

## Phase 6 scientific verification

Phase 6 independently rechecked the central formulas and recursions and made nine thesis-preserving corrections:

1. admissible partition support is defined independently of observed segment likelihood values;
2. exposure-dependent reporting uses a declared reference descriptor, `m_star(theta)=m(theta;d_star)`;
3. unsupported universal quadrature monotonicity and routine-wide rates were removed;
4. sum-product posterior inference and max-sum joint-MAP recovery are separate results;
5. storage and output complexity statements were corrected;
6. additive and Hamming decision rules now state their loss assumptions;
7. fixed-count Poisson interval occupancy uses local odds `exp(Lambda_j)-1`;
8. the latent-group Jensen criterion uses the implementation-scale group weight, including `n_g log gamma_u(i,j)`;
9. nonconjugate probability perturbation and changepoint matching are stated with explicit bounds and one-to-one matching rules.

All 13 independent finite-case or numerical checks passed. The book has 28 established theorem-like statements with 28 immediately following proofs and one proof obligation; the paper has 12 established statements with 12 immediate proofs and one proof obligation.

## Compiled documents

| Document | Pages | Format | SHA-256 |
|---|---:|---|---|
| Technical book | 168 | A4 | `b1736341ef8a8400c491dda9db15649a0bdc257c234208e13f4670e90adb97ec` |
| Main journal paper, two-column | 35 | US letter | `de4205f1f0dcda438b834f883beef6cc8eea13e1ecc7b36813a62e23218707e2` |
| Main journal paper, single-column | 42 | US letter | `b978f68c1b199799659befc6b15b306203febad202ede4fd614307c1f60b8d8b` |
| Executive summary | 12 | US letter | `9db1afb736f91969562cb2308f7162e597ca020f8f2a0b3f60a8831d1a8aefd2` |

The total release contains **257 compiled pages**. All pages were rendered at 150 dpi and inspected through 22 contact sheets. A clean extraction rebuilt all four documents, passed the Phase 6 validator, and produced pixel-identical content on all 257 pages at 72 dpi.

## Bibliography, source, and implementation records

- 38 unique bibliography keys, 38 annotation files, and 38 manifest entries form a one-to-one set.
- 16 coding tasks, 15 experiment protocols, 14 claim records, and 17 result records are synchronized from the canonical implementation specification.
- The repository archive is explicitly incomplete and does not claim that the planned scientific routines are implemented.
- The presentation package contains 15 handoff files, exactly two approved executive diagrams, and no slides.
- The clean source archive contains 320 files and excludes build products, caches, operating-system metadata, and font files.

## Real numerical results

All 53 archived numerical-result assets match the corrective Phase 4R SHA-256 baseline. No research experiment was rerun and no populated archived numerical value was changed.

`RES-BB-CMP-002` remains a real historical execution but is excluded from comparator conclusions because the compared objects use incompatible coordinate axes. `RES-BB-RD-007Q` remains a real historical execution but is excluded from posterior-predictive conclusions because the computation used the wrong observation-family predictive routine and an implicit endpoint rule.

## Implementation-test verification

The immutable historical record `RES-BB-QA-002` remains 179 collected, 157 passed, five skipped, and 17 unresolved. The Phase 6 bounded rerun `RES-BB-QA-003` records 179 collected, 173 passed, five skipped, one unresolved EP timeout under a 20-second per-test cap, and zero failed.

## Explicitly incomplete items

- One routine-specific nonconjugate approximation-rate statement remains a proof obligation.
- The EP logistic-normal comparison did not complete under the Phase 6 per-test cap.
- The previously identified latent-group returned-objective/restart-selection defect was not repaired because scientific package-code changes were outside this manuscript phase.
- Corrected array-CGH comparator and methylation posterior-predictive computations were not executed.
- Independent external changepoint annotations remain unavailable for the applied studies.
- The final journal venue, permanent repository, and data-release locations remain unset.
