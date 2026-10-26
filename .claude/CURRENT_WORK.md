# BayesBreak: authoritative outstanding work

**Working state (2026-10-25):** All E00-E10 confirmatory experiments have been executed and verified. Implementation corrections are complete. Outstanding work is limited to E06 methodological audit, E09 scientific-completeness gaps, and manuscript integration of confirmed results.

## Experiment execution status

| Experiment | Artifact SHA (first 16) | gate_pass | Scientific status |
|---|---|---|---|
| E00 exact oracle | `03a4d1c38bfe08c1` | True | COMPLETE -- 380/380 oracle checks, max error 1.07e-14 |
| E01 Gaussian calibration | `1e84a2673bc5b214` | True | COMPLETE |
| E02 family portability | `621e2f1812b22ffa` | True | COMPLETE |
| E03 irregular design | `e0d11782dcbd95ff` | True | COMPLETE |
| E04 shared boundary | `0d135a601bdf32b1` | True | COMPLETE |
| E05 latent templates | `d33f1c5e067d38ff` | True | COMPLETE |
| E06 nonconjugate | `9739f69c790aae6f` | True | COMPLETE -- pg_vb/jj aliasing documented in manuscript; no independent PG-VB benchmarked |
| E07 misspecification | `a038ac8e7e4a17b9` | True | COMPLETE |
| E08 posterior prediction | `a23d9fe92c02724c` | True | COMPLETE (supersedes `cc26eda02dc82357`) |
| E09 real data | `0f5281c46ee8976e` | True | INCOMPLETE -- see below |
| E10 scaling | `bece422c9d90093f` | True | COMPLETE |

Supplemental results:
- `results/supplemental/E09_ruptures_baselines/1aa2500bdfe56fb9/` -- ruptures v1.1.10 PELT/Dynp/BinSeg/BottomUp on welllog and CGH single_subject0.
- `results/supplemental/E09_boundary_marginals/dc8432589c3fa528/` -- exact boundary-event marginals for 6 resolved E09 configurations; DP invariant verified (sum = k_map - 1 for all).

## Outstanding implementation tasks

None currently blocking manuscript.

## Outstanding scientific tasks

### E09 scientific-completeness gaps

- [ ] **Methylation posterior mode**: Persistently truncated at k_max=120, P(k=120|y)~1.0. True MAP k unknown, likely well above 120. Must report as truncated; do not extrapolate.
- [ ] **CGH pooled posterior mode**: Truncated at k_max=60. True MAP k unknown, k>60 certain. Cannot report MAP segments for pooled CGH.
- [ ] **R baselines (CBS, SMUCE, Fearnhead)**: No R runtime available. These comparisons remain UNAVAILABLE. Do not substitute BayesBreak boundaries as ground truth for any baseline comparison.
- [ ] **CGH genomic annotation**: No probe-coordinate-to-chromosome-arm mapping has been performed. Do not report arm-level CNV inferences without this annotation.
- [ ] **Partition prior normalization under min-seg constraints**: The log C_k normalization constant must be verified to be comparable across different min_seg values before interpreting log-evidence differences as Bayes factors.

### Manuscript integration

- [x] **Paper result slots**: E06, E07, E08, E09, E10 all integrated with canonical artifact numbers. Duplicate labels resolved. Stale pre-E09 real-data section replaced. Appendix agreement tables (pre-E09, superseded) replaced with provenance note.
- [x] **E09 manuscript text**: k_map=83 welllog reported correctly. Truncated fits marked with criterion and unknown-MAP note. CGH arm-level annotation explicitly not performed. SPX excluded per protocol.
- [ ] **Book synchronization**: Audit book chapters against finalized model and paper proofs. No duplicate scientific run for the book.

## Completed implementation work (for reference)

- `src/bayesbreak/families/beta_obs.py`: Removed `likelihood_power_` that was scaling marginal log-likelihoods; BetaObs now computes ordinary marginal likelihoods.
- `src/bayesbreak/families/binomial.py`: Same `likelihood_power_` correction.
- `src/bayesbreak/base.py` (`_min_len_prior`): Fixed to return 1.0 for valid segments (log(1.0)=0.0) and 0.0 for invalid (fails `value > 0` check, assigned -inf). Previously returned 0.0 for valid segments causing all cells to fail.
- Observation masks: explicit support for all-missing blocks with well-defined likelihoods.
- CGH weighted segmentation: `sample_weight` correctly passed per-probe per-subject.

## Release acceptance

Paper and book must compile from `docs/manuscripts/`; the paper must remain single-column with proofs immediately below all theorem-like statements and no em dashes in prose. Present statistical statements in the language of the published Bayesian changepoint, product-partition, conjugate-exponential-family and numerical-integration literature. No stale plan/history directory, duplicate source, speculative figure value or unsupported guarantee may be introduced.
