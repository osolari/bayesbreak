# BayesBreak: authoritative outstanding work

**Working state:** The current paper is the single-column manuscript with completed editorial pass and E00-E10 planned result slots. The book is complete as a source document and compiles, but its detailed scientific synchronization to the latest paper is not finished. Existing experimental numbers have not been regenerated under the proposed E00-E10 protocols.

## Ordered scientific and implementation tasks

- [ ] Correct production Gaussian known-precision and Poisson exposure likelihoods, including posterior-predictive semantics. Keep arbitrary likelihood powers, if needed, as a separately named generalized-Bayes option; do not describe such scores as ordinary marginal likelihoods.
- [ ] Support explicit observation masks and well-defined all-missing blocks. Reject invalid scalar and array descriptors.
- [ ] Verify analytic integrals, posterior moments, partition normalizers, forward/backward recursions, boundary/event marginals, and MAP backtracking against independent small-n oracles (E00).
- [ ] Correct or narrow nonconjugate approximation claims: JJ tangent gap, Laplace remainder, EP evidence normalization, PG/JJ aliasing, numerical quadrature convergence, and uniform-block error conditions (E06).
- [ ] Execute E01-E10 from fresh verified data/configuration sources following the frozen protocol and seed scheme; generate independent uncertainty estimates, method failures and error denominators.
- [ ] Regenerate the weighted CGH inference and all comparisons depending on it. Expand segment-count limits and sensitivity for CGH and methylation. Preserve only the currently supportable interpretations in `RESULT_POLICY.md` until new runs pass.
- [ ] Independently check baseline tuning, coordinate axes and external references. Do not turn agreement with the BayesBreak MAP segmentation into external truth.
- [ ] Generate figures/tables only from completed run manifests; update the paper's Planned slots after checking source hashes, scores, and all required cells.
- [ ] Audit book chapters against the finalized model and paper proofs; synchronize the experiment narrative and verified figures/tables. No duplicate scientific run for the book.
- [ ] Run full package tests, tutorial notebooks, dataset-loader checks, compiled-PDF tests and visual QA. Record actual passed, skipped, failed and blocked checks. Do not invent historical hash baselines.

## Release acceptance

Paper and book must compile from `docs/manuscripts/`; the paper must remain single-column with proofs immediately below all theorem-like statements and no em dashes in prose. Present statistical statements in the language of the published Bayesian changepoint, product-partition, conjugate-exponential-family and numerical-integration literature. No stale plan/history directory, duplicate source, speculative figure value or unsupported guarantee may be introduced.
