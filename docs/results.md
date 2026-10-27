# Results and current evidence scope

All E00–E10 confirmatory experiments have been executed and passed their
execution gate (`gate_pass=True`). E00–E08 and E10 are scientifically
complete. E09 (real-data analysis) passed its execution gate but has
incomplete scientific evaluation; the limitations are documented explicitly
below.

Completed figures and tables are the outputs of specific models and datasets
under the stated priors and configurations. They are not universal validation
of the framework.

## E00: Exactness checks

Exhaustive partition enumeration against analytic reference values.
Four exponential families (Gaussian, Poisson, Binomial, NegBin), sequence
lengths $n \in \{2,\ldots,10\}$, nonuniform prior factors, and moments.

**380 oracle checks — all 380 passed** (artifact `03a4d1c38bfe08c1`).

| Check category | Max error | Tolerance |
|---|---|---|
| Partition sums and normalizers | $1.07 \times 10^{-14}$ | $10^{-8}$ (log scale) |
| Boundary probabilities and first moments | $7.11 \times 10^{-15}$ | $10^{-9}$ |
| Second-moment Bayes curves (Gaussian) | $6.30 \times 10^{-7}$ | $10^{-6}$ |
| Integrated likelihoods vs. 1-D adaptive quadrature | $3.91 \times 10^{-14}$ | $10^{-6}$ (absolute) |

## E01–E02: Gaussian recovery and cross-family portability

ECE 0.00044–0.0081, Brier ≤ 0.020, F1@3 > 0.93 at Δ/σ = 2 (E01).
F1@3 0.88–1.00 at strong effect across Gaussian, Poisson, Binomial,
NegBin (E02). Calibration target: conditional boundary-event marginals
`P(b_i = 1 | y, k_map)`.

## E03–E05: Irregular designs, shared-boundary pooling, latent templates

E03 verifies that gap-aware partition priors correctly down-weight
short candidate segments on irregular grids. E04 confirms exact
shared-boundary posterior pooling for multi-subject sequences. E05
validates the finite-template EM objective and restart-ranking rule.

## E06: Nonconjugate approximation accuracy

Comparison of Laplace, Jaakkola–Jordan, Pólya–Gamma mean field (PG-VB),
expectation propagation, and 1-D Gauss–Hermite quadrature against
analytic Gaussian reference scores (1800 blocks).

**Key finding:** `pg_vb` and `jj` invoke the same JJ numerical routine —
their results are identical by construction. Laplace: median 0.021, max 0.038.
JJ: median 0.222, max 0.977. EP: median 3.779, max 30.001. JJ has zero
lower-bound violations. EP does not satisfy the uniform error bound required
for Theorem 1 in the block-error regime sampled.

## E07: Model misspecification

Misspecification grid confirming that family mismatches degrade calibration
in interpretable ways. Findings document family-specific failure modes rather
than a general robustness claim.

## E08: Posterior prediction

Held-out log-likelihood evaluation confirming that the posterior-predictive
layer correctly dispatches to family-specific support rules. PIT uniformity
verified for conjugate families with closed-form CDFs.

## E09: Real-data analysis (execution complete; scientific completeness incomplete)

The E09 protocol defines **31 cells** (dataset × prior × $k_{\max}$ variant
combinations). Six cells were resolved; 25 were truncated at their declared
$k_{\max}$ and the unrestricted MAP segment counts for those cells are unknown.
The eight rows in Table R (real-data summary) are the one per dataset–prior
combination that reached a final sweep; the full 31-cell protocol is recorded
in artifact `0f5281c46ee8976e`.

| Dataset | Result | Scientific status |
|---|---|---|
| Well-log NMR (n=4050) | $\widehat{k}=23$ | Resolved |
| Array-CGH single (n=2215) | $\widehat{k}=71$ | Resolved |
| Array-CGH pooled (S=43) | Truncated at $k_{\max}=60$ | Unresolved |
| CpG methylation (n=1904) | Truncated at $k_{\max}=120$ (both count priors) | Unresolved |
| S&P 500 volatility | Excluded from authentic real-data experiment | Simulated analog only |

**CGH precision model.** The single-subject CGH analysis uses per-probe
precision weights derived from rolling median absolute deviation (MAD) of
log₂ ratio differences, treating the resulting relative inverse-variance
values as known observation precisions. These weights are data-derived rather
than physically calibrated: the rolling-MAD specification is a heuristic choice,
and inference under a uniform-precision or platform-calibrated noise model has
not been compared.

**Methylation truncation.** Both methylation cells were truncated at
$k_{\max}=120$; sensitivity of the MAP segment count to $k_{\max}$ was not
characterized for that dataset.

**Baselines.** R-based comparators (CBS, SMUCE) are unavailable in this
environment. Python-based comparators via `ruptures` are available but the
planned cross-comparator study (Table P4) has not been executed.

## E10: Computational scaling

Memory and timing measurements at $n \in \{128, 256, 512, 1024, 2048\}$.
Peak memory ≈ 500 MB at $n=2048$; log-log slope 1.94–2.00 consistent with
$\mathcal{O}(n^2)$ dominated by the block-evidence matrix. Forward-backward
slope 1.10–1.24 in practice. End-to-end 3.25 s at $n=2048$, $k_{\max}=20$.
Artifact `bece422c9d90093f`.

## Outstanding limitations

The following limitations apply to the current experimental evidence:

- E09 CGH pooled and methylation MAP counts are unknown (truncated).
- R-based comparator study (CBS, SMUCE) has not been executed.
- Sensitivity of methylation segmentation to $k_{\max}$ was not characterized.
- The rolling-MAD CGH precision model is a heuristic; no comparison against
  uniform-precision or calibrated alternatives has been performed.
- SPX (S&P 500) was excluded from the authentic real-data experiment; reported
  SPX figures use a simulated GARCH-like analog.
- External changepoint annotations are unavailable for all four datasets;
  reported MAP boundaries are not validated against independent truth.

These limitations are stated in full in the Limitations section of the paper
(`docs/manuscripts/paper/sections/10-limitations.tex`).
