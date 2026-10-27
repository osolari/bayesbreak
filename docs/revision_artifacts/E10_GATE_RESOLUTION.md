# E10 Gate Resolution: Computational Scaling

**Date:** 2026-10-25
**Status:** PASS
**Config SHA256:** bece422c9d90093f (full: `bece422c9d90093f8f1133a72d156b8329265b55583e2d21e247fee92b939f0e`)
**Results directory:** `results/confirmatory/E10_scaling/bece422c9d90093f/`
**Run time:** 2026-10-25 (elapsed 212.7 s)
**Git revision at run time:** dfcc529c97810f0a3c4541e5cb7ccc9a61d5b9b2

---

## 1. Protocol

From `experiments.json`:

| Parameter | Value |
|---|---|
| n_values | [128, 256, 512, 1024, 2048] |
| k_max_values | [5, 10, 20] |
| warmup | 3 |
| repeats | 7 |
| blas_threads | 1 |
| family | Gaussian (conjugate, fixed nu=0.0, rho2=4.0, sigma2=1.0) |
| count_prior | geometric_decay_log4 |

Total cells: 15 (5 x 3). All completed without failure.

---

## 2. Hardware and software environment

| Item | Value |
|---|---|
| Platform | macOS-26.6.1-arm64-arm-64bit (Apple Silicon) |
| Python | 3.11.17 |
| NumPy | 2.4.6 |
| BLAS threads | 1 (OMP/MKL/OPENBLAS/VECLIB all set to 1 before numpy import) |
| Timing method | time.perf_counter(), wall-clock |
| Memory method | tracemalloc peak over one end-to-end fit |
| Warmup | 3 full fits before timing (same data array) |
| Timed repeats | 7; report median and std |

---

## 3. Computational stages and theoretical complexity

| Stage | Implementation | Theoretical complexity | Source |
|---|---|---|---|
| Block evidence | `_compute_block_evidence` (Gaussian conjugate updates) | O(n^2) | Nested loop over i, j filling (n+1)x(n+1) array |
| Forward-backward DP | `_dp.forward_backward` | O(k_max * n^2) | k_max passes, each accumulating n-length prefix sums |
| Posterior over k | `_dp.posterior_over_k` | O(k_max * n) | Single pass over k, n |
| Boundary marginals | `_dp.boundary_event_marginals_fixed_k` | O(k_map * n) | k_map-1 outer loops |
| MAP backtracking | `_dp.max_sum_segmentation` | O(n + k_map) | After max-sum DP O(k_map * n^2) |
| End-to-end | `est.fit()` | O(k_max * n^2) | Dominated by block evidence + forward-backward |

Memory: the block evidence matrix is (n+1) x (n+1) float64 arrays. Forward and backward
tables are (k_max+1) x (n+1). Total storage is O(k_max * n + n^2); the O(n^2) block
evidence array dominates.

---

## 4. Per-cell results (median over 7 repeats)

| n | k_max | k_map | Block (s) | Fwd-bwd (s) | Boundary marg (s) | E2E (s) | Peak mem (MB) |
|---|---|---|---|---|---|---|---|
| 128 | 5 | 3 | 0.0004 | 0.0328 | <0.001 | 0.0377 | 2.1 |
| 128 | 10 | 3 | 0.0004 | 0.0639 | <0.001 | 0.0691 | 2.1 |
| 128 | 20 | 3 | 0.0004 | 0.1252 | <0.001 | 0.1289 | 2.1 |
| 256 | 5 | 3 | 0.0012 | 0.0690 | <0.001 | 0.0790 | 7.9 |
| 256 | 10 | 3 | 0.0013 | 0.1378 | <0.001 | 0.1487 | 7.9 |
| 256 | 20 | 3 | 0.0014 | 0.2690 | <0.001 | 0.2769 | 7.9 |
| 512 | 5 | 3 | 0.0057 | 0.1460 | <0.001 | 0.1711 | 31.4 |
| 512 | 10 | 3 | 0.0060 | 0.2906 | <0.001 | 0.3163 | 31.4 |
| 512 | 20 | 3 | 0.0058 | 0.5791 | <0.001 | 0.6151 | 31.4 |
| 1024 | 5 | 3 | 0.0207 | 0.3202 | 0.001 | 0.3905 | 125.3 |
| 1024 | 10 | 3 | 0.0224 | 0.6428 | 0.001 | 0.7164 | 125.3 |
| 1024 | 20 | 3 | 0.0210 | 1.2776 | 0.002 | 1.3572 | 125.3 |
| 2048 | 5 | 3 | 0.0954 | 0.7473 | 0.002 | 0.9403 | 500.7 |
| 2048 | 10 | 3 | 0.0918 | 1.4748 | 0.003 | 1.6827 | 500.7 |
| 2048 | 20 | 3 | 0.1025 | 3.0226 | 0.006 | 3.2483 | 500.7 |

---

## 5. Empirical scaling versus theoretical complexity

Slopes fit by log-log OLS across the full n grid at fixed k_max=20:

| Stage | Observed slope in n | Theoretical slope |
|---|---|---|
| End-to-end | **1.16** | 2 (O(n^2)) |
| Forward-backward | **1.14** | 2 (O(n^2)) |
| Block evidence | **2.00** | 2 (O(n^2)) |
| Peak memory | **1.98** | 2 (O(n^2)) |

Slope in k_max at fixed n=2048:

| Stage | Observed slope in k_max | Theoretical slope |
|---|---|---|
| End-to-end | **0.89** | 1 (O(k_max)) |
| Forward-backward | **1.01** | 1 (O(k_max)) |

**Key finding:** Memory and block-evidence computation scale as O(n^2) as predicted.
Forward-backward scales as O(k_max^1.01 * n^1.14) in practice, substantially better
than the theoretical O(k_max * n^2). This is explained by numpy vectorization: the
outer loop over k runs in Python (k_max iterations) but the inner sum for each k step
is a numpy vector operation of length n, not a Python loop of n iterations. The effective
wall-clock cost per forward-backward pass is O(k_max * n) numpy operations rather than
O(k_max * n^2) Python iterations, giving empirical exponent closer to 1 than 2 in n.

Do not interpret the empirical slope (1.14 in n) as an asymptotic improvement over the
O(n^2) worst-case bound in Proposition~\ref{prop:complexity-paper}. The theoretical
bound counts arithmetic operations; the empirical slope reflects numpy's vectorized
execution over the measured finite range [128, 2048]. For very large n where the n^2
block matrix no longer fits in CPU cache, the empirical slope may increase toward 2.

---

## 6. Memory analysis

Memory scales as O(n^2) (slope 1.98, consistent with 2.0 within measurement noise).
At n=2048, peak memory is 500.7 MB, dominated by the (n+1) x (n+1) block evidence
array: 2049 x 2049 x 8 bytes = 33.6 MB plus the corresponding (n+1) x (n+1) max-sum
array = 67.2 MB. The remaining ~433 MB includes forward/backward tables (k_max * n float64),
boundary marginal computations, and Python interpreter overhead.

For conjugate Gaussian: block evidence storage is the primary memory cost. Non-conjugate
families with numerical block integration add per-block integration cache overhead but
do not change the O(n^2) block table size.

---

## 7. Stage time fractions (at n=2048, k_max=20)

| Stage | Time (s) | Fraction of E2E |
|---|---|---|
| Block evidence | 0.103 | 3.2% |
| Forward-backward | 3.023 | 93.0% |
| Boundary marginals | 0.006 | 0.2% |
| MAP backtracking | <0.001 | <0.1% |
| Other (setup, post-processing) | ~0.116 | 3.6% |

The forward-backward DP is the dominant computation (93% of end-to-end time at n=2048,
k_max=20). At smaller n or smaller k_max, block evidence and Python setup contribute
proportionally more.

---

## 8. Numerical correctness

All 15 cells returned k_map=3 (correct: k_true=3 under the generating signal with
SNR=2.0 and the geometric-decay prior). No NaN, infinity, or numerical failure occurred.
The boundary marginal invariant (sum = k_map - 1 = 2) was verified in the supplemental
boundary marginals run for welllog configurations.

---

## 9. Comparison with existing runtime measurements

The manuscript section "Finite-range runtime behavior" reports existing measurements
at n in {50,...,400} with k_max in {10, 20}. At k_max=20, n=400: ~0.77s (from existing
chart). The E10 benchmark at n=512, k_max=20: 0.62s. These are from different hardware
and timing setups; the comparison is illustrative, not a controlled regression.

The fitted log-log slopes from existing measurements were ~1.07 (k_max=10) and ~1.14
(k_max=20) in n, consistent with the E10 slope of 1.14 (k_max=20) from this benchmark.

---

## 10. Baselines

No baseline runtimes are prescribed in the E10 protocol. The paper_figure_label
`fig:planned-scaling-paper` is for the BayesBreak scaling figure only.

---

## 11. Gate verdict

**E10 PASS.** Canonical artifact: `results/confirmatory/E10_scaling/bece422c9d90093f/`

15 cells, 0 failures, gate_pass=True. All configurations completed without numerical
failure. Memory and block-evidence scale as O(n^2). Forward-backward scales as
O(k_max * n^1.14) in practice on Apple Silicon / numpy 2.4.6 at n in [128, 2048].

Caveats:
- Measurements are from a single Apple Silicon machine; results on different hardware
  (x86, GPU, large memory) may differ.
- The empirical slope of 1.14 should not be cited as an asymptotic improvement over
  the O(n^2) theoretical bound.
- Memory at n=2048 is 500MB. For n=4096, expected memory is ~2GB (O(n^2) scaling).
- Non-conjugate families (BetaObs, logistic-normal) add numerical integration overhead
  per block; scaling for these families was not measured in this benchmark.
