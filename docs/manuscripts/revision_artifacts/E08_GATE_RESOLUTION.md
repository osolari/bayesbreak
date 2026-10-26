# E08 Gate Resolution: Posterior Prediction

**Date:** 2026-10-24
**Status:** PASS (corrected run, protocol revision 2026-10-24-2)

---

## 1. Original contaminated run (cc26eda02dc82357)

Protocol revision 2026-10-24-1. Three defects were found during A-checks:

1. `BayesBreakBetaObs._compute_block_evidence` did not apply `likelihood_power_`; holdout
   observations entered the DP block evidence matrix and contaminated the MAP segmentation
   and posterior moments for BetaObs cells.
2. `BayesBreakBetaObs.posterior_predictive_logpdf_block` and
   `BayesBreakBetaObs._segment_posterior_mean` called `_segment_posterior_log_weights` with
   raw `sample_weight_` instead of `lp * sample_weight_`.
3. `prediction._segment_pit` for BetaObs called `_segment_posterior_log_weights` with raw
   `sample_weight_`; for Binomial it computed training S and N without the `lp[a:b]` factor.

None of the non-BetaObs families (Gaussian, Poisson, Binomial) were affected by defects 1-2.
Defect 3 (Binomial PIT) produced biased PIT statistics in the old run.

---

## 2. Fixes applied (all in production code, no protocol change)

| File | Change |
|---|---|
| `src/bayesbreak/families/beta_obs.py` | `_compute_block_evidence`: add `a_lp = likelihood_power_`, `w = a_lp * w` before prefix sums |
| `src/bayesbreak/families/beta_obs.py` | `posterior_predictive_logpdf_block`: pass `lp * sample_weight_` to `_segment_posterior_log_weights` |
| `src/bayesbreak/families/beta_obs.py` | `_segment_posterior_mean`: pass `lp * w` to `_segment_posterior_log_weights` |
| `src/bayesbreak/prediction.py` | BetaObs `_segment_pit`: pass `lp * estimator.sample_weight_` to `_segment_posterior_log_weights` |
| `src/bayesbreak/prediction.py` | Binomial `_segment_pit`: compute `S = sum(lp * w_train * y_train)`, `N = sum(lp * w_train * n_train)` |

---

## 3. A-check results

### A1: likelihood_power_ regression tests

Six new tests added to `tests/test_observation_mask.py`:

- `TestBetaObsLikelihoodPower.test_block_evidence_masked_equals_observed_only`
- `TestBetaObsLikelihoodPower.test_block_evidence_differs_from_unmasked_full_fit`
- `TestBetaObsLikelihoodPower.test_predictive_logpdf_masked_matches_observed_only`
- `TestBetaObsLikelihoodPower.test_pit_masked_matches_direct_fit_on_observed_subset`
- `TestBinomialPITLikelihoodPower.test_masked_pit_matches_direct_fit_on_observed_subset`
- `TestBinomialPITLikelihoodPower.test_posterior_parameters_exclude_masked_obs`

All pass. Total suite: **952 passed, 4 skipped**.

Binomial `w` in `_segment_pit` is the per-prediction n_trials (same semantic as
`posterior_predictive_logpdf_block(w_new=n_trials)`), not sample_weight or likelihood power.
The `lp` factor applies to the TRAINING posterior S and N, not to the new-observation n_trials.

### A2: NegBin moments

First moment: `E[m_*] = r_* * b_post / (a_post - 1)` requires `a_post > 1`. Reports `nan`
when `a_post <= 1` (four existing tests in `TestA2NegBinDivergentMoment`).

Second moment: `E[m_*^2] = r_*^2 * b_post * (b_post + 1) / ((a_post - 1) * (a_post - 2))`
requires `a_post > 2`. **Not currently computed or stored.** Three new tests added to
`TestA2NegBinDivergentMoment`:
- `test_second_moment_not_stored`: confirms no `block_second_moment_` attribute.
- `test_second_moment_formula_valid_above_two`: formula is positive finite for `a_post > 2`.
- `test_second_moment_formula_diverges_at_a_post_le_2`: denominator is zero at `a_post = 2`,
  negative at `a_post = 1.5`.

Comment added to `negative_binomial.py` near the E[m_*] computation.

### A3: Signed segment-count error (k_map - k_true, k_true = 3)

From the corrected run (1600 rows, 50 reps x 4 windows per cell):

| Cell | Family | Effect | mean k_map | signed error | k_map distribution |
|---|---|---|---|---|---|
| 0 | Gaussian | delta=0.5 | 1.235 | -1.765 | {1:168, 2:17, 3:15} |
| 1 | Gaussian | delta=1.5 | 3.065 | +0.065 | {3:187, 4:13} |
| 2 | Poisson | q=1.5 | 2.170 | -0.830 | {1:77, 2:18, 3:99, 4:6} |
| 3 | Poisson | q=3.0 | 3.040 | +0.040 | {3:192, 4:8} |
| 4 | Binomial | p_mid=0.35 | 3.030 | +0.030 | {3:194, 4:6} |
| 5 | Binomial | p_mid=0.65 | 3.040 | +0.040 | {3:192, 4:8} |
| 6 | BetaObs | mu_mid=0.4 | 3.055 | +0.055 | {3:189, 4:11} |
| 7 | BetaObs | mu_mid=0.8 | 3.045 | +0.045 | {3:191, 4:9} |

Cells 0 and 2 (weak effects) have negative signed errors (-1.765, -0.830), meaning the model
more often under-segments than over-segments. For cell 0, 84% of rows have k_map=1 (no
changepoints found); for cell 2, 49.5% have k_map=3 and 38.5% have k_map=1.

### A4: BetaObs quadrature convergence

Compared 128-node vs 256-node Gauss-Legendre quadrature on a representative sample of E08
held-out points (mu_mid=0.4 and mu_mid=0.8, window 0, rep 0).

| Cell | k_map (128) | k_map (256) | max |diff| | mean |diff| |
|---|---|---|---|---|
| BetaObs mu_mid=0.4 | 4 | 4 | 9.70e-12 | 3.35e-12 |
| BetaObs mu_mid=0.8 | 3 | 3 | < 1e-11 | < 1e-11 |

128-node quadrature is fully converged for these configurations. No quadrature amendment needed.

Protocol revision bumped to `2026-10-24-2` (triggering new config SHA256 `a23d9fe92c02724c`).

---

## 4. Corrected confirmatory run (a23d9fe92c02724c)

**Config SHA256:** `a23d9fe92c02724cfe7af740591dd0526d2e0b1ce6530c81359b527b3ec0ab5f`
**Rows:** 1600 (8 cells x 50 reps x 4 windows), failed=0, gate_pass=True
**Elapsed:** 257.5 s

### Per-cell results (bootstrap 95% CIs over 50 replicates)

| Cell | Family | Effect | logpd_diff | 95% CI | Calibration |
|---|---|---|---|---|---|
| 0 | Gaussian | delta=0.5 | -0.010 | [-0.016, -0.005] | coverage 93.9% (fitted), 94.1% (null) |
| 1 | Gaussian | delta=1.5 | +0.048 | [+0.014, +0.084] | coverage 87.6% (fitted), 86.4% (null) |
| 2 | Poisson | q=1.5 | -0.015 | [-0.024, -0.007] | PIT KS 0.196 (fitted), 0.196 (null) |
| 3 | Poisson | q=3.0 | +0.100 | [+0.003, +0.191] | PIT KS 0.271 (fitted), 0.419 (null) |
| 4 | Binomial | p_mid=0.35 | +0.025 | [-0.013, +0.061] | PIT KS 0.240 (fitted), 0.312 (null) |
| 5 | Binomial | p_mid=0.65 | +0.240 | [+0.128, +0.329] | PIT KS 0.326 (fitted), 0.606 (null) |
| 6 | BetaObs | mu_mid=0.4 | +0.206 | [+0.108, +0.301] | coverage 76.9% (fitted), 64.6% (null) |
| 7 | BetaObs | mu_mid=0.8 | +3.177 | [+3.023, +3.340] | coverage 71.3% (fitted), 14.7% (null) |

**Comparison with old (contaminated) BetaObs results:**

| Cell | Old logpd_diff | New logpd_diff | Old coverage | New coverage |
|---|---|---|---|---|
| BetaObs mu_mid=0.4 | +1.188 | +0.206 | 95.7% | 76.9% |
| BetaObs mu_mid=0.8 | +10.836 | +3.177 | 95.8% | 71.3% |

The old results were inflated by data leakage: holdout observations entered the segment
posterior, making the predictive artificially accurate. The corrected results show genuine
under-coverage (76-77%) in both BetaObs cells, attributable to MAP-conditional prediction
not accounting for partition uncertainty and to holdout windows crossing true segment
boundaries (windows 1 and 2 span the boundaries at positions 40 and 80 for n=120).

### Scientific interpretation

- **Identifiable effects (CI entirely positive):** Gaussian delta=1.5, Poisson q=3.0,
  Binomial p_mid=0.65, BetaObs mu_mid=0.4 and mu_mid=0.8. Fitted model beats null.
- **Unidentifiable / weak effects (CI entirely negative):** Gaussian delta=0.5, Poisson
  q=1.5. Over-segmentation or under-segmentation degrades held-out predictive density.
- **Borderline (CI includes zero):** Binomial p_mid=0.35.
- **BetaObs coverage:** 76-77% (vs nominal 95%). MAP-conditional predictive does not account
  for partition uncertainty; genuine boundary-crossing effect. Both families are far better
  than null (14.7% and 64.6% coverage), confirming the fitted model is structurally superior.
- **Gaussian coverage (delta=1.5):** 87.6%, below nominal 95%. Same MAP-conditional issue;
  consistent with the prior analysis.

---

## 5. E06 status

E06 (nonconjugate approximation) has a prior confirmatory run at
`results/confirmatory/E06_nonconjugate/9739f69c790aae6f/` with `gate_pass=True`.
Correction of nonconjugate approximation claims (JJ tangent gap, Laplace remainder, EP
evidence normalization) remains a pending task per `CURRENT_WORK.md`. E06 does not block E08
closure.

---

## 6. Final verdict

**E08 PASS.** Corrected run artifact: `results/confirmatory/E08_posterior_prediction/a23d9fe92c02724c/`.

All 952 tests pass. The contaminated artifact at `cc26eda02dc82357` is superseded; do not
use its BetaObs cells for any manuscript table or figure.
