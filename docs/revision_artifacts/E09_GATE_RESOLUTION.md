# E09 Gate Resolution: Real-Data Analysis

**Date:** 2026-10-24
**Protocol revision:** 2026-10-24-2
**Config SHA256:** 0f5281c46ee8976e (full: `0f5281c46ee8976e7dcaf194319df529797efb051f215079cea12fe9510b7be9`)
**Results directory:** `results/confirmatory/E09_real_data/0f5281c46ee8976e/`
**Run time:** 2026-10-25 05:52:36 to 06:18:53 UTC (1576.9 s)
**Git revision at run time:** dfcc529c97810f0a3c4541e5cb7ccc9a61d5b9b2

**Status legend:**
- Execution: PASS (31 cells, 0 failures, gate_pass=True in manifest)
- Numerical: INCOMPLETE (25/31 cells truncated; CGH pooled and methylation segment-count posteriors unresolved)
- Scientific completeness: INCOMPLETE (seven comparators unavailable; unresolved posteriors preclude several stated E09 objectives)

Do not interpret gate_pass=True as confirmation that the scientific evaluation described in the E09 protocol is complete. It records that the runner executed every prescribed configuration without exception. See `E09_SCIENTIFIC_STATUS.md` for a full completeness accounting.

---

## 1. Dataset provenance

All datasets loaded via `src/bayesbreak/datasets/`. SHA256 hashes recorded at load time.

| Dataset | Source | n | Data SHA256 |
|---|---|---|---|
| welllog (full) | downloaded | 4050 | a3cbf113784463d0bdca4b9696c7aa239b66a9836a6fc104806fcf4e2631386f |
| welllog (stride-8) | derived from welllog full (every 8th point) | 507 | 85cf633323487550b78cdaa2160189aa22d90d2db5a97e83512c4b45250164e2 |
| CGH | downloaded | 2215 probes x 43 subjects | 5de2d3ba07e341c89e810da566f2770b325399e6d40d24831bd57e7d0167af21 |
| methylation | downloaded | 1904 | 39337da5e43c5d014045e683da7f590a3d72d1b94623955eb8832118e60650e4 |
| SPX | simulated fallback | -- | EXCLUDED |

SPX was excluded: `yfinance` is not installed and `load_spx()` returns a simulated series.
Per protocol: never label simulated data as genuine observational data.

---

## 2. Model parameters

**Robust noise estimate.** `sigma2` for Gaussian fits estimated via median-absolute-deviation
of first differences: `sigma = 1.4826 * median(|diff(y) - median(diff(y))|) / sqrt(2)`.

| Dataset | sigma2 (robust MAD-diff) |
|---|---|
| welllog full | 4,674,808.19 |
| welllog stride-8 | 5,466,698.12 |
| CGH (pooled across 43 subjects) | 0.0057 |

**Hyperparameter strategy.**

- welllog: `estimate_hyper=True` with fixed `sigma2`. BayesBreakGaussian estimates `nu`
  (weighted mean of y) and `rho2` (variance of y) from data. Required because welllog
  values are ~10^5 NMR units; a fixed prior of `nu=0, rho2=4.0` places the prior far
  from the data, collapsing evidence to k_map=1 (see Section 3).
- CGH: fixed `nu=0.0, rho2=4.0`. Appropriate for log2-ratio data (neutral copy number
  at 0, typical CNV excursions up to ±2). Shared prior across 43 subjects.
- methylation: BetaObs family; hyperparameters estimated by the family from data.

**Count priors.** `geometric_decay_log4` and `uniform` tested on all primary fits.

**BetaObs quadrature.** 128 Gauss-Legendre nodes (confirmed sufficient in E08 A4).

**CGH pooling.** `SharedBoundaryReplicatesSegmenter` pools 43 subjects: block log-evidences
from each subject are summed before the DP; one shared MAP boundary vector is returned.

**k_max grid.** Primary: {15, 30, 60}. Extended for truncation resolution:
- welllog full: up to k_max=100
- welllog stride-8: k_max=60
- CGH single_subject0: up to k_max=100
- methylation: up to k_max=120 (truncation not resolved)

---

## 3. Fixes applied before the canonical run

Two correctness defects found and fixed during script development. Both affected the
superseded run (config SHA256 prefix `bae816ac60300df3`).

### Fix 1: `_min_len_prior` returned 0.0 for valid segments

`base.py` checks `if value > 0 and np.isfinite(value)`. Returning `0.0` caused valid
segments to receive `-inf` in the log-g table. All minimum-length sensitivity cells
failed. Fix: return `1.0` for valid segments (log(1.0) = 0.0, no additive penalty),
`0.0` for invalid segments (fails `> 0` check, assigned -inf).

### Fix 2: Wrong Gaussian prior for welllog

The superseded run used `nu=0, rho2=4.0` for welllog. Welllog NMR values are ~10^5;
the prior was far from the data. All welllog cells returned k_map=1. Fix: `estimate_hyper=True`
for welllog, letting the model estimate nu and rho2 from data while fixing sigma2 to
the robust MAD-diff estimate.

---

## 4. Truncation criterion

A cell is reported truncated if `k_map == k_max` OR `P(k=k_max|y) >= 0.01`. Truncation
means substantial posterior mass lies at or above k_max; the unrestricted MAP segment
count is unknown. Do not extrapolate a MAP value from a truncated posterior.

---

## 5. Per-dataset results

### Welllog (full, n=4050)

| k_max | count_prior | k_map | P(k=k_max\|y) | truncated | log_evidence | elapsed (s) |
|---|---|---|---|---|---|---|
| 15 | geometric_decay_log4 | 15 | 1.000000 | yes | -40029.6 | 6.8 |
| 30 | geometric_decay_log4 | 30 | 1.000000 | yes | -38386.6 | 12.8 |
| 60 | geometric_decay_log4 | 60 | 0.99849 | yes | -37958.5 | 24.7 |
| 100 (extended) | geometric_decay_log4 | **83** | 3.2e-6 | **no** | -37906.1 | 39.6 |
| 15 | uniform | 15 | 1.000000 | yes | -40012.6 | 6.7 |
| 30 | uniform | 30 | 1.000000 | yes | -38349.5 | 12.7 |
| 60 | uniform | 60 | 0.99962 | yes | -37880.5 | 24.7 |
| 100 (extended) | uniform | 100 | 0.34437 | **yes** | -37784.7 | 40.9 |

k_map=83 means 83 segments and 82 internal changepoints. The geometric-decay prior
resolves at k_max=100; the uniform prior does not resolve at k_max=100 (P(k=100|y)=0.344).
Do not report a MAP k for the welllog uniform-prior configuration.

### Welllog (stride-8 subsample, n=507)

| k_max | count_prior | k_map | P(k=k_max\|y) | truncated | elapsed (s) |
|---|---|---|---|---|---|
| 15 | geometric_decay_log4 | 15 | 1.000000 | yes | 0.47 |
| 30 | geometric_decay_log4 | 25 | 0.02122 | yes | 0.92 |
| 60 | geometric_decay_log4 | **25** | 1.78e-23 | **no** | 1.71 |
| 15 | uniform | 15 | 1.000000 | yes | 0.47 |
| 30 | uniform | 30 | 0.53093 | yes | 0.92 |
| 60 | uniform | **37** | 1.27e-5 | **no** | 1.75 |

The stride-8 analysis returns fewer segments than the full-data analysis (25 vs 83 under
geometric-decay). This records the model's behavior on the subsampled series. It does not
establish that a quantified fraction of the full-data segment structure is absent from the
subsampled series.

### Welllog minimum segment length sensitivity (geometric_decay_log4, k_max=100)

| min_seg_len | k_map | P(k=k_max\|y) | truncated | log_evidence | elapsed (s) |
|---|---|---|---|---|---|
| 1 (primary) | 83 | 3.2e-6 | no | -37906.1 | 39.6 |
| 5 | 61 | 3.9e-23 | no | -38163.8 | 61.7 |
| 10 | 56 | 2.8e-34 | no | -38854.7 | 60.9 |

Enforcing minimum segment length reduces the MAP segment count. The log-evidence values
are not directly comparable as Bayes factors between models with different minimum-length
constraints unless the partition prior normalizer is verified to be the same under each
constraint. See A2 note on prior normalization.

### CGH (pooled, 2215 probes, 43 subjects)

| k_max | k_map | P(k=k_max\|y) | truncated | log_evidence | elapsed (s) |
|---|---|---|---|---|---|
| 15 | 15 | 1.000000 | yes | -36584.0 | 372.2 |
| 30 | 30 | 1.000000 | yes | 9067.2 | 381.8 |
| 60 | 60 | 1.000000 | yes | 41222.5 | 390.1 |

All pooled CGH cells truncated. No extended run was executed for the pooled model: k_max=60
took 390s; extending to k_max=100 was projected to exceed the 600s per-cell budget.

The unrestricted MAP segment count for the pooled CGH model is unknown from this run.
k > 60 is supported by all three fits; the actual MAP k cannot be determined.

MAP boundary vectors are recorded in `metrics.json` for each k_max. These are the MAP
boundaries under the stated truncated model, not boundaries under the unrestricted model.
Do not interpret the specific boundary positions at k_max=15 as a resolution of the
underlying changepoint structure without noting that the model is constrained to exactly
15 segments.

Do not compare the pooled (truncated, k > 60 unresolved) and single-subject analyses
(resolved at k_map=71) as if they are directly comparable results; the two fits involve
different k ranges and different model support.

### CGH single_subject0 (representative, 2215 probes)

| k_max | k_map | P(k=k_max\|y) | truncated | log_evidence | elapsed (s) |
|---|---|---|---|---|---|
| 15 | 15 | 1.000000 | yes | 880.4 | 2.8 |
| 30 | 30 | 1.000000 | yes | 1488.1 | 5.5 |
| 60 | 60 | 0.65695 | yes | 1693.5 | 10.8 |
| 100 (extended) | **71** | 3.8e-16 | **no** | 1700.4 | 17.1 |

k_map=71 for subject 0 (71 segments, 70 internal changepoints). Subject 0 is representative
but not a summary of all 43 subjects. No genomic coordinate conversion was performed;
results are in probe-index coordinates only.

### Methylation (n=1904, BetaObs)

| k_max | count_prior | k_map | P(k=k_max\|y) | truncated | log_evidence | elapsed (s) |
|---|---|---|---|---|---|---|
| 15 | geometric_decay_log4 | 15 | 1.000000 | yes | -9495.7 | 6.0 |
| 30 | geometric_decay_log4 | 30 | 1.000000 | yes | -4549.8 | 8.2 |
| 60 | geometric_decay_log4 | 60 | 1.000000 | yes | -1047.2 | 12.7 |
| 120 (extended) | geometric_decay_log4 | 120 | ~1.000000 | yes | 1453.8 | 20.8 |
| 15 | uniform | 15 | 1.000000 | yes | -9478.8 | 6.2 |
| 30 | uniform | 30 | 1.000000 | yes | -4512.7 | 8.7 |
| 60 | uniform | 60 | 1.000000 | yes | -969.2 | 12.9 |
| 120 (extended) | uniform | 120 | ~1.000000 | yes | 1614.2 | 21.0 |

Methylation is persistently truncated at all tested k_max up to 120. P(k=120|y) ≈ 1.0
under both priors. The unrestricted MAP segment count is unknown. Do not state or imply
that the true number of methylation segments exceeds 120; the posterior simply has not
been resolved up to 120. Log-evidence increases monotonically with k_max, indicating the
data consistently favor more segments within the tested range.

---

## 6. Boundary marginals

The BayesBreak DP provides exact boundary-event marginals (`boundary_marginals_`) and
boundary location posteriors (`boundary_location_posterior_`) after fitting, conditional
on the MAP segment count. No Markov chain Monte Carlo or variational approximation is
required. These are stored as fitted attributes and can be queried for any resolved
configuration.

For truncated configurations, boundary marginals conditional on k_map are available but
should be interpreted with caution: they condition on a segment count that may not be the
unrestricted MAP, and they reflect the boundary distribution within the constrained model
support. These are not marginals under the unrestricted model.

Detailed boundary-marginal summaries for the resolved configurations (welllog full
geometric_decay_log4 k_max=100, welllog stride-8, welllog min_seg sensitivity, CGH
single_subject0) are reported in `E09_SCIENTIFIC_STATUS.md`.

---

## 7. Baselines availability

The E09 runner script hard-coded all baseline comparators as unavailable without testing
for the packages at runtime. Post-hoc investigation revealed:

| Baseline family | Package | Availability |
|---|---|---|
| PELT, OP, BS, WBS | ruptures v1.1.10 | Available in the `bayesbreak` conda environment |
| CBS (DNAcopy), SMUCE, Fearnhead | rpy2 + R | NOT available (no R runtime installed) |

ruptures was not used in the canonical E09 run because the script did not perform a
dynamic import check. A supplemental baseline analysis using ruptures can be executed
in the bayesbreak conda environment. See `E09_SCIENTIFIC_STATUS.md` for the plan.

---

## 8. SPX exclusion

`load_spx()` returned a simulated fallback (yfinance not installed). Per protocol and
standing constraint: never label simulated data as genuine observational data. SPX is
excluded from E09. It is not counted in any result table.

---

## 9. Scientific findings from resolved configurations

### Welllog (primary: geometric_decay_log4, k_max=100)

The full welllog series (n=4050) has MAP segment count 83 (82 internal changepoints) under
the geometric-decay prior. The stride-8 subsampled series (n=507) has MAP segment count
25 (24 internal changepoints). These are the model's segment-count estimates under the
respective observation sequences; no claim is made about structural information content
or about which changepoints from the full series are absent in the subsampled result.

Minimum segment length sensitivity: enforcing min_seg=5 gives k_map=61; min_seg=10 gives
k_map=56. Both are resolved. The corresponding log-evidence values (-38163.8 and -38854.7
vs -37906.1 for min_seg=1) reflect the model families with and without the segment-length
constraint; they are not directly comparable as Bayes factors unless the partition prior
normalizer is verified to be the same under each constraint.

### CGH

The pooled MAP segment count is not resolved at any tested k_max. The single-subject
analysis of subject 0 gives k_map=71 (70 internal changepoints) at k_max=100. No external
genomic annotation verification was performed; all boundary positions are in probe-index
coordinates. The subject 0 result is not a summary for all 43 subjects.

### Methylation

The MAP segment count is not resolved at any tested k_max up to 120. Candidate drivers
of the truncation include: high per-site read-count precision (max 818 reads), large number
of CpG sites (n=1904), and the BetaObs model's ability to detect small differences under
high precision. These are candidate explanations, not verified diagnoses.

---

## 10. Superseded run

Config SHA256 prefix `bae816ac60300df3` (same session, 2026-10-24), gate_pass=False.
Two defects: wrong welllog priors (k_map=1 for all welllog cells) and `_min_len_prior`
bug (all min_seg>1 cells failed). Do not use for any scientific analysis.

---

## 11. Gate verdict

**E09 PASS (execution only).** Canonical artifact:
`results/confirmatory/E09_real_data/0f5281c46ee8976e/`

31 cells executed, 0 failed, 25 truncated. gate_pass=True records that the runner
completed every configuration. It does not record scientific completeness.

Scientific-completeness status: INCOMPLETE. Unresolved items:
- Methylation posterior mode unknown (truncated at k_max=120).
- CGH pooled posterior mode unknown (truncated at k_max=60).
- Welllog uniform prior not resolved at k_max=100.
- Seven planned comparators: ruptures baselines can be run as a supplemental analysis;
  R-based baselines (CBS, SMUCE, Fearnhead) are not available.
- No genomic annotation verification for CGH boundary positions.
- Partition prior normalization under minimum-segment-length constraints requires verification
  before log-evidence differences are interpreted as model-comparison quantities.

These items are tracked in `E09_SCIENTIFIC_STATUS.md`.
