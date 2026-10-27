# E09 Scientific Status

**Date:** 2026-10-25
**Canonical execution artifact:** `results/confirmatory/E09_real_data/0f5281c46ee8976e/`
**Gate resolution:** `E09_GATE_RESOLUTION.md`

This document records the scientific-completeness status of E09 separately from the
execution gate (gate_pass=True). It tracks which E09 scientific objectives are met,
which are incomplete, and what work remains.

---

## Status summary

| Objective | Status | Notes |
|---|---|---|
| Welllog MAP segmentation (geometric prior) | COMPLETE | k_map=83 at k_max=100, resolved |
| Welllog stride-8 analysis | COMPLETE | k_map=25/37 at k_max=60, resolved |
| Welllog min-seg sensitivity | COMPLETE | k_map=61/56 at k_max=100, resolved |
| CGH single-subject MAP segmentation | COMPLETE | k_map=71 at k_max=100, resolved |
| CGH pooled MAP segmentation | INCOMPLETE | Truncated at k_max=60; mode unknown |
| Methylation MAP segmentation | INCOMPLETE | Truncated at k_max=120; mode unknown |
| SPX analysis | EXCLUDED | Simulated fallback; no real data available |
| ruptures baselines (PELT, OP, BS, WBS) | INCOMPLETE | Package available; supplemental run needed |
| R baselines (CBS, SMUCE, Fearnhead) | UNAVAILABLE | No R runtime installed |
| Boundary marginals for resolved configs | PENDING | Computationally available via DP |
| CGH genomic annotation verification | NOT PERFORMED | Would require external database |
| Partition prior normalization check | NOT PERFORMED | Required for min-seg log-evidence comparison |

---

## Detailed status by objective

### 1. Welllog full (n=4050) -- COMPLETE under geometric-decay prior

k_map=83 (83 segments, 82 internal changepoints), resolved at k_max=100.
P(k=100|y) = 3.2e-6 under geometric_decay_log4 prior.

**Not resolved:** uniform prior configuration (k_map=100, P=0.344, still truncated at k_max=100).
Do not report a MAP k for welllog under the uniform prior.

### 2. Welllog stride-8 (n=507) -- COMPLETE

k_map=25 under geometric-decay, k_map=37 under uniform; both resolved at k_max=60.

### 3. Welllog minimum-segment-length sensitivity -- COMPLETE (numerical)

k_map=61 (min_seg=5) and k_map=56 (min_seg=10); both resolved.

**Unresolved:** Whether the log-evidence differences (-38163.8 vs -37906.1 for min_seg=5/1)
can be interpreted as Bayes factors depends on whether the partition prior normalizer is
the same under each constraint. This requires verification. Until verified, report these
as evidence values for different model families, not as Bayes factors between them.

### 4. CGH single-subject analysis (subject 0) -- COMPLETE

k_map=71 (71 segments, 70 internal changepoints), resolved at k_max=100.
Subject 0 is representative but not a summary of all 43 subjects.
Results are in probe-index coordinates; no genomic coordinate conversion performed.

### 5. CGH pooled analysis (43 subjects) -- INCOMPLETE

All three k_max values (15, 30, 60) truncated. MAP segment count unknown; k > 60 certain.
Extending to k_max=100 for the pooled model is estimated to require >600s and was not run.

**Next steps to resolve:** Run with k_max=100 and budget_seconds_per_cell >= 900, or
run on a machine where the computation is faster. This is a new run, not a rerun of the
existing protocol (different k_max would change the config SHA256).

### 6. Methylation -- INCOMPLETE

Persistent truncation at k_max=120 for both priors; P(k=120|y) ≈ 1.0.
Log-evidence increases monotonically with k_max, confirming strong data support for
additional segments. The unrestricted MAP k is unknown.

Do not state or imply that the true segment count exceeds 120. The posterior simply
has not been resolved within the tested range.

Candidate drivers of truncation (not verified diagnoses):
- High per-site read-count precision (max observed coverage: 818 reads per site)
- n=1904 CpG sites provides many potential boundaries
- BetaObs model sensitivity to small methylation fraction differences under high precision

**Next steps to resolve:** Extend k_max to 200+ (requires a new, amended run). This is
out of scope for the current E09 confirmatory protocol.

### 7. SPX -- EXCLUDED

yfinance not installed; load_spx() returns simulated data. Excluded per protocol.
Slot remains unavailable until yfinance is installed and a real data source is verified.

### 8. ruptures baselines (PELT, OP, BS, WBS) -- INCOMPLETE; supplemental run feasible

ruptures v1.1.10 is installed in the `bayesbreak` conda environment but was not
dynamically checked by the E09 runner script. The script hardcoded all comparators as
unavailable. A supplemental baseline analysis using ruptures can be executed.

**Methodological constraints for a valid supplemental run:**
- Use ruptures v1.1.10 with documented hyperparameters.
- PELT requires a penalty parameter; OP requires a cost model and k specification.
- Do not select penalty parameters to maximize agreement with BayesBreak.
- Use the same probe-index coordinate axis as BayesBreak.
- Score agreement with the BayesBreak MAP partition as a diagnostic, not as accuracy.
- Do not equate ruptures agreement with ground truth.
- Label all comparisons as agreement diagnostics.

**Status:** Not yet executed. Supplemental script to be written as
`scripts/supplemental_e09_ruptures_baselines.py`.

### 9. R-based baselines (CBS/DNAcopy, SMUCE, Fearnhead) -- UNAVAILABLE

No R runtime available in the execution environment (`Rscript` not found). rpy2 not
installed. These comparators cannot be run until R and DNAcopy/SMUCE/Fearnhead packages
are installed.

### 10. Boundary marginals for resolved configurations -- PENDING

The BayesBreak DP computes exact boundary-event marginals and boundary location posteriors
after fitting, via `boundary_marginals_` and `boundary_location_posterior_` attributes.
No MCMC is required.

These are conditioned on k_map (the MAP segment count). For resolved configurations, this
is the MAP k under the full unrestricted model. For truncated configurations, boundary
marginals conditioned on a truncated MAP k are available but should be interpreted
accordingly.

**Resolved configurations for which boundary marginals can be computed:**
- welllog full, geometric_decay_log4, k_max=100 (k_map=83)
- welllog stride-8, geometric_decay_log4, k_max=60 (k_map=25)
- welllog stride-8, uniform, k_max=60 (k_map=37)
- welllog min_seg=5, k_max=100 (k_map=61)
- welllog min_seg=10, k_max=100 (k_map=56)
- CGH single_subject0, k_max=100 (k_map=71)

**Status:** Not yet computed from the canonical run. Refitting with the same parameters
and extracting attributes is required (the canonical run did not serialize boundary marginals
beyond the MAP boundaries stored in metrics.json).

### 11. CGH genomic annotation verification -- NOT PERFORMED

CGH boundary positions are in probe-index coordinates (0-indexed positions in the 2215-probe
array). No conversion to genomic coordinates (chromosome, base position) was performed.
No external CNV annotation database was consulted.

This means no claim about chromosome-arm CNV events, recurrent cancer CNV patterns, or
agreement with published CGH annotations can be made from this run.

### 12. Partition prior normalization under minimum-segment-length constraints -- NOT PERFORMED

When min_seg_len > 1, the set of valid partitions changes. The partition prior normalizer
C_k = |{valid partitions of size k}| changes accordingly. Whether the geometric-decay
count prior is properly normalized under each constraint (so that log-evidence values
reflect comparable models) requires verification before log-evidence differences are
reported as Bayes factors.

---

## Frozen confirmatory artifact

The canonical run `results/confirmatory/E09_real_data/0f5281c46ee8976e/` is immutable.
`manifest.json` and `metrics.json` must not be modified. Supplemental analyses must be
written to new output directories and clearly distinguished from the confirmatory artifact.

---

## Work remaining before E09 scientific objectives are complete

1. CGH pooled: extend k_max (requires new run with budget amendment).
2. Methylation: extend k_max to 200+ (requires new run, outside current protocol scope).
3. ruptures baselines: write and run `scripts/supplemental_e09_ruptures_baselines.py`.
4. Boundary marginals: refit resolved configurations and serialize `boundary_marginals_`.
5. Partition prior normalization: verify C_k under min-seg constraints.
6. CGH genomic annotation: requires external database (out of scope for current run).
7. R-based baselines: requires R runtime installation (environment constraint).
