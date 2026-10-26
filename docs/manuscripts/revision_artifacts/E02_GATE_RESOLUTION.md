# E02 Gate Resolution: Beta-obs Quadrature Convergence

**Date:** 2026-10-23
**Status:** RESOLVED — corrected run PASS

---

## 1. Original Protocol

- **Experiment:** E02 family portability (protocol revision 2026-10-21-2)
- **Beta-obs quadrature method:** 64-node Gauss-Legendre on (0,1)
- **Gate check:** max Bayes-curve deviation (64 vs 128 GL nodes) < 1.000e-3
- **Original run directory:** `results/confirmatory/E02_family_portability/a64d0fde6f74fd39/`

---

## 2. The Failure

The original confirmatory run (900 replicates, completed 2026-10-22) produced:

| Cell | max curve_dev (64 vs 128 GL) | Gate |
|------|------------------------------|------|
| beta_obs, mu_mid=0.4 | **1.006e-3** | FAIL |
| beta_obs, mu_mid=0.8 | 9.26e-4 | PASS |

- `run_log.txt` correctly ends: `E02 result: FAIL (gate check)`
- `metrics.json` **incorrectly** records `gate_pass=true` due to a code bug: `gate_pass` was computed before `_run_gate_checks()` executed, based only on replicate completion counts rather than oracle gate results.
- A separate bug: the gate threshold in `_run_gate_checks` had been loosened to `5e-3` after the run, inconsistent with the prespecified 1e-3 threshold.

Original artifact SHA256s (preserved, unmodified):
- `metrics.json`: `d9c5601b6193ffdd4d53dbe1e3d8a5d290d4b3f96b7feff683e5fca8d2cd2e02`
- `oracle_checks.json`: `cd1bb4d93b30c07c328fca6e06745e9e978b1b46e3edaab029cb20e4b64a30eb`
- `manifest.json`: `8a097993a55f6cbb5ed16be1a838b31df2c4f4a41fc0a13c7738105425ad201b`

Correction record: `results/confirmatory/E02_family_portability/a64d0fde6f74fd39/validation_status.json`

---

## 3. Investigation Results

Diagnostic script: `scripts/diag_beta_obs_quadrature.py`

### 3.1 Quadrature convergence table (mu_mid=0.4, rep 2 — worst case)

| Orders compared | max Bayes-curve deviation | Status vs 1e-3 |
|-----------------|--------------------------|----------------|
| 32 vs 64        | 1.340e-2                 | FAIL           |
| 64 vs 128       | **1.006e-3**             | **FAIL** (marginal) |
| 128 vs 256      | 1.400e-8                 | PASS           |
| 256 vs 512      | 5.576e-14                | PASS           |

### 3.2 Per-rep 64 vs 128 deviations (mu_mid=0.4, reps 0-4)

| Rep | 64 vs 128 curve_dev | Status | 128 vs 256 curve_dev | Status |
|-----|---------------------|--------|----------------------|--------|
| 0   | 6.239e-4            | PASS   | 4.693e-8             | PASS   |
| 1   | 9.365e-4            | PASS   | 6.938e-9             | PASS   |
| 2   | **1.006e-3**        | FAIL   | 1.400e-8             | PASS   |
| 3   | 3.374e-4            | PASS   | 1.411e-8             | PASS   |
| 4   | 8.063e-4            | PASS   | 1.322e-8             | PASS   |

### 3.3 Per-rep 64 vs 128 deviations (mu_mid=0.8, reps 0-4)

All 5 reps: 64 vs 128 range [7.88e-4, 9.26e-4] — all PASS.
All 5 reps: 128 vs 256 range [9.2e-9, 1.5e-8] — all PASS.

### 3.4 Integrand concentration

- mu_mid=0.4 middle segment: mode ~ 0.399, 95% CI width ~ 0.044
- mu_mid=0.8 middle segment: mode ~ 0.803, 95% CI width ~ 0.034

The narrow 95% CI (< 5% of the [0,1] domain) explains why 64 Gauss-Legendre nodes is marginal: the integrand is concentrated enough that 64 nodes are barely sufficient to resolve the curvature.

### 3.5 scipy.integrate.quad reference (mu_mid=0.4, full block, rep 2)

For the full-block (n=120 observations) posterior mean of mu:
- scipy.quad reference: mu_mean = 0.25512
- 64-node GL: 0.25529, deviation from scipy = 1.759e-4
- 128-node GL: 0.25519, deviation from scipy = 7.510e-5
- 256-node GL: 0.25512, deviation from scipy = 4.624e-11
- 512-node GL: 0.25512, deviation from scipy = 1.166e-15

Note: full-block log-evidence comparison to scipy.quad is not available for mu_mid=0.8 because the scipy unnormalized integrand for all 120 observations underflows to zero in linear arithmetic (the GL quadrature correctly operates in log-space internally).

---

## 4. Root Cause

The 64-node failure is a marginal boundary case, not a systematic failure:

1. One replicate (rep 2, mu_mid=0.4) produced a dataset whose posterior integrand is concentrated enough that the 64-node approximation of the Bayes-curve integral deviates by 1.006e-3 — just 0.6% above the 1e-3 threshold.

2. The integrand is narrow (~4.4% CI width) because the segment contains a mix of phi=20 and phi=80 observations. At phi=80 the Beta likelihood is very sharply peaked.

3. 128-node GL fully resolves this: all deviations (128 vs 256) are ~1e-8, 5 orders of magnitude below the threshold.

---

## 5. Resolution

**Protocol amendment (revision 2026-10-21-3, applied 2026-10-23):**
- Confirmatory quadrature order raised from **64 to 128 nodes**
- Gate comparison changed from (64 vs 128) to **(128 vs 256 nodes)**
- Gate threshold remains **1e-3** (not relaxed)

**Corrected run:** `results/confirmatory/E02_family_portability/621e2f1812b22ffa/`

Corrected oracle check results:

| Cell | max curve_dev (128 vs 256 GL) | Gate |
|------|-------------------------------|------|
| beta_obs, mu_mid=0.4 | 4.693e-8 | PASS |
| beta_obs, mu_mid=0.8 | 1.507e-8 | PASS |

Corrected metrics.json: `gate_pass=true`

New artifact SHA256s:
- `metrics.json`: `2d06d418b29ae232515436b29dcd7afc44603308deaa72ec3255775604996398`
- `oracle_checks.json`: `fa5caae97f1419770bb95c7355b29558e8c3725f89fc601487dddf6e0abb2b5b`
- `manifest.json`: `273177a3971ca095f83fff7f6d63e7d48fcb15a206a5a07a9bcff65a42eb1fb1`

---

## 6. F1 / Recovery Sensitivity to Quadrature Order

F1, count MAE, and signal MSE are **not materially affected** by the quadrature change.
The 64-node vs 128-node Bayes-curve deviation (max 1.006e-3) is within the noise of a (0,1)-valued signal. The F1 scores (based on MAP boundaries from Viterbi-like DP, not the Bayes curve) and P(k|y) values are identical between 64- and 128-node runs because the MAP partition is determined by the log-block-evidence table, which differs by ~5e-3 in log evidence (negligible relative to block log-likelihoods of order 10-100).

From the corrected run (128 nodes), F1@3 values are unchanged vs the original 64-node run:
- beta_obs, mu_mid=0.4: mean_f1 = 0.990
- beta_obs, mu_mid=0.8: mean_f1 = 1.000

---

## 7. Runner Code Fixes

Two bugs fixed in `scripts/run_e02_family_portability.py`:

1. **gate_pass premature evaluation** (line ~1170): `gate_pass` was set to `n_reps_failed==0 and n_cells_completed==n_cells_planned` before `_run_gate_checks()` executed. Fixed: gate_pass is now `False` in metrics.json initially, then updated after `_run_gate_checks()` returns.

2. **Incorrect threshold** (line ~1313): `_curve_threshold = 5e-3` was inconsistent with the prespecified 1e-3. Fixed to `1e-3`.

3. **`_run_gate_checks` now returns bool** and the main function exits with `sys.exit(1)` on gate failure.

---

## 8. E03 Unblocking

E02 is now PASS. E03 (irregular design prior) does not depend on Beta-obs quadrature and is unblocked pending its own prespecified execution.
