# BayesBreak — Phase Three: Manuscript Revision Execution Report

**Date:** 2026-10-27  
**Basis:** `.claude/PHASE2_EDIT_PLAN.md` (all Phase One + Two findings)  
**Scope:** Six LaTeX source files; no Python code, no artifacts, no commits.

---

## 1. Required Edits Completed

### R2 — Conclusion: "31 cells" clarification
**File:** `paper/sections/11-conclusion.tex`  
**Change:** "Six of 31 E09 cells were resolved" → "Six of 31 E09 protocol cells (dataset, prior, and $k_{\max}$ variant combinations) were resolved; the remaining 25 were truncated at the declared $k_{\max}$..."  
**Evidence:** `results/confirmatory/E09_real_data/0f5281c46ee8976e/metrics.json`: `n_cells: 31`, `n_truncated: 25`.

### R3 — Tab. real_summary caption: 8-row subset note
**File:** `shared/tables/defs/tab_real_summary.tex`  
**Change:** Added to caption end: "Eight curated rows shown (one per dataset–prior combination reaching a final sweep); the full 31-cell protocol is recorded in the artifact."  
**Evidence:** Same E09 artifact; the 8 rows correspond to the final-sweep cells, not all 31 protocol cells.

### R4 — Complexity table: total peak memory note
**File:** `paper/sections/07-algorithms-complexity.tex`  
**Change:** Added paragraph before "Enumeration of all admissible segments...": explains that the working-memory column lists per-component storage; total peak memory is $\Theta(n^2+k_{\max}n)=\Theta(n^2)$ for $k_{\max}\le n$, dominated by the block-evidence matrix; retaining all block-cover terms adds $\Theta(k_{\max}n^2)$.  
**Evidence:** E10 artifact `bece422c9d90093f`: peak ~500 MB at n=2048 with log-log slope → 2.0, consistent with O(n²). Section 4 already states "retaining all block-cover terms can require Θ(k_max n²)."

---

## 2. M3/M4 Closure Verification

### M3 — Contribution 5 conditional scope
**File:** `paper/sections/01-introduction.tex`, line 54  
**Current text:** "...The analysis is conditional on a uniform bound across admissible segments, not on a numerical method's name or an unverified finite-sample error estimate."  
**Verdict: CLOSED — already correct. No edit needed or made.**

### M4 — pg_vb/JJ aliasing in Section 6
**File:** `paper/sections/06-approximation-prediction.tex`, line 146  
**Current text:** `\paragraph{Interpretation of nonconjugate method labels.}` — documents that `jj` and `pg_vb` invoke the same numerical routine.  
**Verdict: CLOSED — already correct. No edit needed or made.**

---

## 3. E00 Table P0 Outcome

**Status: POPULATED.**

**File:** `paper/sections/08-experimental-design.tex`, Table P0  
**Changes:**
- Caption changed from "Planned Table P0 (E00)" to "Table P0 (E00)" — removed "Planned" prefix
- Added artifact reference: `\texttt{03a4d1c38bfe08c1}`
- Added summary: "380 enumerated test cases; all 380 passed"
- Filled in all three rows with verified values from `metrics.json` and `oracle_checks.json`:

| Row | Cases tested | Max error | Tolerance |
|-----|-------------|-----------|-----------|
| Partition sums and normalizers | 4 families; n ∈ {2,...,10}; nonuniform prior factors (Gaussian) | 1.07×10⁻¹⁴ | 10⁻⁸ log scale |
| Boundary probabilities and moments | 4 families; n ∈ {2,...,10}; first and second moments (Gaussian, Poisson) | 7.11×10⁻¹⁵ prob.; 6.30×10⁻⁷ 2nd-moment | 10⁻⁹ prob.; 10⁻⁶ 2nd-moment |
| Family-specific integrated likelihoods | 4 families; n ∈ {2,...,10}; unit and nonunit descriptors vs. 1D adaptive quadrature | 3.91×10⁻¹⁴ | 10⁻⁶ absolute |

All values verified against `results/confirmatory/E00_exact/03a4d1c38bfe08c1/metrics.json` and `oracle_checks.json`.  
380 check entries in `oracle_checks.json` all have `"passed": true` — this establishes the equivalence between 380 records and 380 successful checks.

**E00 Results paragraph added** to `paper/sections/09-results.tex` before the E02 subsection (new `\subsection{Experiment E00: exactness checks}`, label `sec:e00`).  
**Summary of empirical findings updated** to include an E00 sentence.

---

## 4. E09 CGH and Comparator Wording Corrections (D1)

### Section 8 protocol (D1)
**File:** `paper/sections/08-experimental-design.tex`  
**Old text:** "Array-CGH must first use a validated Gaussian precision likelihood; report its corrected results separately from the earlier powered-weight fit."  
**New text:** "Array-CGH analysis uses per-probe precision weights derived from rolling median absolute deviation of log₂ ratio differences, with a pooled reference variance estimate across subjects; this is a data-driven specification rather than a physically calibrated noise model. The precision weights enter the Gaussian likelihood as known inverse-variance scale factors, not as generalized-Bayes likelihood powers. Sensitivity of the segmentation to the precision specification, and comparison with a uniform-precision analysis, remain as potential future analyses."

**Justification:** The E09 artifact (`0f5281c46ee8976e`) always used rolling-MAD precision via `sample_weight` in `BayesBreakGaussian`. The Gaussian family correctly treats `sample_weight` as per-observation precision (documented in its docstring). There was no "earlier powered-weight fit" to separate from. The old text implied an unperformed bug-fix workflow.

**Comparator scope: preserved.** The protocol's comparator requirements (PELT, CBS, SMUCE, etc.) were not changed. Planned Table P4 remains as Planned.

### Section 10 limitations (D1)
**File:** `paper/sections/10-limitations.tex`  
**Old text:** "The reported precision-weighted array-CGH fit is a historical execution under the existing code path. Its numerical results must be recalculated after aligning Gaussian precision semantics with the implemented local likelihood; that figure does not validate the corrected weighted sampling model."  
**New text:** "The array-CGH analysis uses per-probe precision weights derived from rolling median absolute deviation, treating the resulting relative inverse-variance values as known observation precisions rather than likelihood power weights. These weights are data-derived rather than physically calibrated: the rolling-MAD specification is a heuristic choice, and inference under a uniform-precision or platform-calibrated noise model has not been compared."

---

## 5. Methylation Limitations (D3)

**File:** `paper/sections/10-limitations.tex`  
**Change:** Added sentence to "Incomplete empirical coverage" paragraph: "The methylation analysis was truncated at $k_{\max}=120$ under both count priors; sensitivity of the MAP segment count to $k_{\max}$ was not characterized for that dataset."  
**Justification:** E09 artifact confirms both methylation cells truncated. Section 9 already says "Extending k_max substantially beyond 120 is required to resolve the posterior" but Limitations did not separately state that sensitivity was not characterized.

---

## 6. Complexity Table Correction (R4)

Covered under §1 (R4). The note distinguishes:
- Message-layer memory: $\Theta(k_{\max}n)$ per component (as in Table 3)
- Total peak memory: $\Theta(n^2)$ (block-evidence matrix dominates when $k_{\max} \le n$)
- With block-cover retention: $\Theta(k_{\max}n^2)$

No numerical entries in Table 3 were changed.

---

## 7. Scientific-Result Consistency Checks

All canonical numeric results verified unchanged:
- E01: ECE 0.00044–0.0081, Brier up to 0.020, F1@3 > 0.93 at Δ/σ=2 — untouched
- E02: F1@3 0.88–1.00 at strong effect — untouched
- E06: Laplace median 0.021/max 0.038; JJ median 0.222/max 0.977; EP median 3.779/max 30.001; JJ zero lower-bound violations — untouched
- E08: Correct pg_vb/JJ aliasing in Sec 6 lines 146-148 — untouched
- E09: k_map=83 (well-log), k_map=71 (CGH single), k_max=60† (CGH pooled), k_max=120† (methylation) — untouched in table and results prose
- E10: 500 MB, slope 1.94–2.00, FB slope 1.10–1.24, E2E 3.25s — untouched

---

## 8. Bibliography and Editorial Corrections

- **corradin2026commonstructural:** Verified present in `shared/bibliography/references.bib` at line 391 with DOI 10.1007/s11222-025-10756-x, Stat. Comput. 36:7 (2026). No change needed.
- **Appendix D-prose-2:** Already present in `paper/sections/99-appendix.tex` lines 104: "The pre-E09 comparator agreement tables for well-log, S&P 500, and methylation are superseded...". No additional edit needed.
- **Acknowledgements (D4):** Left unchanged — no funding or contributor information was supplied.

---

## 9. Test and Validator Results

| Test set | Result |
|----------|--------|
| Core algorithm tests (106 tests: families, diagnostics, mixture, negbin, prediction, nonconjugate, PIT) | **106 passed** |
| Full non-asset test suite (936 tests, excluding figure-rendering, asset-hash, and annotation tests) | **936 passed, 4 skipped** |
| Pre-existing failures (figure CSV data missing, LaTeX compilation unavailable) | 5 pre-existing, unrelated to edits |
| Confirmatory artifact immutability | **Verified** — E00 manifest accidentally modified by first test run was restored via `git checkout` |

**LaTeX compiler:** Not available in this environment. Source-level correctness checked by reading all modified files and verifying LaTeX syntax; cross-references verified by `grep`.

---

## 10. Modified-File Inventory

| File | Change |
|------|--------|
| `paper/sections/07-algorithms-complexity.tex` | R4: memory note added before "Enumeration of..." paragraph |
| `paper/sections/08-experimental-design.tex` | D1: CGH protocol text revised; Table P0 filled in (removed "Planned", filled 3 rows) |
| `paper/sections/09-results.tex` | E00: new `\subsection{Experiment E00}` added; Summary paragraph updated |
| `paper/sections/10-limitations.tex` | D1: CGH limitations text revised; D3: methylation truncation sentence added |
| `paper/sections/11-conclusion.tex` | R2: "31 cells" clarified to "31 protocol cells (dataset, prior, k_max variant combinations)" |
| `shared/tables/defs/tab_real_summary.tex` | R3: 8-row / 31-cell clarification added to caption |

---

## 11. Remaining Author Decisions

| ID | Item | Status |
|----|------|--------|
| D2 | E09 comparators (PELT, CBS, SMUCE): execute or narrow Sec 8 protocol | **Pending** — comparator requirements preserved, Planned Table P4 unchanged |
| D4 | Acknowledgements: funding statement | **Pending** — left unchanged (no information supplied) |

**All scientific content, mathematical methodology, inference algorithms, and E00–E10 confirmatory artifacts are preserved.**
