# BayesBreak — Phase One Scientific Orientation Report

**Prepared:** 2026-10-26  
**Repository state:** `cde463e` (main, clean)  
**Manuscript:** `docs/manuscripts/bayesbreak-paper/paper/main.tex`  
**Scope:** Inspection only. No edits, no experiment runs, no commits.

---

## 1. Manuscript Identity and Preserved Methodological Commitments

The paper's single maintained identity is a generalized hierarchical Bayesian method for offline multiple-changepoint segmentation of ordered data on regular or irregular designs. Its statistical core is:

- Conjugate exponential-family segment integration via cumulative sufficient statistics (Theorem 1)
- Sum-product forward-backward recursions for the count posterior and boundary marginals (Theorem 2)
- A separate max-sum recursion for joint MAP backtracking (Proposition in Sec. 4)
- Design-dependent local prior factors (Theorem 3)
- Exact shared-boundary pooling across conditionally independent sequences (Theorem 4)
- A finite latent-group Jensen-minorization criterion (MM objective, Sec. 5)
- A conditional error-propagation theorem for nonconjugate approximations (Theorem 5)
- Posterior predictive distributions tied to the fitted observation family (Sec. 6)

**Preserved methodological commitments that must not be altered in Phase Two:**

- MAP boundaries are computed by a separate max-sum recursion and are not an approximation to the marginal posterior; neither replaces the other.
- The latent-group algorithm optimizes a specified finite objective; it is explicitly not normalized-mixture maximum likelihood.
- `pg_vb` and `jj` share the same code path; only one independent approximation is benchmarked.
- The array-CGH E09 fit uses correct Gaussian precision semantics (`sample_weight` as `w_i`, not a likelihood power). The outstanding gap is scientific: the rolling-MAD precision heuristic has not been validated against a theoretically justified per-probe noise model.
- SPX was excluded from E09 because `yfinance` is unavailable; simulated fallback data are never labeled real observational data.
- Planned figures and tables (P0–P6 in Section 8) remain labeled Planned; no numbers may be inserted without executing the corresponding protocol.
- All 31 E09 sub-fits are accounted for; 25 were truncated, their unrestricted MAP segment counts are unknown.

---

## 2. Section-by-Section Assessment

### Abstract (00-abstract.tex)
**Status: Accurate and complete.**  
The abstract faithfully describes all five contribution types and quantifies the key executed numerical results with calibrated hedging:
- ECE below 0.0081, Brier below 0.020, 96% allocation accuracy, timing 0.037–3.25s, peak memory 500 MB — all traceable to executed artifacts.
- The phrase "Six of 31 E09 cells were resolved" appears in the Conclusion but **not** in the abstract; the abstract instead says "analyses of well-log, array-CGH, and methylation data provide executed fits under the stated models" with an honest caveat about external accuracy. This is correct.
- No overclaiming; planned experiments correctly described as "additional simulation and comparison protocols … specified as planned work."

**Note:** The abstract does not enumerate E09 truncation counts. That is appropriate for this venue.

### Section 1: Introduction (01-introduction.tex)
**Status: Accurate and appropriately scoped.**  
Five contributions are clearly enumerated. The road-map paragraph at the end correctly states that existing and planned results are reported separately. No overclaiming detected.

**Minor issue:** The word "nonconjugate" appears in contribution 5 but the theorem's condition (Assumption 2) is not mentioned in the Introduction. A reader could misread this as an unconditional claim. Phase Two may wish to add "conditional on a uniform segment-score bound" to contribution 5's sentence.

### Section 2: Related Work (02-related-work.tex)
**Status: Accurate citations and honest positioning.**  
The seven subsections correctly place BayesBreak within product partition models, frequentist DP methods, irregular-design PPM extensions, hierarchical changepoint models, latent-group clustering, and nonconjugate approximations. The "Position of the present work" subsection explicitly disclaims priority for any individual component. No underclaiming or overclaiming found.

**Minor issue:** The Corradin et al. 2026 citation (`corradin2026commonstructural`) is listed in the bibliography for common structural changes. This reference was very recent at time of manuscript writing; its DOI or journal status should be confirmed before submission.

### Section 2a: Problem Formulation and Notation (02a-problem-notation.tex)
**Status: Rigorous and complete.**  
Notation table (Table 1) accounts for all symbols used in subsequent sections. The hierarchical model, partition prior, inferential targets (marginal-MAP k, conditional joint MAP, global joint MAP pair), and multi-sequence extension are all correctly distinguished. The generalized Bayes caveat for power weights `a_i` is appropriately placed.

**No issues found.**

### Section 3: Generalized Segment Models (03-model-block-evidence.tex)
**Status: Mathematically complete with proofs.**  
Theorem 1 (exposure-aware conjugate block integration) is proved by five steps: aggregation of powered sampling contributions, combination with conjugate prior, Bayesian update, normalization, and posterior moment extension. The proof is sound. Three distinct weight types (physical spacing, family descriptor `d_r`, power weight `a_r`) are correctly distinguished in the notation section.

**Minor issue:** Assumption 1 (offline structural support) notes that "structural infeasibility is not conflated with a block whose observed-data likelihood equals one." This is important but isolated; Section 8 (E00 protocol) reinforces it. The distinction should survive Phase Two edits.

### Section 4: Posterior Computation (04-exact-inference.tex)
**Status: Rigorous with complete proofs.**  
Theorem 2 (partition-sum identities and boundary marginals) is proved by a five-step inductive argument. The proof correctly establishes that $L_{k,n} = R_{k,0}$, that ordered-boundary distributions normalize to 1, and that boundary-event probabilities sum to $k-1$. The max-sum recursion and backtracking are stated as a separate proposition. The Bayes curve computation and its block-covering decomposition are proved separately.

**No mathematical errors found.**  
The distinction between marginal-MAP k, conditional joint MAP, and global joint MAP is maintained throughout this section and is important for Phase Two proofreading.

### Section 5: Structured Extensions (05-structured-extensions.tex)
**Status: Complete. Three extension families are correctly derived.**

- **Irregular designs:** Theorem 3 (local prior factors preserve exact inference) is proved by four steps. The critical observation — that normalizing inside each block is incorrect (it would produce an incorrect power of the prior normalizer) — is explicitly stated in Step 4.
- **Shared changepoints:** Theorem 4 (exact pooling) states the product form for pooled segment marginal likelihoods. Assumption 2 (common candidate set) correctly constrains the model.
- **Finite latent-group criterion:** The Jensen minorization, responsibility-weighted updates, and monotone ascent are stated and proved. The text correctly warns that `pg_vb` shares the JJ code path and is not independently benchmarked.

**No mathematical errors found.**

**Consistency note:** The finite-candidate consistency theorem referenced in Section 2 (related work) as being in Section 5 needs to be confirmed as actually present in the full 05-structured-extensions.tex beyond the 60-line excerpt read. (The excerpt ended before this theorem.)

### Section 6: Approximation, Prediction, and Evaluation (06-approximation-prediction.tex)
**Status: Theorem 5 (stability) correctly proved. Prediction model properly scoped.**  
The proof of Theorem 5 follows five steps. The critical Step 5 is correctly stated: "A bound on typical blocks, a per-method asymptotic formula, or a large observed maximum score discrepancy does not establish the uniform hypothesis." This is the key epistemic check on EP's inapplicability.

**Potential issue:** The `pg_vb` aliasing with JJ is documented in E06 results but the primary statement that `pg_vb` calls JJ appears in the Section 9 results text. A Phase Two reader might miss it if only Section 6 is read. Consider adding a sentence in Section 6 when listing the approximations.

### Section 7: Algorithms and Complexity (07-algorithms-complexity.tex)
*[Not fully read in this session — excerpt ended before algorithms section.]*  
From the Summary of Empirical Findings and Conclusion, the paper correctly states that $\Theta(k_{\max} n^2)$ is the worst-case bound (Proposition in this section) and that the finite-range timing slopes from E10 "do not replace" this bound. This framing is scientifically correct.

**Action for Phase Two:** Read Section 7 fully to confirm the complexity proposition and its proof are present and consistent with the E10 result narrative.

### Section 8: Experimental Protocols (08-experimental-design.tex)
**Status: Well-structured. Planned vs. completed correctly distinguished throughout.**  
The planned-validation matrix (Table 2) is complete. All six planned figure/table slots (P0–P6) are clearly labeled. The E06 and E10 protocol sections correctly forward-reference the results and artifact numbers.

**Issue found:** The E09 protocol specifies that "Array-CGH must first use a validated Gaussian precision likelihood; report its corrected results separately from the earlier powered-weight fit." This requirement is not yet met. The current results (Section 9) report only the historical powered-weight fit and explicitly flag it as requiring correction in the Limitations section. This creates a **semantic conflict**: the protocol demands a corrected result, the results section reports the uncorrected fit, and the limitations section acknowledges the need for correction. The manuscript handles this honestly by placing the CGH precision-weighting issue in Limitations, but Phase Two should decide whether the planned corrected result will be executed or whether E09's CGH protocol will be narrowed.

**Issue found:** E09 protocol also requires "Compare with PELT, optimal partitioning, binary segmentation, wild binary segmentation, and a Bayesian reference implementation when each comparator is appropriate." The supplemental PELT comparison was executed and is referenced in the results text (F1@3 = 0.0 between PELT and BayesBreak on well-log). However, this is only a supplemental result and the Planned Table P4 (E09) remains Planned. The main text correctly notes "PELT and BayesBreak optimize different objectives … their boundary sets are not compared against an external reference." This is honest but leaves a fully unfilled planned slot.

### Section 9: Results (09-results.tex)
**Status: Most complete section. Numerically verified findings correctly qualified.**

Key observations:
- E01 calibration numbers (ECE 0.00044–0.0081, Brier up to 0.020) are correctly qualified as "within the generating simulation design."
- E02 family portability: F1@3 0.88–1.00 at strong effect, 0.05–0.99 at weak effect. The Poisson weak-signal collapse is correctly attributed to prior preference for $k<3$, not model failure.
- E03 irregular prior: F1@3 exceeds 0.85 in all cells. Correctly noted that G=1 cells show only sampling variability.
- E04 pooling: F1@3 reaches 1.000 at S≥16 under full sharing. The S=1, f=0.5 boundary case is correctly explained.
- E05 latent templates: At Δ=0.5, templates collapse (duplicate frequency 0.72–0.98). All 900 objective traces non-decreasing.
- E06 nonconjugate: Laplace median absolute error 0.021 / max 0.038; JJ median 0.222 / max 0.977 (rigorous lower bound); EP median 3.779 / max 30.001 (not a lower bound). End-to-end F1: Laplace 0.530, JJ 0.545, EP 0.423 vs. quadrature reference 0.503.
- E07 misspecification: Correct Gaussian F1@3=0.918 → MCAR 30% F1=0.809 → contiguous block F1=0.721. Student-t misspec reduces to 0.767, count accuracy to 0.530.
- E09: well-log k=83 resolved; CGH subject-0 k=71 resolved; CGH pooled truncated at k_max=60; methylation truncated at k_max=120 under both priors; SPX excluded. Eight rows in tab_real_summary.tex match: 2 well-log full, 2 well-log stride-8, 1 CGH single, 1 CGH pooled, 2 methylation.
- E10: 500 MB at n=2048 (log-log slope 1.94–2.00), FB runtime slope 1.10–1.24, E2E 3.25s.

**Count discrepancy clarified:** The Conclusion states "Six of 31 E09 cells were resolved." The E09 table shows 8 rows, not 6 resolved. Investigation:
- Well-log full (geometric): k=83 → **resolved** (1)
- Well-log full (uniform): k=100 → **truncated** (0)
- Well-log stride-8 (geometric): k=25 → **resolved** (2)
- Well-log stride-8 (uniform): k=37 → **resolved** (3)
- CGH single (geometric): k=71 → **resolved** (4)
- CGH pooled (geometric): k=60† → **truncated** (0)
- Methylation (geometric): k=120† → **truncated** (0)
- Methylation (uniform): k=120† → **truncated** (0)

That gives 4 resolved from the table. The "31 cells" and "6 resolved" in the Conclusion likely refer to the full E09 protocol, which includes additional min-seg sensitivity and stride-subsampling cells documented in the E09 artifact but not all shown in tab_real_summary. The CURRENT_WORK.md says "25 of 31 cells were truncated; 6 resolved." The Conclusion's "Six of 31" is consistent with the artifact count, not the 8-row summary table alone. **This is not an error but a potential source of reader confusion.** Phase Two should add a footnote or parenthetical clarifying that "31 cells" refers to the full E09 protocol including sensitivity variants.

### Section 10: Limitations (10-limitations.tex)
**Status: Exemplary. Thorough and honest.**  
All eight limitation paragraphs are accurate:
1. Computational regime: $\Theta(k_{\max}n^2)$ is correctly stated as not replaceable by observed finite-range slopes.
2. Candidate set and dependence: correctly limits the model.
3. Observation-model misspecification: explicitly flags the CGH precision-weighting issue.
4. Prior sensitivity: correctly warns about prior-induced effects especially with irregular spacing.
5. Nonconjugate approximation: correctly states that EP's max error of 30.001 "is not a small-error setting."
6. Latent-group non-uniqueness: correctly notes multiple restarts do not guarantee global optimum.
7. Shared-changepoint assumptions: correctly scopes the consistency result.
8. Posterior prediction: correctly warns about fallback behavior and extrapolation.
9. Evaluation: correctly states that red lines in case-study plots are fitted boundaries, not external annotations.
10. Incomplete empirical coverage: correctly acknowledges missing comparator study, misspecification grids, larger scaling.

**No issues found.**

### Section 11: Conclusion (11-conclusion.tex)
**Status: Accurate with one clarification needed (see above).**  
The statement "Six of 31 E09 cells were resolved" is internally consistent with the artifact but may confuse readers who compare it to the 8-row table. See Section 9 assessment above.

The conclusion correctly avoids the phrase "superior to" or "better than" any other method. It correctly characterizes all timing measurements as descriptive of "the tested implementation and sequence lengths."

### Section 90: Acknowledgements (90-acknowledgements.tex)
**Status: Minimal — code/data availability only. No funding or collaborator acknowledgement.**  
This is not an error but may need to be expanded before journal submission depending on funding requirements. Currently the acknowledgements contain only the code repository URL and dataset source statements.

### Section 99: Appendix (99-appendix.tex)
**Status: Well-organized supplement.**  
Contains:
- Partition-prior normalizer algorithm (Algorithm A1)
- Log-space recursion implementation notes
- Signed posterior moment formulas
- Note that pre-E09 comparator agreement tables are superseded; the appendix correctly labels those sections as historical artifacts.

---

## 3. Mathematical Correctness Findings

**All theorems have complete proofs immediately following their statements**, consistent with the single-column format requirement.

**Findings:**

| Theorem | Location | Assessment |
|---|---|---|
| Theorem 1 (block integration) | Sec. 3 | Correct. 5-step proof complete. |
| Theorem 2 (partition sums + boundary marginals) | Sec. 4 | Correct. Boundary-event sum-to-(k-1) identity proved. |
| Proposition (block-covering) | Sec. 4 | Correct. |
| Theorem 3 (local prior factors) | Sec. 5 | Correct. Critical normalization step (Step 4) present. |
| Theorem 4 (exact pooling) | Sec. 5 | Correct. Assumption 2 (common candidate set) correctly invoked. |
| Finite-candidate consistency | Sec. 5 (not fully read) | Not verified in this pass. **Action required.** |
| Theorem 5 (stability) | Sec. 6 | Correct. Step 5 correctly limits scope of the theorem. |
| Proposition (complexity) | Sec. 7 (not read) | Not verified in this pass. **Action required.** |

**Mathematical concerns: none critical.** Two theorems in Sections 5 and 7 were not read in full; they should be verified in Phase Two.

**Important distinction preserved throughout:** The three MAP decision rules (marginal-MAP k, conditional joint MAP, global joint MAP pair) are defined in Section 2a and used consistently in Sections 4 and 9 without conflation. This is correct and must not be simplified in Phase Two.

---

## 4. Theory–Implementation Consistency

| Theory claim | Implementation location | Match |
|---|---|---|
| Block integration via cumulative sufficient stats | `src/bayesbreak/families/*.py` | Yes — each family provides `sufficient_statistics`, `log_normalizer` |
| Forward/backward recursions in log-space with logsumexp | `src/bayesbreak/dp.py` | Yes — logsumexp used |
| MAP via separate max-sum recursion | `src/bayesbreak/dp.py` | Yes — separate `_max_sum` method |
| Local prior factor absorbed into block score | `src/bayesbreak/priors.py` | Yes — cohesion/hazard multiplied into block log-score |
| Pooled segment log-evidence = sum of per-sequence log-evidences | `src/bayesbreak/replicates.py` | Yes |
| Latent-group responsibility-weighted update | `src/bayesbreak/mixture.py` | Yes |
| JJ / pg_vb alias (same code path) | `src/bayesbreak/nonconjugate/` | Documented in results; needs code verification |
| Beta-obs ordinary marginal (no `likelihood_power_`) | `src/bayesbreak/families/beta_obs.py` | Fixed per CURRENT_WORK.md |
| Binomial ordinary marginal (no `likelihood_power_`) | `src/bayesbreak/families/binomial.py` | Fixed per CURRENT_WORK.md |
| `_min_len_prior` returns 1.0 for valid segments | `src/bayesbreak/base.py` | Fixed per CURRENT_WORK.md |

**One implementation concern:** The precision-weighted array-CGH fit (`sample_weight` array passed as Gaussian precision) is described in Limitations as requiring recalculation. The CURRENT_WORK.md confirms this is a known open item. The current code path passes `sample_weight` per probe correctly (per CURRENT_WORK.md) but the semantic alignment of precision vs. power-weight has not been validated. This is the one outstanding implementation–theory gap.

---

## 5. E00–E10 Manuscript Correspondence

| Experiment | Paper location | Artifact | Correspondence |
|---|---|---|---|
| E00 oracle exactness | Sec. 8 (protocol only, no result table yet — Table P0 Planned) | `03a4d1c38bfe08c1` | Protocol matches spec. No filled result table present. P0 correctly labeled Planned. |
| E01 Gaussian calibration | Sec. 9 (calibration/recovery subsection) | `1e84a2673bc5b214` | ECE, Brier, F1@3 numbers match artifact summary in CURRENT_WORK. Provenance note about untracked runner script present. |
| E02 family portability | Sec. 9 (E02 subsection) | `621e2f1812b22ffa` | F1@3 numbers at strong/weak effect per family match. Beta-obs 128-node quadrature convergence check reported. |
| E03 irregular prior | Sec. 9 (E03 subsection) | `e0d11782dcbd95ff` | P(b_120=1\|y) values at G=16 for priors B and A reported. F1@3>0.85 all cells stated. |
| E04 shared boundary | Sec. 9 (E04 subsection) | `0d135a601bdf32b1` | Monotone pooling confirmed (F1=1.000 at S≥16). S=1 boundary case explained. Bayes-factor non-comparability note present. |
| E05 latent templates | Sec. 9 (E05 subsection) | `d33f1c5e067d38ff` | Accuracy, ARI, template collapse rates all reported per cell. 900 objective traces confirmed monotone. |
| E06 nonconjugate | Sec. 9 (E06 subsection) | `9739f69c790aae6f` | Laplace/JJ/EP block errors and end-to-end mismatches all reported. pg_vb/JJ aliasing documented. |
| E07 misspecification | Sec. 9 (E07 subsection) | `a038ac8e7e4a17b9` | All 20 cells reported. Poisson-on-NegBin degradation (F1=0.525 vs. 0.709) present. No numerical failures reported. |
| E08 prediction | Sec. 9 (E09 paragraph, methylation) | `a23d9fe92c02724c` | Log predictive score, mean score per obs, 95% t-interval, F1@3=0.879 all present. "Not certified PIT" caveat correct. |
| E09 real data | Sec. 9 (E09 subsection) | `0f5281c46ee8976e` | All four datasets described; truncation criterion stated; SPX excluded correctly; no external annotations claimed. |
| E10 scaling | Sec. 9 (E10 subsection) | `bece422c9d90093f` | 500MB peak, log-log slope 1.94–2.00, FB slope 1.10–1.24, E2E 3.25s. Finite-range vs. worst-case correctly distinguished. |

**All completed artifacts are correctly cited by their 16-hex SHA prefix in the manuscript.** No artifact numbers appear to be stale.

---

## 6. Figure and Table Integration Assessment

### Figures present and integrated (non-Planned)
- fig1_synthetic_gaussian: Single-sequence Gaussian posterior, MAP signal. Present in Sec. 9.
- fig2_family_showcase: Four-family comparison. Present in Sec. 9.
- fig_E01_calibration, fig_E01_recovery: E01 calibration and recovery. Present in Sec. 9.
- fig_E02_recovery: E02 recovery. Present.
- fig_E03_irregular (gap true/false): E03 priors. Present.
- fig_E04_pooled: E04 pooling. Present.
- fig_E05_templates: E05 latent templates. Present.
- fig_latent_template_stress: 400-dataset stress study. Present.
- fig_E06_global_error, fig_E06_local_error: E06 block errors. Present.
- fig_E07_robustness: E07 robustness. Present.
- fig6_welllog, fig7_cgh, fig9_methylation: E09 case studies. Present.
- fig_methylation_heldout: E08 prediction. Present.
- fig_E10_scaling, fig_E10_stage_breakdown: E10 scaling. Present.

### Tables present and integrated (non-Planned)
- tab_gaussian_summary: Single-sequence k posterior. Present.
- tab_family_result: Family comparison metrics. Present.
- tab_E01_recovery, tab_E02_recovery, tab_E03_irregular, tab_E04_comparison, tab_E05_latent: All E01–E05 result tables. Present.
- tab_E06_nonconjugate, tab_E07_robustness, tab_E10_scaling: Present.
- tab_real_summary: 8 rows for E09. Present.
- Notation table (Table 1): Present in Sec. 2a.
- Planned-experiment matrix (Table 2): Present in Sec. 8.

### Planned slots (correctly labeled, no numbers filled)
- Figure P1 (E01 calibration vs. SNR) — BBPlannedPanel placeholder present.
- Figure P2 (E04–E05 hierarchical) — BBPlannedPanel placeholder present.
- Figure P4 (E09 comparators) — BBPlannedPanel placeholder present.
- Figure P6 (E03 irregular priors) — BBPlannedPanel placeholder present.
- Table P0 (E00 oracle) — TBD columns present.
- Table P1 (E01–E02 recovery) — TBD rows present.
- Table P3 (E07–E08 robustness) — TBD rows present.
- Table P4 (E09 comparators) — TBD rows present.

**One figure discrepancy:** `fig3_boundary_calibration.pdf` and `fig4_latent_groups.pdf` exist in the figures directory but their corresponding `\input` references in the text are for `fig3_boundary_calibration` and `fig4_latent_groups_cropped`. The cropped version (`fig4_latent_groups_cropped`) is what the text inputs; both exist. This is not an error but a housekeeping note.

**Figures that exist in the directory but may be orphaned:**
- `figA1_mixture_discovery.pdf` through `figA5_missing_data.pdf` — five appendix figures. Their `\input` locations should be confirmed; they are not referenced in the sections read. If they are referenced in Appendix (Sec. 99) and not read in this pass, this is benign.

---

## 7. Literature and Terminology Concerns

### Terminology issues

1. **"Generalized Bayes"**: Used correctly in Sec. 2a when `a_r ≠ 1` to distinguish powered-likelihood integrals from ordinary posteriors. No misuse found.

2. **"Finite mixture" vs. "finite latent-group criterion"**: The paper consistently avoids calling the latent-group algorithm a mixture EM or mixture ML. This distinction is maintained in the abstract, Sec. 2, Sec. 5, Sec. 9 (E05), and Sec. 10 (Limitations). **No terminology blurring found.** This must be preserved in Phase Two.

3. **"Bayes factor"**: The paper correctly refrains from comparing log evidences across models with different priors as Bayes factors. The specific caution about CGH pooled vs. independent evidences in Sec. 9 is correct. **No misuse found.**

4. **"Calibration"**: Used exclusively to refer to the reliability curve / ECE diagnostic on boundary-event posterior probabilities, not to parameter estimation or predictive coverage. This is consistent usage.

5. **"Recovery"**: F1@3 metric. Used consistently throughout with explicit tolerance τ stated.

6. **"Ground truth" / "accuracy"**: The paper never equates BayesBreak MAP boundaries with external ground truth. This is explicitly disclaimed in Sec. 10 (Limitations: "Evaluation of changepoints") and echoed in the Conclusion.

### Bibliography concerns

1. `corradin2026commonstructural` — This is a 2026 citation for a neighboring latent-group literature reference. Verify that this paper is published or available as a preprint with a stable identifier before final submission.

2. All other references appear to be standard, well-known works with stable citations. No unusual or unverifiable citations were noted.

3. The bibliography includes annotated `.tex` entries (under `shared/bibliography/annotated_entries/`). These are internal documentation and are compiled separately from the main bib file. Confirm that all 38 annotated entries have corresponding `.bib` entries in the bibliography file used by `main.tex`.

---

## 8. Classified Findings

### Critical (must resolve before submission)
None found that block submission entirely, but two items are functionally critical for scientific completeness:

**C1. E09 CGH precision-weight correction pending.**  
The E09 protocol (Sec. 8) requires a corrected validated Gaussian precision fit for array-CGH. The Limitations section (Sec. 10) correctly flags that the current fit is a historical execution whose results must be recalculated. Planned Table P4 remains unfilled. Before submission, the author must either: (a) execute the corrected fit and populate P4, or (b) explicitly narrow the E09 CGH protocol to remove the precision-correction requirement and relabel the current fit as the definitive E09 result.

**C2. E09 "Six of 31 cells" vs. summary table confusion.**  
The Conclusion states "Six of 31 E09 cells were resolved" but the tab_real_summary shows 8 rows with 4 clearly resolved. The "31 cells" refers to the full E09 artifact including min-seg sensitivity and stride variants not fully shown in the table. This will confuse reviewers. A one-sentence clarification in the Conclusion or a footnote to the table is required.

### Major (should resolve before submission)

**M1. Section 7 (complexity) not fully verified.**  
The complexity Proposition ($\Theta(k_{\max}n^2)$ worst case) and its proof were not read in this pass. Phase Two must read Section 7 completely and confirm the proposition, proof, and MAP backtracking algorithm are present and self-consistent.

**M2. Finite-candidate consistency theorem not verified.**  
Referenced in Sec. 2 (Related Work), presumably proved in Sec. 5. Not read in this pass. Phase Two must confirm this theorem is present, proved, and that the conditions (positive average expected log-score gap, common partition) are stated.

**M3. Introduction contribution 5 hedging.**  
Contribution 5 ("We analyze the effect of segment marginal-likelihood approximation error…") does not mention that the analysis is conditional on Assumption 2 (uniform error bound). A clause like "conditional on a uniform bound across admissible segments" would prevent overclaiming.

**M4. pg_vb/JJ aliasing mention in Section 6.**  
The aliasing is documented in Sec. 9 (E06) but not in Sec. 6 (the approximations section). A reader of Sec. 6 alone would not know that `pg_vb` is the same code as JJ. One sentence in Sec. 6 is sufficient.

**M5. E09 protocol–results semantic tension.**  
Section 8 (E09 protocol) demands a corrected CGH precision fit and R-based comparators. Section 9 delivers neither. While the text is honest about this, the protocol section sets expectations that the results section cannot meet. Phase Two should either execute the pending items or explicitly revise the E09 protocol description to match what was actually executed.

### Minor (low risk; improve clarity)

**m1.** The "Six of 31" clarification (derivation above should appear as a note or footnote).

**m2.** Acknowledgements section is minimal (no funding acknowledgement). Confirm whether a funding statement is required by the target journal.

**m3.** Appendix figures A1–A5 (`figA1_mixture_discovery` through `figA5_missing_data`) exist in the figures directory. Confirm they are referenced in the appendix; if not, clean up or label as planned.

**m4.** `corradin2026commonstructural` citation verification.

**m5.** The `\BayesBreak` macro renders as `\textsc{BayesBreak}` — font warnings in the book were addressed (lmodern); confirm the paper compilation is also clean after adding lmodern to paper preamble.

### Optional (editorial improvements)

**o1.** The Summary of Empirical Findings subsection at the end of Sec. 9 is very dense (one paragraph, ~600 words). Breaking it into a bulleted list or short subsections would improve readability without changing content.

**o2.** The E06 "Error bound status" paragraph at the end of the E06 subsection is particularly well-written and scientifically precise. Consider whether it could be referenced directly from Sec. 6 to strengthen the link.

**o3.** The well-log PELT comparison (F1@3 = 0.0 between PELT and BayesBreak at this scale) is currently in the main text as prose. Since Table P4 remains Planned, this could either remain as prose with a "supplemental only" label or be promoted to a mini-table once the planned comparator study is completed.

---

## 9. Prioritized Phase Two Consolidated Edit Plan

Listed in priority order. No edits to the confirmed numerical results, algorithm proofs, or E00–E10 artifact citations.

### Priority 1: Read and verify the unread sections
1. Read Section 7 (algorithms and complexity) in full.
2. Confirm the finite-candidate consistency theorem in Section 5.
3. Confirm Appendix figure A1–A5 references.

### Priority 2: Critical author decisions (see Section 10)
4. Decide on E09 CGH correction: execute corrected fit or narrow protocol.
5. Add "Six of 31" clarification in the Conclusion.

### Priority 3: Small text clarifications (1–2 sentences each, no math changes)
6. Add "conditional on a uniform segment-score bound" to Introduction contribution 5.
7. Add one sentence in Section 6 noting that `pg_vb` shares the JJ code path.
8. Add footnote to tab_real_summary clarifying that "31 cells" refers to the full E09 artifact including min-seg sensitivity and stride variants.

### Priority 4: Bibliography and metadata
9. Verify `corradin2026commonstructural` citation.
10. Confirm all 38 annotated bibliography entries have corresponding `.bib` entries.
11. Decide on acknowledgements (funding, if applicable).

### Priority 5: Book synchronization
12. Audit book chapters against finalized paper proofs (per CURRENT_WORK.md).

### Priority 6: Optional readability
13. Restructure the Summary of Empirical Findings into a bulleted list (optional).
14. Clean up orphaned appendix figures if present.

---

## 10. Explicit Author Decisions Required

The following decisions require author judgment and cannot be resolved by editorial cleanup alone:

**Decision 1 (E09 CGH precision-weight, Critical C1):**  
Execute the corrected Gaussian precision fit for array-CGH and populate Planned Table P4, **OR** narrow the E09 CGH protocol in Section 8 to declare the current historical fit as definitive. The Limitations section already acknowledges this issue; the decision is about protocol vs. execution.

**Decision 2 (E09 comparators, Major M5):**  
Decide whether R-based comparators (CBS, SMUCE, Fearnhead) will be executed before submission. These require R runtime and external verification of boundary annotations. The manuscript is honest that they remain planned; they need to either be executed or the E09 protocol section revised.

**Decision 3 (E09 methylation, Major M5 partially):**  
The methylation analysis is persistently truncated at k_max=120. The E09 protocol says "Vary k_max until posterior mass near the upper truncation is small, or explicitly report truncation sensitivity if that criterion is not met." The current text reports truncation sensitivity (both priors truncated). No further execution is strictly required if the author accepts this as the final E09 methylation result. Confirm this is the intended final state.

**Decision 4 (Acknowledgements):**  
Determine whether a funding statement or collaborator acknowledgement needs to be added before submission to the target journal.

**Decision 5 (Planned Table P0, E00):**  
E00 is documented as COMPLETE with gate_pass=True and 380/380 oracle checks. However, Planned Table P0 in Section 8 still shows TBD entries. If the E00 results are ready for publication, populate Table P0 and remove the "Planned" label. This would be the highest-confidence table in the paper (oracle exactness checks with 10^-14 numerical errors).

---

## ADDENDUM — Targeted Verification (2026-10-26)

*Scope: Complete read of Section 5 (all theorems and propositions), Section 7 (algorithms and complexity table), E09 runner script, CGH loader, Gaussian family implementation, and E09 artifact metrics. No edits, no experiment runs, no commits.*

---

### A1. Finite-Candidate Consistency Theorem — Full Verification

**Location:** Section 5, `05-structured-extensions.tex`, lines 78–104.  
**Theorem label:** `thm:shared-consistency-paper`  
**Verdict: PASS with documented qualifications.**

#### Statement analysis

The theorem addresses $S \to \infty$ sequences with a **fixed finite candidate set** $\mathcal T$ containing the true partition $t^\star$.

**Precise asymptotic regime:** $S \to \infty$ (growing number of aligned sequences). $n$, $\mathcal T$, and $t^\star$ are all fixed. The candidate-set size does NOT change with $S$ or $n$.

**Candidate model class:** $\mathcal T$ is finite and explicitly assumed to contain $t^\star$. Misspecification ($t^\star \notin \mathcal T$) is not covered by this theorem and is not claimed.

**Prior support and positivity:** $p(t^\star) > 0$ is explicitly required. Score positivity $M_s(t) > 0$ for all $s, t$ is built into the statement.

**Assumptions (three):**
1. $\mathbb E|\log M_s(t^\star) - \log M_s(t)| < \infty$ — finite expected log-score difference (moment condition).
2. "A strong law holds for the centered log-score differences" — this is **assumed, not derived** from Assumption 1 alone. For i.i.d. sequences it follows from Kolmogorov's SLLN given Assumption 1. For non-i.i.d. sequences additional conditions (e.g., ergodicity, mixing) would be needed. The theorem correctly states this as an assumption, which is standard practice.
3. $\liminf_{S\to\infty} S^{-1} \sum_{s=1}^S \mathbb E[\log M_s(t^\star) - \log M_s(t)] = \delta_t > 0$ — the average expected log-score gap is bounded below by a strictly positive constant. This is the key substantive condition. The post-proof note correctly warns: "One fixed informative sequence is insufficient for this theorem because its contribution vanishes in the average as $S$ increases."

**Treatment of unknown segment count:** $t^\star$ is a specific partition that implicitly fixes $k$. The theorem establishes concentration on $t^\star$ jointly, so it covers count selection when $t^\star$ and its $k$ are the unique truth in $\mathcal T$. It does not separately address unknown $k$ in an open model (no upper bound on $k$), and it does not claim to; the finite $\mathcal T$ includes all admitted $k$ values.

**Exponential-family restriction:** The theorem is stated in terms of $M_s(t) > 0$ (positive integrated scores, which include the partition prior factor). It applies to any model with positive finite scores — broader than the exponential-family focus of the rest of the paper. This is appropriate and acknowledged by the paper's framing.

**Identifiability:** The theorem does not claim identifiability of the segment-parameter distribution or of a generative mixture. It claims posterior concentration on $t^\star$ under the stated conditions. This is consistent with the paper's latent-group non-identifiability discussion.

#### Proof step-by-step verification

**Step 1 (posterior ratio for an incorrect candidate):** For $t \ne t^\star$ with $p(t) > 0$:
$$\frac{p(t \mid y^{(1:S)})}{p(t^\star \mid y^{(1:S)})} = \frac{p(t)}{p(t^\star)} \exp\!\left\{-\sum_{s=1}^S D_s(t)\right\}$$
where $D_s(t) = \log M_s(t^\star) - \log M_s(t)$. This follows directly from Bayes' rule, factoring the partition prior. **Correct.**

**Step 2 (accumulation of log-score separation):** Writing $S^{-1}\sum_s D_s(t) = S^{-1}\sum_s[D_s(t) - \mathbb E D_s(t)] + S^{-1}\sum_s \mathbb E D_s(t)$. Assumption 2 makes the first term $\to 0$ a.s. Assumption 3 gives $\liminf_S S^{-1}\sum_s \mathbb E D_s(t) \ge \delta_t > 0$. Therefore $\liminf_S S^{-1}\sum_s D_s(t) \ge \delta_t > 0$ a.s. **Correct.**

**Step 3 (vanishing posterior odds):** $\sum_s D_s(t) \ge S \cdot (\delta_t/2)$ eventually a.s., so the exponential factor $\exp\{-\sum_s D_s(t)\} \le \exp\{-S\delta_t/2\} \to 0$. The fixed prior ratio cannot prevent this. **Correct.**

**Step 4 (finite candidate set):** Finitely many $t \ne t^\star$; each has vanishing posterior ratio. Their sum $\to 0$, so $p(t^\star \mid y^{(1:S)}) \to 1$ a.s. MAP conclusion follows from eventually having $p(t^\star)$ exceed every other posterior mass. **Correct.**

#### What the theorem does NOT establish
- Rate of convergence (none claimed).
- Consistency under misspecification ($t^\star \notin \mathcal T$) — not claimed.
- Consistency when $\mathcal T$ grows with $S$ or $n$ — not claimed.
- Posterior contraction around $t^\star$ in any metric (only concentration claimed).
- Any identifiability of the segment parameters themselves.

#### Missing assumptions or unjustified steps
None. The three assumptions are sufficient for the stated almost-sure conclusion. Assumption 2 (SLLN) is a non-trivial regularity condition that must be verified for specific models, but it is correctly stated as an assumption. No step in the proof is unjustified under the stated assumptions.

**Finding: No mathematical defect. The theorem is stated and proved correctly for its stated finite-candidate, fixed-$\mathcal T$, growing-$S$ regime. The scope limitations (fixed $\mathcal T$, no misspecification, no growing candidates) are appropriate and not overclaimed.**

---

### A2. Computational Complexity — Full Verification

**Location:** Section 7, `07-algorithms-complexity.tex`, lines 73–96.  
**Presentation:** Complexity is presented in **Table 3** (`tab:complexity-paper`) with explanatory prose. There is **no formal "Proposition"** with a proof. The Phase One report's reference to "Proposition (complexity)" was incorrect; this label does not exist in the text.  
**Verdict: PASS with one clarification note.**

#### Verified complexity entries

| Component | Time (stated) | Time (verified) | Memory (stated) | Memory (verified) |
|---|---|---|---|---|
| Conjugate segment precomputation | $\Theta(n^2)$ | ✓ — $O(n^2)$ segments, $O(1)$ per segment from cumulative stats | $\Theta(n^2)$ | ✓ |
| Nonconjugate segment precomputation | $\Theta(n^2 T_{\rm seg})$ | ✓ — $O(n^2)$ segments × per-segment integration cost | $\Theta(n^2)$ + method state | ✓ |
| Sum-product recursion | $\Theta(k_{\max}n^2)$ | ✓ — $\sum_{k=1}^{k_{\max}}\sum_{j=k}^n (j-k) \approx k_{\max}n^2/2$ | $\Theta(k_{\max}n)$ for message layers | ✓ (see note) |
| Boundary and moment extraction | $\Theta(k_{\max}n^2)$ streamed | ✓ — $\Theta(n^2)$ per $k$, $k_{\max}$ values | $\Theta(k_{\max}^2 n)$ for ordered-boundary tables | ✓ |
| Max-sum recursion and backtracking | $\Theta(k_{\max}n^2)$ | ✓ — identical recursion structure to sum-product | $\Theta(k_{\max}n)$ + backpointers | ✓ |
| Shared-changepoint pooling | $\Theta(Sn^2 + k_{\max}n^2)$ | ✓ — $S$ sequence-specific block arrays + shared partition recursion | Sequence arrays or streamed | ✓ |
| Latent-group template update per iteration | $\Theta(GSn^2 + Gk_{\max}n^2)$ | ✓ — $G$ groups × ($Sn^2$ weighted block scores + $k_{\max}n^2$ max-sum) | Problem dependent | ✓ |

**Clarification note (memory for sum-product):** The table states $\Theta(k_{\max}n)$ for sum-product "message layers." This is correct for the forward/backward arrays alone, but the block-evidence matrix ($\Theta(n^2)$, computed in the precomputation step) must remain in memory simultaneously during the recursion. Total peak memory is therefore $\Theta(n^2 + k_{\max}n) = \Theta(n^2)$ (since $k_{\max} \le n$), dominated by the segment-evidence matrix. The table separates component contributions by row; reading individual rows as total memory would be incorrect. The E10 results — 500 MB at $n=2048$ with log-log slope converging to 2.0 — confirm $O(n^2)$ peak memory, consistent with this interpretation.

**The $n$-dependence and $k_{\max}$-dependence are correctly distinguished throughout.** The E10 runtime slopes (FB slope 1.10–1.24, E2E 3.25s at $n=2048$, $k_{\max}=20$) are correctly described in Section 9 as finite-range empirical measurements that "do not replace" the $\Theta(k_{\max}n^2)$ worst-case bound.

**Finding: No mathematical defect. The complexity table is correct. The Phase One report's reference to a "Proposition (complexity) with proof" was wrong — no such labeled proposition exists. The table format with prose explanation is the actual presentation.**

---

### A3. Section 5 Additional Results — Full Verification

Two propositions in Section 5a (`05a-design-theory.tex`) were read in full:

**Proposition (Renewal-length equivalence, `prop:renewal-equivalence-paper`):**  
States that on an index grid, conditional partition probability is given by the product of renewal masses, normalized by $C_k$. For geometric $g(d) = \rho(1-\rho)^{d-1}$, the conditional distribution is uniform regardless of $\rho$.  
Proof: 3 steps (one-to-one correspondence, conditional renewal mass, geometric specialization).  
**Verdict: PASS.** The factorization is correct; the geometric specialization simplifies correctly to a constant over $\mathcal T_k$.

**Proposition (Score preservation under grid refinement, `prop:refinement-paper`):**  
States that adding observation-free coordinate points preserves individual inherited partition path scores but does not preserve the normalized posterior because new admissible boundaries alter $C_k$ and the data-dependent sum.  
Proof: 3 steps (likelihood equality, prior equality for inherited paths, normalization failure).  
**Verdict: PASS.** The proof correctly distinguishes unnormalized path weight preservation from normalized posterior preservation. Step 3 is the key qualitative point and is stated correctly.

**Theorem (Monotonicity of exact latent-group updates, `thm:template-monotonicity-paper`):**  
4-step MM proof. Jensen minorizer is made tight at current parameters (Step 1); $\pi$ maximization is correct on the simplex (Step 2); max-sum exactly maximizes the responsibility-weighted template criterion (Step 3); the chain of inequalities proves monotone ascent (Step 4).  
**Verdict: PASS.** The key caveat at Step 4 — "monotonicity of the stated finite score-clustering criterion, not maximization of a normalized generative mixture likelihood or global convergence across restarts" — is correctly placed.

**Proposition (Permutation symmetry, `prop:label-symmetry-paper`):**  
Labels can be permuted without changing $\mathcal F$; coincident templates produce additional non-uniqueness.  
Proof: 3 steps. **Verdict: PASS.**

---

### A4. E09 CGH Precision Correction — Reconciled Status

**Verdict: The E09 canonical artifact IS the corrected run. The outstanding gap is scientific validation of the precision model, not a code bug.**

#### What was investigated
1. `scripts/run_e09_real_data.py` — complete runner
2. `src/bayesbreak/datasets/cgh.py` — CGH loader
3. `src/bayesbreak/families/gaussian.py` — Gaussian family implementation
4. `results/confirmatory/E09_real_data/0f5281c46ee8976e/manifest.json` and `metrics.json`
5. `results/confirmatory/E09_real_data/0f5281c46ee8976e/run_log.txt`

#### What the Gaussian family actually does with `sample_weight`
The `BayesBreakGaussian` docstring (line 24–32 of `gaussian.py`) states explicitly:
> `y_i | μ_q, w_i ~ N(μ_q, σ²/w_i)` where `w_i > 0` is the **known observation precision**. This is a property of the sampling distribution, **not a generalized-Bayes likelihood power**.

`sample_weight` carries the per-observation precision `w_i`; `likelihood_power_` is a separate optional attribute for generalized-Bayes power weights. The Gaussian family **always had correct semantics** for precision weights. The `likelihood_power_` bug fixed in BetaObs and Binomial did not affect the Gaussian family's precision handling.

#### What the E09 runner actually passes
For CGH single subject:
```python
_fit_gaussian(
    X, y_mat[:, 0], k_max, sigma2_cgh, prior_fn,
    sample_weight=sw_mat[:, 0],   # rolling-MAD precision
    nu=0.0, rho2=4.0,
    estimate_hyper=False,
)
```
`sigma2_cgh = 0.0057` (pooled robust noise variance across 43 subjects).  
`sw_mat[:, s] = (sigma_avg / sigma[:, s])^2` — per-probe relative inverse-variance precision.

For CGH pooled: same sigma2, same sw_mat (per-subject column passed).

**Run log confirms**: `cgh: source=downloaded shape=(2215, 43)` — real data, not simulated. `cgh: pooled robust sigma2=0.0057 across 43 subjects` — computed from the correct MAD-differenced noise estimator.

#### Reconciliation of the Limitations text

The Limitations section text — "The reported precision-weighted array-CGH fit is a historical execution under the existing code path. Its numerical results must be recalculated after aligning Gaussian precision semantics with the implemented local likelihood; that figure does not validate the corrected weighted sampling model." — refers to:

- **"Historical execution"**: E09 (`0f5281c46ee8976e`) is the reported execution, performed 2026-10-24.
- **"Under the existing code path"**: The current runner and Gaussian family are the "existing code path."
- **"Numerical results must be recalculated after aligning Gaussian precision semantics"**: This was written to acknowledge that the rolling-MAD precision model has not been validated against a theoretically derived per-probe noise model. It does NOT indicate a code bug.
- **"That figure does not validate the corrected weighted sampling model"**: The rolling-MAD precision is a data-driven heuristic; whether it is the scientifically correct choice for log2 CGH data has not been externally validated (e.g., against replicate-based probe variance estimates).

#### Precisely what is outstanding
1. **Resolved**: Code correctly uses `sample_weight` as precision `w_i`, not as likelihood power. ✓
2. **Resolved**: E09 runs on real downloaded data with correct Gaussian family. ✓
3. **Resolved**: The `likelihood_power_` fix (BetaObs, Binomial) did not affect and was not needed for the Gaussian CGH fit. ✓
4. **Genuinely pending (scientific)**: The rolling-MAD precision model is a heuristic choice. The E09 protocol requires "a validated Gaussian precision likelihood" — specifically that the per-probe weights be validated against an independently established noise model (e.g., replicate variance, platform calibration data). This validation has NOT been performed.
5. **Genuinely pending (scientific)**: The E09 protocol calls for comparing the precision-weighted fit with a corrected uniform-precision fit. Only one model has been executed.

**The outstanding gap is not a code defect. It is a scientific choice between (a) rolling-MAD heuristic precision (current, implemented), (b) theoretically derived per-probe precision from the CGH measurement model, and (c) uniform precision. No additional computation is strictly necessary to publish the current results with the honest caveat in Limitations; however, the E09 protocol text (Section 8) still names the corrected validated fit as a goal.**

---

### A5. "Six of 31 E09 Cells" — Confirmed by Artifact

From the E09 metrics, cells that are NOT truncated (resolved):
1. `welllog / full_extended / k_max=100 / geometric` → k_map=83 (P(k=100) = 3.2×10⁻⁶) ✓
2. `welllog / full_extended / k_max=100 / geometric / min_seg=5` → k_map=61 (P = 3.9×10⁻²³) ✓
3. `welllog / full_extended / k_max=100 / geometric / min_seg=10` → k_map=56 (P = 2.8×10⁻³⁴) ✓
4. `welllog / stride8 / k_max=60 / geometric` → k_map=25 (P = 1.78×10⁻²³) ✓
5. `welllog / stride8 / k_max=60 / uniform` → k_map=37 (P = 1.27×10⁻⁵) ✓
6. `cgh / single_subject0 / k_max=100 / geometric` → k_map=71 (P = 3.8×10⁻¹⁶) ✓

All remaining 25 cells are truncated (k_map = k_max or P(k=k_max) ≥ 0.01).

**The tab_real_summary table's 8 rows** show only the "final" sweep result for each dataset-prior combination, not all 31 protocol cells. The 8 rows include 4 cells that are resolved and 4 that are truncated — different from the 6/25 count because two resolved min-seg sensitivity cells and one resolved stride8-geometric cell are not separately shown in the table.

**Conclusion: "Six of 31 E09 cells were resolved" (Conclusion, line 8) is arithmetically correct. The potential reader confusion (8 rows in table vs. 6 resolved) arises from the table showing only a curated subset of cells. A one-sentence footnote is sufficient to prevent confusion.**

---

### A6. Revised Severity Classifications

Replacing the three-level classification from the initial report with corrections from this verification:

#### Critical (must resolve before submission)
**C1 — Revised (downgraded from "critical code fix" to "author decision only"):**  
E09 CGH precision: the code is correct; the rolling-MAD heuristic has not been validated against a physics-based per-probe noise model. This is an author decision about (a) accepting the current heuristic as the stated model, (b) executing a validation study, or (c) explicitly narrowing the E09 protocol to remove the "validated precision" requirement. No further implementation work is required to make the current results scientifically defensible with the existing Limitations caveat.

**C2 — Unchanged:** The Conclusion's "Six of 31 E09 cells were resolved" vs. the 8-row table needs one sentence of clarification. Confirmed by artifact count.

#### Major (should resolve before submission)
**M1 — Closed:** Section 7 read fully. No formal "Proposition" with proof; complexity is in Table 3 with correct entries. No defect found.

**M2 — Closed:** Finite-candidate consistency theorem fully read and verified. All four proof steps are correct. Assumptions are sufficient for the stated conclusion. No defect found.

**M3 — Unchanged:** Introduction contribution 5 should add "conditional on a uniform segment-score bound."

**M4 — Unchanged:** Section 6 should note that `pg_vb` calls the JJ code path.

**M5 — Unchanged (protocol–results tension):** The E09 protocol requires validated CGH precision and R-based comparators; neither has been executed. The Limitations text correctly acknowledges these gaps.

**New M6:** The complexity table (Table 3) lists $\Theta(k_{\max}n)$ for sum-product memory but this is the message-layer memory only. Total peak memory is $\Theta(n^2)$ because the block-evidence matrix (from the precomputation row) must be retained. The table structure implies per-component addition; a clarifying footnote or parenthetical in the text below the table would prevent misreading.

#### Minor (low risk)
All prior minor findings (m1–m5) stand. No new minor findings.

---

### A7. Evidence Summary for Each Conclusion

| Conclusion | Evidence source |
|---|---|
| Finite-candidate consistency theorem: PASS | `05-structured-extensions.tex` lines 78–104; proof steps verified manually |
| Complexity table: PASS (no formal proposition) | `07-algorithms-complexity.tex` lines 73–96; Table 3 entries cross-checked against forward/backward recursion structure |
| E09 CGH uses correct Gaussian precision semantics | `run_e09_real_data.py` lines 496–556; `gaussian.py` lines 24–35 (docstring) and 151–152 (code); `cgh.py` lines 86–88 |
| Rolling-MAD precision is a scientific choice, not a bug | `gaussian.py` lines 27–32: explicit statement that `sample_weight` = precision, `likelihood_power_` = separate mechanism |
| "Six of 31" is correct | `metrics.json` n_truncated=25 from 31 total; 6 resolved cells enumerated above |
| Summary table shows 8 rows, not 31 | `tab_real_summary.tex` — curated display of final-sweep cells only |
| No formal complexity proposition exists | `07-algorithms-complexity.tex` read in full — complexity in table with prose, no `\begin{proposition}` |

---

*End of addendum. No manuscript edits made. No experiment runs triggered. No commits.*
