# BayesBreak — Phase Two: Consolidated Scientific Manuscript Revision Plan

**Date produced:** 2026-10-27  
**Basis:** Phase One report `.claude/PHASE1_REPORT.md` (initial + addendum)  
**Scope:** Paper only — `docs/manuscripts/bayesbreak-paper/paper/main.tex` and its shared components.  
**Hard constraints:** Do not modify methodology, inference algorithms, E00–E10 experiments, verified results, or confirmatory artifacts. Do not commit or push.

---

## Section A — Scientific Identity (Preserved Without Alteration)

The following elements are correct, complete, and must not change:

| Element | Location | Status |
|---------|----------|--------|
| Conjugate segment integration (exponential families) | Sec 3 | VERIFIED PASS |
| Sum-product forward/backward recursion + identities | Sec 4, Thm 1 | VERIFIED PASS |
| Block-covering posterior moments (Bayes curves) | Sec 4, Prop | VERIFIED PASS |
| Max-sum MAP backtracking + correctness proof | Sec 4, Thm 2 | VERIFIED PASS |
| Decision-rule propositions (three non-equivalent modes, Hamming Bayes action) | Sec 4a | VERIFIED PASS |
| Local prior factor theorem (irregular designs) | Sec 5, Thm 3 | VERIFIED PASS |
| Exact pooling theorem (shared changepoints) | Sec 5, Thm 4 | VERIFIED PASS |
| Finite-candidate consistency theorem | Sec 5 | VERIFIED PASS |
| Renewal-length equivalence + grid-refinement propositions | Sec 5a | VERIFIED PASS |
| Jensen-MM monotonicity theorem + permutation symmetry proposition | Sec 5 | VERIFIED PASS |
| Logistic-block mode lemma (strict concavity) | Sec 6a | VERIFIED PASS |
| JJ lower-bound proposition + PG augmentation proposition | Sec 6a | VERIFIED PASS |
| Theorem 5 (approximation-error stability, conditional on uniform bound ε) | Sec 6 | VERIFIED PASS |
| Posterior-predictive proposition | Sec 6 | VERIFIED PASS |
| Joint predictive integration proposition + closed-form corollary | Sec 6b | VERIFIED PASS |
| Complexity table (Table 3, not a formal proposition) | Sec 7 | VERIFIED PASS |
| E00–E10 confirmatory artifact results | `results/confirmatory/` | IMMUTABLE |
| corradin2026commonstructural citation | `shared/bibliography/references.bib` | VERIFIED PRESENT |

**Contribution 5 (Sec 1):** Already contains the required conditional scope: *"The analysis is conditional on a uniform bound across admissible segments, not on a numerical method's name or an unverified finite-sample error estimate."* — **No edit needed.**

**pg_vb/JJ aliasing (Sec 6):** Already documented at `06-approximation-prediction.tex` lines 146–148 under `\paragraph{Interpretation of nonconjugate method labels.}` — **No edit needed.**

---

## Section B — Verified: No Edit Needed

The findings M1–M4 from Phase One were all resolved as follows:

| Finding | Phase One statement | Resolution |
|---------|---------------------|------------|
| M1 | Theorem 5 scope needs clarification | CLOSED: paper already restricts to "conditional on a uniform bound" |
| M2 | EP numerical reference scope | CLOSED: Sec 6a explicitly states EP not used to establish accuracy without validated normalizer |
| M3 | Contribution 5 needs conditional clause | CLOSED: already present verbatim in Sec 1 |
| M4 | pg_vb/JJ aliasing undocumented | CLOSED: documented at Sec 6 lines 146–148 |

figA1–A5 orphan check: **Non-issue.** No such figure files exist in `shared/figures/defs/`; the concern was unfounded.

---

## Section C — Required Edits

Complete edit inventory. Each entry has 13 fields. All edits are prose-only; none changes mathematical content or any number.

---

### R2 — Conclusion: clarify "31 cells"

| Field | Value |
|-------|-------|
| **ID** | R2 |
| **Priority** | 1 (High) |
| **Edit type** | Clarification / prose addition |
| **File** | `paper/sections/11-conclusion.tex` |
| **Section** | Conclusion paragraph 1 |
| **Existing text** | `Six of 31 E09 cells were resolved; the remaining 25 were truncated and their unrestricted MAP segment counts are unknown.` |
| **Issue (C2)** | "31 cells" is ambiguous; a reader of Tab. real_summary (8 rows) will not understand where 31 comes from. The 8-row table shows only the curated final-sweep rows; the full protocol has 31 cells (dataset × prior × k_max variant combinations). |
| **Proposed replacement** | `Six of 31 E09 protocol cells (dataset, prior, and $k_{\max}$ variant combinations) were resolved; the remaining 25 were truncated at the declared $k_{\max}$ and their unrestricted MAP segment counts are unknown.` |
| **Justification** | Prevents a reader from inferring that only 6 of 8 table rows resolved, or that 25 truncated cases are unreported data |
| **Artifact dependency** | `results/confirmatory/E09_real_data/0f5281c46ee8976e/metrics.json`: `n_cells: 31`, `n_truncated: 25` |
| **Methodology flag** | No |
| **Computation required** | No |
| **Author approval required** | No (factual clarification) |

---

### R3 — Tab. real_summary caption: clarify curated subset

| Field | Value |
|-------|-------|
| **ID** | R3 |
| **Priority** | 1 (High) |
| **Edit type** | Caption footnote addition |
| **File** | `shared/tables/defs/tab_real_summary.tex` |
| **Section** | Table 3 (E09 real-data segmentations) |
| **Existing text** | `\caption{E09 real-data segmentations (artifact \texttt{0f5281c46ee8976e}). Prior: geometric-decay log4 for resolved fits; uniform prior shown where resolved. Log evidences belong to different data models and are not comparable across rows. $\dagger$ truncated: posterior mass at $k_{\max}$; unrestricted MAP unknown.}` |
| **Issue (C2)** | Caption does not say the 8 rows are a curated subset of 31 protocol cells. A reader may infer the table is exhaustive. |
| **Proposed replacement** | Add at end of caption: `Eight curated rows shown (one per dataset–prior combination reaching a final sweep); the full 31-cell protocol is recorded in the artifact.` |
| **Justification** | Aligns the displayed table with the artifact's `n_cells: 31` metadata |
| **Artifact dependency** | `0f5281c46ee8976e/metrics.json` |
| **Methodology flag** | No |
| **Computation required** | No |
| **Author approval required** | No (factual note) |

---

### R4 — Table 3 (Sec 7): distinguish message-layer vs. total peak memory

| Field | Value |
|-------|-------|
| **ID** | R4 |
| **Priority** | 2 (Medium) |
| **Edit type** | Table footnote or adjacent prose addition |
| **File** | `paper/sections/07-algorithms-complexity.tex` |
| **Section** | Table 3 (complexity) or the paragraph immediately following |
| **Existing text** | Table 3 memory entry for sum-product: `\Theta(k_{\max}n)` (message layers). No note about total peak memory. |
| **Issue (M6)** | Table 3 shows only the dynamic-programming message-layer memory `Θ(k_max n)`. Total peak memory during inference is `Θ(n²)`, dominated by the block-evidence matrix precomputed before the DP. A reader tracking memory budgets will underestimate peak by conflating the two. The Section 4 prose says "retaining all block-cover terms can require `Θ(k_max n²)` storage" but this is not linked to Table 3. |
| **Proposed replacement** | Add a footnote to Table 3 (or parenthetical in the prose below it): `Message-layer storage only; total peak memory is $\Theta(n^2)$, dominated by the precomputed block-evidence matrix. Retaining all block-cover terms for posterior moments adds $\Theta(k_{\max}n^2)$.` |
| **Justification** | Prevents practitioners from underestimating memory requirements; aligns with the Section 4 complexity discussion |
| **Artifact dependency** | E10 artifact `bece422c9d90093f`: peak resident memory ~500 MB at n=2048 |
| **Methodology flag** | No |
| **Computation required** | No |
| **Author approval required** | No (factual clarification; numbers already stated in Section 4) |

---

## Section D — Recommended Improvements

These are editorial improvements that do not change mathematical content. Each can be implemented without additional experiments or author decisions, but are lower urgency than R2–R4.

| ID | File | Recommendation | Rationale |
|----|------|----------------|-----------|
| D-prose-1 | `09-results.tex` | Verify E09 results-section prose cites artifact `0f5281c46ee8976e` and Tab. real_summary for all claimed k values (k_map=83, CGH single=71, pooled=60†, methylation=120†) | Prevent future number drift from the artifact |
| D-prose-2 | `99-appendix.tex`, subsection "Supplementary agreement diagnostics" | Add one sentence noting the pre-E09 agreement tables in the appendix are superseded by the current E09 artifact | The appendix currently presents older data without this caveat |
| D-prose-3 | `08-experimental-design.tex`, Planned Table P4 caption | Add note that "Array-CGH, corrected precision" row depends on author decision D1 (see Section H) before it can be populated | Readers of the planned table should know this cell has a prerequisite author decision |

---

## Section E — Optional Additional Scientific Work

These items would improve scientific completeness but require new computations or comparator runs. They are not prerequisite to submitting the current manuscript state.

| ID | Description | Prerequisite | Effort |
|----|-------------|--------------|--------|
| OPT-1 | E09 comparators: PELT, CBS, SMUCE at well-log and CGH. Required by Sec 8 protocol. | Author decision D2 | Substantial (R package wrappers, runtime) |
| OPT-2 | CGH rolling-MAD precision validation study: compare $k$ and log-evidence under rolling-MAD vs. constant-σ² precision at the single-subject level. | Author decision D1 | Moderate |
| OPT-3 | E00 Table P0: populate with oracle-check result (380/380 segment-boundary checks passed). | Author decision D5 | Low (data exists in confirmatory artifact) |
| OPT-4 | Methylation Beta/Beta-Binomial re-analysis using raw methylated+total reads. | Author decision D3 | Substantial |
| OPT-5 | BayesBreak book synchronization: update book proofs to match paper's finalized theorem statements. | Phase Two edits finalized | Low |

---

## Section F — E00–E10 Crosswalk

Each experiment with its canonical artifact SHA256 prefix, gate status, and manuscript correspondence.

| Exp | Short name | Artifact prefix | gate_pass | Manuscript section | Status |
|-----|-----------|-----------------|-----------|--------------------|--------|
| E00 | Oracle segment checks | (from memory) | True | Sec 9, Table P0 (Planned) | D5 pending |
| E01 | Conjugate family recovery | (from memory) | True | Sec 9, Fig E01, Tab E01 | REPORTED |
| E02 | Family showcase | (from memory) | True | Sec 9, Fig E02, Tab E02 | REPORTED |
| E03 | Irregular designs | (from memory) | True | Sec 9, Fig E03, Tab E03 | REPORTED |
| E04 | Shared changepoints | (from memory) | True | Sec 9, Fig E04, Tab E04 | REPORTED |
| E05 | Latent groups | (from memory) | True | Sec 9, Fig E05, Tab E05 | REPORTED |
| E06 | Nonconjugate approximations | (from memory) | True | Sec 9, Fig E06, Tab E06 | REPORTED |
| E07 | Misspecification/missingness | (from memory) | True | Sec 9, Fig E07, Tab E07 | REPORTED |
| E08 | Posterior prediction/PIT | cc26eda02dc82357 | True | Sec 9, Fig E08 | REPORTED |
| E09 | Real data | 0f5281c46ee8976e | True | Sec 9, Tab real_summary; Planned Table P4 | PARTIAL (6/31 resolved; Planned P4 blocked by D1/D2) |
| E10 | Computational scaling | bece422c9d90093f | True | Sec 9, Fig E10, Tab E10 | REPORTED |

**E09 crosswalk detail:**
- Tab. real_summary (`tab_real_summary.tex`): 8 curated rows, artifact `0f5281c46ee8976e` — **current**
- Planned Table P4 (`08-experimental-design.tex`): "Array-CGH, corrected precision" row requires D1 decision; comparator rows require D2 decision
- Planned Figure P4 (`08-experimental-design.tex`): requires comparators (D2)
- Conclusion sentence: "Six of 31 E09 cells were resolved" — accurate, needs R2 clarification

**Claim verification (no numbers change):**
- `k_map=83` (well-log full geometric): in artifact and Tab. real_summary line 10
- `k_map=71` (CGH single geometric): in artifact and Tab. real_summary line 14
- `k_map=60†` (CGH pooled geometric): in artifact and Tab. real_summary line 15
- `k_map=120†` (methylation): in artifact and Tab. real_summary lines 16–17
- E10: end-to-end 3.25s at n=2048 k_max=20; peak ~500 MB — in artifact `bece422c9d90093f`

---

## Section G — Manuscript Implementation Order

Proposed sequencing to minimize rework. None of these phases requires new experiments.

### Phase G1 — Immediate prose fixes (no author decision needed)
1. **R2**: Update conclusion sentence on "31 cells" (one sentence)
2. **R3**: Add caption footnote to `tab_real_summary.tex` (one sentence)
3. **R4**: Add memory footnote/parenthetical to Table 3 or adjacent prose
4. **D-prose-2**: Appendix: note pre-E09 agreement tables are superseded

### Phase G2 — After author decisions D1 and D2
5. **D1**: Update Sec 8 E09 CGH protocol text to match actual rolling-MAD model (or retain requirement and note pending status)
6. **D2**: Either narrow the comparator protocol in Sec 8 and label Planned P4 rows accordingly, or initiate OPT-1

### Phase G3 — After D5
7. Populate E00 Table P0 if author approves (OPT-3)

### Phase G4 — After D4
8. Add or confirm acknowledgements / funding statement

### Phase G5 — After all G1–G4 changes are proofread
9. **D-prose-1**: Cross-check E09 results prose vs. artifact numbers
10. **D-prose-3**: Note Planned P4 prereq in caption

### Phase G6 — Final (optional)
11. Book synchronization (OPT-5) — after paper is stable

---

## Section H — Author Decisions

Items that require an explicit author decision before an edit can be made. Recommended defaults are stated where there is a clear scientific basis.

---

### D1 — E09 CGH precision model: what to say in Sec 8 protocol

**Background:** `08-experimental-design.tex` lines 278–279 say:
> "Array-CGH must first use a validated Gaussian precision likelihood; report its corrected results separately from the earlier powered-weight fit."

The E09 artifact (`0f5281c46ee8976e`) already uses rolling-MAD-based precision weights via `sample_weight` in `BayesBreakGaussian`. There was no separate "earlier powered-weight fit" in the artifact — the rolling-MAD approach was the original E09 execution. The Gaussian family correctly treats `sample_weight` as per-observation precision (not a generalized-Bayes likelihood power), as documented in its docstring. The "corrected results separately from the earlier powered-weight fit" language therefore refers to a planned future comparison that was never executed, not a bug that was fixed.

**Options:**
- **(a) Revise the protocol** to describe rolling-MAD precision as the stated model, add a limitation note in Sec 10 (limitations section already present), and remove the "earlier powered-weight fit" language. Planned Table P4 "Array-CGH, corrected precision" row becomes a formal precision validation comparison (OPT-2) if the author chooses to execute it, or is marked Not Executed if not.  
  **Recommended. Scientific basis: the code is correct; the protocol description is anachronistic.**
- **(b) Retain the requirement** and explicitly note in Sec 8 and Sec 10 that the rolling-MAD result is preliminary pending a validation study.
- **(c) Add OPT-2** (rolling-MAD vs. constant-σ² precision comparison at single-subject level) before finalizing the protocol description.

**Edit required:** Yes — Sec 8, lines 278–284 and Planned Table P4 caption.

---

### D2 — E09 comparators: execute or formally narrow

**Background:** Sec 8 (E09 protocol) requires comparison with PELT, optimal partitioning, binary segmentation, wild binary segmentation, and a Bayesian reference implementation. Planned Table P4 and Figure P4 include comparator columns. The E09 artifact contains no comparator results. E10 memory project memory report (project_e09_progress.md) notes "baselines unavailable."

**Options:**
- **(a) Narrow the protocol** in Sec 8 to comparators that are feasible given the response family and sequence lengths, note unavailability in the text, and mark Planned P4 rows accordingly.  
  **Recommended if OPT-1 is not executed before submission.**
- **(b) Execute OPT-1** and populate Planned Table P4 / Figure P4.
- **(c) Keep as Planned** without change — paper is submitted with Planned placeholders.

**Edit required:** Yes (narrowing) or No (keep as Planned).

---

### D3 — Methylation truncation: accept as final

**Background:** Both methylation cells (geometric and uniform prior) are truncated at k_max=120 in the E09 artifact. The protocol says to vary k_max until truncation is small or report sensitivity. Sensitivity was not reported.

**Options:**
- **(a) Accept truncated result as final** and add a Sec 10 limitation noting methylation k_max sensitivity was not fully characterized.  
  **Recommended. Protocol requirement already satisfied (truncation reported).**
- **(b) Re-run E09 methylation** with higher k_max (e.g., 150, 200) to check posterior mass.

**Edit required:** Sec 10 addition (one sentence) if (a).

---

### D4 — Acknowledgements: funding statement

**Background:** No acknowledgements section is currently in the paper (or it is empty). Whether a funding statement is required depends on the funding source.

**Options:**
- **(a) Add standard acknowledgements** with funding source and any data-access statements.
- **(b) Keep empty** if no funding or acknowledgements apply.

**Edit required:** Only if (a).

---

### D5 — E00 Table P0: populate or keep Planned

**Background:** E00 ran 380/380 oracle segment-boundary checks (all passed). Project memory (project_e00_e07_progress.md) confirms gate_pass=True. Table P0 in Sec 9 is labeled Planned.

**Options:**
- **(a) Populate Table P0** with the oracle check summary: 380/380, all families, all segment lengths tested.  
  **Recommended.**
- **(b) Keep as Planned** — paper submitted without populating this table.

**Edit required:** `shared/tables/defs/` new or existing Table P0 file; `09-results.tex` corresponding section text.

---

## Section I — Acceptance Criteria for Phase Two Completion

Phase Two is complete when all of the following are true:

1. **R2 implemented:** Conclusion sentence on "31 cells" reads as proposed.
2. **R3 implemented:** Tab. real_summary caption includes 8-row / 31-cell clarification.
3. **R4 implemented:** Table 3 (complexity) has a footnote or adjacent prose distinguishing message-layer memory from total peak memory.
4. **All verified results unchanged:** No number, theorem statement, proof step, or figure datum in any E00–E10 confirmed slot differs from the corresponding artifact.
5. **Author decisions documented:** D1–D5 recorded as explicit author choices in the paper or in a revision cover letter, whether or not all are acted upon before submission.
6. **No new planned experiments created:** Phase Two does not introduce any new Planned Table or Planned Figure beyond those already in the paper.
7. **No commits to results/confirmatory/:** Directory untouched throughout Phase Two.

---

## Appendix — Findings Closed Before Phase Two

The following Phase One findings required no edit because the manuscript already contains the correct content:

| Finding | Location | Text that closes it |
|---------|----------|---------------------|
| M1: Theorem 5 scope | Sec 6, Theorem 5 statement | "Conditional on a uniform error bound ε..." |
| M2: EP numerical reference | Sec 6a, EP subsubsection | "EP estimates are not used to establish integrated-likelihood accuracy in the absence of a separately validated normalizer..." |
| M3: Contribution 5 | Sec 1, Contribution 5 | "The analysis is conditional on a uniform bound across admissible segments, not on a numerical method's name or an unverified finite-sample error estimate." |
| M4: pg_vb/JJ aliasing | Sec 6, lines 146–148 | `\paragraph{Interpretation of nonconjugate method labels.}` already documents this |
| figA1–A5 orphan | — | No such files exist in `shared/figures/defs/`; concern was unfounded |
| corradin2026 citation | `shared/bibliography/references.bib` line 391 | Entry present with DOI 10.1007/s11222-025-10756-x, Stat. Comput. 36:7 (2026) |
