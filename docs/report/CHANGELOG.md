# CHANGELOG

## Summary

This pass produced a publication-development-ready BayesBreak project and an updated
coding-agent handoff package from the approved consolidated edit plan (Phase Two). The revised
project preserves the existing LaTeX structure, document class, bibliography style, macros,
labels, figure and table environments, placeholder real-data figures, placeholder real-data
tables, planned experiments, expected outputs, projected outputs, implementation plans, and the
manuscript's theoretical and methodological direction. No completed empirical results, benchmark
scores, figures, tables, or observed findings were fabricated. New theoretical material is a
single proposed corollary derived from assumptions already in force, plus clearly labeled remarks
and one explicitly labeled exploratory appendix note.

Unlike the previous pass, a full LaTeX toolchain was available in this environment, so the
revised project was compiled end to end.

### Applied change counts by category

- **A. Technical errors and inaccuracies:** 8 items (4 bibliographic corrections, 3 new-citation
  insertions tied to corrected attributions, 1 prose qualifier).
- **B. Theoretical development:** 4 items (1 proposed corollary with supporting remark, 2
  clarifying remarks, 1 tightened theorem assumption).
- **C. Rigor and completeness improvements:** 4 items (forward reference, checklist linkage,
  per-family sign statement, planned sensitivity metric).
- **D. Clarity and organization:** 3 applied items (abstract reordering, positioning-claim
  softening, modular-comparator positioning); 1 no-op item retained for transparency.
- **E. Planned/expected/projected/placeholder material:** 3 items (9 placeholder captions
  prefixed with "(PLANNED)", 1 planned-baseline extension).
- **F. New theory/methodology/experiment/handoff additions:** 4 items (12 new bibliography
  entries with prose integration, 1 proposed corollary, 1 exploratory appendix note, handoff
  verification tasks).
- **G. Other substantive additions:** 2 items (ruptures software citation, this changelog
  rewrite).
- **H. Minor edits and LaTeX fixes:** 4 items (2 bibliography entry refinements, 2 overfull-box
  fixes; 2 no-op/flag-only items retained for transparency).

### Preservation statement

Planned-experiment, expected-result, projected-result, implementation-plan, and placeholder
material was preserved. The four real-data placeholder case studies (well-log, array-CGH,
S&P 500, methylation) remain in the project with their placeholder figures and tables intact.
Real-data figures and tables remain placeholders until finalized pipelines generate observed
outputs; their captions are now prefixed "(PLANNED)" per the approved status convention.
Cautious or conditional language is retained only where it distinguishes planned outputs from
observed results, assumptions from conclusions, and approximation guarantees from exact conjugate
inference.

### Baseline build status

A full TeX Live toolchain was available. The original project did not compile out of the box in
this environment because three packages/options were unavailable on the minimal install
(`lmodern.sty`, `bbm.sty`, and the `latin` Babel option) and `microtype` font expansion fails on
non-scalable fonts. Reversible portability shims (item Build-A1) were added so the project
compiles on minimal installs while remaining no-ops on a full install. With those shims, the
**original project compiled to 92 pages** with zero errors, zero undefined references, zero
undefined citations, and zero multiply-defined labels.

### Revised build status

The revised project compiles end to end via `pdflatex -> bibtex -> pdflatex -> pdflatex` with
zero errors, zero undefined references, zero undefined citations, and zero multiply-defined
labels. The **revised PDF has 98 pages** after the CG integration pass. The Phase Three build was 96 pages, the Phase Four build 97 pages; the additional page reflects the new CG-1 remark, CG-2 corollary with proof, CG-3 contract paragraph, the two CG-4 limitations paragraphs, and the resolved appendix recipe. Box warnings: 15 overfull and 50 underfull `\hbox` (down from 17 overfull at baseline).
No avoidable new warnings were introduced; the remaining overfull boxes are all under 9 pt and
are font-substitution artifacts (headings and run-in paragraph headers) expected to shrink
further on a full install with the intended `lmodern` fonts.

### Files added or changed

- **Changed source files:** `bayesbreak.tex`, `saim.cls`, `reference/cite.bib`,
  `sections/0-abstract.tex`, `sections/1.intro.tex`, `sections/2.problem.tex`,
  `sections/4.method.tex`, `sections/5.algorithms.tex`, `sections/6.evaluation.tex`,
  `sections/8.appendix.tex`.
- **Updated deliverables:** `CHANGELOG.md`, `CODING_AGENT_HANDOFF.md`.
- **New bibliography entries:** 12 (see Bibliography section below). **No new package, macro,
  figure file, or table file was added.** Two portability shims reference packages
  (`lmodern`, `bbm`) only through `\IfFileExists` guards with fallbacks, so no new hard package
  dependency was introduced.

---

## Front Matter, Root File, and Build Configuration

### Build-A1 -- `saim.cls`, `bayesbreak.tex`

- Category: A / H (compilation).
- Change: Added reversible, well-commented portability shims so the project compiles on minimal
  TeX installs: `microtype` loaded with `expansion=false`; the unused `latin` Babel option
  dropped; `lmodern` loaded only via `\IfFileExists` with a graceful fallback to default
  Computer Modern; `bbm` loaded via `\IfFileExists` with a `\mathbbm` -> `\mathbb` fallback for
  the single `\mathbbm` use in the manuscript.
- Rationale: The original project required packages/options absent on minimal installs. The
  shims are no-ops on a full installation and guarantee compilation otherwise; they do not alter
  the manuscript's content, template, or formatting conventions.

### Bib-F1 -- `reference/cite.bib`

- Category: H / F (bibliography).
- Change: Corrected the `hutter2006bpcr` entry title to "Exact Bayesian Regression of Piecewise
  Constant Functions" (the verified published title), keeping year 2007 and venue
  *Bayesian Analysis* 2(4):635--664. The citation key was left unchanged to avoid silent
  breakage.
- Rationale: The entry title did not match the published paper; the year/title were internally
  inconsistent.

---

## Abstract

### 0-D1 -- `sections/0-abstract.tex`

- Category: D (clarity).
- Change: Reordered the final two sentences so the completed-synthetic versus planned-real-data
  status reads first; no content was added or removed.
- Rationale: Surfaces the empirical-status distinction earlier without weakening the technical
  summary.

---

## Section 1: Introduction, Related Work, and Positioning

### 1-A1 / 1-A2 / 1-A3 -- `reference/cite.bib`

- Category: A (technical errors).
- Change: Corrected three bibliography entries against verified sources: `auger1989segment`
  (title and venue corrected to "Algorithms for the Optimal Identification of Segment
  Neighborhoods," *Bulletin of Mathematical Biology* 51(1):39--54); `frick2014smuce` (venue
  corrected to *JRSS-B* 76(3):495--580); `bleakley2011groupfused` (converted to an `@misc`
  arXiv:1106.4199 entry, with the venue flagged for author verification in the handoff).
- Rationale: The original entries listed incorrect titles or venues; the prose characterizations
  were accurate and were retained.

### 1-A4 -- `sections/1.intro.tex`, `reference/cite.bib`

- Category: A / F (technical errors, new additions).
- Change: Added one sentence to the "Bayesian offline segmentation and product partitions"
  paragraph acknowledging Hartigan (1990) as the abstract origin of partition models, Chib
  (1998) as the intermediate hidden-Markov Bayesian multiple-changepoint formulation, and Wyse,
  Friel & Rue (2011) as exact-DP segmentation with within-segment dependence. Three new
  bibliography entries (`hartigan1990partition`, `chib1998estimation`, `wyse2011approximate`).
- Rationale: Closes a genuine gap in the historical positioning that a reviewer would notice.

### 1-D1 -- `sections/1.intro.tex`

- Category: D (clarity, positioning accuracy).
- Change: Softened the "fragmented literature" claim to the verified-defensible phrasing and
  added an explicit clause acknowledging the multi-sequence and dependence exceptions (Fearnhead
  & Liu 2011, Fan & Mackey 2017, Quinlan et al. 2024).
- Rationale: The original blanket claim was slightly overstated relative to the literature.

### 1-D2 / G-1 -- `sections/1.intro.tex`, `reference/cite.bib`

- Category: D / G (clarity, positioning).
- Change: Added a sentence to the "Positioning of BayesBreak" paragraph positioning the
  framework relative to the closest modular comparators, the `ruptures` Python library and the
  `changepoint` R package, noting these are optimization-based and frequentist. New bibliography
  entries `truong2018ruptures` and `killick2014changepoint`.
- Rationale: A reviewer would expect explicit engagement with the closest modular toolkits.

### 5-A1 -- `sections/1.intro.tex`, `reference/cite.bib`

- Category: A (technical correctness).
- Change: Added an expected-case-and-conditions qualifier to the PELT complexity mention (linear
  cost is expected-case under a changepoint-density condition; worst case remains quadratic) and
  cited the functional-pruning algorithms FPOP and SNIP. New bibliography entry
  `maidstone2017optimal`.
- Rationale: Prevents an unqualified linear-cost claim and engages modern pruned-DP work.

---

## Section 2: Problem Formulation and Inferential Targets

### 2-C1 -- `sections/2.problem.tex`

- Category: C (rigor).
- Change: Added a forward reference in the "Computational scope" subsection acknowledging the
  optional resegmentation-mode prediction cost of `O(k_max m^2)` per group, which is otherwise
  introduced only much later in the prediction section.
- Rationale: Completeness of the computational-scope statement.

---

## Section 4: Method (EF Block Evidence, DP, Irregular Designs, Pooling, Latent Groups, Families, Non-Conjugate Blocks, Prediction)

### 4-B1 / F-2 -- `sections/4.method.tex`

- Category: B / F (theoretical development, new addition).
- Change: Inserted a new corollary (`cor:abs-prob`, "Absolute probability error for the
  segment-count posterior") immediately after the boundary-ranking corollary, with a full proof
  and a status remark (`rem:abs-prob-status`). The corollary converts the existing
  posterior-odds stability bound into an explicit absolute-probability and total-variation bound
  for the segment-count posterior, derived entirely from assumptions already in force for
  Proposition `prop:stability`. The closing sentence of `prop:stability` was updated to point to
  the new corollary instead of leaving the conversion implicit.
- Rationale: A proposed strengthening that pre-empts an obvious reviewer request; the odds-level
  guarantee alone does not give the absolute-probability statement that downstream thresholding
  rules require.

### 4-B2 -- `sections/4.method.tex`

- Category: B (theoretical development).
- Change: Added a remark (`rem:marg-eq-joint`, "When the two summaries agree") after the
  marginal-versus-joint-MAP counterexample, stating the boundary case in which independently
  maximizing the marginals does recover the joint MAP segmentation.
- Rationale: The manuscript stated the discrepancy but never the converse condition.

### 4-B3 -- `sections/4.method.tex`

- Category: B (rigor, theory).
- Change: Tightened the wording of Laplace-approximation assumption (A4) and added a clarifying
  sentence separating what strict log-concavity provides (uniqueness of the maximizer) from what
  it does not (tail control), which must instead come from coercivity or compact-domain
  truncation.
- Rationale: Removes a slight informality in the relationship between (A4) and the
  log-concavity lemma.

### 4-C1 -- `sections/4.method.tex`

- Category: C (rigor, clarity).
- Change: Rewrote the "Approximation-validation checklist" paragraph so each checklist item is
  explicitly tied to the specific failure mode it guards against (Remark `rem:failure-modes`),
  and so that the relationship to Corollary `cor:ranking` is made explicit.
- Rationale: Operationalizes the failure-mode discussion and the stability theory.

### 6-A1 -- `sections/6.evaluation.tex` (well-log subsection)

- Category: A (technical errors), with `reference/cite.bib` and `sections/8.appendix.tex`.
- Change: Corrected the well-log NMR dataset attribution. The body subsection now cites the
  primary source (O Ruanaidh & Fitzgerald 1996) alongside the downstream users, states that the
  cleaned version is used, and references Fearnhead & Rigaill (2019) for the raw/cleaned
  distinction. New bibliography entries `oruanaidh1996numerical`, `lai2005comparative`,
  `fearnhead2019changepoint`.
- Rationale: The dataset was attributed only to a downstream user, and the appendix recipe used
  a dataset object name (`Lai2005fig4`) that actually refers to a different (array-CGH)
  benchmark.

### 6-A2 -- `sections/6.evaluation.tex` (array-CGH and methylation subsections)

- Category: A (technical errors), with `reference/cite.bib`.
- Change: Converted prose-only attributions to proper citations: Snijders et al. (2001) for the
  Coriell array-CGH panel and Loyfer et al. (2023) for the methylation atlas. New bibliography
  entries `snijders2001assembly`, `loyfer2023atlas`, `killick2014changepoint`.
- Rationale: These works were named in prose but had no bibliography entries, so they produced
  no formatted references.

### 4-D1 -- `sections/4.method.tex`

- Category: D (organization).
- Change: No-op, retained for transparency. The method file is long, but per Objective 8 the
  project layout was not reorganized.
- Rationale: Preservation of project structure.

---

## Section 5: Algorithms, Complexity, and Numerical Considerations

### 5-C1 -- `sections/5.algorithms.tex`

- Category: C (rigor, handoff).
- Change: Added a per-family statement of moment-numerator sign to the log-block-evidence
  storage paragraph: the conjugate-family moment numerators are strictly positive and may be
  stored directly in log space, with the signed-log path needed only for sign-changing targets
  such as a centered Gaussian mean or a non-conjugate Laplace test function.
- Rationale: Tells an implementer precisely when the signed-log representation is required.

---

## Section 6: Experiments and Results

### 6-E1 / 6-E2 -- `sections/6.evaluation.tex`

- Category: E (planned/placeholder material).
- Change: Prefixed all nine real-data placeholder figure and table captions
  (`tab:realdata-status`, `fig:welllog`, `tab:real_welllog`, `fig:cgh`, `tab:real_cgh`,
  `fig:spx`, `tab:real_spx`, `fig:methylation`, `tab:real_methylation`) with "(PLANNED)" per the
  approved status convention; removed the now-redundant inline "(placeholder)" wording. All
  placeholder content, `---` cells, and pipeline cross-references were preserved.
- Rationale: Applies the explicit "(PLANNED)" caption convention for placeholder artifacts.

### 6-E3 -- `sections/6.evaluation.tex`, `reference/cite.bib`

- Category: E / F (planned material, new addition).
- Change: Extended the planned pre-submission frequentist baseline list to include the
  functional-pruning algorithms FPOP and SNIP (Maidstone et al. 2017). Remains explicitly
  planned work.
- Rationale: Strengthens the planned comparison set with modern pruned-DP baselines.

### 6-C1 -- `sections/6.evaluation.tex`

- Category: C (rigor, experiment plan).
- Change: Added a planned prior-sensitivity diagnostic to the evaluation-protocol description --
  recording variation of `P(k|y)` and the boundary marginals under prior perturbation -- tied to
  the already-planned ablations over `p(k)` and `g`. Marked as planned.
- Rationale: The planned ablations had no associated robustness metric.

---

## Section 8: Appendix

### App-A1 -- `sections/8.appendix.tex`

- Category: A (technical errors).
- Change: Converted informal prose attributions in the real-data appendix to proper citations
  (O Ruanaidh & Fitzgerald 1996, Snijders et al. 2001, Loyfer et al. 2023, Killick & Eckley 2014).
  Corrected the well-log R recipe to remove the incorrect `Lai2005fig4` dataset object name,
  replacing it with a placeholder object name and an explicit author-verification note, and
  added an explicit warning that `Lai2005fig4` is the array-CGH example of Lai et al. (2005).
  Flagged the methylation atlas distribution channel (GitHub path, GEO accession) as an
  author-verification task.
- Rationale: Removes a misleading dataset attribution and links prose references to the
  bibliography.

### App-B1 -- `sections/8.appendix.tex`

- Category: B (rigor, theory).
- Change: Added a remark (`rem:renewal-scope`, "Scope of the renewal equivalence") after the
  renewal-equivalence proposition, formally delimiting the boundary case: for general
  irregular designs the factorized prior remains a valid product-partition prior but is no
  longer an i.i.d. renewal law, so the renewal-process language is heuristic motivation rather
  than an exact equivalence.
- Rationale: Prevents the renewal interpretation from being read as more general than it is.

### F-3 -- `sections/8.appendix.tex`

- Category: F (new addition, exploratory).
- Change: Added a new appendix subsection (`app:latent-template-positioning`, "Exploratory note:
  relation of the latent-template mixture to multi-sequence Bayesian segmentation"), explicitly
  labeled as a future-direction discussion rather than a completed contribution. It contrasts
  the latent-template mixture with BASIC (Fan & Mackey 2017) and JRPM (Quinlan et al. 2024)
  along three axes (discrete assignment versus shared-prior parameter; exact max-sum DP versus
  MCMC; finite template-mixture objective versus full posterior averaging) and records the
  fuller comparison as planned work.
- Rationale: The research flagged that the latent-template-EM novelty claim is defensible but
  must be explicitly contrasted with the closest comparators; the note is kept exploratory so it
  does not derail the main narrative.

### F-1 -- `sections/8.appendix.tex` (annotated literature table)

- Category: F (new additions).
- Change: Folded the high-value missing references into the annotated literature review table:
  Hartigan (1990) and Chib (1998) into the foundational and exact-DP rows; Wyse, Friel & Rue
  (2011) into the exact-DP row; Maidstone et al. (2017) into the frequentist partitioning row;
  and two new rows for Fearnhead & Rigaill (2019) (robust segmentation / the well-log benchmark)
  and Jewell et al. (2022) (post-selection inference for changepoints). New bibliography entry
  `jewell2022testing`.
- Rationale: Closes genuine positioning gaps identified during the literature investigation.

---

## Bibliography, Tables, Figures, and Source Hygiene

### H-1 -- `reference/cite.bib`

- Category: H (bibliography).
- Change: Refined the `rigaill2010pruned` entry to an `@misc` form with the arXiv identifier
  (arXiv:1004.0887) and a note pointing to the later journal version; flagged for author
  verification in the handoff.
- Rationale: The original entry's venue and year did not clearly correspond to a single
  published artifact.

### H-2 -- `reference/cite.bib`

- Category: H (bibliography).
- Change: Pinned the `punskaya2002bayesian` entry to the IEEE Transactions on Signal Processing
  journal version (50(3):747--758), with a note on the conference version; flagged for author
  verification in the handoff.
- Rationale: The original entry used a vague "also circulated as a technical report" note.

### H-3 -- `sections/8.appendix.tex`

- Category: H (LaTeX fixes).
- Change: Fixed the two largest overfull `\hbox` warnings (46.9 pt and 36.7 pt) by converting
  two long inline mathematical expressions in the appendix -- the block ELBO inequality and the
  aggregated sufficient-statistics definitions -- into displayed equations. Overfull-box count
  fell from 17 to 15; the remaining boxes are all under 9 pt and are font-substitution artifacts.
- Rationale: Removes the only genuinely disruptive overfull boxes; the remaining sub-9 pt boxes
  are headings and run-in headers that do not warrant rewriting.

### H-4 / H-5 -- `math_commands.tex`, `bayesbreak.tex`

- Category: H.
- Change: No-op / flag-only, retained for transparency. `math_commands.tex` was reviewed and
  found defect-free (it already comments out its `\eqref` override in favor of `amsmath`). The
  stale commented-out author/affiliation scaffolding lines in `bayesbreak.tex` were left in
  place as harmless; their removal can be done at the author's discretion.
- Rationale: No corrective action required.

### New bibliography entries (12 total)

`oruanaidh1996numerical`, `lai2005comparative`, `snijders2001assembly`, `loyfer2023atlas`,
`killick2014changepoint`, `hartigan1990partition`, `chib1998estimation`, `wyse2011approximate`,
`maidstone2017optimal`, `fearnhead2019changepoint`, `jewell2022testing`, `truong2018ruptures`.
All entries were verified against reliable sources during the Phase Two literature
investigation. No citation was invented; entries whose details remain uncertain are flagged for
author verification in `CODING_AGENT_HANDOFF.md` (Section 8).

---

## Phase Four: Independent Audit and Confirmed Fixes

An independent review of the full revised manuscript was conducted as the Phase Four audit. The
audit found no substantive defects in the manuscript's theory, methodology, claims, structure,
or compilation integrity. The new mathematical material was verified: the
marginal-versus-joint-MAP counterexample is arithmetically correct, and the new
absolute-probability corollary (`cor:abs-prob`) has a sound proof. All Phase Three edits were
confirmed applied without contradiction or claim drift, and all planned, expected, projected, and
placeholder material was confirmed intact and correctly labeled. Two minor polish items were
surfaced and confirmed for application.

### P4-H1 -- `sections/6.evaluation.tex`

- Category: H (minor cross-reference).
- Change: Replaced a vague section-level cross-reference in the prior-sensitivity diagnostic
  sentence (added under 6-C1) with a label-free phrase pointing to the planned pre-submission
  additions described later in the same subsection.
- Rationale: The original `\ref` resolved to the evaluation section as a whole rather than to the
  specific planned-additions discussion; the label-free phrasing is unambiguous.

### P4-H2 -- `sections/8.appendix.tex`

- Category: H (minor wording).
- Change: Added a brief parenthetical to the exploratory appendix note
  (`app:latent-template-positioning`) linking the planned BASIC/JRPM comparison to the
  corresponding open-questions item and external-baseline experiment in
  `CODING_AGENT_HANDOFF.md`.
- Rationale: Improves traceability between the manuscript's exploratory note and the handoff
  document.

### Phase Four build

After applying the two confirmed fixes, the project was recompiled end to end via
`pdflatex -> bibtex -> pdflatex -> pdflatex` with zero errors, zero undefined references, zero
undefined citations, and zero multiply-defined labels. The PDF page count moved from 96 to 97
pages: the two single-sentence wording changes pushed content across one page boundary. Box
warnings are unchanged at 15 overfull and 50 underfull `\hbox`.

## Integration Pass: External-Draft Cross-Comparison (CG Items)

After Phase Four delivery, an external ChatGPT-revised package was supplied for comparison. The
external package itself did not compile (missing portability guards plus 167 PGF errors from
corrupted TikZ plate diagrams in `sections/4.method.tex`), so it could not be merged
wholesale. A targeted comparison identified four substantive ideas worth integrating into the
working copy as discrete, identified edits, plus one bibliography correction surfaced by a
companion verification research task.

### CG-1 -- `sections/4.method.tex`

- Category: B / D (theoretical clarification).
- Change: Inserted a new remark (`rem:score-matrix-exactness`, "Exact DP conditional on the
  score matrix") at the opening of the dynamic-programming section. The remark draws an explicit
  line between the algebraic exactness of the DP recursions for any supplied admissible
  block-score matrix and their interpretation as exact Bayesian recursions for the
  data-generating model only when those entries are true marginal likelihoods under the stated
  product-partition prior. The remark cross-references `eq:A0`, `prop:stability`, and the new
  `cor:abs-prob`.
- Rationale: The distinction between exact-DP-on-supplied-scores and exact-DP-on-true-evidences
  is the most reviewer-relevant conceptual sharpening identified in the external comparison. It
  reinforces, rather than overlaps with, the absolute-probability bound added in Phase Three.

### CG-2 -- `sections/4.method.tex`

- Category: B / F (theoretical addition).
- Change: Inserted a new corollary (`cor:boundary-event-sum`, "Fixed-count boundary-event
  normalization") with a one-line proof, stating that the boundary-event marginals
  $\sum_{i=1}^{n-1} P(b_i=1\mid y,k) = k-1$ for any fixed $k$ with positive evidence. The
  corollary is placed immediately after the existing posterior-evidence sanity-checks paragraph
  and is followed by a sentence explaining its use as an implementation localization check.
- Rationale: The identity was previously stated only as an embedded clause within the DP
  correctness theorem. Elevating it to a labeled, citable corollary improves the paper's
  presentation as an implementation-oriented framework and gives a clean named target for the
  per-DP boundary-sum check.

### CG-3 -- `sections/3.setup.tex`

- Category: C / D (rigor, coding-agent readiness).
- Change: Added an explicit "Block-score contract" paragraph immediately before the
  renewal-process pointer, formalizing the per-family invariant that block routines return
  finite log-evidence on admissible blocks and the sentinel $-\infty$ on inadmissible blocks,
  that $\log C_k$ is computed under the same admissibility mask as the score array, and that
  exact and surrogate scores must not be mixed silently. No change was needed in
  `sections/5.algorithms.tex` because the Phase Three sanity-check paragraph already aligns
  with the new contract.
- Rationale: Consolidates scattered guidance into one labeled passage that an implementer can
  cite. Pre-empts the most common class of implementation bugs (admissibility-mask mismatch
  between the score matrix and the prior normalizer).

### CG-4 -- multiple files

- Category: A / D / F (technical correctness, clarity, new content).
- Change: Four coupled edits informed by an independent bibliographic verification research
  task that checked four references introduced by the external draft.
  (a) Added a "Limitations and manuscript-development status" paragraph at the end of
  `sections/1.intro.tex` (before the Paper-organization paragraph) and a parallel
  "Limitations and manuscript-development status" paragraph at the end of
  `sections/7.conclusion.tex`. Both paragraphs scope the conjugate-DP guarantees, the
  approximation-controlled non-conjugate extension, and the EM caveats, and identify the
  real-data case studies as planned pipelines.
  (b) Updated the well-log dataset source paragraph in `sections/6.evaluation.tex` and the
  well-log appendix recipe in `sections/8.appendix.tex` to add Fearnhead & Clifford (2003)
  alongside Ó Ruanaidh & Fitzgerald (1996) as the popularizing changepoint-literature
  reference, and to resolve the previous `<welllog_object>` placeholder with the verified
  package-and-object pair `changepoint.influence::welldata` (a length-4050 numeric vector).
  (c) Added one new bibliography entry, `fearnhead2003particle`, verified against the JRSS-B
  publisher page and the CRAN `changepoint.influence` package manual: P.~Fearnhead and
  P.~Clifford (2003), "On-line inference for hidden Markov models via particle filters,"
  *JRSS-B* 65(4):887--899, DOI 10.1111/1467-9868.00421.
  (d) Corrected the methylation atlas GitHub repository attribution in
  `sections/8.appendix.tex`: replaced the previous `nloyfer/meth_atlas` pointer (which the
  research verification showed implements the older Moss et al.\ 2018 array-deconvolution
  method, not the 2023 atlas) with `nloyfer/wgbs_tools` and `nloyfer/UXM_deconv`, the
  companion software for the 2023 atlas paper.
- Rationale: The verification research confirmed the four ChatGPT-introduced references as
  pointing to real works; the Fearnhead-Clifford entry adds genuine content (a popularizing
  reference for the well-log series and a concrete package-and-object name) while the Loyfer
  GitHub correction prevents the manuscript from pointing readers to the wrong code base.

### Build after CG integration

The project recompiles via `pdflatex -> bibtex -> pdflatex -> pdflatex` with zero errors, zero
undefined references, zero undefined citations, and zero multiply-defined labels. The PDF is
**98 pages** (one more than the 97-page Phase Four build), and the bibliography now has 48
entries (one more than the 47-entry Phase Three count). Box warnings are unchanged at 15
overfull and 50 underfull, all under 9 points and font-substitution artifacts.

## Proposed changes not applied

None. All items on the Phase Two approval checklist were approved and applied, both Phase Four
fixes were confirmed and applied, and the four CG integration items (CG-1 through CG-4) were
approved and applied. The four no-op / flag-only items (4-D1, H-4, H-5, and the
structural-preservation aspects of Build-A1) are recorded above for transparency.

## New material requiring future work

- The proposed corollary `cor:abs-prob` is a complete result (statement and proof) and requires
  no further work, but its conservative `k_max` bound could be tightened in a later iteration.
- The exploratory appendix note `app:latent-template-positioning` records a planned head-to-head
  comparison against BASIC-style and JRPM-style inference; this is logged in
  `CODING_AGENT_HANDOFF.md`.
- The planned prior-sensitivity diagnostic (6-C1) and the FPOP/SNIP baseline extension (6-E3)
  are planned experiment-plan additions logged in `CODING_AGENT_HANDOFF.md`.
- Author-verification tasks for several bibliography entries (Bleakley & Vert venue;
  Denison-Mallick-Smith 1998 identity; Punskaya venue; Rigaill version; Müller-Quintana-Rosner
  end page; well-log dataset object name; methylation atlas accession/repository) are logged in
  `CODING_AGENT_HANDOFF.md`, Section 8.

## Compilation record

- Build command: `pdflatex bayesbreak.tex` -> `bibtex bayesbreak` -> `pdflatex bayesbreak.tex`
  (x2).
- Bibliography tool: BibTeX with `plainnat` style.
- Result: zero errors; zero undefined references; zero undefined citations; zero
  multiply-defined labels.
- Original PDF: 92 pages. Revised PDF after Phase Three: 96 pages. Revised PDF after Phase Four:
  97 pages. Revised PDF after CG integration pass: 98 pages.
- Warnings: 15 overfull and 50 underfull `\hbox` (baseline: 17 overfull). No avoidable new
  warnings introduced; remaining boxes are font-substitution artifacts.
