# BayesBreak manuscript — CHANGELOG

This changelog records substantive changes applied in the current revision pass. It excludes grammar
edits and formatting-only changes. Entries are listed in document order with the stable identifiers
defined in the Phase Two consolidated edit plan.

Build status after applied changes (final `pdflatex → bibtex → pdflatex → pdflatex` cycle, with one
extra settling pass): exit 0; 0 LaTeX errors; 0 undefined references; 0 undefined citations; 0
overfull/underfull h/v boxes; 0 LaTeX warnings; 110 pages compiled. Baseline before this pass: 100
pages. Net additions: 10 pages of new theoretical, methodological, empirical, and literature
content, with no shortening of existing material.

---

## 2. Problem formulation and inferential targets — `sections/2.problem.tex`

**[2-C02]** Inserted formal `\begin{definition}[Posterior summaries: marginal boundaries, segment
count, and joint MAP]` with label `def:posterior-summaries`. This consolidates the three downstream
posterior summaries — marginal boundary posterior $\mathrm{MB}_i(k)$, posterior over segment count
$\mathrm{PK}(k)$, and joint MAP segmentation $\widehat t^{\mathrm{MAP}}(k)$ — as first-class
labelled objects. The definition is referenced by the new metric definitions in §6 (`def:metric-ece`)
and by the decision flowchart in §5b. The definition also makes explicit that the three objects are
returned by distinct DP recursions and that their modes are not in general consistent with each
other.

---

## 4. Method — `sections/4.method.tex`

**[4-B02]** Added `\begin{proposition}[Forward--backward duality and total-evidence identity]` with
label `prop:fb-duality` immediately after the sum-product recursion equation \eqref{eq:LR}. The
proposition states (i) the boundary-condition support constraints, (ii) the total-evidence
identity $\widetilde L_{k,n}=\widetilde R_{k,0}$, and (iii) the boundary-marginal product form
$p(t_p=h\mid y,k)=\widetilde L_{p,h}\widetilde R_{k-p,h}/\widetilde L_{k,n}$. A constructive proof
by induction on $k$ is supplied in-place. The identity was previously checked numerically in the
DP-invariant test of `sec:alg-checklist` but never proved formally; this proposition closes the
gap.

**[4-B05]** Added `\begin{proposition}[Shared-boundary identifiability]` with label
`prop:shared-boundary-identifiability` immediately after the proof of
Theorem~\ref{thm:multisubject} (Exact pooling of block evidences). The proposition states that
the population log-likelihood aggregated under shared boundaries is uniquely maximized at the
true boundary vector, up to the trivial relabeling $t_0=0$, $t_k=n$. A KL-based proof is supplied
in-place. A companion `\begin{remark}[Conditions that make the identifying-block hypothesis
automatic]` (label `rem:identifying-block`) records the support condition under which the
non-degeneracy hypothesis is satisfied automatically. Both close the gap noted in the Phase Two
plan that the shared-boundary identifiability statement was previously only implicit.

**[4-C03 + Bib-01]** Added `\begin{remark}[Finite-mixture identifiability and the
overspecified-$G$ counterexample]` with label `rem:teicher-overspec` immediately after the
existing "Exact EM versus regularized or floored variants" remark in §10. The remark wires in
the previously unused bib entry `teicher1963identifiability` (Bib-01) as the canonical reference
for the saturated $G=G^\star$ identifiability theorem, and supplies an explicit
overspecified-$G>G^\star$ counterexample showing that two distinct $(\pi,\tau)$ configurations
can yield the same mixture density. Practical mitigations (held-out predictive selection of $G$,
cluster-stability diagnostics across restarts, mixing-weight floor) are documented.

---

## 5. Algorithms — `sections/5.algorithms.tex`

**[5-C01 + GLO-D01]** Promoted the previously prose-only multi-subject pooling description to a
formal `\begin{algorithm}` block titled "Pooled-DP for shared-boundary multi-subject inference
(log-space)" with label `alg:pool-shared-boundary`. Explicit `\KwIn` (per-subject log-block-evidence
arrays, length-prior log-cohesion, $k_{\max}$, group labels), `\KwOut` (per-group pooled
posteriors, MAP segmentations, segment-conditional parameter posteriors), step-by-step loop,
complexity comment ($\Theta(\sum_g(|\mathcal{S}_g|n^2 + k_{\max}n^2))$), and numerical-considerations
comment (log-space pooling, `logsumexp`, $-\infty$ absorption) are all present. This matches the
header-consistency standard required by GLO-D01.

**[5-C02 + GLO-D01]** Promoted the prose latent-template EM implementation notes to a formal
`\begin{algorithm}` block titled "Latent-template EM with exact responsibility and exact max-sum
M-step (log-space)" with label `alg:latent-em-detail`. Explicit `\KwIn` (per-subject log-block
evidences, $G$, $k_{\max}$, restart count $R$, optional mixing-weight floor), `\KwOut` (mixing
weights, templates, responsibilities, achieved $\ell_\star$), full restart loop with explicit
E-step (responsibility update) and two-part M-step (mixing-weight update then exact max-sum
template update), per-iteration complexity comment ($\Theta(GSn^2+Gk_{\max}n^2)$), and
numerical-considerations comment. Convergence criterion (relative $|\Delta\ell_\star|<10^{-6}$ and
template invariance under deterministic tie-breaking) is preserved as in the surrounding prose.

---

## 5b. Limitations — `sections/5b.limitations.tex`

**[5b-C01]** Inserted two new named failure-mode paragraphs after the "Out-of-scope settings"
paragraph. The "Identifiability failures (named)" paragraph documents three identifiability
failure modes — (i) label switching at saturated $G=G^\star$, cross-linked to
Proposition~\ref{prop:latent-identifiability} and \citet{teicher1963identifiability}; (ii)
overspecified-$G$ redundancy, cross-linked to Remark~\ref{rem:teicher-overspec}; (iii)
shared-boundary degeneracy, cross-linked to Proposition~\ref{prop:shared-boundary-identifiability}
and Definition~\ref{def:admissible-blocks}. The "Non-conjugate approximation outside the
small-$\varepsilon$ regime" paragraph names the failure mode triggered by the EP row of
Table~\ref{tab:nonconj_tradeoff} (empirical $\max|\Delta\log A^{(0)}|\approx 14.5$, well outside
the stability-bound regime) and cross-references the diagnostic-based fallback rule.

**[5b-C02]** Inserted a "Decision flowchart: which branch to use" paragraph with a 5-item
enumerated checklist mapping problem class (single-sequence conjugate, single-sequence
non-conjugate, shared-boundary multi-subject, heterogeneous-boundary multi-subject, prediction on
new data) to the appropriate BayesBreak branch and verification step. Cross-links to the relevant
theorems, algorithms, definitions, and tables in the body.

---

## 6. Experiments and results — `sections/6.evaluation.tex`

**[6-C01]** Inserted four formal metric definitions in §6.1 ("Metrics") immediately after the
Table~\ref{tab:metrics} protocol overview, so the metrics referenced throughout the section are
labelled cross-reference targets rather than informal terms.
- `\begin{definition}[Boundary precision, recall, and $F_1$ at tolerance $\tau$]` with label
  `def:metric-f1` — defines the one-to-one matching rule and $\mathrm{TP}_\tau/\mathrm{FP}_\tau/\mathrm{FN}_\tau$
  accounting, with explicit zero-denominator conventions.
- `\begin{definition}[Boundary-location mean absolute error]` with label `def:metric-mae`.
- `\begin{definition}[Expected calibration error for boundary marginals]` with label
  `def:metric-ece` — defines the bin-based ECE for boundary posteriors, cross-referenced to
  `def:posterior-summaries(a)`.
- `\begin{definition}[Held-out predictive log-likelihood]` with label `def:metric-loglik` —
  defines the per-observation and held-out variants used in §6 and §8.

**[6-E01a — Author decision A1-a]** Promoted `tab:real_welllog` from `planned` to
`partially-real`. The "Index-uniform prior" row is now populated with $\widehat k=23$ and MAP
log-evidence $-4989.28$, both taken verbatim from `figures/realdata_metrics.json` (the same
archived fit that produced Figure~\ref{fig:welllog}); the "Length-aware prior" row retains
`---` because the JSON records `needs_refit` for that configuration. Caption changed from
`(PLANNED)` to `(Partially populated)` with explicit row-by-row provenance. The surrounding prose
paragraph at line 236 is rewritten in mixed voice: row 1 is described in indicative voice (it is
populated), row 2 in projection voice (it is reserved pending the refit and reference-boundary
load). Header column name `MAP evidence` → `MAP log-evidence` for technical correctness.

**[6-E01b — Author decision A1-a]** Promoted `tab:real_cgh` from `planned` to `partially-real`.
Pooled log-evidence column populated for both rows: $76359.80$ for shared-boundaries (joint
pooled log-evidence at $k_{\mathrm{MAP}}=15$) and $109617.70$ for independent per-subject
(sum of per-subject log $A^{(0)}_{0,n}$ values under separate $\texttt{BayesBreakGaussian}(k_{\max}=15)$
fits). Both numbers are from `figures/realdata_metrics.json`. The Boundary F1 / MAE / Runtime
columns retain `---` because they require external Snijders-2001 annotations or a new timing
record. Caption rewritten with explicit per-row provenance and the note that the independent
strategy yields no single $\widehat k$. Surrounding prose at line 277 updated to mixed voice.

**[6-E01c — Author decision A1-a]** Promoted `tab:real_spx` from `planned` to `partially-real`.
The "Gaussian on $\log r_t^2$" primary row is now populated with $\widehat k=29$ and log
evidence $-1296.65$ from the same archived fit as Figure~\ref{fig:spx} ($n=566$). The "Poisson on
threshold crossings" row retains `---` because the JSON records `needs_refit`. Visual-alignment
column populated for row 1 with the qualitative descriptor and a forward-pointer to
Figure~\ref{fig:spx}; row 2 remains `---`. Caption and surrounding prose updated.

**[6-E01d — Author decision A1-a]** Promoted `tab:real_methylation` from `planned` to
`partially-real`. The "Region A, cell type 1" row is now populated with $\widehat k=15$ and
held-out log-predictive $-387.50$ from the archived chr21 \texttt{methylKit} \texttt{test1.myCpG}
fit ($n=1904$ CpGs, held-out $m=381$); the "Region A, cell type 2" row retains `---` because it
requires the Loyfer-2023 atlas pipeline (GEO~GSE186458 + \texttt{wgbs\_tools}/\texttt{UXM\_deconv})
which is not yet wired into the reproduction repository. The Boundary F1 vs.\ atlas column
retains `---` across both rows pending verified atlas annotations. Caption and surrounding prose
updated.

---

## 7. Conclusion — `sections/7.conclusion.tex`

**[7-D01]** Expanded the conclusion from 37 lines to roughly 85 lines, adding four new
substantive paragraphs without removing existing content.
- "What is exact and what is approximate." A three-regime taxonomy distinguishing
  Bayesian-exact-at-machine-precision (conjugate DP, forward--backward duality, pooled
  shared-boundary DP, max-sum backtracking), exact-for-a-finite-template-objective (latent-template
  EM with Teicher-identifiability caveat), and approximate-at-block-exact-at-DP (the
  $\varepsilon$-quantified non-conjugate routines).
- "Deferred empirical work." Four explicitly named deferred extensions (frequentist baselines
  PELT/WBS/SMUCE, Bayesian baselines RJMCMC/Fearnhead, partition-prior and pooling ablations
  populating reserved table rows, and a tens-of-thousands-$n$ scaling study).
- "Downstream applications." Explicit map from the four real-data case studies to their
  intended downstream targets (ctDNA CNA calling, methylation tissue-of-origin segmentation,
  cross-domain non-genomic application).
- "Reproducibility statement." Forward-pointer to Appendix~\ref{app:real-data} and to each
  archived JSON sidecar with its \texttt{y\_hash} and DP-diagnostic status.

Also corrected the previously planned-table claim "are planned benchmarking outputs rather than
completed measurements" to the new partial-population status reflecting the A1-a promotion of
Tables~\ref{tab:real_welllog}--\ref{tab:real_methylation}.

---

## 8. Appendix — `sections/8.appendix.tex`

**[8-Lit01 — Author decision A3-a]** Appended five modern-thread scaffold rows to
`tab:annotated-lit` covering: scalable Bayesian changepoint detection; survey and benchmark
literature (Aminikhanghahi & Cook 2017; Truong, Oudre & Vayatis 2020); multivariate and
multichannel changepoint methods; kernel/nonparametric and generalized-Bayes changepoint;
ctDNA / copy-number bioinformatics applications. Each row uses the project annotation pattern
(theme, representative papers, BayesBreak-relative annotation, limitation of cited work) and is
explicitly marked `[Phase-Four verification pending]` so the bibliographic metadata can be
verified in the Phase Four research-mode pass before final commit. No new bib entries were added
in Phase Three; this is deferred to Phase Four where venue/DOI/author-list verification is
required.

**[8-C01 — Author decision A2-c]** Added an "Additional diagnostic plots not shown in the body"
paragraph to Appendix~\ref{app:code} listing the six orphan PDFs on disk
(`fig6_mixture_discovery`, `fig7_snr_sensitivity`, `fig8_multivariate_shared`,
`fig9_model_selection`, `fig10_missing_data`, `fig4_latent_groups` uncropped variant) with a
one-line description of each. The paragraph explicitly notes that none of these diagnostics is
load-bearing for any body claim. This matches A2-c (document, do not wire).

**[8-D01]** Added a "Source-stability note: verify before use" paragraph at the head of
Appendix~\ref{app:real-data}. The paragraph names the moving-target external sources (CRAN
packages, Bioconductor releases, Yahoo Finance via \texttt{yfinance}, NCBI GEO accessions,
GitHub source-distribution repositories) and instructs the reader to (i) re-verify the package
version that ships the cited data object, (ii) re-verify the download endpoint/schema, (iii)
re-verify the GEO accession and series-matrix file naming, and (iv) re-verify the per-CpG
coverage convention. Cross-references the archived JSON sidecar \texttt{y\_hash} fields as the
cheapest cross-check available.

**[8-D01 follow-up]** Updated the four `\paragraph{Planned table output.}` headings inside the
per-dataset appendix subsections to `\paragraph{Partially-populated table output.}` to inherit
the partial-population status from the A1-a table promotion. Each paragraph is rewritten to (i)
describe in indicative voice the cells that were populated in §6, and (ii) describe in projection
voice the cells that remain reserved, with an explicit reason for each reservation.

---

## Build hygiene — no source-side edits

**[Build-H01]** No source-side changes. The clean four-pass build cycle (`pdflatex → bibtex →
pdflatex → pdflatex → pdflatex` for cross-reference settling) requires only a one-time
`mktexlsr` invocation if the TeX Live `lmodern` package was installed without registering its
ls-R database; this environmental step is documented in the new `CODING_AGENT_HANDOFF.md`.

---

## Deferred to Phase Four

- **[1-Lit01, 8-Lit01]** All `[Phase-Four verification pending]` markers in `tab:annotated-lit`
  require external verification of author, year, venue, and DOI metadata before the
  corresponding bib entries can be committed. The scaffold rows are content-complete in
  annotation style; only the citation keys are deferred.

## Items considered and deliberately not applied

- **[GLO-C01 / 3-C01]** Notation table at head of §3 — was already present in the prior pass at
  `3.setup.tex` L30–89 (`tab:notation`, 9 grouped sections, ~60 symbols). Audited; no further
  additions needed.
- **[GLO-C02]** Hypothesis-discipline pass — audit found existing theorems and propositions already
  state their hypotheses explicitly via `\begin{theorem}` / `\begin{proposition}` blocks and
  `\begin{assumption}` cross-references. No silent claims were found requiring rewriting.
- **[GLO-H01]** Single "well known" instance at `1.intro.tex` L8 — left in place: it is a
  literature-framing claim followed by four immediate citations, not a hidden proof step. Distinct
  from the "clearly/obviously/easy to see" anti-pattern that the journal-rigor checklist targets.
- **[1-D01, 1-D02]** Introduction polish — current contributions and paper-organization paragraphs
  are concise and structured; no measurable scientific gain from rewording.
- **[H01]** Commented `\pdfobjcompresslevel=0` at `bayesbreak.tex` L1 — purely cosmetic; left
  for the author to decide whether to retain or drop.
- **[8-B01, 8-B02]** Appendix proof headers (max-sum DP correctness, variational bounds) — audited;
  proof blocks already use `\begin{proof}` / `\end{proof}` delimiters correctly, no missing `\qed`.

---

## 2026-05-21 — Figure refresh and empirical-headline pass

This pass refreshed every bundled figure to publication quality and propagated the new
empirical headlines into the abstract, the relevant `§6` subsections, the `§8` supplementary
figures list, and the runtime companion table.

**[FIG-R01]** All bundled figure scripts in `scripts/figures/` and `scripts/figures/supplementary/`
were rewritten or restyled around the upgraded `_style.py` module (saim brand colors, larger
`axes.titlepad`, panel-label/title baseline alignment so labels never collide with titles).
Every figure under `docs/report/figures/` was re-rendered. Highlights:
* `fig4_latent_groups.pdf` — six-panel composition (per-group examples; boundary marginals;
  responsibility heatmap with truth stripe; group-averaged signal; assignment-confidence
  diagnostic that plots $r_{s,y_s}$ per sequence with the 0.5 decision threshold shaded and
  mis-assigned sequences circled). Default $\sigma$ raised from 0.35 to 1.0 so the EM faces a
  non-trivial problem (96% hard accuracy, one mis-assignment).
* `fig5_runtime_scaling.pdf` — extended to $n=800$, added a panel-B $k_{\max}$ sweep, and now
  annotates the empirical log-log slopes in both panels alongside theoretical $\mathcal{O}(n^2)$
  / linear-in-$k_{\max}$ reference lines.
* `fig7_snr_sensitivity.pdf` (script `figA2_snr_sensitivity.py`) — F1 with IQR band and a
  critical-$\sigma$ annotation, a column-normalised 2D histogram of selected $\hat k$, and an MSE
  panel against a $\sigma^2$ noise-oracle reference.
* `fig9_model_selection.pdf` (script `figA4_model_selection.py`) — single tight row of three
  posterior panels with shared $y$-axis, per-panel posterior-mode $\hat k$ annotations, and an
  inline rotated "true $k^{\star}=3$" label on the leftmost panel.

**[ABS-E01]** `sections/0-abstract.tex`: appended one sentence carrying the empirical
headlines now backed by the refreshed figures (ECE ≈ 0.010, Brier ≈ 0.011, 96% latent-mixture
accuracy at $\sigma=1.0$, principled posterior-$k$ drift under high noise, and the empirical
near-linear runtime slopes $\approx 1.07$ and $\approx 1.14$ over $n\in[50,800]$).

**[6-C04]** `sections/6.evaluation.tex` §6.4 (Calibration): corrected the quoted calibration
number to $\mathrm{ECE}\approx 0.010$ with Brier $\approx 0.011$ to match the refreshed `fig3`
inset.

**[6-E03]** `sections/6.evaluation.tex` §6.5 (Latent-group pooling): rewrote the prose, caption,
and supporting paragraph to describe the refreshed `fig4` ($\sigma=1.0$, 96% accuracy, six-panel
companion description with the new assignment-confidence diagnostic).

**[6-R01]** `sections/6.evaluation.tex` §6.7 (Runtime scaling): rewrote the prose and figure
caption to describe both panels of the refreshed `fig5`, quoted the empirical log-log slopes
($\approx 1.07$ at $k_{\max}=10$, $\approx 1.14$ at $k_{\max}=20$, $\approx 0.88$ for the
$k_{\max}$ sweep), and explained the sub-quadratic empirical behavior through cumulative
sufficient-statistics block caching.

**[6-T01]** `tables/table1_runtime_scaling.{tex,md,csv}`: extended to include the new $n=800$
row at $k_{\max}=20$, taken from the refreshed `fig5` CSV sidecar.

**[8-S01]** `sections/8.appendix.tex` "Additional diagnostic plots" subsection: enriched the
entries for `fig6_mixture_discovery.pdf`, `fig7_snr_sensitivity.pdf`, `fig8_multivariate_shared.pdf`,
`fig9_model_selection.pdf`, `fig10_missing_data.pdf`, and the uncropped `fig4_latent_groups.pdf`
so each entry describes the panel structure and the interpretive headline of the corresponding
refreshed figure.

**[5b-C01]** `sections/5b.limitations.tex`: replaced a stale "(Table~\ref{tab:real_welllog},
planned)" parenthetical with "(Table~\ref{tab:real_welllog}; length-aware row reserved)" since
the index-uniform row is now populated.
