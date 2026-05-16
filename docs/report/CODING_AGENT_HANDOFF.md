# BayesBreak — Coding Agent Handoff

This document hands off the BayesBreak manuscript repository to the next implementation or
verification agent. It is paired with `CHANGELOG.md` (which records the substantive edits of the
current pass) and with the compiled `bayesbreak.pdf` (110 pages, clean build).

---

## 1. Project overview

BayesBreak is an exact-Bayesian piecewise-constant segmentation framework for ordered data, with
a modular block-evidence interface that supports conjugate exponential-family blocks
(Gaussian, Poisson, Binomial, Negative-Binomial, Beta-observation), a uniform error-propagation
treatment for non-conjugate blocks (Laplace, JJ, EP, PG mean field, 1D quadrature), exact
shared-boundary multi-subject pooling, a finite latent-template EM extension for unknown
groups, and a first-class prediction layer for pointwise / set-valued / vector-valued new data.

The manuscript directory contains:
- `bayesbreak.tex` — top-level main file using a custom `saim.cls` class.
- `math_commands.tex` — shared macro definitions.
- `sections/` — 10 section files (0-abstract through 8-appendix; 4-method.tex is largest at
  ~2,335 lines after this pass; the body is now ~4,640 lines).
- `tables/` — five `\input`-able table source files plus companion `.csv` artifacts.
- `figures/` — 9 referenced real PDFs with JSON sidecars + 6 orphan PDFs (documented in
  `app:code`) + `realdata_metrics.json/.txt` aggregator.
- `assets/` — author-side image assets.
- `reference/cite.bib` — 36-entry bibliography (all entries have complete canonical fields).
- `saim.cls`, vendored `natbib.sty`, `plainnat.bst`, `fancyhdr.sty`, `iclr2026_conference.bst`.

---

## 2. Build instructions

The manuscript compiles with TeX Live 2023 or newer. Build cycle:

```bash
pdflatex -interaction=nonstopmode bayesbreak.tex
bibtex bayesbreak
pdflatex -interaction=nonstopmode bayesbreak.tex
pdflatex -interaction=nonstopmode bayesbreak.tex
pdflatex -interaction=nonstopmode bayesbreak.tex   # one extra settling pass
```

After this cycle the build is clean: 0 LaTeX errors, 0 undefined references, 0 undefined
citations, 0 overfull or underfull h/v boxes, 0 LaTeX warnings, 110 pages.

### Environmental note: `lmodern` package registration

On a fresh TeX Live install where the `texmf-dist/ls-R` database has not been refreshed since
`lmodern` was installed, `pdflatex` will fail with `! LaTeX Error: File 'lmodern.sty' not found.`
Resolve once with `mktexlsr` (or `sudo mktexlsr` if the relevant tree is system-managed). This is
an environmental issue, not a source-side issue; no `.tex` file requires editing.

### Build invariants (do not break)

- All `\ref{...}`, `\eqref{...}`, and `\cref{...}` calls resolve. The cite-key audit reports 35
  used and 36 defined; the one unused entry (`teicher1963identifiability`) is intentionally wired
  into Remark `rem:teicher-overspec` and Limitation paragraph 5b-C01 by the current pass.
- No new `\usepackage{...}` calls have been added. The package set is whatever `saim.cls`
  declares plus the explicit `\usepackage{...}` lines at the head of `bayesbreak.tex`.
- The `algorithm2e` environment is used throughout for algorithms; new algorithms in this pass
  (`alg:pool-shared-boundary`, `alg:latent-em-detail`) follow the same `\KwIn` / `\KwOut` /
  `\tcp{...}` conventions.

---

## 3. Implementation tasks (what the next agent should do)

In priority order:

### 3.1 Phase Four: bibliographic metadata verification

Five new rows in `tab:annotated-lit` (`sections/8.appendix.tex`) carry the marker
`[Phase-Four verification pending]`. The annotation prose is content-complete; the bibliographic
metadata for the cited works is not yet committed. The next pass should, via web research:

- Resolve author/year/venue/DOI for representative scalable Bayesian changepoint detection
  references (scalable BOCPD; SVI-based segmentation).
- Resolve metadata for the two surveys explicitly named: **Aminikhanghahi & Cook (2017)** and
  **Truong, Oudre & Vayatis (2020)**. Both have stable canonical citations.
- Resolve metadata for representative multivariate / multichannel changepoint references.
- Resolve metadata for kernel/nonparametric and generalized-Bayes CP references.
- Resolve metadata for ctDNA / CNA / methylation-atlas bioinformatics references (ichorCNA,
  `meth_atlas`, `wgbs_tools`, UXM_deconv).

For each verified reference, add a `@article{...}` or `@inproceedings{...}` entry to
`reference/cite.bib` with complete canonical fields (author, title, journal/booktitle, year,
volume, number, pages, doi when available), and replace the `[Phase-Four verification pending]`
marker in the appendix table with an explicit `\citet{key}` invocation.

### 3.2 Real-data table completion (paired with reproduction pipelines)

Each of the four real-data tables in §6 is now `partially-real` rather than `planned`. The
remaining `---` cells fall into three categories:

| Cell type | Why it's reserved | What's needed |
|---|---|---|
| Length-aware-prior row in `tab:real_welllog` | `realdata_metrics.json` records `needs_refit: "length-aware prior not in fit cache; refit on real welllog when datasets extras are installed"` | run the length-aware-prior fit on the well-log data via `python -m bayesbreak.experiments.realdata --dataset welllog --prior length-aware` |
| ECE / Boundary F1 / Boundary MAE columns | require external ground-truth boundaries (well-log reference boundaries; Snijders-2001 array-CGH annotations; methylation-atlas transition annotations) | load verified annotations into the reproduction pipeline; recompute `def:metric-f1` and `def:metric-ece` |
| Poisson-on-threshold-crossings row in `tab:real_spx` | `realdata_metrics.json` records `needs_refit: "threshold-crossings variant not in fit cache; refit on real SPX when yfinance is available"` | run the Bernoulli/Poisson threshold-crossings refit; the appendix code block already documents the pipeline |
| Region-A-cell-type-2 row in `tab:real_methylation` | `realdata_metrics.json` records the need for the **Loyfer-2023 atlas pipeline (GEO~GSE186458 + `wgbs_tools` / `UXM_deconv`)** which is not yet wired into the repository | wire the Loyfer-2023 pipeline; one extra row in `tab:real_methylation` |
| Runtime (s) columns | require a fresh benchmarking-pipeline timing record under fixed hardware | run the existing reproduction pipelines on the target machine with timing capture enabled |

When each cell becomes available, edit the table in place (in `sections/6.evaluation.tex` and the
corresponding `\paragraph{Partially-populated table output.}` paragraph in
`sections/8.appendix.tex`) and update the row provenance in the caption. The current pass has
already wired in the cells that come from `figures/realdata_metrics.json` and softened captions
to `(Partially populated)` for honesty about what's measured and what's reserved.

### 3.3 Deferred empirical extensions (named in §7 "Deferred empirical work")

- Head-to-head comparison against frequentist baselines PELT, wild binary segmentation (WBS),
  SMUCE on the four real-data case studies, under a fixed boundary-tolerance protocol consistent
  with `def:metric-f1`.
- Head-to-head comparison against Bayesian baselines RJMCMC (`green1995rjMCMC` in the bib) and
  Fearnhead's exact DP (`fearnhead2006exact` in the bib).
- A scaling study pushing $n$ into the tens of thousands to exercise the memory--time
  trade-offs of Remark `rem:mem-time`. The current `tab:runtime_scaling` reports $n\in\{50,100,
  200,400\}$ at $k_{\max}=20$; extending this to $n\in\{10^3,10^4,10^5\}$ is the natural next step.

None of these requires modifying the BayesBreak inferential core. Each requires a corresponding
reproduction-pipeline harness.

---

## 4. Experiment plan (referenced from §6 and Appendix `app:real-data`)

The four real-data datasets, their block models, and their reproduction recipes are documented
canonically in `sections/8.appendix.tex` subsections `app:real-data-welllog`,
`app:real-data-cgh`, `app:real-data-spx`, `app:real-data-methylation`. The "Source-stability
note: verify before use" paragraph at the head of `app:real-data` documents the moving-target
external sources (CRAN/Bioconductor releases, yfinance endpoint, GEO accession schemas, GitHub
source repositories).

The archived JSON sidecars (`figures/fig6_welllog.json`, `figures/fig7_cgh.json`,
`figures/fig8_spx.json`, `figures/fig9_methylation.json`) carry the input-series
`y_hash`, run timestamp, DP-diagnostic status (`4/4 checks passed` for each archived run), and
`figure_path`. The simplest cross-check that a reproduction has succeeded is to recompute the
`y_hash` of the reproduced input and confirm a match before reading downstream numbers.

---

## 5. Figures and tables to generate

The manuscript references 13 figures and 14 tables. Status after the current pass:

| Asset | Status | Where to find it |
|---|---|---|
| `fig:plate-ef`, `fig:plate-replicates`, `fig:plate-latent-em`, `fig:plate-nonconj` | theoretical (TikZ in-source) | `sections/4.method.tex` |
| `fig:single_synth` through `fig:runtime` (5 synthetic) | real | `figures/fig{1,2,3,4_cropped,5}_*.pdf` |
| `fig:welllog`, `fig:cgh`, `fig:spx`, `fig:methylation` | real | `figures/fig{6,7,8,9}_*.pdf` with JSON sidecars |
| `tab:metrics`, `tab:family-summary`, `tab:prediction-outputs`, `tab:complexity-summary` | theoretical | inline in `sections/{6.evaluation,4.method,5.algorithms}.tex` |
| `tab:posterior_summary`, `tab:single_quant`, `tab:nonconj_tradeoff`, `tab:runtime_scaling` | real | `tables/table{2,3,4,1}*.tex` with CSV companions |
| `tab:real_welllog`, `tab:real_cgh`, `tab:real_spx`, `tab:real_methylation` | partially-real (this pass) | inline in `sections/6.evaluation.tex` |
| `tab:annotated-lit` | real (5 new modern-thread scaffold rows added; bib metadata Phase-Four) | `sections/8.appendix.tex` longtable |

Six orphan PDFs are documented but not wired in (matches author decision A2-c); see
"Additional diagnostic plots not shown in the body" paragraph in `app:code`.

---

## 6. Projected / expected results

The deferred empirical extensions of §3.3 should yield, by construction of the BayesBreak
framework:

- **Frequentist baselines.** PELT/WBS/SMUCE will return point segmentations under penalty.
  BayesBreak's posterior-quantity output (calibrated boundary marginals, $P(k\mid y)$, Bayes
  curves) is the differentiator. Boundary F1 at a fixed tolerance is the natural cross-method
  comparison.
- **Bayesian baselines.** RJMCMC and Fearnhead's exact DP target the same posterior quantities
  as BayesBreak. Differentiators are (a) closed-form rather than simulation-based posterior
  summaries, (b) the modular block-evidence interface that lets the same DP layer accept any of
  five non-conjugate routines, and (c) the exact shared-boundary multi-subject pooling.
- **Scaling study.** The exact DP scales as $\Theta(k_{\max}n^2)$ time and at least
  $\Theta(k_{\max}n)$ memory. For $n=10^4$ at $k_{\max}=20$, the projected runtime is on the
  order of single-digit seconds on a modern laptop; for $n=10^5$, on the order of single-digit
  minutes. Memory becomes the binding constraint at $n\approx 10^5$ if boundary marginals are
  retained; the appendix `app:complexity-proofs` records the proof.

These projections are restatements of the explicit complexity results in the manuscript
(Proposition `prop:bb-complexity`) and are not new empirical claims.

---

## 7. Theory → code connections

The next agent should preserve the cross-reference network established in this pass. Key labels
for the implementation to test against:

| Theoretical label | Algorithm | What the code should compute |
|---|---|---|
| `prop:fb-duality` (this pass) | `alg:dp-core`, `alg:dp-log` | $\widetilde L_{k,n}=\widetilde R_{k,0}$ identity; boundary-marginal product form |
| `thm:ef-integral` | `alg:block-precompute` | conjugate block evidence as a ratio of normalizers $Z(\alpha,\beta)$ |
| `thm:multisubject` | `alg:pool-shared-boundary` (this pass) | log-pooling of per-subject block evidences |
| `prop:shared-boundary-identifiability` (this pass) | unit test on the pooled DP | population log-likelihood uniquely maximized at $t^\star$ up to trivial relabeling |
| `thm:em-monotone` | `alg:latent-em-detail` (this pass) | finite-mixture objective $\ell_\star$ non-decreasing across EM iterations |
| `rem:teicher-overspec` (this pass) | unit test at $G>G^\star$ | two distinct $(\pi,\tau)$ configurations yield equal mixture density |
| `prop:stability`, `prop:uniform-bounds` | non-conjugate routine wrappers | uniform $\varepsilon$-bound on $\log\widehat A^{(0)}_{ij}-\log A^{(0)}_{ij}$ |
| `thm:map-correctness` | `alg:map-forward`, `alg:map-backtrack` | joint MAP segmentation matches exhaustive maximization on small $n$ |
| `def:metric-f1`, `def:metric-mae`, `def:metric-ece`, `def:metric-loglik` (this pass) | evaluation harness | metric implementations whose unit tests reference the manuscript definitions |
| `def:posterior-summaries` (this pass) | DP output extractor | the three posterior summaries returned by distinct DP recursions |

The "DP-invariant test" in `sec:alg-checklist` already enumerates the runtime sanity checks the
code path enforces; the new `prop:fb-duality` is the algebraic basis of those checks.

---

## 8. Open technical questions

- **Phase-Four bib metadata.** The five new rows in `tab:annotated-lit` are scaffold-only; the
  `\citet{key}` invocations have been deliberately left out of the row content to keep the build
  warning-free. Phase Four will add the verified bib entries.
- **Length-aware welllog refit.** The needed `needs_refit` configuration is documented in
  `figures/realdata_metrics.json` but not yet run. The reproduction recipe is in
  `app:real-data-welllog`.
- **Loyfer-2023 atlas wiring.** The methylation pipeline currently uses `methylKit`'s
  `test1.myCpG` (single chr21 region, $n=1904$ CpGs). A second-row population in
  `tab:real_methylation` requires wiring in the Loyfer-2023 atlas
  (GEO~GSE186458 + `wgbs_tools` / `UXM_deconv`). This is documented in the JSON aggregator
  `figures/realdata_metrics.txt`.
- **Orphan PDF inventory.** The six orphan PDFs are documented in `app:code` but not wired in.
  Author decision A2-c made this deliberate; if a future pass wants to wire them, each should
  carry a corresponding figure caption and a forward-link to the analysis it supports.
- **Mixing-weight floor $\pi_{\min}$.** Algorithm `alg:latent-em-detail` documents an optional
  mixing-weight floor; the manuscript's Remark on "Exact EM versus regularized or floored
  variants" notes that this is a regularized variant. Code should report a warning when
  $\pi_{\min}>0$ is used and label the achieved $\ell_\star$ accordingly.

---

## 9. Files changed or added in the current pass

Seven section source files modified:

| File | Edits |
|---|---|
| `sections/2.problem.tex` | 2-C02 inserted `def:posterior-summaries` |
| `sections/4.method.tex` | 4-B02 inserted `prop:fb-duality`; 4-B05 inserted `prop:shared-boundary-identifiability` and `rem:identifying-block`; 4-C03 inserted `rem:teicher-overspec` (wires `teicher1963identifiability` from the bib) |
| `sections/5.algorithms.tex` | 5-C01 promoted pooling description to `alg:pool-shared-boundary`; 5-C02 promoted EM-implementation description to `alg:latent-em-detail` |
| `sections/5b.limitations.tex` | 5b-C01 appended "Identifiability failures (named)" and "Non-conjugate approximation outside the small-$\varepsilon$ regime" paragraphs; 5b-C02 appended "Decision flowchart: which branch to use" paragraph |
| `sections/6.evaluation.tex` | 6-C01 inserted 4 new metric definitions; 6-E01a/b/c/d populated 4 partially-real tables with caption + prose status softening |
| `sections/7.conclusion.tex` | 7-D01 expanded with 4 new paragraphs (regime taxonomy, deferred work, downstream applications, reproducibility statement); corrected stale planned-table claim |
| `sections/8.appendix.tex` | 8-Lit01 appended 5 modern-thread scaffold rows to `tab:annotated-lit`; 8-C01 added orphan-PDF inventory in `app:code`; 8-D01 added source-stability note in `app:real-data` and updated 4 "Planned table output" → "Partially-populated table output" paragraphs |

Files added (new): `CHANGELOG.md`, `CODING_AGENT_HANDOFF.md`.

Files NOT changed: `bayesbreak.tex` (main wrapper), `math_commands.tex`, `reference/cite.bib`
(despite `teicher1963identifiability` being newly used, the entry was already present), all
`tables/*.tex`, all `figures/*.pdf`/`*.json`/`*.csv`, `saim.cls`, vendored `.sty`/`.bst` files.

Prior-pass artifacts `CHANGELOG.prior.md` and `CODING_AGENT_HANDOFF.prior.md` are preserved
unchanged as historical record.

---

## 10. Do-not-change constraints

These constraints are imposed by the manuscript revision protocol and must be respected by any
future automated pass:

- **Numerical content is read-only.** No new pass may re-derive or re-report any of the
  measured numbers in `tab:posterior_summary`, `tab:single_quant`, `tab:nonconj_tradeoff`,
  `tab:runtime_scaling`, or the populated cells of the four real-data tables. If a reproduction
  changes any of those numbers, the cell must be updated together with a provenance note and the
  archived JSON sidecar updated correspondingly.
- **No silent invented data.** Every cell that was populated in this pass carries explicit
  per-row provenance in the caption ("populated from the same archived fit as Figure X").
  Future passes must preserve that provenance discipline.
- **No scope shortening.** The protocol forbids shortening any existing section without explicit
  author approval. Expansions are permitted; deletions and condensations are not.
- **Status-framing rule.** Real assets use indicative voice; partially-real assets use mixed
  voice with explicit per-cell or per-row honesty; planned assets use projection voice with a
  `(PLANNED)` caption prefix; theoretical assets use derivation/protocol voice with the note "no
  source CSV is associated with it". Any prose claim that cites an asset inherits the asset's
  status.
- **No new bib entries via inference.** The Phase Four metadata verification must use external
  research; bib entries may not be invented or reconstructed from memory. Each new entry must
  carry author, title, journal/booktitle, year, and (where available) DOI.
- **Build hygiene.** The manuscript must remain at 0 errors / 0 undefined refs / 0 undefined
  cites / 0 overfull or underfull boxes / 0 warnings after each pass. The four-pass
  `pdflatex → bibtex → pdflatex → pdflatex → pdflatex` cycle is the standard.
- **Cross-reference network.** The labels listed in section 7 above are now referenced from
  multiple places in the manuscript. A future pass may rename a label only if all references
  are updated in the same commit. Removing a label that the limitations section, conclusion, or
  appendix references is forbidden.
