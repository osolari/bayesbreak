# Changelog

## Phase 6 Report Adoption - 2026-08-05

- Replaced the legacy `docs/report/` manuscript wholesale with the complete Phase 6
  source under `report/`, including the technical book, journal layouts, executive
  summary, canonical registries, coding handoff, repository skeleton, presentation
  handoffs, and verification records.
- Added all signed release artifacts under `report/releases/phase6/` and verified them
  against the supplied SHA-256 manifest.
- Preserved all 53 archived numerical assets; each matches the Phase 4R read-only hash
  baseline. The two excluded historical computations retain their recorded status.
- Redirected figure and table generation to `results/` so new executions cannot
  overwrite archived report assets.
- Recorded the source archive's case-colliding bibliography manifests and the normalized
  macOS adoption in `report/revision_artifacts/adoption/ADOPTION_LEDGER.md`.

## [2.0.0-rc3] — Unreleased

Aligns the implementation with the 2026-05-15 Phase Three manuscript
(now in `docs/report/`, replacing the May 14 draft). The Phase Three
draft adds a dedicated §5b limitations section and several named
theorems / propositions / assumptions / definitions, and rolls back some
of the prior draft's broader literature additions. This release updates
docstrings and code surfaces to track those changes.

### Added

- `bayesbreak.mixture._canonical_template_order` and matching
  `canonical_permutation_` attribute on `BayesBreakMixtureClassifier` —
  templates, mixing weights, and responsibilities are now reported in
  ascending `k_g`, then lexicographic order on `t^(g)`, anchoring the
  permutation indeterminacy of `prop:latent-identifiability`
  (cf. `ex:label-switch-counterexample`). Two new tests
  (`test_canonical_template_ordering_after_fit`,
  `test_canonical_ordering_stable_across_seeds`).
- `run_non_conjugate_diagnostics` now records `approx_routine`,
  `theoretical_rate`, and `theoretical_rate_violated` from
  `prop:uniform-bounds` (§4): Laplace/JJ/PG → `O(n^{-1})` on reachable
  blocks, Quadrature → `O(Q^{-2r})` for `C^{2r}` integrands, true EP →
  not uniformly bounded.
- `bayesbreak.baselines.run_smuce` — SMUCE (Frick, Munk & Sieling 2014)
  via the R package `stepR`, driven through `rpy2`. Registered as
  `"smuce"` in the dispatch registry; lazy import; readable
  `ImportError` when the upstream is missing.

### Changed

- `DiagnosticCheck.failure_mode` tags renamed for the Phase Three draft:
  `"abs-prob-tv-bound"` → `"tv-bound"`. Check name
  `pk_tv_bound_cor_abs_prob` → `pk_tv_bound_check`. The math is unchanged
  (the bound `exp(2 k_max ε) − 1` is derivable directly from
  `prop:stability`); only the label naming follows the manuscript, which
  no longer ships a separate `cor:abs-prob` corollary.
- Module docstrings in `bayesbreak.dp` and `bayesbreak.diagnostics`, and
  the `_compute_block_evidence` docstring in `bayesbreak.base`, now
  reference `prop:stability` + `ass:uniform-block-error` +
  `prop:uniform-bounds` instead of the removed `rem:score-matrix-exactness`
  / `cor:boundary-event-sum` / `cor:abs-prob` labels.
- `bayesbreak.datasets.welllog` and `bayesbreak.datasets.methylation`
  docstrings reverted to the Phase Three text (`changepoint::Lai2005fig4`
  / `nloyfer/meth_atlas` recipes). Code-comment caveats record the
  prior-draft alternatives (`changepoint.influence::welldata`,
  `nloyfer/wgbs_tools` + `nloyfer/UXM_deconv`) so a future agent can
  re-raise them if the rolled-back recipes fail at run time.
- `docs/api.md` cross-references the Phase Three labels:
  `prop:bb-complexity`, `thm:map-correctness`, `prop:gaussian-block` /
  `prop:poisson-block` / `prop:binomial-block` / `prop:negbin-block` /
  `prop:beta-block`, `def:exported-segmentation`, `def:bayes-curve`,
  `def:prediction-cases`, `ass:uniform-block-error`,
  `prop:uniform-bounds`, `prop:latent-identifiability`.

## [2.0.0-rc2] — Unreleased

Aligns the implementation with the 2026-05-14 revised manuscript in
`docs/report/` (replaces the earlier May 9 draft). Surfaces the
"Block-score contract" of §`sec:setup`, adds the absolute-probability TV
diagnostic from Corollary `cor:abs-prob`, declares per-family
moment-sign contracts (§5 5-C1), adds a prior-sensitivity diagnostic
(§6 6-C1), corrects real-data source provenance docstrings, and
introduces a status-aware `(PLANNED)` helper for figures that fall back
to simulated data.

### Added

- `BayesBreakSegmenter.admissibility_mask_` — boolean mask of the
  finite-evidence cells in `log_block_evidence_`. Materializes the
  §`sec:setup` "Block-score contract" that the DP and `compute_log_C_k`
  share the same admissibility convention.
- `MOMENT_SIGN_CONTRACT` class attribute on every family. Gaussian
  declares `"signed"`; Poisson, Binomial, Beta, BetaObs, Bernoulli,
  NegBin, LogisticNormal declare `"nonneg"` (§5 paragraph 5-C1).
- `DiagnosticCheck.failure_mode` — optional short tag tying each check
  in the non-conjugate diagnostic report to a failure mode in the
  rewritten approximation-validation checklist (§4 4-C1).
- `run_non_conjugate_diagnostics` now reports two new fields:
  `pk_tv_empirical` and `pk_tv_upper_bound`, plus a passing check
  `pk_tv_bound_cor_abs_prob` enforcing the absolute-probability TV
  bound `exp(2·k_max·ε) − 1` of Corollary `cor:abs-prob`.
- `run_prior_sensitivity(estimator, ...)` — planned diagnostic from
  §6 6-C1; reports variation of `P(k|y)` and the fixed-`k_map`
  boundary marginals under `p(k)` and `g` perturbations.
- `scripts.figures._realdata.planned_caption()` helper and a
  `planned: bool | None` argument on `make_realdata_figure` — infers
  status from `bundle.source` by default so figures populated from a
  verified download do **not** carry a `(PLANNED)` tag, and only the
  simulated-fallback path is marked planned.
- `bayesbreak.baselines` subpackage — thin wrappers around upstream
  baseline libraries (no re-implementation). PELT, optimal partitioning,
  BS, WBS via `ruptures` (`pip install bayesbreak[baselines]`); CBS via
  Bioconductor `DNAcopy` driven through `rpy2`
  (`pip install bayesbreak[baselines-r]`). Public API:
  `segment_with(algorithm, y, **tuning)` returning a `BaselineResult`
  with boundaries, segment count, upstream package + version, and the
  tuning kwargs used.
- New tests: `test_score_matrix_passthrough` (DP-on-supplied-matrix
  exactness, `rem:score-matrix-exactness`),
  `test_admissibility_mask_matches_log_block_evidence`,
  `test_abs_prob_tv_bound` (Corollary `cor:abs-prob`),
  `TestMomentSignContract` (per-family sign), and
  `test_prior_sensitivity_reports_variation_per_variant`.

### Changed

- Dataset loader docstrings corrected against the new manuscript:
  - `datasets.welllog`: cites Ó Ruanaidh & Fitzgerald 1996 +
    Fearnhead & Clifford 2003 and the `changepoint.influence::welldata`
    object (length 4050). Warns explicitly that
    `changepoint::Lai2005fig4` is the array-CGH example of
    Lai et al. 2005, **not** the well-log NMR series.
  - `datasets.methylation`: replaces the older `nloyfer/meth_atlas`
    pointer (which implements Moss et al. 2018, not the 2023 atlas)
    with `nloyfer/wgbs_tools` + `nloyfer/UXM_deconv`, the companion
    software for `loyfer2023atlas`. Records GEO accession
    `GSE186458` as the canonical raw source.
- Manuscript caption hygiene in `docs/report/sections/6.evaluation.tex`:
  the four real-data figures (`fig:welllog`, `fig:cgh`, `fig:spx`,
  `fig:methylation`) no longer carry the `(PLANNED)` prefix —
  populated PDFs are reported as observed. `tab:realdata-status`
  rewritten to reflect that figures are populated and only the
  metrics tables remain placeholders. The four metric tables retain
  `(PLANNED)` because their cells are still `---`.

## [2.0.0-rc1] — Earlier May 2026

This release aligns the implementation with the May 2026 revision of the
manuscript at `docs/report/bayesbreak.pdf`. The whole DP layer, latent-group
EM, prediction layer, real-data pipelines, and figure aesthetic now match
the report's terminology, conventions, and reproducibility constraints.

### Breaking

- **DP layer rebuilt around the design-aware partition prior.**
  `forward_backward`, `posterior_over_k`, `bayes_regression_curve_*`, and
  `max_sum_segmentation` now take an optional `log_g_table` (the per-block
  log-cohesion `log g(Δ_x(i, j))`). New helper `compute_log_C_k` evaluates
  the partition normalizer `C_k = Σ_t ∏_q g(Δ_x)` via a second DP. The old
  `boundary_event_marginals(L, R, log_post_k, n, k_max)` is split into the
  conditional-on-k `boundary_event_marginals_fixed_k(L, R, n, k)` (the §6
  calibration target) and the marginalized `boundary_event_marginals_marginalised`.
- **`BayesBreakSegmenter` constructor gains `length_prior`,
  `boundary_coordinates`, and `prior_k`.** All conjugate / non-conjugate
  families forward these to the base class. `boundary_marginals_` now stores
  `P(b_i = 1 | y, k_map_)` (conditional on k), matching §6.
- **EM mixture rewritten around the report's finite-template objective `ℓ_⋆`**
  (Algorithm `multi-em`). The M-step is now a responsibility-weighted
  **max-sum** DP per template with the count offset
  `n_g (log p(k) - log C_k)`. The legacy `regression_curve` /
  `prior_k="geometric"` switches are removed; pass `length_prior` and
  `prior_k` callables directly. Convergence criterion gained the
  deterministic-tie-handling re-emit certification (criterion (iii) in
  §`latent-em`).
- **`BayesBreakGroupedClassifier` switches to exported-MAP scoring
  exclusively.** `score_samples` now routes new sequences through each
  group's exported MAP segmentation via `posterior_predictive_logpdf`; the
  legacy resegmentation mode is gone.
- **`BayesBreakBetaObs.phi`** is now per-observation. The constant-φ
  fallback is the special case `phi=scalar`. The deprecated `quad_points`
  alias is removed (use `quadrature_points`).

### Added

- **`BayesBreakNegBin`** — Beta-NegBin block family (§`sec:nb-block`).
  Moment numerators target the *observation-mean* `m_*(p) = r_* (1-p)/p`,
  not parameter-scale Beta moments. New `r_predict` constructor argument
  exposes the predict-time dispersion.
- **`SharedBoundaryReplicatesSegmenter`** — exact boundary-posterior pooling
  for multi-subject 1-D sequences (Theorem `multisubject`).
- **`bayesbreak.diagnostics`** — IMP-13 module. `run_dp_diagnostics`
  verifies the four DP invariants from §4.2; `run_non_conjugate_diagnostics`
  reports max / q95 / median block error over **reachable** blocks plus
  posterior sensitivity vs. a reference fit.
- **`bayesbreak.experiments.{synthetic,realdata}`** — `python -m`
  reproducibility entry points, with the manuscript's appendix command
  shape `python -m bayesbreak.experiments.realdata --dataset {welllog,cgh,spx,methyl}`.
  CLI `bayesbreak synthetic` / `bayesbreak realdata` dispatch to these.
- **Real-data placeholder mode + `--verified` flag.** Real-data figures
  default to a watermarked placeholder render with a sidecar JSON capturing
  raw-data hash, preprocessing hash, fit hyperparameters, and DP
  diagnostics. `--verified` (or `BAYESBREAK_VERIFIED=1`) is required to
  flip the figure to the finalized look — only after author approval.
  Simulated-fallback bundles are *always* placeholders regardless of flag.
- **Real CGH download** — `load_cgh()` pulls `cran/ecp/ACGH.RData` (2215
  probes × 43 subjects), parses with `rdata`, and returns a multi-subject
  bundle with rolling-MAD per-probe precision. `fig7_cgh.py` switches to
  `SharedBoundaryReplicatesSegmenter`.
- **Real methylation download** — `load_methylation()` pulls the
  methylKit chr21 example (`test1.myCpG.txt`) and propagates per-CpG read
  coverage as `phi_t` into `BayesBreakBetaObs`.
- **Prediction layer expansion.** New `bayesbreak.prediction.Unit` Case-B
  set-valued container; `predict_group` / `predict_map_signal` matching
  Algorithms `predict-group` / `predict-map`; `pit_residuals` for
  closed-CDF families (Gaussian / Bernoulli / Poisson / Beta / Binomial).
- **Cropped fig4 variant.** `scripts/figures/fig4_latent_groups.py` emits
  both the full 3-panel figure and the 2-panel `fig4_latent_groups_cropped.{png,pdf}`
  used in §6.
- **Test suite expansion.** Seven new test files —
  `test_design_prior`, `test_negbin`, `test_replicates`,
  `test_em_template`, `test_stability_bound`, `test_pit`,
  `test_diagnostics` — bringing the conceptual-correctness coverage up to
  133 passing tests.

### Changed

- **Figure aesthetic** — full spines on every axes, ~14 pt body / ~12 pt
  ticks, Paul Tol's *muted* (colour-blind-safe) palette. Applies to
  fig1-9, fig4-cropped, and the supplementary figures via
  `scripts/figures/_style.py`.
- **`table4_nonconj_tradeoff`** — block-error column now reports the max,
  95th-percentile, and median over **reachable** blocks (those usable by
  some `k`-segmentation with `k ≤ k_max`), per Proposition `stability`.
- **DP terminology** — "exact pooling" renamed to "exact boundary-posterior
  pooling" in `replicates` docstrings; "observed-data log-likelihood"
  renamed to "finite-template mixture objective `ℓ_⋆`" in `mixture`.
- Joint-MAP-vs-marginal-mode counterexample test refreshed to the report's
  `n=5, k=3, (1,4) vs (2,4)` example with explicit posterior weights
  `(0.30, 0.28, 0.22, 0.20)`.

### Removed

- `BayesBreakSegmenter`'s old `regression_curve="mix_k"` was *never*
  produced by the EM mixture; the now-removed `regression_curve` constructor
  parameter on `BayesBreakMixtureClassifier` is gone.
- `bayesbreak reproduce {figures,tables,all}` legacy CLI subcommand. Use
  `bayesbreak synthetic` and `bayesbreak realdata` instead.

## [1.0.0] — Unreleased

### Breaking

- Strict scikit-learn API. `fit(X, y)` is the only accepted signature (`X` is
  the design matrix, `y` is the response). `fit(y)` / `fit(X=y)` are removed.
- Renamed the abstract base class `BayesBreakBase` → `BayesBreakSegmenter`.
- Renamed `BayesBreakGrouped` → `BayesBreakGroupedClassifier` and
  `BayesBreakMixture` → `BayesBreakMixtureClassifier`.
- Split `BayesBreakMultivariate(combine="shared"|"independent")` into two
  dedicated classes: `SharedBoundaryMultivariateSegmenter` and
  `IndependentMultivariateSegmenter`.
- Replaced old fitted attributes with report-consistent names:
  `k_ml_ → k_map_`, `boundary_post_ → boundary_marginals_`,
  `pc_fit_ → map_curve_`, `brc_ → bayes_curve_mean_`,
  `lA0_ → log_block_evidence_`, `A1_ → block_first_moment_`,
  `L_ → log_left_`, `R_ → log_right_`, `C_ → k_posterior_`.
- `score(X, y)` now returns the **mean posterior-predictive log-density**
  (sklearn CV-compatible). The old log marginal likelihood is available via
  `log_evidence_` / `get_log_evidence()`.
- Removed aliases `BayesBreak = BayesBreakGaussian` and `make_model`.

### Added

- `bayesbreak.dp` — standalone DP module exposing `forward_backward`,
  `posterior_over_k`, `boundary_event_marginals`,
  `boundary_location_posterior`, `max_sum_segmentation`,
  `bayes_regression_curve_{fixed,mixed}_k`, `marginal_boundary_modes`.
- `bayesbreak.dp.max_sum_segmentation` — exact joint MAP segmentation via
  max-sum DP with backtracking (the report's §4.4 object).
- `bayesbreak.prediction` — posterior-predictive scoring matching §8 of the
  report; per-family `posterior_predictive_logpdf_block` hooks on every
  estimator.
- `bayesbreak.validation` — `check_segmentation_input`, `check_sample_weight`,
  `require_fitted`.
- `bayesbreak.interface.BlockEvidence` protocol and
  `SegmentationPosterior` dataclass.
- `bayesbreak.reproduce` + `bayesbreak` CLI with `reproduce {figures,tables,all}`.
- Conceptual-correctness tests: brute-force DP comparisons, closed-form
  predictive checks, MAP-vs-marginal-topk counterexample, EM behaviour.
- Conda recipe at `conda-recipe/meta.yaml`.

### Fixed

- CI previously ran `pytest test/` — corrected to `pytest tests/` and a multi-
  Python matrix.
- `conda-publish.yml` no longer contains the `<your-anaconda-username>`
  placeholder.

## [0.1.0]

Initial public release.
