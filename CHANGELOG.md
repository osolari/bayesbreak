# Changelog

## [2.0.0-rc1] — Unreleased

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
