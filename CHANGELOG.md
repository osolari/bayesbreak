# Changelog

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
