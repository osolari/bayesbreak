# BayesBreak

Exact Bayesian segmentation with a scikit-learn compatible API.

BayesBreak separates two concerns:

- **Block evidence** — a family-specific integrated single-segment marginal
  likelihood on every candidate block `(i, j]` (Gaussian, Poisson, Binomial,
  Bernoulli, Beta, Beta-observation, Logistic-Normal).
- **Dynamic programming** — a distribution-agnostic engine that consumes the
  triangular block-evidence matrix and produces

  - the marginal evidence `log p(y)` and segment-count posterior `P(k | y)`,
  - boundary-event marginals `P(b_i = 1 | y)`,
  - the **joint** MAP segmentation (max-sum DP + backtracking — distinct from
    marginal-topk summaries),
  - the Bayesian regression curve (expected latent signal).

The reference report (`docs/report/bayesbreak.pdf`) develops the framework in
detail; this documentation is the practical user guide.

- [Quickstart](quickstart.md)
- [Model families](models.md)
- [API reference](api.md)
- [Math notes](math.md)
- [Reproducibility](reproducibility.md)
