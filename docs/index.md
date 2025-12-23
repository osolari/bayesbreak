# BayesBreak

BayesBreak is a small, focused implementation of Bayesian piecewise-constant
regression (segmentation) using dynamic programming.

## Key features

- **Conjugate families:** Gaussian (Normal--Normal), Poisson (Gamma--Poisson),
  Binomial (Beta--Binomial), and Beta-valued observations via a fractional
  Beta--Binomial construction.
- **Posterior over number of segments:** uniform prior over ``k`` by default.
- **Boundary posterior scores:** marginal posterior mass of a changepoint at each
  interior location.
- **Optional Bayesian regression curve:** posterior mean curve under a fixed
  ``k`` or mixture over ``k``.

For examples, see the `examples/` directory and the reproducibility scripts in
`scripts/`.
