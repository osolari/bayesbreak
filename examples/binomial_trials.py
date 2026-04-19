"""Binomial segmentation with per-observation trial counts.

Run:
    python examples/binomial_trials.py
"""

from __future__ import annotations

import numpy as np

from bayesbreak import BayesBreakBinomial


def main() -> int:
    rng = np.random.default_rng(1)
    n = 120
    X = np.arange(n).reshape(-1, 1)

    n_trials = rng.integers(low=10, high=50, size=n)
    p = np.r_[np.full(40, 0.2), np.full(40, 0.6), np.full(40, 0.35)]
    y = rng.binomial(n_trials, p)

    model = BayesBreakBinomial(k_max=10, n_trials=n_trials, regression_curve="mix_k").fit(X, y)

    print("k_map          :", model.k_map_)
    print("MAP boundaries :", model.map_boundaries_)
    print("hyperparameters:", model.hyper_)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
