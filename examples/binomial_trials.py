"""Binomial example with per-observation trial counts.

Run:
    python examples/binomial_trials.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Allow running the example from a source checkout without installation.
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from bayesbreak import BayesBreakBinomial  # noqa: E402


def main() -> int:
    rng = np.random.default_rng(1)
    n = 120

    # number of trials varies per position
    n_trials = rng.integers(low=10, high=50, size=n)

    p = np.r_[np.full(40, 0.2), np.full(40, 0.6), np.full(40, 0.35)]
    y = rng.binomial(n_trials, p)

    m = BayesBreakBinomial(k_max=10, n_trials=n_trials, regression_curve="mix_k").fit(y)

    print("Selected k:", m.get_segment_count())
    print("Boundaries:", m.get_boundaries())
    print("Hyperparameters:", m.hyper_)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
