"""Basic usage example for BayesBreak.

Run:
    python examples/basic_usage.py

This script prints key posterior quantities and saves a small diagnostic plot
under `results/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Allow running the example from a source checkout without installation.
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from bayesbreak import BayesBreakGaussian  # noqa: E402


def main() -> int:
    rng = np.random.default_rng(0)

    mu = np.r_[np.zeros(60), 1.5 * np.ones(40), -0.75 * np.ones(50)]
    y = mu + 0.3 * rng.standard_normal(mu.size)

    model = BayesBreakGaussian(k_max=12, regression_curve="mix_k").fit(y)

    print("Selected k:", model.get_segment_count())
    print("Boundaries:", model.get_boundaries())
    print("Log-evidence:", model.score())

    yhat = model.predict()

    outdir = Path("results"); outdir.mkdir(exist_ok=True)

    plt.figure(figsize=(10, 3))
    plt.plot(y, label="y")
    plt.plot(yhat, label="pc-fit")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "example_basic_usage.png", dpi=200)
    plt.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
