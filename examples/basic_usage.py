"""Basic BayesBreak usage.

Run:
    python examples/basic_usage.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bayesbreak import BayesBreakGaussian


def main() -> int:
    rng = np.random.default_rng(0)
    mu = np.r_[np.zeros(60), 1.5 * np.ones(40), -0.75 * np.ones(50)]
    y = mu + 0.3 * rng.standard_normal(mu.size)
    X = np.arange(mu.size).reshape(-1, 1)

    model = BayesBreakGaussian(k_max=12, regression_curve="mix_k").fit(X, y)

    print("k_map            :", model.k_map_)
    print("MAP boundaries   :", model.map_boundaries_)
    print("log p(y)         :", model.log_evidence_)
    print("held-out score   :", model.score(X, y))

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot.")
        return 0

    outdir = Path("docs/report/figures")
    outdir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 3))
    plt.plot(y, label="y")
    plt.plot(model.predict(X), label="MAP fit")
    if model.bayes_curve_mean_ is not None:
        plt.plot(model.bayes_curve_mean_, label="Bayes curve", lw=1.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "example_basic_usage.png", dpi=200)
    plt.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
