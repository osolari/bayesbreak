"""Deterministic simulated analogs for the four real-data figures.

Every simulation is seed-pinned so the generated data (and therefore the
downstream figure) is byte-identical across runs on the same NumPy version.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _from_levels(levels: list[float], lengths: list[int]) -> NDArray[np.floating]:
    return np.concatenate(
        [np.full(L, m, dtype=float) for m, L in zip(levels, lengths, strict=True)]
    )


def simulate_welllog(seed: int = 0, n: int = 4050) -> dict:
    """Piecewise-constant NMR-like signal with additive Gaussian noise."""

    rng = np.random.default_rng(seed)
    # 10 segments with realistic NMR tool-response levels.
    segment_means = [112.0, 115.0, 118.5, 116.2, 113.8, 119.4, 121.0, 118.1, 114.9, 117.6]
    # Proportional lengths that sum to n.
    props = np.array([0.08, 0.12, 0.10, 0.14, 0.09, 0.07, 0.13, 0.11, 0.08, 0.08])
    lengths = (props / props.sum() * n).astype(int)
    lengths[-1] = n - int(lengths[:-1].sum())
    sig = _from_levels(segment_means, lengths.tolist())
    y = sig + 1.6 * rng.standard_normal(n)
    boundaries = [0, *np.cumsum(lengths).tolist()]
    return {
        "X": np.arange(n, dtype=float).reshape(-1, 1),
        "y": y,
        "sample_weight": None,
        "true_boundaries": boundaries,
        "name": "welllog",
        "source": "simulated",
        "description": "NMR-like well-log analog (simulated, n=4050, 10 segments).",
    }


def simulate_cgh(seed: int = 0) -> dict:
    """Array-CGH-like log2-ratios with three amplified / deleted regions."""

    rng = np.random.default_rng(seed)
    baseline_noise = 0.18
    segs = [
        (0.00, 380),  # normal
        (0.72, 85),  # amplification
        (0.00, 210),  # normal
        (-0.55, 60),  # deletion
        (0.00, 150),  # normal
        (0.48, 115),  # amplification
        (0.00, 100),
    ]
    levels = [m for m, _ in segs]
    lengths = [L for _, L in segs]
    sig = _from_levels(levels, lengths)
    n = sig.size
    y = sig + baseline_noise * rng.standard_normal(n)
    # Mild heteroscedasticity: edges slightly noisier (simulated probe quality).
    w = np.ones(n, dtype=float)
    w[:50] = 0.6
    w[-50:] = 0.6
    boundaries = [0, *np.cumsum(lengths).tolist()]
    return {
        "X": np.arange(n, dtype=float).reshape(-1, 1),
        "y": y,
        "sample_weight": w,
        "true_boundaries": boundaries,
        "name": "cgh",
        "source": "simulated",
        "description": (
            "Array-CGH-like log2 ratios (simulated, n=1100, two amplifications + one deletion)."
        ),
    }


def simulate_spx(seed: int = 0, n: int = 1500) -> dict:
    """GARCH-like regime-switching log-squared returns."""

    rng = np.random.default_rng(seed)
    vol_regimes = [0.010, 0.028, 0.012, 0.038, 0.015]
    lengths = [int(n * f) for f in (0.28, 0.16, 0.22, 0.14, 0.20)]
    lengths[-1] = n - sum(lengths[:-1])
    sigmas = _from_levels(vol_regimes, lengths)
    returns = sigmas * rng.standard_normal(n)
    y = np.log(returns**2 + 1e-8)  # log-squared returns are ~Gaussian in regime
    boundaries = [0, *np.cumsum(lengths).tolist()]
    return {
        "X": np.arange(n, dtype=float).reshape(-1, 1),
        "y": y,
        "sample_weight": None,
        "true_boundaries": boundaries,
        "name": "spx",
        "source": "simulated",
        "description": "S&P-500-like volatility regimes (simulated, n=1500, 5 regimes).",
    }


def simulate_methylation(seed: int = 0, n: int = 600) -> dict:
    """Beta-distributed methylation rates with three plateau levels."""

    rng = np.random.default_rng(seed)
    levels = [0.18, 0.82, 0.42, 0.88, 0.25]
    lengths = [140, 110, 130, 100, n - 480]
    mu = _from_levels(levels, lengths)
    concentration = 70.0
    y = rng.beta(concentration * mu, concentration * (1.0 - mu))
    y = np.clip(y, 1e-3, 1.0 - 1e-3)
    boundaries = [0, *np.cumsum(lengths).tolist()]
    return {
        "X": np.arange(n, dtype=float).reshape(-1, 1),
        "y": y,
        "sample_weight": None,
        "true_boundaries": boundaries,
        "name": "methylation",
        "source": "simulated",
        "description": "CpG-atlas-like methylation fractions (simulated, n=600, 5 plateaus).",
    }
