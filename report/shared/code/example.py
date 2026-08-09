"""Minimal deterministic program used by the book template."""

from __future__ import annotations

import numpy as np


def normalise(values: np.ndarray) -> np.ndarray:
    """Return zero-mean, unit-standard-deviation values."""
    values = np.asarray(values, dtype=np.float64)
    scale = values.std()
    if scale == 0.0:
        raise ValueError("normalise requires non-constant input")
    return (values - values.mean()) / scale


def main() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    output = normalise(values)

    print(f"input mean: {values.mean():.3f}")
    print(f"output mean: {output.mean():.3f}")

    assert np.isclose(output.mean(), 0.0)
    assert np.isclose(output.std(), 1.0)
    print("checks: PASS")


if __name__ == "__main__":
    main()
