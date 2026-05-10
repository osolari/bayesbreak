"""Smoke tests for the real-data loaders.

All tests run in ``simulated=True`` mode so the suite is fully offline.
"""

from __future__ import annotations

import numpy as np
import pytest

from bayesbreak.datasets import (
    DatasetBundle,
    load_cgh,
    load_methylation,
    load_spx,
    load_welllog,
)

_LOADERS = [load_welllog, load_cgh, load_spx, load_methylation]


@pytest.mark.parametrize("loader", _LOADERS)
def test_simulated_bundle_schema(loader):
    b = loader(simulated=True)
    assert isinstance(b, DatasetBundle)
    assert b.source == "simulated"
    assert b.is_simulated
    assert b.X.ndim == 2 and b.X.shape[1] == 1
    assert b.y.ndim == 1
    assert b.X.shape[0] == b.y.shape[0]
    assert np.all(np.isfinite(b.y))
    # Simulated loaders should know the ground-truth boundary vector.
    assert len(b.true_boundaries) >= 2
    assert b.true_boundaries[0] == 0
    assert b.true_boundaries[-1] == b.y.size


def test_simulated_determinism():
    """Two calls return byte-identical arrays (seed-pinned simulation)."""

    a = load_welllog(simulated=True)
    b = load_welllog(simulated=True)
    assert np.array_equal(a.y, b.y)


def test_methylation_domain_is_open_unit_interval():
    b = load_methylation(simulated=True)
    assert np.all((b.y > 0.0) & (b.y < 1.0))


def test_cgh_sample_weight_present():
    b = load_cgh(simulated=True)
    assert b.sample_weight is not None
    assert b.sample_weight.shape == b.y.shape
    assert np.all(b.sample_weight > 0)


def test_methylation_returns_valid_bundle_when_csv_missing(tmp_path):
    """A missing csv_path does not crash: the loader falls through to the real
    methylKit mirror and, if that is also unreachable, to the simulated analog."""

    b = load_methylation(csv_path=tmp_path / "does_not_exist.csv")
    assert isinstance(b, DatasetBundle)
    assert b.y.ndim == 1
    assert np.all((b.y > 0.0) & (b.y < 1.0))
    assert b.source in {"downloaded", "simulated"}


@pytest.mark.parametrize(
    "loader,family_name",
    [
        (load_welllog, "gaussian"),
        (load_cgh, "gaussian"),
        (load_spx, "gaussian"),
        (load_methylation, "beta"),
    ],
)
def test_bundle_fits_corresponding_family(loader, family_name):
    """Every simulated bundle is compatible with the family it targets."""

    from bayesbreak import make_bayesbreak

    b = loader(simulated=True)
    # Limit sample size for speed when testing welllog and spx.
    n_max = 200
    X = b.X[:n_max]
    y = b.y[:n_max]
    est = make_bayesbreak(family_name, k_max=6)
    est.fit(X, y, sample_weight=(b.sample_weight[:n_max] if b.sample_weight is not None else None))
    assert est.k_map_ >= 1
    assert len(est.map_boundaries_) == est.k_map_ + 1
