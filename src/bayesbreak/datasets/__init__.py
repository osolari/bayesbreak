"""Real-data loaders shipped with BayesBreak.

Every loader returns a :class:`DatasetBundle` with a consistent schema:

- ``X``           : ``(n, 1)`` design points / locations.
- ``y``           : ``(n,)``    response (univariate) or ``(n, d)`` multivariate.
- ``sample_weight``: ``(n,)`` or ``None``.
- ``true_boundaries`` : list of ints when ground truth is available (always for
  simulated fallbacks; often unknown for real data).
- ``name``        : short identifier (``"welllog"``, ``"cgh"``, …).
- ``source``      : ``"downloaded"`` or ``"simulated"``.
- ``description`` : human-readable one-liner.

Loaders prefer a real, cached download (via ``pooch`` in the
``bayesbreak[datasets]`` extra) and automatically fall back to the deterministic
simulated analog in :mod:`bayesbreak.datasets._simulate` when the download is
not possible (missing dep, offline, hash mismatch, …) or when the caller passes
``simulated=True``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._cache import banner, cache_dir, describe_fallback

FloatArray = NDArray[np.floating]


@dataclass
class DatasetBundle:
    """Normalised schema returned by every ``load_*`` loader."""

    X: FloatArray
    y: FloatArray
    sample_weight: FloatArray | None
    true_boundaries: list[int]
    name: str
    source: str  # "downloaded" or "simulated"
    description: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_simulated(self) -> bool:
        return self.source == "simulated"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DatasetBundle:
        return cls(
            X=np.asarray(d["X"], dtype=float),
            y=np.asarray(d["y"], dtype=float),
            sample_weight=(
                None
                if d.get("sample_weight") is None
                else np.asarray(d["sample_weight"], dtype=float)
            ),
            true_boundaries=list(d.get("true_boundaries") or []),
            name=str(d["name"]),
            source=str(d["source"]),
            description=str(d["description"]),
            metadata=dict(d.get("metadata") or {}),
        )


from .base import DatasetCard, load_with_provenance  # noqa: E402
from .cgh import load_cgh  # noqa: E402
from .methylation import load_methylation  # noqa: E402
from .spx import load_spx  # noqa: E402
from .welllog import load_welllog  # noqa: E402

__all__ = [
    "DatasetBundle",
    "DatasetCard",
    "banner",
    "cache_dir",
    "describe_fallback",
    "load_cgh",
    "load_with_provenance",
    "load_methylation",
    "load_spx",
    "load_welllog",
]
