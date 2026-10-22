"""Uniform result type for baseline changepoint algorithms.

Each upstream package exposes its own segmentation API (``ruptures`` Python
objects, ``DNAcopy::segment`` R dataframes, etc.). We do not re-implement
any of these algorithms; the wrappers in this subpackage simply call the
upstream library and normalize its output into a :class:`BaselineResult`.

Conventions
-----------

- ``boundaries`` lists **interior** boundary indices on the original-data
  index axis, i.e. integers in ``[1, n - 1]``. The implied segmentation is
  ``(0, boundaries[0], boundaries[1], ..., boundaries[-1], n)``. Empty
  ``boundaries`` means one segment.
- ``k`` is the number of segments, equal to ``len(boundaries) + 1``.
- ``algorithm`` identifies the algorithm family (e.g. ``"pelt"``, ``"cbs"``).
- ``package`` records the upstream library name and version actually used.
- ``tuning`` records the input arguments (penalty value, jump, minimum-length
  constraints, etc.) so the run is reproducible without re-reading the
  caller's code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class BaselineResult:
    """Result of a single baseline segmentation run."""

    algorithm: str
    package: str
    package_version: str
    n: int
    boundaries: NDArray[np.intp]
    tuning: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def k(self) -> int:
        return int(self.boundaries.size) + 1

    @property
    def map_boundaries(self) -> list[int]:
        """``[0, b_1, ..., b_{k-1}, n]`` for callers that expect endpoints."""
        return [0, *self.boundaries.tolist(), int(self.n)]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["boundaries"] = self.boundaries.tolist()
        d["k"] = self.k
        return d
