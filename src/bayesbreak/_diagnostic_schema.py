"""Machine-readable schema for the bayesbreak diagnostic reports.

Separated from :mod:`bayesbreak.diagnostics` so that downstream tooling can
depend on the report shape without importing the full DP-diagnostic runtime.
"""

from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class DiagnosticCheck:
    """Single invariant check.

    ``failure_mode`` is an optional short tag identifying which failure mode
    the check targets, aligned with the §4 approximation-validation
    checklist and the §5b limitations section. Examples used by
    :func:`bayesbreak.diagnostics.run_non_conjugate_diagnostics`:

    - ``"short-segment-laplace"``: Laplace expansion inaccuracy on short blocks;
    - ``"mf-vb-variance"``: mean-field variational variance underestimation;
    - ``"method-sensitivity"``: posterior summary depending heavily on the
      choice of approximation method;
    - ``"ep-nonconvergence"``: EP iteration failing to converge or oscillating
      (the canonical EP failure mode per ``prop:uniform-bounds`` (v));
    - ``"tv-bound"``: total-variation bound on ``P(k|y)`` from
      Corollary ``cor:probability-error-conversion`` (derivable from
      Proposition ``prop:stability``).
    """

    name: str
    passed: bool
    detail: str
    measured: float | None = None
    tolerance: float | None = None
    failure_mode: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class DiagnosticReport:
    """Bundle of checks for one fitted estimator."""

    estimator_class: str
    n: int
    k_max: int
    k_map: int | None
    checks: list[DiagnosticCheck] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def summary(self) -> str:
        n_pass = sum(c.passed for c in self.checks)
        return f"{n_pass}/{len(self.checks)} checks passed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimator_class": self.estimator_class,
            "n": self.n,
            "k_max": self.k_max,
            "k_map": self.k_map,
            "checks": [c.to_dict() for c in self.checks],
            "extra": _json_safe(self.extra),
            "created_at": self.created_at,
            "passed": self.passed,
            "summary": self.summary,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=_json_default)


def _json_default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _json_safe(d: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _json_safe(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, np.generic):
            out[k] = v.item()
        else:
            out[k] = v
    return out
