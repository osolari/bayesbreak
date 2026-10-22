"""Data carriers for the latent-template mixture EM.

Separated from :mod:`bayesbreak.mixture` so downstream callers (and tests)
can depend on the state shape without importing the full EM runtime.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _GroupState:
    """Internal per-group container."""

    hyper: dict[str, float]
    template: list[int]
    k_g: int
    log_score_offset: float  # log p(k_g) - log C_{k_g}


@dataclass(frozen=True)
class RestartDiagnostic:
    """Terminal status for one latent-template optimization restart."""

    restart: int
    seed: int
    status: str
    final_objective: float | None
    n_iterations: int
    reason: str | None = None
