"""Baseline wrappers around the upstream ``ruptures`` library.

We do not re-implement the algorithms. Each function loads ``ruptures``
lazily, drives the corresponding ``ruptures`` class
(``Pelt``/``Dynp``/``Binseg``), and reshapes the output into a
:class:`~bayesbreak.baselines._types.BaselineResult`.

Coverage:

- :func:`run_pelt` — Killick, Fearnhead & Eckley (2012). Penalty-based exact
  DP with pruning; expected linear cost under a changepoint-density
  condition, ``O(n^2)`` worst case (the qualifier added in §1 paragraph
  5-A1 of the new manuscript).
- :func:`run_dynp` — Jackson et al. (2005) optimal partitioning, with a
  fixed number of segments ``k_max`` exposed by ``ruptures.Dynp``.
- :func:`run_binseg` — classical binary segmentation, capped by ``k_max``.
- :func:`run_wbs` — wild binary segmentation (Fryzlewicz 2014), driven by
  ``ruptures.Binseg`` with random window sub-sampling on top.

All wrappers accept either a 1-D ``y`` (univariate) or a 2-D ``y`` of shape
``(n, d)`` (multivariate). The default cost model is ``"l2"``; pass
``cost_model="rbf"``, ``"linear"``, etc. through to ``ruptures``.

If ``ruptures`` is not installed, every call raises
``ImportError("ruptures is required ...; pip install bayesbreak[baselines]")``
with a single message instead of an opaque attribute error.
"""

from __future__ import annotations

import importlib
import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ._types import BaselineResult

_RUPTURES_HINT = (
    "ruptures is required for the PELT / Dynp / Binseg / WBS wrappers; "
    "install with `pip install bayesbreak[baselines]` or `pip install ruptures`."
)


def _load_ruptures():
    """Import ``ruptures`` lazily with a single readable error message."""
    try:
        return importlib.import_module("ruptures")
    except ImportError as exc:  # pragma: no cover - env-specific
        raise ImportError(_RUPTURES_HINT) from exc


def _as_signal(y: ArrayLike) -> tuple[np.ndarray, int]:
    arr = np.asarray(y, dtype=float)
    if arr.ndim == 1:
        arr2d = arr.reshape(-1, 1)
    elif arr.ndim == 2:
        arr2d = arr
    else:
        raise ValueError(f"y must be 1-D or 2-D; got ndim={arr.ndim}.")
    n = int(arr2d.shape[0])
    return arr2d, n


def _strip_endpoint(breaks: list[int], n: int) -> np.ndarray:
    """``ruptures`` returns boundary lists that end with ``n``; drop it."""
    interior = [int(b) for b in breaks if 0 < int(b) < n]
    return np.asarray(interior, dtype=np.intp)


def run_pelt(
    y: ArrayLike,
    *,
    penalty: float,
    cost_model: str = "l2",
    min_size: int = 2,
    jump: int = 1,
) -> BaselineResult:
    """Killick, Fearnhead & Eckley (2012) PELT via ``ruptures.Pelt``.

    Parameters
    ----------
    y : array-like
        1-D or 2-D signal.
    penalty : float
        Linear penalty per changepoint passed to ``Pelt.predict(pen=...)``.
        Tuning the penalty is the standard PELT calibration knob (BIC-like
        choices are common; see Killick & Eckley 2014).
    cost_model : {"l1", "l2", "rbf", ...}, default "l2"
        ``ruptures`` cost-function identifier.
    min_size : int, default 2
        Minimum segment length.
    jump : int, default 1
        Candidate-changepoint subgrid step.
    """
    rpt = _load_ruptures()
    sig, n = _as_signal(y)
    algo = rpt.Pelt(model=cost_model, min_size=int(min_size), jump=int(jump)).fit(sig)
    breaks = algo.predict(pen=float(penalty))
    return BaselineResult(
        algorithm="pelt",
        package="ruptures",
        package_version=getattr(rpt, "__version__", "unknown"),
        n=n,
        boundaries=_strip_endpoint(breaks, n),
        tuning={
            "penalty": float(penalty),
            "cost_model": cost_model,
            "min_size": int(min_size),
            "jump": int(jump),
        },
    )


def run_dynp(
    y: ArrayLike,
    *,
    n_bkps: int,
    cost_model: str = "l2",
    min_size: int = 2,
    jump: int = 1,
) -> BaselineResult:
    """Jackson et al. (2005) optimal partitioning via ``ruptures.Dynp``.

    ``n_bkps`` is the number of interior changepoints (segments minus one).
    """
    rpt = _load_ruptures()
    sig, n = _as_signal(y)
    algo = rpt.Dynp(model=cost_model, min_size=int(min_size), jump=int(jump)).fit(sig)
    breaks = algo.predict(n_bkps=int(n_bkps))
    return BaselineResult(
        algorithm="optimal_partitioning",
        package="ruptures",
        package_version=getattr(rpt, "__version__", "unknown"),
        n=n,
        boundaries=_strip_endpoint(breaks, n),
        tuning={
            "n_bkps": int(n_bkps),
            "cost_model": cost_model,
            "min_size": int(min_size),
            "jump": int(jump),
        },
    )


def run_binseg(
    y: ArrayLike,
    *,
    n_bkps: int | None = None,
    penalty: float | None = None,
    cost_model: str = "l2",
    min_size: int = 2,
    jump: int = 1,
) -> BaselineResult:
    """Classical binary segmentation via ``ruptures.Binseg``.

    Pass either ``n_bkps`` (fixed segment count) or ``penalty`` (BIC-like
    stopping criterion); ``ruptures.Binseg.predict`` accepts whichever is
    supplied.
    """
    if (n_bkps is None) == (penalty is None):
        raise ValueError("Provide exactly one of `n_bkps` or `penalty`.")
    rpt = _load_ruptures()
    sig, n = _as_signal(y)
    algo = rpt.Binseg(model=cost_model, min_size=int(min_size), jump=int(jump)).fit(sig)
    kwargs: dict[str, Any] = {}
    if n_bkps is not None:
        kwargs["n_bkps"] = int(n_bkps)
    else:
        kwargs["pen"] = float(penalty)
    breaks = algo.predict(**kwargs)
    return BaselineResult(
        algorithm="binary_segmentation",
        package="ruptures",
        package_version=getattr(rpt, "__version__", "unknown"),
        n=n,
        boundaries=_strip_endpoint(breaks, n),
        tuning={
            "n_bkps": kwargs.get("n_bkps"),
            "penalty": kwargs.get("pen"),
            "cost_model": cost_model,
            "min_size": int(min_size),
            "jump": int(jump),
        },
    )


def run_wbs(
    y: ArrayLike,
    *,
    n_bkps: int | None = None,
    penalty: float | None = None,
    cost_model: str = "l2",
    n_random_windows: int = 100,
    min_size: int = 2,
    jump: int = 1,
    random_state: int | None = 0,
) -> BaselineResult:
    """Wild Binary Segmentation (Fryzlewicz 2014).

    Implemented on top of ``ruptures.Binseg`` by sampling random windows and
    taking the union of detected breakpoints, then re-fitting BS on the
    pooled candidates. This follows the standard WBS-with-CUSUM recipe; the
    randomness is seeded by ``random_state``.
    """
    if (n_bkps is None) == (penalty is None):
        raise ValueError("Provide exactly one of `n_bkps` or `penalty`.")
    rpt = _load_ruptures()
    sig, n = _as_signal(y)
    rng = np.random.default_rng(random_state)

    candidates: set[int] = set()
    for _ in range(int(n_random_windows)):
        lo = int(rng.integers(0, max(1, n - 2 * min_size)))
        hi = int(rng.integers(lo + 2 * min_size, n + 1))
        window = sig[lo:hi]
        if window.shape[0] < 2 * min_size:
            continue
        try:
            bs = rpt.Binseg(model=cost_model, min_size=int(min_size), jump=int(jump)).fit(window)
            local_breaks = bs.predict(n_bkps=1)
        except Exception:
            continue
        for b in local_breaks:
            if 0 < int(b) < window.shape[0]:
                candidates.add(int(b) + lo)

    if not candidates:
        # Fall back to plain BS so we always return a result.
        return run_binseg(
            y,
            n_bkps=n_bkps,
            penalty=penalty,
            cost_model=cost_model,
            min_size=min_size,
            jump=jump,
        )

    breaks = _select_wbs_candidates(
        rpt,
        sig,
        candidates,
        n_bkps=n_bkps,
        penalty=penalty,
        cost_model=cost_model,
        min_size=int(min_size),
    )

    return BaselineResult(
        algorithm="wild_binary_segmentation",
        package="ruptures",
        package_version=getattr(rpt, "__version__", "unknown"),
        n=n,
        boundaries=_strip_endpoint(breaks, n),
        tuning={
            "n_bkps": n_bkps,
            "penalty": penalty,
            "cost_model": cost_model,
            "n_random_windows": int(n_random_windows),
            "min_size": int(min_size),
            "jump": int(jump),
            "random_state": random_state,
        },
        extra={
            "n_candidate_breakpoints": len(candidates),
            "candidate_selection": "candidate-constrained-dynamic-programming",
        },
    )


def _select_wbs_candidates(
    rpt: Any,
    signal: np.ndarray,
    candidates: set[int],
    *,
    n_bkps: int | None,
    penalty: float | None,
    cost_model: str,
    min_size: int,
) -> list[int]:
    """Minimize the upstream cost over partitions supported by WBS candidates."""

    n = int(signal.shape[0])
    cost = rpt.costs.cost_factory(model=cost_model).fit(signal)
    required_length = max(int(min_size), int(getattr(cost, "min_size", 1)))
    positions = np.asarray([0, *sorted(candidates), n], dtype=np.intp)
    max_segments = min(len(positions) - 1, n // required_length)
    if n_bkps is not None:
        target_segments = int(n_bkps) + 1
        if target_segments < 1:
            raise ValueError("n_bkps must be nonnegative")
        if target_segments > max_segments:
            raise ValueError(
                f"WBS candidates support at most {max_segments - 1} breakpoints; requested {n_bkps}"
            )
        segment_limit = target_segments
    else:
        penalty_value = float(penalty)
        if not math.isfinite(penalty_value) or penalty_value < 0:
            raise ValueError("penalty must be finite and nonnegative")
        segment_limit = max_segments

    costs = np.full((segment_limit + 1, positions.size), np.inf, dtype=float)
    predecessors = np.full((segment_limit + 1, positions.size), -1, dtype=np.intp)
    costs[0, 0] = 0.0
    for segment_count in range(1, segment_limit + 1):
        for stop_index in range(1, positions.size):
            stop = int(positions[stop_index])
            for start_index in range(stop_index):
                if not math.isfinite(float(costs[segment_count - 1, start_index])):
                    continue
                start = int(positions[start_index])
                if stop - start < required_length:
                    continue
                try:
                    segment_cost = float(cost.error(start, stop))
                except Exception:
                    continue
                candidate_cost = float(costs[segment_count - 1, start_index]) + segment_cost
                if candidate_cost < costs[segment_count, stop_index]:
                    costs[segment_count, stop_index] = candidate_cost
                    predecessors[segment_count, stop_index] = start_index

    if n_bkps is not None:
        selected_segments = target_segments
    else:
        objectives = costs[1:, -1] + float(penalty) * np.arange(segment_limit)
        if not np.any(np.isfinite(objectives)):
            raise ValueError("No feasible WBS candidate partition")
        selected_segments = int(np.argmin(objectives)) + 1
    if not math.isfinite(float(costs[selected_segments, -1])):
        raise ValueError("No feasible WBS candidate partition")

    selected: list[int] = []
    stop_index = positions.size - 1
    for segment_count in range(selected_segments, 0, -1):
        start_index = int(predecessors[segment_count, stop_index])
        if start_index < 0:
            raise RuntimeError("WBS candidate backtracking failed")
        if start_index > 0:
            selected.append(int(positions[start_index]))
        stop_index = start_index
    return [*sorted(selected), n]
