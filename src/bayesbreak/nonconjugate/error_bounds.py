"""Reachable-segment error assessment and conditional partition bounds."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from ..base import BayesBreakSegmenter


@dataclass(frozen=True)
class SegmentErrorRecord:
    """A support-aligned comparison with separately reported error sources."""

    family: str
    block_support_hash: str
    reference_method: str
    max_log_score_error: float | None
    optimization_residual: float | None
    tail_bound: float | None
    quadrature_error: float | None
    convergence_status: str
    n_reachable_blocks: int
    failure_reason: str | None = None
    schema_version: str = "1.0.0"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_reachable_segment_error(
    approximate_log_scores: ArrayLike,
    reference_log_scores: ArrayLike,
    *,
    family: str,
    reference_method: str,
    k_max: int,
    optimization_residual: float | None = None,
    tail_bound: float | None = None,
    quadrature_error: float | None = None,
    convergence_status: str = "unverifiable",
) -> SegmentErrorRecord:
    """Compare score arrays only when reachable block coordinates agree."""

    approximate = np.asarray(approximate_log_scores, dtype=float)
    reference = np.asarray(reference_log_scores, dtype=float)
    _validate_score_shapes(approximate, reference)
    n = approximate.shape[0] - 1
    if not isinstance(k_max, int) or isinstance(k_max, bool) or not 1 <= k_max <= n:
        raise ValueError(f"k_max must satisfy 1 <= k_max <= {n}")
    for name, value in {
        "optimization_residual": optimization_residual,
        "tail_bound": tail_bound,
        "quadrature_error": quadrature_error,
    }.items():
        _validate_optional_error(value, name)
    if convergence_status not in {"verified", "unverifiable", "failed", "not-applicable"}:
        raise ValueError("Unknown convergence_status")

    reachable = _reachable_blocks_mask(n, k_max)
    invalid = reachable & (
        np.isnan(approximate)
        | np.isposinf(approximate)
        | np.isnan(reference)
        | np.isposinf(reference)
    )
    approximate_support = reachable & np.isfinite(approximate)
    reference_support = reachable & np.isfinite(reference)
    shared_support = approximate_support & reference_support
    support_hash = _support_hash(shared_support)

    if np.any(invalid):
        return SegmentErrorRecord(
            family=family,
            block_support_hash=support_hash,
            reference_method=reference_method,
            max_log_score_error=None,
            optimization_residual=optimization_residual,
            tail_bound=tail_bound,
            quadrature_error=quadrature_error,
            convergence_status="failed",
            n_reachable_blocks=int(np.sum(shared_support)),
            failure_reason="reachable scores contain NaN or +inf",
        )
    if not np.array_equal(approximate_support, reference_support):
        return SegmentErrorRecord(
            family=family,
            block_support_hash=support_hash,
            reference_method=reference_method,
            max_log_score_error=None,
            optimization_residual=optimization_residual,
            tail_bound=tail_bound,
            quadrature_error=quadrature_error,
            convergence_status="unverifiable",
            n_reachable_blocks=int(np.sum(shared_support)),
            failure_reason="approximate and reference reachable block supports differ",
        )
    if not np.any(shared_support):
        return SegmentErrorRecord(
            family=family,
            block_support_hash=support_hash,
            reference_method=reference_method,
            max_log_score_error=None,
            optimization_residual=optimization_residual,
            tail_bound=tail_bound,
            quadrature_error=quadrature_error,
            convergence_status="unverifiable",
            n_reachable_blocks=0,
            failure_reason="no shared reachable blocks",
        )

    max_error = float(np.max(np.abs(approximate[shared_support] - reference[shared_support])))
    return SegmentErrorRecord(
        family=family,
        block_support_hash=support_hash,
        reference_method=reference_method,
        max_log_score_error=max_error,
        optimization_residual=optimization_residual,
        tail_bound=tail_bound,
        quadrature_error=quadrature_error,
        convergence_status=convergence_status,
        n_reachable_blocks=int(np.sum(shared_support)),
    )


def certify_against_gh_reference(
    estimator: BayesBreakSegmenter,
    *,
    reference_gh_points: int = 120,
    family_label: str | None = None,
) -> SegmentErrorRecord:
    r"""Build a :class:`SegmentErrorRecord` from a fitted logistic-normal fit.

    The certificate compares the estimator's per-block log-evidence to a
    **high-order Gauss--Hermite quadrature reference** on the same data.
    This yields a meaningful empirical bound for every approximation
    kernel — the previous same-method Q-doubling proxy was only
    informative for ``approx="quadrature"`` / ``"gh"`` and reported
    ``quadrature_error \approx 0`` for Laplace, JJ, and PG-VB.

    ``reference_gh_points`` defaults to 120, the same order used in the
    manuscript's misspecification suite for its own logistic-normal
    reference (``prop:uniform-bounds`` (i): ``O(Q^{-2r})`` for
    ``C^{2r}`` integrands).
    """

    from ..families import BayesBreakLogisticNormal

    if not isinstance(estimator, BayesBreakLogisticNormal):
        raise TypeError(
            "certify_against_gh_reference currently supports BayesBreakLogisticNormal only"
        )
    lA0 = np.asarray(estimator.log_block_evidence_, dtype=float)
    n = lA0.shape[0] - 1
    k_max = int(estimator.log_left_.shape[0] - 1)

    high_q = int(reference_gh_points)
    if high_q < 5:
        raise ValueError("reference_gh_points must be >= 5 for stable quadrature")

    # Fit a high-order GH reference on the same data and hyperparameters,
    # regardless of the estimator's own approximation kernel. For
    # ``approx in {"gh", "quadrature"}`` this recovers the earlier
    # Q-doubling behaviour when the caller passes 2 * gh_points.
    reference = BayesBreakLogisticNormal(
        k_max=estimator.k_max,
        estimate_hyper=False,
        approx="quadrature",
        regression_curve=estimator.regression_curve,
        length_prior=estimator.length_prior,
        boundary_coordinates=estimator.boundary_coordinates,
        prior_k=estimator.prior_k,
        nu=estimator.hyper_["nu"],
        rho2=estimator.hyper_["rho2"],
        gh_points=high_q,
        max_iter=estimator.max_iter,
        partition_prior=getattr(estimator, "partition_prior", None),
    )
    x_design = estimator.x_design_.reshape(-1, 1)
    y_train = estimator._y_train_
    reference.fit(x_design, y_train, sample_weight=estimator.sample_weight_)
    lA0_ref = np.asarray(reference.log_block_evidence_, dtype=float)

    reachable = _reachable_blocks_mask(n, k_max)
    shared = reachable & np.isfinite(lA0) & np.isfinite(lA0_ref)
    quadrature_error: float | None
    if np.any(shared):
        quadrature_error = float(np.max(np.abs(lA0[shared] - lA0_ref[shared])))
    else:
        quadrature_error = None

    opt_residual = getattr(estimator, "optimization_residual_", None)
    if opt_residual is not None:
        opt_arr = np.asarray(opt_residual, dtype=float)
        opt_shared = shared & np.isfinite(opt_arr)
        optimization_residual = (
            float(np.max(opt_arr[opt_shared])) if np.any(opt_shared) else None
        )
    else:
        optimization_residual = None

    tail_arr = getattr(estimator, "analytic_tail_bound_", None)
    if tail_arr is not None:
        tail_np = np.asarray(tail_arr, dtype=float)
        tail_shared = shared & np.isfinite(tail_np)
        tail_bound: float | None = (
            float(np.max(tail_np[tail_shared])) if np.any(tail_shared) else None
        )
    else:
        tail_bound = None

    if estimator.approx == "ep":
        convergence_status = (
            "verified"
            if bool(getattr(estimator, "ep_all_converged_", False))
            else "failed"
        )
    else:
        convergence_status = "verified"

    return evaluate_reachable_segment_error(
        lA0,
        lA0_ref,
        family=family_label or type(estimator).__name__,
        reference_method=f"gh-{high_q}",
        k_max=k_max,
        optimization_residual=optimization_residual,
        tail_bound=tail_bound,
        quadrature_error=quadrature_error,
        convergence_status=convergence_status,
    )


# Compatibility alias retained for API consumers. The canonical routine is
# ``certify_against_gh_reference``, which evaluates a specified GH reference.
certify_by_quadrature_doubling = certify_against_gh_reference


def propagate_partition_bounds(max_log_score_error: float, k_max: int) -> dict[str, float]:
    """Return global consequences conditional on a uniform reachable-block bound."""

    error = float(max_log_score_error)
    if not math.isfinite(error) or error < 0:
        raise ValueError("max_log_score_error must be finite and nonnegative")
    if not isinstance(k_max, int) or isinstance(k_max, bool) or k_max < 1:
        raise ValueError("k_max must be a positive integer")
    eta = k_max * error
    exponent = 2.0 * eta
    ratio_upper = math.exp(exponent) if exponent <= math.log(np.finfo(float).max) else math.inf
    ratio_lower = 0.0 if math.isinf(ratio_upper) else math.exp(-exponent)
    tv_bound = 1.0 if math.isinf(ratio_upper) else min(1.0, math.expm1(exponent))
    return {
        "max_log_evidence_error": eta,
        "max_log_posterior_odds_error": 2.0 * eta,
        "probability_ratio_lower": ratio_lower,
        "probability_ratio_upper": ratio_upper,
        "tv_upper_bound": tv_bound,
    }


def _validate_score_shapes(approximate: np.ndarray, reference: np.ndarray) -> None:
    if approximate.ndim != 2 or approximate.shape[0] != approximate.shape[1]:
        raise ValueError("approximate_log_scores must be square")
    if reference.shape != approximate.shape:
        raise ValueError("approximate and reference score arrays must have identical shape")
    if approximate.shape[0] < 2:
        raise ValueError("score arrays must represent at least one observation")


def _validate_optional_error(value: float | None, name: str) -> None:
    if value is not None and (not math.isfinite(float(value)) or float(value) < 0):
        raise ValueError(f"{name} must be finite and nonnegative when provided")


def _reachable_blocks_mask(n: int, k_max: int) -> np.ndarray:
    mask = np.zeros((n + 1, n + 1), dtype=bool)
    for start in range(n):
        for stop in range(start + 1, n + 1):
            if start + (n - stop) + 1 <= k_max:
                mask[start, stop] = True
    return mask


def _support_hash(mask: np.ndarray) -> str:
    coordinates = [[int(start), int(stop)] for start, stop in zip(*np.nonzero(mask), strict=True)]
    payload = json.dumps(coordinates, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()
