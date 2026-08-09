r"""Diagnostic checks for fitted BayesBreak estimators (IMP-13).

The report enumerates a small set of *required* invariants and sensitivity
diagnostics in §4.2 (Posterior-evidence sanity checks), §4.8
(Approximation-validation checklist), and §5 (Numerical-implementation
checklist). This module collects them in a single place so that every
experiment can emit a machine-readable diagnostic record alongside its
figures and tables.

Two top-level entry points:

- :func:`run_dp_diagnostics` — invariants of the sum-product / max-sum DP
  on a fitted segmenter, replicates segmenter, or mixture classifier.
- :func:`run_non_conjugate_diagnostics` — block-error and posterior
  sensitivity vs. a reference (typically a higher-accuracy quadrature fit).

Both return a :class:`DiagnosticReport` with a pass/fail summary and a
``to_json()`` serializer. Failures are reported but do not raise — the
caller decides what to do with them.
"""

from __future__ import annotations

import dataclasses
import json
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from . import dp as _dp
from .nonconjugate import evaluate_reachable_segment_error, propagate_partition_bounds
from .utils import logsumexp as _logsumexp

if TYPE_CHECKING:
    pass


FloatArray = NDArray[np.floating]


# -----------------------------------------------------------------------------
# Report schema
# -----------------------------------------------------------------------------


@dataclass
class DiagnosticCheck:
    """Single invariant check.

    ``failure_mode`` is an optional short tag identifying which failure mode
    the check targets, aligned with the §4 approximation-validation
    checklist and the §5b limitations section. Examples used by
    :func:`run_non_conjugate_diagnostics`:

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
            "passed": self.passed,
            "summary": self.summary,
            "checks": [c.to_dict() for c in self.checks],
            "extra": _json_safe(self.extra),
            "created_at": self.created_at,
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


# -----------------------------------------------------------------------------
# DP-layer invariants (§4.2 sanity checks; §5 numerical checklist)
# -----------------------------------------------------------------------------


def run_dp_diagnostics(
    estimator: Any,
    *,
    atol: float = 1e-7,
) -> DiagnosticReport:
    """Run the four DP invariants on a fitted estimator.

    Works on :class:`BayesBreakSegmenter`,
    :class:`SharedBoundaryReplicatesSegmenter`, and any object that exposes
    ``log_left_``, ``log_right_``, ``log_block_evidence_`` (or ``lA0_pool``),
    ``k_posterior_``, ``boundary_marginals_``, ``map_boundaries_``,
    ``log_joint_map_``, ``log_g_table_``, ``k_map_``, ``n_``.

    The four checks are:

    1. ``Σ_k P(k | y) = 1``;
    2. ``L̃[k_map, n] = R̃[k_map, 0]`` — the forward/backward
       total-evidence identity of Proposition ``prop:fb-duality``;
    3. ``Σ_i P(b_i = 1 | y, k_map) = k_map − 1`` (boundary-event sum
       identity, stated inline in the DP correctness theorem);
    4. max-sum backtracked score equals the stored ``log_joint_map_``
       (Theorem ``thm:map-correctness``).
    """

    n = int(estimator.n_)
    k_map = int(estimator.k_map_)
    k_max = int(estimator.log_left_.shape[0] - 1)
    klass = type(estimator).__name__

    log_left = np.asarray(estimator.log_left_, dtype=float)
    log_right = np.asarray(estimator.log_right_, dtype=float)
    p_k = np.asarray(estimator.k_posterior_, dtype=float)
    boundary_marginals = np.asarray(estimator.boundary_marginals_, dtype=float)
    log_joint_map = float(estimator.log_joint_map_)
    map_boundaries = list(estimator.map_boundaries_)
    log_g_table = getattr(estimator, "log_g_table_", None)

    # Where to find the per-block log-evidence: depends on class.
    lA0 = getattr(estimator, "log_block_evidence_", None)
    if lA0 is None:
        lA0 = getattr(estimator, "lA0_pool_", None)
    if lA0 is None:
        # Reconstruct from L (this only happens for the mixture's group state).
        raise ValueError(
            "Estimator has no log_block_evidence_ attribute; "
            "run_dp_diagnostics expects a segmenter or replicates estimator."
        )
    lA0 = np.asarray(lA0, dtype=float)

    checks: list[DiagnosticCheck] = []

    # 1. Posterior over k sums to 1.
    s = float(np.sum(p_k))
    checks.append(
        DiagnosticCheck(
            name="posterior_k_normalised",
            passed=abs(s - 1.0) < atol,
            detail=f"|Σ P(k|y) − 1| = {abs(s - 1.0):.3e}",
            measured=s,
            tolerance=atol,
        )
    )

    # 2. Forward/backward total-evidence identity (prop:fb-duality).
    fwd = float(log_left[k_map, n])
    bwd = float(log_right[k_map, 0])
    delta = abs(fwd - bwd)
    checks.append(
        DiagnosticCheck(
            name="forward_backward_agreement",
            passed=delta < 1e-6 or delta < 1e-9 * max(1.0, abs(fwd)),
            detail=f"|L̃[{k_map},n] − R̃[{k_map},0]| = {delta:.3e}",
            measured=delta,
            tolerance=1e-6,
        )
    )

    # 3. Boundary-event marginals sum to k_map − 1.
    expected = max(0, k_map - 1)
    measured = float(np.sum(boundary_marginals))
    checks.append(
        DiagnosticCheck(
            name="boundary_event_sum",
            passed=abs(measured - expected) < 1e-6,
            detail=f"Σ P(b_i=1|y,k_map) = {measured:.6f}, expected {expected}",
            measured=measured - expected,
            tolerance=1e-6,
        )
    )

    # 4. MAP backtrack score equals stored terminal value.
    score = 0.0
    for a, b in zip(map_boundaries[:-1], map_boundaries[1:], strict=False):
        score += float(lA0[int(a), int(b)])
        if log_g_table is not None:
            score += float(log_g_table[int(a), int(b)])
    delta_map = abs(score - log_joint_map)
    checks.append(
        DiagnosticCheck(
            name="map_backtrack_consistent",
            passed=delta_map < 1e-6 or delta_map < 1e-9 * max(1.0, abs(log_joint_map)),
            detail=f"|backtrack − stored| = {delta_map:.3e}",
            measured=delta_map,
            tolerance=1e-6,
        )
    )

    return DiagnosticReport(
        estimator_class=klass,
        n=n,
        k_max=k_max,
        k_map=k_map,
        checks=checks,
    )


# -----------------------------------------------------------------------------
# Non-conjugate block-error / posterior-sensitivity diagnostics (§4.8)
# -----------------------------------------------------------------------------


def run_non_conjugate_diagnostics(
    estimator: Any,
    reference: Any,
    *,
    quantiles: tuple[float, ...] = (0.5, 0.95, 1.0),
) -> DiagnosticReport:
    """Block-error and posterior-sensitivity diagnostics for non-conjugate fits.

    Compares ``estimator.log_block_evidence_`` to a reference (typically a
    high-accuracy quadrature fit on the same data) over the **reachable**
    blocks (those usable by some k-segmentation with ``k ≤ k_max``).

    Returns a report with:

    - ``block_error_max`` / ``block_error_q95`` / ``block_error_median``:
      estimate of the uniform error ``ε`` named in
      Assumption ``ass:uniform-block-error`` and bounded routine-by-routine
      in Proposition ``prop:uniform-bounds``;
    - ``k_posterior_l1`` between the two fits;
    - ``boundary_marginal_l1`` at the chosen ``k_map``;
    - ``map_path_overlap`` (Jaccard of map_boundaries_);
    - ``pk_tv_upper_bound`` = ``min(1, exp(2 k_max ε_max) − 1)``, the worst-case
      total-variation bound on ``P(k | y)`` from
      Corollary ``cor:probability-error-conversion`` (derivable directly
      from Proposition ``prop:stability``); the bound is conservative
      since it uses the worst-case ``k_max`` in place of the per-state ``k``;
    - ``pk_tv_empirical`` = ``0.5 · |P̂(k|y) − P_ref(k|y)|_1``, the
      empirical TV distance against the reference posterior. The
      ``pk_tv_bound_check`` passes when the empirical TV is dominated by
      the bound, modulo a small numerical slack.

    Each check carries an explicit ``failure_mode`` tag aligned with the
    §4 approximation-validation checklist and the §5b limitations on the
    non-conjugate approximation regime: block-error checks target
    short-segment Laplace inaccuracy and MF-VB variance underestimation;
    the boundary-marginal and MAP-path checks target method sensitivity
    and EP non-convergence (the canonical failure mode flagged by the EP
    row of Table ``nonconj_tradeoff`` and by
    Proposition ``prop:uniform-bounds`` part (v)).
    """

    klass = type(estimator).__name__
    n = int(estimator.n_)
    k_map = int(estimator.k_map_)
    k_max = int(estimator.log_left_.shape[0] - 1)

    lA0_a = np.asarray(estimator.log_block_evidence_, dtype=float)
    lA0_r = np.asarray(reference.log_block_evidence_, dtype=float)
    if lA0_a.shape != lA0_r.shape:
        raise ValueError("Estimator and reference must share log_block_evidence_ shape.")

    reachable = _reachable_blocks_mask(n, k_max)
    finite = np.isfinite(lA0_a) & np.isfinite(lA0_r)
    mask = reachable & finite
    diffs = np.abs(lA0_a[mask] - lA0_r[mask])
    measured_q = {f"q{int(q * 100):02d}": float(np.quantile(diffs, q)) for q in quantiles}
    median = float(np.median(diffs))
    max_err = float(np.max(diffs)) if diffs.size else 0.0

    # Posterior-over-k L1 distance and worst-case TV upper bound from
    # Proposition ``prop:stability``.
    p_k_a = np.asarray(estimator.k_posterior_, dtype=float)
    p_k_r = np.asarray(reference.k_posterior_, dtype=float)
    pk_len = min(p_k_a.size, p_k_r.size)
    p_k_l1 = float(np.sum(np.abs(p_k_a[:pk_len] - p_k_r[:pk_len])))
    pk_tv_empirical = 0.5 * p_k_l1
    approx = getattr(estimator, "approx", None)
    reference_method = str(getattr(reference, "approx", None) or type(reference).__name__)
    convergence_status = "unverifiable"
    if (
        str(approx).lower().replace("-", "_") == "ep"
        and getattr(estimator, "ep_all_converged_", None) is False
    ):
        convergence_status = "failed"
    error_record = evaluate_reachable_segment_error(
        lA0_a,
        lA0_r,
        family=klass,
        reference_method=reference_method,
        k_max=k_max,
        convergence_status=convergence_status,
    )
    if error_record.max_log_score_error is None:
        pk_tv_upper_bound = None
        partition_bounds = None
    else:
        partition_bounds = propagate_partition_bounds(error_record.max_log_score_error, k_max)
        pk_tv_upper_bound = partition_bounds["tv_upper_bound"]

    # Boundary-marginal L1 at the reference's k_map (avoids comparing different k's).
    bm_a = _boundary_marginals_at_k(estimator, int(reference.k_map_))
    bm_r = np.asarray(reference.boundary_marginals_, dtype=float)
    bm_len = min(bm_a.size, bm_r.size)
    bm_l1 = float(np.sum(np.abs(bm_a[:bm_len] - bm_r[:bm_len])))

    # MAP path Jaccard (interior only).
    map_a = {int(b) for b in estimator.map_boundaries_[1:-1]}
    map_r = {int(b) for b in reference.map_boundaries_[1:-1]}
    union = map_a | map_r
    jaccard = len(map_a & map_r) / max(1, len(union))

    # TV check passes when empirical TV ≤ Corollary-bound + numerical slack.
    tv_slack = 1e-9
    tv_passed = bool(
        pk_tv_upper_bound is not None and pk_tv_empirical <= pk_tv_upper_bound + tv_slack
    )

    checks = [
        DiagnosticCheck(
            name="block_error_max",
            passed=True,  # informational; caller decides threshold
            detail=f"max |Δ log A0| over reachable blocks = {max_err:.4f}",
            measured=max_err,
            failure_mode="short-segment-laplace",
        ),
        DiagnosticCheck(
            name="block_error_quantiles",
            passed=True,
            detail=" ".join(f"{k}={v:.4f}" for k, v in measured_q.items()),
            measured=median,
            failure_mode="mf-vb-variance",
        ),
        DiagnosticCheck(
            name="k_posterior_l1",
            passed=True,
            detail=f"|P̂(k) − P_ref(k)|_1 = {p_k_l1:.4f}",
            measured=p_k_l1,
            failure_mode="method-sensitivity",
        ),
        DiagnosticCheck(
            name="boundary_marginal_l1_at_ref_k_map",
            passed=True,
            detail=f"|P̂(b|y,k) − P_ref(b|y,k)|_1 at k = {reference.k_map_} = {bm_l1:.4f}",
            measured=bm_l1,
            failure_mode="method-sensitivity",
        ),
        DiagnosticCheck(
            name="map_path_jaccard",
            passed=True,
            detail=f"Jaccard(MAP_a, MAP_r) = {jaccard:.4f}",
            measured=jaccard,
            failure_mode="ep-nonconvergence",
        ),
        DiagnosticCheck(
            name="pk_tv_bound_check",
            passed=tv_passed,
            detail=(
                f"TV(P̂(k|y), P_ref(k|y)) = {pk_tv_empirical:.4f}; "
                f"conditional cor:stability-paper bound "
                f"min(1, exp(2·k_max·ε_max) − 1) = "
                f"{pk_tv_upper_bound if pk_tv_upper_bound is not None else 'unverifiable'} "
                f"for k_max={k_max}, ε_max={max_err:.4f}"
            ),
            measured=pk_tv_empirical,
            tolerance=pk_tv_upper_bound,
            failure_mode="tv-bound",
        ),
    ]

    theoretical_rate = "not established; routine-specific certification remains required"
    rate_violated = None

    extra = {
        "block_error_quantiles": measured_q,
        "block_error_max": max_err,
        "block_error_median": median,
        "n_reachable_blocks": int(np.sum(mask)),
        "k_posterior_l1": p_k_l1,
        "boundary_marginal_l1": bm_l1,
        "map_path_jaccard": jaccard,
        "k_map_estimator": k_map,
        "k_map_reference": int(reference.k_map_),
        "pk_tv_empirical": pk_tv_empirical,
        "pk_tv_upper_bound": pk_tv_upper_bound,
        "approx_routine": approx,
        "theoretical_rate": theoretical_rate,
        "theoretical_rate_violated": rate_violated,
        "segment_error_record": error_record.to_dict(),
        "conditional_partition_bounds": partition_bounds,
    }

    return DiagnosticReport(
        estimator_class=klass,
        n=n,
        k_max=k_max,
        k_map=k_map,
        checks=checks,
        extra=extra,
    )


# -----------------------------------------------------------------------------
# Prior-sensitivity diagnostic (§6 paragraph 6-C1)
# -----------------------------------------------------------------------------


def run_prior_sensitivity(
    estimator: Any,
    *,
    pk_perturbations: tuple[Callable[[int, int], float], ...] | None = None,
    g_variants: tuple[str, ...] = ("uniform", "length-proportional"),
) -> DiagnosticReport:
    r"""Record variation of ``P(k|y)`` and the fixed-``k_map`` boundary marginals
    under perturbations of the segment-count prior ``p(k)`` and the length
    factor ``g``. Planned-diagnostic from §6 (6-C1): used alongside the
    already-planned ablations over ``p(k)`` and ``g``.

    The block evidences and the cumulative sufficient statistics are
    independent of the prior, so the perturbed posteriors are evaluated by
    rerunning the DP on the existing ``log_block_evidence_`` array with the
    perturbed ``log_p_k`` and/or ``log_g_table`` rather than refitting.

    Parameters
    ----------
    estimator : fitted segmenter
        Provides ``log_block_evidence_``, ``log_left_``, ``log_right_``,
        ``log_g_table_``, ``log_C_k_``, ``k_posterior_``,
        ``boundary_marginals_``, ``k_map_``, ``n_``.
    pk_perturbations : tuple of callables (k_max, k) -> log_p_k, optional
        Each callable returns a ``log_p_k`` vector of shape ``(k_max + 1,)``.
        Defaults to three perturbations: uniform on ``{1,…,k_max}``, a mild
        geometric ``p(k) ∝ 0.8^k``, and ``p(k) ∝ k^{-1}``.
    g_variants : tuple of str
        Length-factor variants to evaluate. ``"uniform"`` is the index-uniform
        prior (``g ≡ 1``); ``"length-proportional"`` is ``g(ℓ) ∝ ℓ`` derived
        from the boundary coordinates of the fitted estimator.

    Returns
    -------
    DiagnosticReport
        ``extra`` records per-variant ``Δ p(k|y)`` (max absolute, total
        variation) and ``Δ P(b_i=1|y, k_map)`` (max absolute, L1), with one
        :class:`DiagnosticCheck` per variant carrying the
        ``"prior-sensitivity"`` failure mode tag.
    """

    n = int(estimator.n_)
    k_map = int(estimator.k_map_)
    k_max = int(estimator.log_left_.shape[0] - 1)
    klass = type(estimator).__name__

    lA0 = np.asarray(estimator.log_block_evidence_, dtype=float)
    raw_log_g = getattr(estimator, "log_g_table_", None)
    if raw_log_g is None:
        log_g_base = np.zeros((n + 1, n + 1), dtype=float)
    else:
        log_g_base = np.asarray(raw_log_g, dtype=float)
    pk_base = np.asarray(estimator.k_posterior_, dtype=float)
    bm_base = np.asarray(estimator.boundary_marginals_, dtype=float)
    log_p_k_base = _log_p_k_from_estimator(estimator, k_max)

    def _eval(
        log_p_k: FloatArray, log_g_table: FloatArray, tag: str
    ) -> tuple[DiagnosticCheck, dict[str, Any]]:
        L, R = _dp.forward_backward(lA0, n, k_max, log_g_table=log_g_table)
        log_C_k = _dp.compute_log_C_k(log_g_table, n, k_max)
        _, post_k, _ = _dp.posterior_over_k(L, n, k_max, log_C_k=log_C_k, log_p_k=log_p_k)
        bm = _dp.boundary_event_marginals_fixed_k(L, R, n, k_map)

        d_pk = post_k - pk_base
        d_pk_max = float(np.max(np.abs(d_pk)))
        d_pk_tv = 0.5 * float(np.sum(np.abs(d_pk)))
        d_bm = bm - bm_base
        d_bm_max = float(np.max(np.abs(d_bm))) if d_bm.size else 0.0
        d_bm_l1 = float(np.sum(np.abs(d_bm)))

        summary = {
            "variant": tag,
            "delta_pk_max": d_pk_max,
            "delta_pk_tv": d_pk_tv,
            "delta_bm_max": d_bm_max,
            "delta_bm_l1": d_bm_l1,
        }
        check = DiagnosticCheck(
            name=f"prior_sensitivity[{tag}]",
            passed=True,  # informational; report variation magnitudes
            detail=(
                f"max|Δ p(k|y)| = {d_pk_max:.4f}, TV = {d_pk_tv:.4f}; "
                f"max|Δ P(b_i|y,k_map)| = {d_bm_max:.4f}, L1 = {d_bm_l1:.4f}"
            ),
            measured=d_pk_tv,
            failure_mode="prior-sensitivity",
        )
        return check, summary

    if pk_perturbations is None:

        def _uniform(km: int, _k: int) -> float:
            return -math.log(float(km))

        def _geom(_km: int, k: int) -> float:
            return k * math.log(0.8)

        def _inv(_km: int, k: int) -> float:
            return -math.log(float(k))

        pk_perturbations = (_uniform, _geom, _inv)

    checks: list[DiagnosticCheck] = []
    per_variant: list[dict[str, Any]] = []

    # p(k) perturbations under the fitted length factor.
    for fn in pk_perturbations:
        log_p_k = np.full(k_max + 1, -np.inf, dtype=float)
        for k in range(1, k_max + 1):
            log_p_k[k] = float(fn(k_max, k))
        # Normalize.
        denom = float(_logsumexp(log_p_k[1:])) if k_max >= 1 else 0.0
        log_p_k[1:] -= denom
        tag = f"pk={getattr(fn, '__name__', 'fn')}"
        check, summary = _eval(log_p_k, log_g_base, tag)
        checks.append(check)
        per_variant.append(summary)

    # g variants under the fitted p(k).
    for g_tag in g_variants:
        if g_tag == "uniform":
            log_g = np.zeros((n + 1, n + 1), dtype=float)
        elif g_tag == "length-proportional":
            u = np.asarray(estimator.boundary_coordinates_, dtype=float)
            d = u[None, :] - u[:, None]
            with np.errstate(divide="ignore", invalid="ignore"):
                log_g = np.where(d > 0, np.log(np.where(d > 0, d, 1.0)), -np.inf)
        else:
            raise ValueError(f"Unknown g variant: {g_tag!r}")
        tag = f"g={g_tag}"
        check, summary = _eval(log_p_k_base, log_g, tag)
        checks.append(check)
        per_variant.append(summary)

    extra = {
        "variants": per_variant,
        "k_map": k_map,
        "n": n,
        "k_max": k_max,
    }
    return DiagnosticReport(
        estimator_class=klass,
        n=n,
        k_max=k_max,
        k_map=k_map,
        checks=checks,
        extra=extra,
    )


# -----------------------------------------------------------------------------
# Held-out G selection for the latent-template mixture (§5b 'Identifiability
# failures'; mitigates rem:teicher-overspec)
# -----------------------------------------------------------------------------


def select_n_groups_by_holdout(
    base_estimator: Any,
    sequences: Any,
    *,
    g_grid: tuple[int, ...] = (1, 2, 3, 4, 5),
    n_folds: int = 5,
    random_state: int = 0,
    n_restarts: int = 3,
    max_iter: int = 30,
    **mixture_kwargs: Any,
) -> DiagnosticReport:
    r"""Held-out G selection for ``BayesBreakMixtureClassifier``.

    Implements the §5b "Identifiability failures (named)" mitigation: the
    saturated-``G`` identifiability of ``prop:latent-identifiability`` does
    not prevent the overspecified-``G`` redundancy of
    ``rem:teicher-overspec``, where two distinct ``(π, τ)`` configurations
    induce identical mixture densities at ``G > G^*``. The recommended
    response is to choose ``G`` by held-out predictive log-likelihood per
    ``def:metric-loglik``.

    K-fold splits operate over the **sequence** axis (each fold holds out
    a stratified sample of sequences). For each candidate ``G`` and each
    fold:

    1. Fit ``BayesBreakMixtureClassifier(base_estimator, n_groups=G, ...)``
       on the train sequences.
    2. Compute per-sequence marginal log-likelihood
       ``log p(y^{(s)})`` on the held-out sequences via
       :meth:`BayesBreakMixtureClassifier.sequence_log_likelihood`.
    3. Average over held-out sequences and across folds.

    Parameters
    ----------
    base_estimator : BayesBreakSegmenter
        The block family used inside the mixture.
    sequences : list of 1-D arrays or 2-D array of shape (S, n)
        Subject sequences. Must all share the same length ``n``.
    g_grid : tuple of int
        Candidate group counts to evaluate.
    n_folds : int
        Number of K-fold splits. Capped at ``S``.
    random_state : int
        Seed for the fold-shuffle RNG (deterministic across calls).
    n_restarts, max_iter : int
        Passed through to ``BayesBreakMixtureClassifier``; the defaults are
        slightly larger than the class defaults to make CV scores more
        stable.
    **mixture_kwargs
        Forwarded to ``BayesBreakMixtureClassifier`` (e.g. ``k_max``,
        ``length_prior``, ``prior_k``).

    Returns
    -------
    DiagnosticReport
        ``extra`` contains:

        - ``g_grid``: candidate group counts.
        - ``mean_test_loglik``: mean per-held-out-sequence
          ``log p(y)`` across folds, one entry per candidate ``G``.
        - ``std_test_loglik``: stdev across folds.
        - ``fold_logliks``: full ``(len(g_grid), n_folds)`` matrix.
        - ``best_g``: ``g_grid[argmax(mean_test_loglik)]``.
        - ``n_sequences``: number of sequences used.

        ``checks`` contains one diagnostic per ``G``, all with
        ``failure_mode="teicher-overspec"``; the check on ``best_g`` is
        the only one marked passing.
    """
    # Local import to avoid a circular dependency.
    from .mixture import BayesBreakMixtureClassifier  # noqa: PLC0415

    # Coerce sequences into a list of 1-D arrays.
    if isinstance(sequences, np.ndarray):
        if sequences.ndim == 2:
            seqs = [np.asarray(row, dtype=float) for row in sequences]
        elif sequences.ndim == 1:
            seqs = [np.asarray(sequences, dtype=float)]
        else:
            raise ValueError(f"sequences must be 1-D or 2-D; got ndim={sequences.ndim}")
    else:
        seqs = [np.asarray(s, dtype=float).ravel() for s in sequences]
    S = len(seqs)
    if S < 2:
        raise ValueError(f"Need at least 2 sequences for G-selection; got {S}.")
    n = int(seqs[0].shape[0])
    if any(s.shape[0] != n for s in seqs):
        raise ValueError("All sequences must share the same length.")
    nf = min(int(n_folds), S)
    if nf < 2:
        raise ValueError("n_folds clamped below 2; supply more sequences.")

    rng = np.random.default_rng(int(random_state))
    perm = rng.permutation(S)
    folds = np.array_split(perm, nf)

    fold_logliks = np.full((len(g_grid), nf), np.nan, dtype=float)

    for g_idx, G in enumerate(g_grid):
        for f_idx, test_idx in enumerate(folds):
            test_set = {int(i) for i in test_idx}
            train = [seqs[i] for i in range(S) if i not in test_set]
            test = [seqs[i] for i in range(S) if i in test_set]
            if len(train) < max(1, int(G)) or not test:
                continue
            try:
                est = BayesBreakMixtureClassifier(
                    base_estimator,
                    n_groups=int(G),
                    n_restarts=int(n_restarts),
                    max_iter=int(max_iter),
                    random_state=int(random_state) + f_idx,
                    **mixture_kwargs,
                ).fit(train)
                log_lik = est.sequence_log_likelihood(test)
                fold_logliks[g_idx, f_idx] = float(np.mean(log_lik))
            except Exception as exc:  # pragma: no cover - rare bad-fold failure
                fold_logliks[g_idx, f_idx] = float("-inf")
                _ = exc  # surface via the report if needed

    mean_test_loglik = np.nanmean(fold_logliks, axis=1)
    std_test_loglik = np.nanstd(fold_logliks, axis=1)
    best_idx = int(np.argmax(np.where(np.isfinite(mean_test_loglik), mean_test_loglik, -np.inf)))
    best_g = int(g_grid[best_idx])

    checks: list[DiagnosticCheck] = []
    for g_idx, G in enumerate(g_grid):
        m = float(mean_test_loglik[g_idx])
        s = float(std_test_loglik[g_idx])
        checks.append(
            DiagnosticCheck(
                name=f"holdout_loglik_G={int(G)}",
                passed=(int(G) == best_g),
                detail=(f"mean held-out log p(y) = {m:.4f} (±{s:.4f} across {nf} folds)"),
                measured=m,
                failure_mode="teicher-overspec",
            )
        )

    extra = {
        "g_grid": [int(g) for g in g_grid],
        "mean_test_loglik": mean_test_loglik.tolist(),
        "std_test_loglik": std_test_loglik.tolist(),
        "fold_logliks": fold_logliks.tolist(),
        "best_g": best_g,
        "n_sequences": int(S),
        "n_folds": nf,
        "n": int(n),
    }
    return DiagnosticReport(
        estimator_class="BayesBreakMixtureClassifier",
        n=int(n),
        k_max=int(getattr(base_estimator, "k_max", 0)),
        k_map=None,
        checks=checks,
        extra=extra,
    )


def _log_p_k_from_estimator(estimator: Any, k_max: int) -> FloatArray:
    """Recover ``log p(k)`` from the fitted estimator, normalized over k=1..k_max."""

    log_p_k = np.full(k_max + 1, -np.inf, dtype=float)
    fn = getattr(estimator, "prior_k", None)
    if fn is None:
        log_p_k[1:] = -math.log(float(k_max))
        return log_p_k
    for k in range(1, k_max + 1):
        v = float(fn(k))
        log_p_k[k] = math.log(v) if v > 0 else -np.inf
    denom = float(_logsumexp(log_p_k[1:]))
    log_p_k[1:] -= denom
    return log_p_k


def _reachable_blocks_mask(n: int, k_max: int) -> NDArray[np.bool_]:
    """Mask of blocks ``(i, j]`` reachable by some k-segmentation with k ≤ k_max.

    A block ``(i, j]`` is reachable iff there is at least one ``k ∈ [1, k_max]``
    such that ``i`` is reachable as the ``(p−1)``th boundary of a length-``k``
    segmentation of ``[0, n]``, i.e. ``p − 1 ≤ i ≤ n − (k − p + 1)`` for some
    ``p ∈ {1, ..., k}``. Equivalently, ``(i, j]`` is reachable if there exist
    integers ``a ≥ 0`` and ``b ≥ 0`` with ``a + 1 + b ≤ k_max``,
    ``a ≤ i`` and ``b ≤ n − j``.

    The simplest reachability for any ``k ≥ 1`` reduces to: ``i ≥ 0``,
    ``j ≤ n``, ``j > i``, and ``i + (n − j) + 1 ≤ k_max`` (room for the head
    and tail segments plus this one). Below ``k_max = n`` every ``i < j``
    block is reachable.
    """

    mask = np.zeros((n + 1, n + 1), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n + 1):
            if i + (n - j) + 1 <= k_max:
                mask[i, j] = True
    return mask


def _boundary_marginals_at_k(estimator: Any, k: int) -> FloatArray:
    """Return P(b_i = 1 | y, k) computed from a fitted estimator's stored DP tables."""

    if k <= 1:
        return np.zeros(estimator.n_ - 1, dtype=float)
    log_left = np.asarray(estimator.log_left_, dtype=float)
    log_right = np.asarray(estimator.log_right_, dtype=float)
    n = int(estimator.n_)
    return _dp.boundary_event_marginals_fixed_k(log_left, log_right, n, k)
