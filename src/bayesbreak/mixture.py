"""bayesbreak.mixture

Latent-group pooling (mixture model) for BayesBreak.

The paper's latent-group setting treats each observed sequence as arising from one of
G latent groups, where each group induces its own segmentation structure. In practice,
we implement an EM-like coordinate ascent scheme:

1) **M-step (group update):**
   - Update mixture weights ``pi_g`` from responsibilities.
   - For each group, compute responsibility-weighted pooled block evidences
     ``log A^0_{ij,g}`` and pooled first-moment stats ``A^1_{ij,g}``.
   - Run the BayesBreak DP on the pooled evidences to obtain group posterior boundary
     probabilities and group MAP boundaries.

2) **E-step (responsibility update):**
   - Score each sequence under each group using an evidence-weighted compatibility
     score derived from the group's segmentation posterior.
   - Convert scores into responsibilities with a softmax.

This approach is designed to be:
  - **Family-agnostic**: it works with any :class:`~bayesbreak.base.BayesBreakBase`
    family that can compute per-segment evidences.
  - **Sklearn-friendly**: ``fit/predict/predict_proba/score`` are provided.

Assumptions
-----------
* All sequences must share the same length ``n`` (and thus the same indexing grid).
  This matches the multi-sequence setups in the paper.
* The base estimator is assumed to be *scalar-output* (``y`` is 1D). For multivariate
  outputs, wrap the base estimator with :class:`~bayesbreak.multivariate.BayesBreakMultivariate`.

Notes
-----
The mixture objective used in ``score`` is a practical lower bound-like objective.
It is not claimed to be the exact observed-data marginal likelihood of a fully
specified generative mixture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_random_state

from .base import BayesBreakBase
from .families import (
    BayesBreakBernoulli,
    BayesBreakBeta,
    BayesBreakBetaObs,
    BayesBreakBinomial,
    BayesBreakGaussian,
    BayesBreakLogisticNormal,
    BayesBreakPoisson,
)
from .groups import BayesBreakGrouped
from .utils import log_binom, logsumexp


ArrayLike1D = Union[np.ndarray, Sequence[float]]
SequenceInput = Union[np.ndarray, Sequence[np.ndarray]]


@dataclass
class _GroupState:
    """Internal container for per-group fitted quantities."""

    hyper: Dict[str, float]
    lA0: np.ndarray
    A1: np.ndarray
    L: np.ndarray
    R: np.ndarray
    logC: np.ndarray
    C: np.ndarray
    log_evidence: float
    k_ml: int
    boundary_post: np.ndarray
    boundaries: List[int]
    brc: Optional[np.ndarray]
    # Segment posterior probabilities for the fixed-k model used in E-step scoring.
    # Shape (n+1, n+1) on the upper triangle (i<j); zeros elsewhere.
    seg_post: np.ndarray


def _as_list_of_1d_arrays(X: SequenceInput, *, name: str) -> List[np.ndarray]:
    """Coerce input into a list of 1D float arrays."""
    if isinstance(X, np.ndarray) and X.ndim == 2:
        return [np.asarray(row, dtype=float) for row in X]
    if isinstance(X, np.ndarray) and X.ndim == 1:
        return [np.asarray(X, dtype=float)]
    if isinstance(X, (list, tuple)):
        out: List[np.ndarray] = []
        for i, arr in enumerate(X):
            a = np.asarray(arr, dtype=float)
            if a.ndim != 1:
                raise ValueError(f"{name}[{i}] must be 1D, got shape {a.shape}.")
            out.append(a)
        if not out:
            raise ValueError(f"{name} must contain at least one sequence.")
        return out
    raise TypeError(f"Unsupported type for {name}: {type(X)!r}.")


def _as_list_of_weight_arrays(
    sample_weight: Optional[SequenceInput], *, n_seq: int, n: int
) -> List[Optional[np.ndarray]]:
    """Coerce sample_weight into a list of per-sequence 1D arrays (or None)."""
    if sample_weight is None:
        return [None] * n_seq
    if isinstance(sample_weight, np.ndarray) and sample_weight.ndim == 2:
        if sample_weight.shape != (n_seq, n):
            raise ValueError(
                f"sample_weight must have shape ({n_seq}, {n}), got {sample_weight.shape}."
            )
        return [np.asarray(row, dtype=float) for row in sample_weight]
    if isinstance(sample_weight, np.ndarray) and sample_weight.ndim == 1:
        if n_seq != 1:
            raise ValueError("1D sample_weight is only valid when a single sequence is provided.")
        if sample_weight.shape[0] != n:
            raise ValueError(f"sample_weight length must be {n}, got {sample_weight.shape[0]}.")
        return [np.asarray(sample_weight, dtype=float)]
    if isinstance(sample_weight, (list, tuple)):
        if len(sample_weight) != n_seq:
            raise ValueError(
                f"sample_weight must have length {n_seq}, got {len(sample_weight)}."
            )
        out: List[Optional[np.ndarray]] = []
        for i, w in enumerate(sample_weight):
            if w is None:
                out.append(None)
                continue
            ww = np.asarray(w, dtype=float)
            if ww.ndim != 1 or ww.shape[0] != n:
                raise ValueError(
                    f"sample_weight[{i}] must be shape ({n},), got {ww.shape}."
                )
            out.append(ww)
        return out
    raise TypeError(f"Unsupported type for sample_weight: {type(sample_weight)!r}.")


def _pool_flattened(
    ys: List[np.ndarray],
    ws: List[Optional[np.ndarray]],
    r: np.ndarray,
    g: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Flatten sequences for responsibility-weighted hyper estimation.

    Returns
    -------
    y_flat : ndarray, shape (S*n,)
    w_flat : ndarray or None, shape (S*n,)
        Observation weights multiplied by responsibilities r[:, g].
    """
    y_flat = np.concatenate(ys, axis=0)
    if all(w is None for w in ws):
        w_flat = None
    else:
        w_parts: List[np.ndarray] = []
        for s, y in enumerate(ys):
            rs = float(r[s, g])
            if ws[s] is None:
                w_parts.append(np.full_like(y, rs, dtype=float))
            else:
                w_parts.append(rs * np.asarray(ws[s], dtype=float))
        w_flat = np.concatenate(w_parts, axis=0)
    return y_flat, w_flat


def _pool_hyper_by_family(
    template: BayesBreakBase,
    ys: Sequence[np.ndarray],
    ws: Sequence[Optional[np.ndarray]],
) -> Dict[str, float]:
    """Estimate hyperparameters from multiple sequences.

    For some families, naive concatenation can produce incorrect estimates
    (notably for the Gaussian sigma^2 estimator that uses first differences).
    Where available, we reuse the same pooling rules as
    :class:`bayesbreak.groups.BayesBreakGrouped`.
    """

    # Gaussian: avoid concatenation because the sigma2 estimator depends on
    # within-sequence differences.
    if isinstance(template, BayesBreakGaussian):
        return BayesBreakGrouped._pool_gaussian_hyper(template, list(ys), list(ws))

    # Poisson: pool alpha/beta via the group's moment heuristics.
    if isinstance(template, BayesBreakPoisson):
        return BayesBreakGrouped._pool_poisson_hyper(template, list(ys), list(ws))

    # Binomial / Bernoulli: pool Beta prior parameters. Requires scalar n_trials.
    if isinstance(template, (BayesBreakBinomial, BayesBreakBernoulli)):
        n_trials = getattr(template, "n_trials", 1.0)
        if not np.isscalar(n_trials):
            raise ValueError(
                "BayesBreakMixture hyper pooling for Binomial/Bernoulli currently "
                "requires scalar n_trials."
            )
        return BayesBreakGrouped._pool_beta_binom_hyper(
            template, list(ys), list(ws), float(n_trials)
        )

    # Beta surrogate (fractional) is treated as Beta prior pooling.
    if isinstance(template, BayesBreakBeta):
        return BayesBreakGrouped._pool_beta_hyper(template, list(ys), list(ws))

    # BetaObs and LogisticNormal do not rely on within-sequence differencing; we
    # can safely pool by concatenation.
    if isinstance(template, (BayesBreakBetaObs, BayesBreakLogisticNormal)):
        y_flat = np.concatenate([np.asarray(y, dtype=float) for y in ys], axis=0)
        if all(w is None for w in ws):
            w_flat = None
        else:
            w_flat = np.concatenate(
                [
                    np.ones_like(y, dtype=float) if w is None else np.asarray(w, dtype=float)
                    for y, w in zip(ys, ws)
                ],
                axis=0,
            )
        return template._estimate_global_params(y_flat, sample_weight=w_flat)

    # Generic fallback.
    y_flat = np.concatenate([np.asarray(y, dtype=float) for y in ys], axis=0)
    if all(w is None for w in ws):
        w_flat = None
    else:
        w_flat = np.concatenate(
            [
                np.ones_like(y, dtype=float) if w is None else np.asarray(w, dtype=float)
                for y, w in zip(ys, ws)
            ],
            axis=0,
        )
    return template._estimate_global_params(y_flat, sample_weight=w_flat)


def _safe_weighted_sum_lA0(
    mats: List[np.ndarray], weights: np.ndarray
) -> np.ndarray:
    """Responsibility-weighted sum of log-evidence matrices.

    Avoids ``0 * (-inf) -> nan`` by masking finite entries.
    """
    if not mats:
        raise ValueError("mats must be non-empty")
    out = np.zeros_like(mats[0])
    mask = np.isfinite(mats[0])
    for m, w in zip(mats, weights):
        if w == 0.0:
            continue
        out[mask] += w * m[mask]
    out[~mask] = -np.inf
    return out


def _safe_weighted_sum_A1(mats: List[np.ndarray], weights: np.ndarray) -> np.ndarray:
    if not mats:
        raise ValueError("mats must be non-empty")
    out = np.zeros_like(mats[0])
    for m, w in zip(mats, weights):
        if w == 0.0:
            continue
        out += w * m
    return out


def _run_dp_from_stats(
    lA0: np.ndarray,
    A1: np.ndarray,
    *,
    k_max: int,
    regression_curve: str,
    prior_k: Literal["uniform", "geometric"] = "uniform",
    geom_p: float = 0.5,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    int,
    np.ndarray,
    List[int],
    Optional[np.ndarray],
]:
    """Run the BayesBreak DP using precomputed segment statistics."""
    n = lA0.shape[0] - 1
    kk = min(max(1, n), int(k_max))

    L, R = BayesBreakBase._compute_left_right_recursions(lA0, n, kk)

    # Posterior over k with an optional prior.
    #
    # IMPORTANT: For a fixed k, BayesBreak treats changepoint *sets* uniformly.
    # That induces a combinatorial normalizer C_k = binom(n-1, k-1) for the sum
    # over ordered partitions. Consequently
    #     log p(y|k) = log(sum over partitions) - log C_k
    # which we implement exactly as in BayesBreakBase._posterior_over_k.
    logC_raw = np.full(kk + 1, -np.inf, dtype=float)
    for k in range(1, kk + 1):
        logC_raw[k] = L[k, n] - log_binom(n - 1, k - 1)

    if prior_k == "geometric":
        p = float(geom_p)
        if not (0.0 < p < 1.0):
            raise ValueError("geom_p must be in (0, 1) for prior_k='geometric'.")
        logC_raw[1 : kk + 1] = logC_raw[1 : kk + 1] + (np.arange(1, kk + 1) - 1) * np.log(p)
    elif prior_k != "uniform":
        raise ValueError("prior_k must be one of {'uniform','geometric'}."
                         f" Got {prior_k!r}.")

    logE = float(logsumexp(logC_raw[1 : kk + 1]))
    logC = logC_raw.copy()
    logC[1 : kk + 1] = logC[1 : kk + 1] - logE
    C = np.zeros_like(logC)
    C[1 : kk + 1] = np.exp(logC[1 : kk + 1])

    # k selection around E[k] (mirrors BayesBreakBase.fit)
    ek = float(np.sum((np.arange(1, kk + 1)) * C[1 : kk + 1]))
    valid = np.arange(1, kk + 1)
    k_ml = int(valid[np.argmin((valid - ek) ** 2)])

    d1 = BayesBreakBase._boundary_posteriors_marginal(L, R, logC, n, kk)
    boundaries = BayesBreakBase._select_boundaries_from_scores(d1, k_ml, n)

    brc: Optional[np.ndarray] = None
    if regression_curve == "fixed_k":
        brc = BayesBreakBase._bayes_regression_curve_fixed_k(L, R, A1, n, k_ml)
    elif regression_curve == "mix_k":
        brc = BayesBreakBase._bayes_regression_curve_mixed_k(L, R, A1, n, kk, C)
    return L, R, logC, C, float(logE), k_ml, d1, boundaries, brc


def _segment_posterior_fixed_k(
    *,
    L: np.ndarray,
    R: np.ndarray,
    lA0: np.ndarray,
    n: int,
    k: int,
) -> np.ndarray:
    """Posterior probability that (i,j] is a segment under fixed k.

    Returns a matrix ``P`` of shape (n+1, n+1) where only the strict upper triangle
    (i<j) is non-zero.
    """
    denom = L[k, n]
    P = np.zeros((n + 1, n + 1), dtype=float)
    for i in range(0, n):
        Li = L[0:k, i]
        for j in range(i + 1, n + 1):
            Rj = R[k - 1 :: -1, j]
            # log sum_{p=0..k-1} L[p,i] + R[k-1-p,j]
            log_pref_suf = logsumexp(Li + Rj)
            log_p = log_pref_suf + lA0[i, j] - denom
            if np.isfinite(log_p):
                P[i, j] = float(np.exp(log_p))
    return P


class BayesBreakMixture(BaseEstimator, ClassifierMixin):
    """Latent-group BayesBreak via an EM-like procedure.

    Parameters
    ----------
    base_estimator:
        A fitted-family BayesBreak estimator (e.g., :class:`~bayesbreak.families.BayesBreakGaussian`).
        The estimator is cloned internally.
    n_groups:
        Number of latent groups.
    k_max:
        Maximum number of segments used in each DP run.
    max_iter:
        Maximum number of EM iterations.
    tol:
        Convergence tolerance on the relative change of the objective.
    regression_curve:
        Whether to compute a Bayesian regression curve per group.
        One of {"none", "fixed_k", "mix_k"}.
    prior_k:
        Prior on the number of segments *k* used when selecting each group's
        MAP segment count. "uniform" reproduces the base BayesBreak selection;
        "geometric" applies an exponentially decaying prior p(k) ∝ geom_p^(k-1).
    geom_p:
        Geometric prior parameter in (0, 1). Smaller values penalize larger k more.
    random_state:
        Random seed for responsibility initialisation.
    verbose:
        If True, store the objective trajectory and print progress.

    Attributes
    ----------
    responsibilities_:
        Array of shape (S, G) with final responsibilities.
    pi_:
        Mixture weights of shape (G,).
    group_states_:
        List of per-group fitted states.
    objective_:
        List of objective values per iteration.
    """

    def __init__(
        self,
        base_estimator: BayesBreakBase,
        n_groups: int = 2,
        k_max: int = 50,
        max_iter: int = 50,
        tol: float = 1e-4,
        regression_curve: str = "none",
        prior_k: str = "uniform",
        geom_p: float = 0.5,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        self.base_estimator = base_estimator
        self.n_groups = int(n_groups)
        self.k_max = int(k_max)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.regression_curve = regression_curve
        self.prior_k = str(prior_k)
        self.geom_p = float(geom_p)
        self.random_state = random_state
        self.verbose = bool(verbose)

        # fitted
        self.responsibilities_: Optional[np.ndarray] = None
        self.pi_: Optional[np.ndarray] = None
        self.group_states_: Optional[List[_GroupState]] = None
        self.objective_: Optional[List[float]] = None
        self.n_: Optional[int] = None
        self.n_seq_: Optional[int] = None

    # ----------------------- sklearn API -----------------------

    def fit(
        self,
        X: SequenceInput,
        y: Optional[SequenceInput] = None,
        sample_weight: Optional[SequenceInput] = None,
    ) -> "BayesBreakMixture":
        # sklearn convention: if y is None, treat X as the observed sequences.
        Y_in = X if y is None else y
        ys = _as_list_of_1d_arrays(Y_in, name="y")
        S = len(ys)
        n = int(ys[0].shape[0])
        if any(int(arr.shape[0]) != n for arr in ys):
            raise ValueError("All sequences must have the same length for BayesBreakMixture.")
        ws = _as_list_of_weight_arrays(sample_weight, n_seq=S, n=n)

        if self.n_groups < 1:
            raise ValueError("n_groups must be >= 1.")
        G = self.n_groups

        rng = check_random_state(self.random_state)
        # Responsibilities initialisation: random Dirichlet with mild symmetry breaking.
        # Prefer a *hard* random initialization (with a small amount of
        # smoothing) to avoid symmetry-induced label collapse.
        init_labels = rng.randint(0, G, size=S)
        # Ensure every group is represented at least once.
        if S >= G:
            init_labels[:G] = np.arange(G)
            rng.shuffle(init_labels)
        r = np.eye(G, dtype=float)[init_labels]
        r = 0.999 * r + 0.001 / G

        pi = np.full(G, 1.0 / G, dtype=float)

        objective: List[float] = []
        group_states: List[_GroupState] = []

        prev_obj = -np.inf
        for it in range(self.max_iter):
            group_states = []
            # Cache per-sequence block evidences under each group's current hyper
            # (used in the E-step scoring).
            lA0_cache: List[List[np.ndarray]] = [[None for _ in range(S)] for _ in range(G)]
            # M-step: update group hyperparameters and pooled DPs.
            for g in range(G):
                # --- hyper (empirical Bayes on responsibility-weighted pooled observations) ---
                est_h = clone(self.base_estimator)
                # Force hyper estimation on pooled data regardless of base_estimator.estimate_hyper.
                est_h.set_params(estimate_hyper=True)

                ws_scaled: List[np.ndarray] = []
                for s in range(S):
                    w_s = ws[s]
                    if w_s is None:
                        w_s = np.ones(n, dtype=float)
                    ws_scaled.append(w_s * float(r[s, g]))

                hyper_g = _pool_hyper_by_family(est_h, ys, ws_scaled)

                # --- per-sequence segment stats under hyper_g ---
                lA0_sg: List[np.ndarray] = []
                A1_sg: List[np.ndarray] = []
                for s in range(S):
                    est_s = clone(self.base_estimator)
                    est_s.set_params(estimate_hyper=False)
                    # Best-effort: push hyper keys into estimator params when supported.
                    for key, val in hyper_g.items():
                        if key in est_s.get_params(deep=False):
                            est_s.set_params(**{key: float(val)})
                    w_s = ws[s]
                    if w_s is None:
                        w_s = np.ones(n, dtype=float)
                    lA0, A1 = est_s._compute_single_segment_stats(ys[s], hyper_g, sample_weight=w_s)
                    lA0_cache[g][s] = lA0
                    lA0_sg.append(lA0)
                    A1_sg.append(A1)

                # --- pooled stats ---
                lA0_g = _safe_weighted_sum_lA0(lA0_sg, r[:, g])
                A1_g = _safe_weighted_sum_A1(A1_sg, r[:, g])

                # --- run DP for the group ---
                L, R, logC, C, logE, k_ml, d1, boundaries, brc = _run_dp_from_stats(
                    lA0_g,
                    A1_g,
                    k_max=self.k_max,
                    regression_curve=self.regression_curve,
                    prior_k=self.prior_k,
                    geom_p=self.geom_p,
                )

                # Segment posterior under fixed-k (k_ml) for E-step scoring.
                seg_post = _segment_posterior_fixed_k(L=L, R=R, lA0=lA0_g, n=n, k=k_ml)

                group_states.append(
                    _GroupState(
                        hyper=hyper_g,
                        lA0=lA0_g,
                        A1=A1_g,
                        L=L,
                        R=R,
                        logC=logC,
                        C=C,
                        log_evidence=logE,
                        k_ml=k_ml,
                        boundary_post=d1,
                        boundaries=boundaries,
                        brc=brc,
                        seg_post=seg_post,
                    )
                )

            # Update mixture weights.
            pi = np.maximum(1e-12, r.mean(axis=0))
            pi = pi / pi.sum()

            # E-step: update responsibilities by scoring each sequence under the
            # current group *MAP* segmentation. This is more discriminative than
            # using fully-marginalised segment posteriors (which can become too
            # diffuse early in EM, leading to collapsed solutions).
            log_resp = np.zeros((S, G), dtype=float)
            for g in range(G):
                gs = group_states[g]
                bnds = gs.boundaries
                for s in range(S):
                    lA0_s = lA0_cache[g][s]
                    # Sum block evidences along the group's MAP partition.
                    score_sg = 0.0
                    for a, b in zip(bnds[:-1], bnds[1:]):
                        score_sg += float(lA0_s[a, b])
                    log_resp[s, g] = np.log(pi[g]) + score_sg

            # Normalise.
            log_norm = logsumexp(log_resp, axis=1, keepdims=True)
            r = np.exp(log_resp - log_norm)

            # Objective: sum_s log sum_g pi_g exp(score_sg)
            obj = float(np.sum(log_norm))
            objective.append(obj)

            if self.verbose:
                print(f"[BayesBreakMixture] iter={it+1:03d} obj={obj:.6f}")

            if it > 0:
                # Relative improvement.
                denom = max(1.0, abs(prev_obj))
                if abs(obj - prev_obj) / denom < self.tol:
                    break
            prev_obj = obj

        self.responsibilities_ = r
        self.pi_ = pi
        self.group_states_ = group_states
        self.objective_ = objective
        self.n_ = n
        self.n_seq_ = S
        return self

    def predict_proba(
        self,
        X: SequenceInput,
        y: Optional[SequenceInput] = None,
        sample_weight: Optional[SequenceInput] = None,
    ) -> np.ndarray:
        if self.group_states_ is None or self.pi_ is None:
            raise RuntimeError("Call fit() first.")
        Y_in = X if y is None else y
        ys = _as_list_of_1d_arrays(Y_in, name="y")
        S = len(ys)
        n = int(ys[0].shape[0])
        if self.n_ is not None and n != self.n_:
            raise ValueError(f"Expected sequences of length {self.n_}, got {n}.")
        ws = _as_list_of_weight_arrays(sample_weight, n_seq=S, n=n)

        G = len(self.group_states_)
        log_resp = np.zeros((S, G), dtype=float)
        for g in range(G):
            gs = self.group_states_[g]
            for s in range(S):
                est_s = clone(self.base_estimator)
                est_s.set_params(estimate_hyper=False)
                for key, val in gs.hyper.items():
                    if key in est_s.get_params(deep=False):
                        est_s.set_params(**{key: float(val)})
                w_s = ws[s]
                if w_s is None:
                    w_s = np.ones(n, dtype=float)
                lA0_s, _A1_s = est_s._compute_single_segment_stats(ys[s], gs.hyper, sample_weight=w_s)
                score_sg = 0.0
                for a, b in zip(gs.boundaries[:-1], gs.boundaries[1:]):
                    score_sg += float(lA0_s[a, b])
                log_resp[s, g] = np.log(self.pi_[g]) + score_sg

        log_norm = logsumexp(log_resp, axis=1, keepdims=True)
        return np.exp(log_resp - log_norm)

    def predict(
        self,
        X: SequenceInput,
        y: Optional[SequenceInput] = None,
        sample_weight: Optional[SequenceInput] = None,
    ) -> np.ndarray:
        proba = self.predict_proba(X, y=y, sample_weight=sample_weight)
        return np.argmax(proba, axis=1)

    def score(
        self,
        X: SequenceInput,
        y: Optional[SequenceInput] = None,
        sample_weight: Optional[SequenceInput] = None,
    ) -> float:
        """Return the mixture objective used during fitting (higher is better)."""
        proba = self.predict_proba(X, y=y, sample_weight=sample_weight)
        # Convert back to log normalizer by re-scoring to avoid exposing internals.
        # This is not the exact marginal likelihood; it's a practical scoring rule.
        return float(np.mean(np.log(np.maximum(1e-300, proba.max(axis=1)))))

    # -------------------- convenience helpers --------------------

    def get_group_boundaries(self, g: int) -> List[int]:
        if self.group_states_ is None:
            raise RuntimeError("Call fit() first.")
        return list(self.group_states_[g].boundaries)

    def get_group_boundary_posteriors(self, g: int) -> np.ndarray:
        if self.group_states_ is None:
            raise RuntimeError("Call fit() first.")
        return self.group_states_[g].boundary_post.copy()

    def get_group_regression_curve(self, g: int) -> Optional[np.ndarray]:
        if self.group_states_ is None:
            raise RuntimeError("Call fit() first.")
        brc = self.group_states_[g].brc
        return None if brc is None else brc.copy()

    def map_signal(
        self,
        X: SequenceInput,
        group: Optional[Union[int, Sequence[int], np.ndarray]] = None,
        *,
        mode: str = "refit",
        return_curve: str = "pc",
        sample_weight: Optional[SequenceInput] = None,
    ) -> np.ndarray:
        """Compute MAP/Bayes signal evaluations conditional on group membership.

        Parameters
        ----------
        X:
            One or more sequences.
        group:
            Group label(s). If None, uses ``predict``.
        mode:
            - ``"refit"`` (default): refit BayesBreak on each sequence with group
              hyperparameters fixed, and return the resulting reconstruction.
            - ``"template"``: use the group's MAP boundaries and compute segment
              posterior means for the new sequence (no DP on the new sequence).
        return_curve:
            ``"pc"`` for the piecewise-constant posterior mean under MAP boundaries,
            or ``"brc"`` to return the Bayesian regression curve (requires
            ``regression_curve != 'none'`` for the refit estimator).

        Returns
        -------
        ndarray
            Array of shape (S, n) (or (n,) for a single sequence).
        """
        if self.group_states_ is None:
            raise RuntimeError("Call fit() first.")
        ys = _as_list_of_1d_arrays(X, name="X")
        S = len(ys)
        n = ys[0].shape[0]
        ws = _as_list_of_weight_arrays(sample_weight, n_seq=S, n=n)

        if group is None:
            grp = self.predict(ys)
        else:
            if isinstance(group, (int, np.integer)):
                grp = np.full(S, int(group), dtype=int)
            else:
                grp = np.asarray(group, dtype=int)
                if grp.shape != (S,):
                    raise ValueError(f"group must be scalar or shape ({S},), got {grp.shape}.")

        out = np.zeros((S, n), dtype=float)
        for s in range(S):
            g = int(grp[s])
            gs = self.group_states_[g]
            if mode == "refit":
                est = clone(self.base_estimator)
                est.set_params(estimate_hyper=False)
                for key, val in gs.hyper.items():
                    if key in est.get_params(deep=False):
                        est.set_params(**{key: float(val)})
                # Preserve the mixture object's curve preference.
                if "regression_curve" in est.get_params(deep=False):
                    est.set_params(regression_curve=self.regression_curve)
                est.fit(ys[s], sample_weight=ws[s])
                if return_curve == "brc":
                    curve = est.get_regression_curve()
                    if curve is None:
                        raise ValueError(
                            "return_curve='brc' requested, but regression_curve is 'none'."
                        )
                    out[s] = curve
                else:
                    out[s] = est.predict()
            elif mode == "template":
                boundaries = gs.boundaries
                # Compute posterior mean per segment under group hyper.
                est = clone(self.base_estimator)
                est.set_params(estimate_hyper=False)
                for key, val in gs.hyper.items():
                    if key in est.get_params(deep=False):
                        est.set_params(**{key: float(val)})
                y_arr = np.asarray(ys[s], dtype=float)
                w_arr = None if ws[s] is None else np.asarray(ws[s], dtype=float)
                for a, b in zip(boundaries[:-1], boundaries[1:]):
                    mu = est._segment_posterior_mean(a, b, y_arr, gs.hyper, sample_weight=w_arr)
                    out[s, a:b] = float(mu)
            else:
                raise ValueError("mode must be one of {'refit','template'}")
        return out[0] if isinstance(X, np.ndarray) and X.ndim == 1 else out
