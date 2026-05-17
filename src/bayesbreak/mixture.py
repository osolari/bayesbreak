r"""Latent-group **template-mixture** EM (§``latent-em`` / Algorithm ``multi-em``).

The mixture model places a single boundary template :math:`\tau_g = (k_g, t^{(g)})`
behind each latent group, integrates out subject-specific segment parameters
inside per-subject block evidences, and alternates:

- **E-step** (eq. ``template-resp``): exact responsibilities for the current
  templates,

  .. math::
      r_{sg} \propto \pi_g \, S_g(y^{(s)}; \tau_g),\qquad
      S_g(y; \tau_g) = \frac{p(k_g)}{C_{k_g}} \prod_{q=1}^{k_g} \tilde A^{(0,s)}_{t^{(g)}_{q-1} t^{(g)}_q}.

- **M-step**: closed-form simplex update on :math:`\pi`, then for each group
  an exact responsibility-weighted **max-sum** segmentation update with score

  .. math::
      B^{(g)}_{ij} = n_g \log g(\Delta_x(i, j))
                    + \sum_{s=1}^S r_{sg} \log A^{(0,s)}_{ij},

  and count offset :math:`n_g (\log p(k) - \log C_k)` when selecting
  :math:`k_g`. Templates are joint MAP backtracks per group.

This is the only objective that Theorem ``em-monotone`` covers; the
**finite-template mixture objective**

.. math::
    \ell_\star(\pi, \tau) = \sum_s \log\Big(\sum_g \pi_g\, S_g(y^{(s)}; \tau_g)\Big)

is non-decreasing under the iteration. ``ℓ_⋆`` is *not* the Bayesian
observed-data marginal likelihood — it is a finite-mixture optimization
objective over template scores. There is no `sum-product` / `geometric`
legacy switch: callers wanting an alternative ``p(k)`` or design-aware
prior pass them via the standard ``prior_k`` and ``length_prior`` arguments.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_random_state

from . import dp as _dp
from .base import BayesBreakSegmenter
from .utils import logsumexp
from .validation import require_fitted

FloatArray = NDArray[np.floating]
SequenceInput = np.ndarray | Sequence[np.ndarray]


@dataclass
class _GroupState:
    """Internal per-group container."""

    hyper: dict[str, float]
    template: list[int]
    k_g: int
    log_score_offset: float  # log p(k_g) - log C_{k_g}


def _canonical_template_order(group_states: list[_GroupState]) -> list[int]:
    """Return a permutation of ``group_states`` that anchors the
    permutation indeterminacy of Proposition ``prop:latent-identifiability``.

    Per ``ex:label-switch-counterexample`` the latent-template mixture is
    only identifiable up to a permutation of the group labels. To make
    label-level reporting reproducible across restarts we adopt the
    deterministic anchoring described in §5b: order groups first by ``k_g``
    (smallest segment count first), then lexicographically on the boundary
    vector ``t^{(g)}``. Returns the index permutation that puts the input
    list into canonical order.

    The anchoring covers the saturated-``G`` (``G = G^star``) case of
    ``prop:latent-identifiability``. The overspecified-``G`` redundancy
    (``rem:teicher-overspec``), where ``G > G^star`` admits distinct
    parameter tuples with identical mixture densities, is *not* resolved
    by this anchor; the recommended mitigation in §5b is to choose ``G``
    by held-out predictive log-likelihood
    (``def:metric-loglik``).
    """
    return sorted(
        range(len(group_states)),
        key=lambda i: (group_states[i].k_g, tuple(group_states[i].template)),
    )


def _as_list_of_1d(X: SequenceInput, *, name: str) -> list[FloatArray]:
    """Coerce to a list of 1-D float arrays of identical length."""

    if isinstance(X, np.ndarray) and X.ndim == 2:
        return [np.asarray(row, dtype=float) for row in X]
    if isinstance(X, np.ndarray) and X.ndim == 1:
        return [np.asarray(X, dtype=float)]
    if isinstance(X, list | tuple):
        out = []
        for i, arr in enumerate(X):
            a = np.asarray(arr, dtype=float)
            if a.ndim != 1:
                raise ValueError(f"{name}[{i}] must be 1-D; got shape {a.shape}.")
            out.append(a)
        if not out:
            raise ValueError(f"{name} must contain at least one sequence.")
        return out
    raise TypeError(f"Unsupported type for {name}: {type(X)!r}.")


def _as_list_of_weights(
    sample_weight: SequenceInput | None, *, n_seq: int, n: int
) -> list[FloatArray | None]:
    if sample_weight is None:
        return [None] * n_seq
    if isinstance(sample_weight, np.ndarray) and sample_weight.ndim == 2:
        if sample_weight.shape != (n_seq, n):
            raise ValueError(f"sample_weight shape must be ({n_seq}, {n}).")
        return [np.asarray(row, dtype=float) for row in sample_weight]
    if isinstance(sample_weight, np.ndarray) and sample_weight.ndim == 1:
        if n_seq != 1 or sample_weight.shape[0] != n:
            raise ValueError("1-D sample_weight only valid for a single sequence.")
        return [np.asarray(sample_weight, dtype=float)]
    if isinstance(sample_weight, list | tuple):
        if len(sample_weight) != n_seq:
            raise ValueError(f"sample_weight must have length {n_seq}.")
        out: list[FloatArray | None] = []
        for i, w in enumerate(sample_weight):
            if w is None:
                out.append(None)
                continue
            ww = np.asarray(w, dtype=float)
            if ww.ndim != 1 or ww.shape[0] != n:
                raise ValueError(f"sample_weight[{i}] must be 1-D length {n}.")
            out.append(ww)
        return out
    raise TypeError(f"Unsupported type for sample_weight: {type(sample_weight)!r}.")


def _build_log_g_table(
    length_prior: Callable[[float], float] | None,
    u: FloatArray,
    n: int,
) -> FloatArray | None:
    if length_prior is None:
        return None
    log_g = np.full((n + 1, n + 1), -np.inf, dtype=float)
    for i in range(n):
        for j in range(i + 1, n + 1):
            d = float(u[j] - u[i])
            if d <= 0:
                continue
            gv = float(length_prior(d))
            if gv > 0 and np.isfinite(gv):
                log_g[i, j] = float(np.log(gv))
    return log_g


def _build_log_p_k(prior_k: Callable[[int], float] | None, k_max: int) -> FloatArray:
    if prior_k is None:
        return np.full(k_max + 1, -np.log(k_max), dtype=float)
    vals = np.array([float(prior_k(k)) for k in range(1, k_max + 1)], dtype=float)
    if np.any(vals < 0):
        raise ValueError("prior_k must be non-negative.")
    total = float(np.sum(vals))
    if total <= 0:
        raise ValueError("prior_k must put positive mass somewhere.")
    log_p = np.log(np.maximum(vals / total, 1e-300))
    full = np.full(k_max + 1, -np.inf, dtype=float)
    full[1:] = log_p
    return full


def _midpoint_u(x_design: FloatArray, n: int) -> FloatArray:
    if n == 1:
        return np.array([x_design[0] - 0.5, x_design[0] + 0.5], dtype=float)
    mids = 0.5 * (x_design[:-1] + x_design[1:])
    first_gap = x_design[1] - x_design[0]
    last_gap = x_design[-1] - x_design[-2]
    u0 = float(x_design[0] - 0.5 * first_gap)
    un = float(x_design[-1] + 0.5 * last_gap)
    return np.concatenate(([u0], mids, [un])).astype(float)


def _template_log_score(
    log_A0_s: FloatArray, template: list[int], log_g: FloatArray | None
) -> float:
    """``Σ_q log Ã^{(0,s)}_{t_{q-1} t_q}`` along the boundary vector ``template``."""

    total = 0.0
    for a, b in zip(template[:-1], template[1:], strict=False):
        v = float(log_A0_s[int(a), int(b)])
        if log_g is not None:
            v = v + float(log_g[int(a), int(b)])
        if not np.isfinite(v):
            return float("-inf")
        total += v
    return total


class BayesBreakMixtureClassifier(BaseEstimator, ClassifierMixin):
    """Latent-group BayesBreak via the template-mixture EM (Algorithm ``multi-em``).

    Parameters
    ----------
    base_estimator : BayesBreakSegmenter
        Family template; cloned per group. Subject-level block evidences use
        the group's empirical-Bayes hyperparameters.
    n_groups : int
    k_max : int
    max_iter : int
    tol : float
        Relative convergence tolerance on the finite-mixture objective.
    random_state : int or None
        Seed for responsibility / template initialisation.
    n_restarts : int
        Number of random restarts; the best (highest objective) is retained.
    length_prior : callable or None
    boundary_coordinates : array-like of shape (n+1,) or None
    prior_k : callable or None
    verbose : bool

    Attributes
    ----------
    pi_ : ndarray of shape (G,)
    responsibilities_ : ndarray of shape (S, G)
    group_states_ : list of ``_GroupState``
    objective_ : list of float
        Finite-template mixture objective ``ℓ_⋆`` trajectory of the *winning*
        restart. ``ℓ_⋆`` is the optimization objective from §``latent-em``,
        not the Bayesian observed-data marginal likelihood.
    """

    def __init__(
        self,
        base_estimator: BayesBreakSegmenter,
        n_groups: int = 2,
        k_max: int = 50,
        max_iter: int = 50,
        tol: float = 1e-5,
        random_state: int | None = None,
        n_restarts: int = 5,
        length_prior: Callable[[float], float] | None = None,
        boundary_coordinates: ArrayLike | None = None,
        prior_k: Callable[[int], float] | None = None,
        verbose: bool = False,
    ):
        self.base_estimator = base_estimator
        self.n_groups = int(n_groups)
        self.k_max = int(k_max)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state
        self.n_restarts = int(n_restarts)
        self.length_prior = length_prior
        self.boundary_coordinates = boundary_coordinates
        self.prior_k = prior_k
        self.verbose = bool(verbose)

        # fitted
        self.pi_: FloatArray | None = None
        self.responsibilities_: FloatArray | None = None
        self.group_states_: list[_GroupState] | None = None
        self.objective_: list[float] | None = None
        self.n_: int | None = None
        self.n_seq_: int | None = None

    # ---- private helpers ---------------------------------------------

    def _per_subject_log_evidence(
        self, ys: list[FloatArray], ws: list[FloatArray | None]
    ) -> tuple[list[FloatArray], list[dict[str, float]]]:
        """Compute per-subject log block-evidence tables under each subject's own EB hyper."""

        out_lA0: list[FloatArray] = []
        out_hyper: list[dict[str, float]] = []
        for s, y in enumerate(ys):
            est_s = clone(self.base_estimator)
            w = ws[s] if ws[s] is not None else np.ones(y.size, dtype=float)
            hyper = est_s._estimate_hyperparameters(y, w)
            lA0_s, _ = est_s._compute_block_evidence(y, hyper, w)
            out_lA0.append(lA0_s)
            out_hyper.append(hyper)
        return out_lA0, out_hyper

    def _maximise_template_for_group(
        self,
        n: int,
        k_max: int,
        n_g: float,
        log_A0_subjects: list[FloatArray],
        responsibilities_g: FloatArray,
        log_g_table: FloatArray | None,
        log_C_k: FloatArray,
        log_p_k: FloatArray,
    ) -> tuple[list[int], int, float, float]:
        """Run the responsibility-weighted max-sum DP over all ``k`` and return the best template.

        Returns
        -------
        template : list[int]
        k_g : int
        log_score : float
            ``M_{k_g, n} + n_g (log p(k_g) - log C_{k_g})``.
        log_offset : float
            ``log p(k_g) - log C_{k_g}`` itself (used by the E-step S_g formula).
        """

        # B^{(g)}_{ij} = n_g log g(Δ) + Σ_s r_{sg} log A^{(0,s)}_{ij}
        # We supply the responsibility-weighted log A^(0) table to max_sum
        # directly, *without* the n_g log g term, since max_sum_segmentation
        # accepts log_g_table as an additive contribution scaled by `1` only.
        # Per Algorithm ``multi-em`` the length factor enters with weight ``n_g``
        # because it is part of S_g and S_g is raised to the power r_sg per
        # subject under the auxiliary Q_g — i.e. the resulting per-block term
        # is n_g · log g(Δ). We bake that scaling into the score table here.
        S = len(log_A0_subjects)
        # responsibility-weighted subject log evidences
        # (mask -inf entries shared across all subjects).
        finite_mask = np.ones_like(log_A0_subjects[0], dtype=bool)
        for la in log_A0_subjects:
            finite_mask &= np.isfinite(la)
        B = np.zeros_like(log_A0_subjects[0])
        for s in range(S):
            r = float(responsibilities_g[s])
            if r == 0.0:
                continue
            B[finite_mask] += r * log_A0_subjects[s][finite_mask]
        if log_g_table is not None:
            # Note: max_sum adds log_g once per block; we want n_g · log g(Δ) per
            # block. Scale here.
            B[finite_mask] += float(n_g) * log_g_table[finite_mask]
        B[~finite_mask] = -np.inf

        # Solve max_k { M_k + n_g (log p(k) - log C_k) }; pass log_g_table=None
        # because the length-factor contribution is already baked into B.
        best_score = -np.inf
        best_k = 1
        best_template: list[int] = [0, n]
        for k in range(1, k_max + 1):
            try:
                template, M_k = _dp.max_sum_segmentation(B, k, log_g_table=None)
            except RuntimeError:
                continue
            offset = float(n_g) * (float(log_p_k[k]) - float(log_C_k[k]))
            score = float(M_k) + offset
            # Deterministic tie-break: prefer smaller k on ties.
            if (score > best_score + 1e-12) or (abs(score - best_score) <= 1e-12 and k < best_k):
                best_score = score
                best_k = k
                best_template = template
        log_offset = float(log_p_k[best_k]) - float(log_C_k[best_k])
        return best_template, best_k, best_score, log_offset

    def _fit_one_restart(
        self,
        rng: np.random.Generator,
        ys: list[FloatArray],
        ws: list[FloatArray | None],
        n: int,
        k_max: int,
        log_g_table: FloatArray | None,
        log_C_k: FloatArray,
        log_p_k: FloatArray,
    ) -> tuple[float, FloatArray, FloatArray, list[_GroupState], list[float], list[FloatArray]]:
        S = len(ys)
        G = int(self.n_groups)

        # Per-subject log A^(0) tables (do NOT depend on group; we estimate
        # subject-specific hyper inside subject's own EB. The paper's
        # template-mixture model integrates out per-(g,s) segment parameters
        # under subject-specific priors.)
        log_A0_subjects, _ = self._per_subject_log_evidence(ys, ws)

        # Initialise hard responsibilities. The §latent-em recommendation is to
        # try several restarts from (a) k-means on per-subject block-evidence
        # matrices, (b) templates drawn from the partition prior, or (c) a
        # warm-started known-groups solution. We default to (a-lite): hard
        # k-means++-style assignment where each group is anchored to a
        # randomly-chosen "seed" subject and the remaining subjects are
        # assigned to the group whose seed is closest under the per-subject
        # log-A^(0) table treated as a flat feature vector. This breaks the
        # symmetric-init pathology that otherwise collapses every group's
        # template to the global pooled template.
        feats = np.stack([np.where(np.isfinite(la), la, 0.0).ravel() for la in log_A0_subjects])
        seeds = list(rng.choice(S, size=min(G, S), replace=False))
        # Greedy farthest-point augmentation if S < G is impossible; we cap.
        anchors = feats[seeds]  # (G, n*n)
        # Squared distances from each subject to each anchor.
        dists = np.linalg.norm(feats[:, None, :] - anchors[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        r = np.eye(G)[labels].astype(float)
        r = 0.99 * r + 0.01 / G
        pi = (r.sum(axis=0) + 1e-9) / (S + G * 1e-9)

        objective: list[float] = []
        prev_obj = -np.inf
        group_states: list[_GroupState] = []
        prev_templates: list[list[int]] | None = None
        for it in range(int(self.max_iter)):
            # ---------- M-step on pi ----------
            n_g_vec = r.sum(axis=0)
            # Add a tiny pseudocount to keep every group active.
            pi = (n_g_vec + 1e-9) / (S + G * 1e-9)

            # ---------- M-step on tau_g via max-sum DP ----------
            group_states = []
            for g in range(G):
                template, k_g, _, log_offset = self._maximise_template_for_group(
                    n,
                    k_max,
                    float(n_g_vec[g]),
                    log_A0_subjects,
                    r[:, g],
                    log_g_table,
                    log_C_k,
                    log_p_k,
                )
                group_states.append(
                    _GroupState(
                        hyper={},  # subject-specific hyper held in subject lA0 already
                        template=template,
                        k_g=k_g,
                        log_score_offset=log_offset,
                    )
                )

            # ---------- E-step: r_{sg} ∝ π_g · S_g(y^(s); τ_g) ----------
            log_u = np.full((S, G), -np.inf, dtype=float)
            for g, gs in enumerate(group_states):
                lp = np.log(max(pi[g], 1e-300))
                offset = gs.log_score_offset
                for s in range(S):
                    score = _template_log_score(log_A0_subjects[s], gs.template, log_g_table)
                    log_u[s, g] = lp + offset + score

            log_norm = logsumexp(log_u, axis=1, keepdims=True)
            r = np.exp(log_u - log_norm)
            obj = float(np.sum(log_norm))
            objective.append(obj)

            if self.verbose:
                print(f"[mixture] iter {it+1:03d} obj={obj:.6f}")

            current_templates = [list(gs.template) for gs in group_states]
            if it > 0:
                denom = max(1.0, abs(prev_obj))
                templates_stable = (
                    prev_templates is not None and prev_templates == current_templates
                )
                obj_stable = abs(obj - prev_obj) / denom < self.tol
                if templates_stable and obj_stable:
                    # §latent-em criterion (iii): re-run the M-step max-sum
                    # under deterministic tie-breaking to certify the same
                    # templates re-emerge from the current responsibilities.
                    n_g_vec = r.sum(axis=0)
                    recertified = True
                    for g in range(G):
                        tmpl, k_g, _, _ = self._maximise_template_for_group(
                            n,
                            k_max,
                            float(n_g_vec[g]),
                            log_A0_subjects,
                            r[:, g],
                            log_g_table,
                            log_C_k,
                            log_p_k,
                        )
                        if list(tmpl) != current_templates[g] or k_g != group_states[g].k_g:
                            recertified = False
                            break
                    if recertified:
                        break
            prev_obj = obj
            prev_templates = current_templates

        return prev_obj, pi, r, group_states, objective, log_A0_subjects

    # ---- sklearn API --------------------------------------------------

    def fit(
        self,
        X: SequenceInput,
        y: SequenceInput | None = None,
        sample_weight: SequenceInput | None = None,
    ) -> BayesBreakMixtureClassifier:
        Y_in = X if y is None else y
        ys = _as_list_of_1d(Y_in, name="y")
        S = len(ys)
        n = int(ys[0].shape[0])
        if any(a.shape[0] != n for a in ys):
            raise ValueError("All sequences must share the same length.")
        ws = _as_list_of_weights(sample_weight, n_seq=S, n=n)

        if self.n_groups < 1:
            raise ValueError("n_groups must be >= 1.")
        k_max = min(max(1, n), int(self.k_max))

        # Length prior, p(k), C_k.
        if self.boundary_coordinates is not None:
            u = np.asarray(self.boundary_coordinates, dtype=float).ravel()
            if u.size != n + 1 or not np.all(np.diff(u) > 0):
                raise ValueError("boundary_coordinates must be strictly increasing of length n+1.")
        else:
            x_design = np.arange(n, dtype=float)
            u = _midpoint_u(x_design, n)
        log_g = _build_log_g_table(self.length_prior, u, n)
        log_C_k = _dp.compute_log_C_k(log_g, n, k_max)
        log_p_k = _build_log_p_k(self.prior_k, k_max)
        self.boundary_coordinates_ = u
        self.log_C_k_ = log_C_k
        self.log_g_table_ = log_g

        rng_master = check_random_state(self.random_state)
        best = None
        for restart in range(max(1, int(self.n_restarts))):
            seed = rng_master.randint(0, 2**31 - 1)
            rng = np.random.default_rng(seed)
            try:
                obj_final, pi, r, gs, traj, lA0_subj = self._fit_one_restart(
                    rng, ys, ws, n, k_max, log_g, log_C_k, log_p_k
                )
            except Exception as exc:
                if self.verbose:
                    print(f"[mixture] restart {restart} failed: {exc}")
                continue
            if best is None or obj_final > best[0]:
                best = (obj_final, pi, r, gs, traj, lA0_subj, seed)
        if best is None:
            raise RuntimeError("All EM restarts failed.")
        _, pi, r, group_states, traj, lA0_subj, seed = best

        # Anchor the label-permutation indeterminacy of
        # ``prop:latent-identifiability`` deterministically (§5b
        # "Latent-group identifiability"): sort groups by k_g, then
        # lexicographically by template t^(g). The unordered multiset is
        # identifiable; this anchor makes label-level reporting reproducible
        # across restarts and across runs.
        perm = _canonical_template_order(group_states)
        pi = pi[perm]
        r = r[:, perm]
        group_states = [group_states[i] for i in perm]
        self.canonical_permutation_ = np.asarray(perm, dtype=int)

        self.pi_ = pi
        self.responsibilities_ = r
        self.group_states_ = group_states
        self.objective_ = traj
        self.n_ = n
        self.n_seq_ = S
        self.boundary_coordinates_ = u
        self._log_A0_subjects_ = lA0_subj
        self.seed_used_ = int(seed)
        return self

    def predict_proba(
        self,
        X: SequenceInput,
        y: SequenceInput | None = None,
        sample_weight: SequenceInput | None = None,
    ) -> FloatArray:
        require_fitted(self, ["group_states_", "pi_"])
        Y_in = X if y is None else y
        ys = _as_list_of_1d(Y_in, name="y")
        S = len(ys)
        n = int(ys[0].shape[0])
        if self.n_ is not None and n != self.n_:
            raise ValueError(f"Expected length {self.n_}; got {n}.")
        ws = _as_list_of_weights(sample_weight, n_seq=S, n=n)
        log_A0_subjects, _ = self._per_subject_log_evidence(ys, ws)

        G = len(self.group_states_)
        log_u = np.full((S, G), -np.inf, dtype=float)
        for g, gs in enumerate(self.group_states_):
            lp = np.log(max(float(self.pi_[g]), 1e-300))
            for s in range(S):
                score = _template_log_score(log_A0_subjects[s], gs.template, self.log_g_table_)
                log_u[s, g] = lp + gs.log_score_offset + score
        log_norm = logsumexp(log_u, axis=1, keepdims=True)
        return np.exp(log_u - log_norm)

    def predict(
        self,
        X: SequenceInput,
        y: SequenceInput | None = None,
        sample_weight: SequenceInput | None = None,
    ) -> FloatArray:
        return np.argmax(self.predict_proba(X, y=y, sample_weight=sample_weight), axis=1)

    def sequence_log_likelihood(
        self,
        X: SequenceInput,
        y: SequenceInput | None = None,
        sample_weight: SequenceInput | None = None,
    ) -> FloatArray:
        r"""Per-sequence marginal mixture log-likelihood ``log p(y^{(s)})``.

        Returns the vector
        ``log Σ_g π_g · S_g(y^{(s)}; τ_g)`` where ``S_g`` is the
        prior-and-cohesion-adjusted template score. This is the quantity
        named ``def:metric-loglik`` in §6 and is the recommended target
        for held-out ``G`` selection (mitigating the overspecified-``G``
        redundancy of ``rem:teicher-overspec``).
        """
        require_fitted(self, ["group_states_", "pi_"])
        Y_in = X if y is None else y
        ys = _as_list_of_1d(Y_in, name="y")
        S = len(ys)
        n = int(ys[0].shape[0])
        if self.n_ is not None and n != self.n_:
            raise ValueError(f"Expected length {self.n_}; got {n}.")
        ws = _as_list_of_weights(sample_weight, n_seq=S, n=n)
        log_A0_subjects, _ = self._per_subject_log_evidence(ys, ws)

        G = len(self.group_states_)
        log_u = np.full((S, G), -np.inf, dtype=float)
        for g, gs in enumerate(self.group_states_):
            lp = np.log(max(float(self.pi_[g]), 1e-300))
            for s in range(S):
                score = _template_log_score(log_A0_subjects[s], gs.template, self.log_g_table_)
                log_u[s, g] = lp + gs.log_score_offset + score
        return np.asarray(logsumexp(log_u, axis=1), dtype=float)

    def score(
        self,
        X: SequenceInput,
        y: SequenceInput | None = None,
        sample_weight: SequenceInput | None = None,
    ) -> float:
        proba = self.predict_proba(X, y=y, sample_weight=sample_weight)
        return float(np.mean(np.log(np.maximum(1e-300, proba.max(axis=1)))))

    # ---- diagnostics --------------------------------------------------

    def get_group_template(self, g: int) -> list[int]:
        require_fitted(self, ["group_states_"])
        return list(self.group_states_[g].template)

    def get_group_boundary_marginals(self, g: int) -> FloatArray:
        """Conditional ``P(b_i = 1 | y, k_g)`` under the group's pooled responsibility-weighted score table.

        Computed by running the sum-product DP on the same ``B^{(g)}`` matrix
        used for the M-step max-sum update.
        """

        require_fitted(self, ["group_states_", "_log_A0_subjects_", "responsibilities_"])
        gs = self.group_states_[g]
        S = len(self._log_A0_subjects_)
        n = self._log_A0_subjects_[0].shape[0] - 1
        n_g = float(self.responsibilities_[:, g].sum())
        finite_mask = np.ones_like(self._log_A0_subjects_[0], dtype=bool)
        for la in self._log_A0_subjects_:
            finite_mask &= np.isfinite(la)
        B = np.zeros_like(self._log_A0_subjects_[0])
        for s in range(S):
            B[finite_mask] += (
                float(self.responsibilities_[s, g]) * self._log_A0_subjects_[s][finite_mask]
            )
        if self.log_g_table_ is not None:
            B[finite_mask] += n_g * self.log_g_table_[finite_mask]
        B[~finite_mask] = -np.inf

        L, R = _dp.forward_backward(B, n, gs.k_g)
        return _dp.boundary_event_marginals_fixed_k(L, R, n, gs.k_g)
