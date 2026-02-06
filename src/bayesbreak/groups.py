from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Hashable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone

from .base import BayesBreakBase
from .families import (
    BayesBreakBernoulli,
    BayesBreakBeta,
    BayesBreakBinomial,
    BayesBreakGaussian,
    BayesBreakPoisson,
)
from .multivariate import BayesBreakMultivariate
from .utils import check_sample_weight, logsumexp, require_fitted

Array1D = np.ndarray
Array2D = np.ndarray


def _is_multivariate_sequence(y: np.ndarray) -> bool:
    return y.ndim == 2


def _as_sequence_list(X: Any) -> List[np.ndarray]:
    """Normalize X into a list of arrays.

    Supported inputs:
      - list of ndarrays (each 1D or 2D)
      - 2D array (treated as a single multivariate sequence if shape (n, d))
      - 1D array (treated as a single univariate sequence)
    """
    if isinstance(X, list):
        ys = [np.asarray(a) for a in X]
        if len(ys) == 0:
            raise ValueError("X must contain at least one sequence.")
        return ys
    arr = np.asarray(X)
    if arr.ndim in (1, 2):
        return [arr]
    raise ValueError("X must be a list of 1D/2D arrays or a 1D/2D array.")


def _as_weight_list(sample_weight: Any, ys: List[np.ndarray]) -> List[Optional[np.ndarray]]:
    if sample_weight is None:
        return [None] * len(ys)
    if isinstance(sample_weight, list):
        if len(sample_weight) != len(ys):
            raise ValueError("If sample_weight is a list, it must match len(X).")
        return [np.asarray(w) if w is not None else None for w in sample_weight]

    # broadcast a single weight array/scalar to all sequences
    w = np.asarray(sample_weight)
    return [w for _ in ys]


def _normalize_weights_for_sequence(y: np.ndarray, w: Optional[np.ndarray]) -> np.ndarray:
    if w is None:
        if y.ndim == 1:
            return check_sample_weight(None, int(y.shape[0]))
        # multivariate: share weights across channels by default
        w1 = check_sample_weight(None, int(y.shape[0]))
        return np.repeat(w1[:, None], int(y.shape[1]), axis=1)

    w = np.asarray(w, dtype=float)
    n = int(y.shape[0])
    if y.ndim == 1:
        return check_sample_weight(w, n)

    # y is (n, d)
    d = int(y.shape[1])
    if w.ndim == 1:
        w1 = check_sample_weight(w, n)
        return np.repeat(w1[:, None], d, axis=1)
    if w.ndim == 2 and w.shape == (n, d):
        out = np.asarray(w, dtype=float)
        if np.any(~np.isfinite(out)) or np.any(out < 0):
            raise ValueError("sample_weight contains NaN/inf or negative values.")
        return out
    raise ValueError(
        f"sample_weight must be shape (n,) or (n,d); got {w.shape} for y shape {y.shape}."
    )


@dataclass(frozen=True)
class _GroupModel:
    label: Hashable
    prior_logprob: float
    estimator_prototype: BaseEstimator


def _fit_with_fixed_hyper(
    est_proto: BaseEstimator, y: np.ndarray, w: Optional[np.ndarray]
) -> BaseEstimator:
    # We always clone for thread-safety and to avoid overwriting cached attrs.
    est = clone(est_proto)
    if hasattr(est, "fit"):
        est.fit(y, sample_weight=w)  # type: ignore[arg-type]
        return est
    raise TypeError("Estimator prototype does not implement fit().")


class BayesBreakGrouped(BaseEstimator, ClassifierMixin):
    """Group-membership scoring + MAP signal evaluation.

    This estimator implements a simple but effective interface used throughout the
    paper's prediction experiments:

    - **Training**: estimate group-specific hyperparameters from labeled training
      sequences and construct a fixed-hyper BayesBreak model per group.
    - **Scoring**: for a new sequence, compute the marginal likelihood (log
      evidence) under each group model; combine with group priors to obtain
      posterior group probabilities.
    - **MAP signal evaluation**: conditional on a group label (true or predicted),
      refit BayesBreak using the group's hyperparameters and return the MAP-like
      piecewise-constant reconstruction (and optionally a Bayesian regression
      curve if enabled in the prototype).

    Notes
    -----
    * The implementation is deliberately *application-agnostic*: sequences are
      generic arrays, and the method does not assume a genomics-specific
      coordinate system.
    * Group membership is defined by the Bayesian marginal likelihood under each
      group model. This does not require point-estimated boundaries shared across
      the group; boundaries remain sequence-specific.
    """

    def __init__(
        self,
        base_estimator: Union[BayesBreakBase, BayesBreakMultivariate],
        class_prior: Literal["empirical", "uniform"] = "empirical",
    ):
        self.base_estimator = base_estimator
        self.class_prior = class_prior

        # fitted
        self.classes_: Optional[np.ndarray] = None
        self.group_models_: Optional[List[_GroupModel]] = None

    # ---------------------------------------------------------------------
    # Hyperparameter pooling helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _pool_gaussian_hyper(
        tmpl: BayesBreakGaussian, ys: List[np.ndarray], ws: List[np.ndarray]
    ) -> Dict[str, float]:
        # pooled weighted mean
        tot_w = 0.0
        tot_wy = 0.0
        for y, w in zip(ys, ws, strict=False):
            tot_w += float(np.sum(w))
            tot_wy += float(np.sum(w * y))
        if tot_w <= 0:
            # fall back to unweighted pooling
            tot_w = float(sum(y.size for y in ys))
            tot_wy = float(sum(np.sum(y) for y in ys))
        nu = tot_wy / max(tot_w, 1e-12)

        # sigma^2 from within-sequence weighted finite differences
        num = 0.0
        den = 0.0
        for y, w in zip(ys, ws, strict=False):
            if y.size <= 1:
                continue
            dy = np.diff(y)
            wdiff = 0.5 * (w[:-1] + w[1:])
            num += float(np.sum(wdiff * dy * dy))
            den += float(np.sum(wdiff))
        sigma2 = num / max(2.0 * den, 1e-12) if den > 0 else 1e-8

        if tmpl.rho_estimation == "cov":
            num = 0.0
            den = 0.0
            for y, w in zip(ys, ws, strict=False):
                if y.size <= 1:
                    continue
                y0 = y[:-1] - nu
                y1 = y[1:] - nu
                wdiff = 0.5 * (w[:-1] + w[1:])
                num += float(np.sum(wdiff * y0 * y1))
                den += float(np.sum(wdiff))
            rho2 = abs(num / max(den, 1e-12)) if den > 0 else 1e-8
        else:
            num = 0.0
            den = 0.0
            for y, w in zip(ys, ws, strict=False):
                yc = y - nu
                num += float(np.sum(w * yc * yc))
                den += float(np.sum(w))
            rho2 = num / max(den, 1e-12) if den > 0 else 1e-8

        return {
            "nu": float(nu),
            "rho2": float(max(rho2, 1e-12)),
            "sigma2": float(max(sigma2, 1e-12)),
        }

    @staticmethod
    def _pool_poisson_hyper(
        tmpl: BayesBreakPoisson, ys: List[np.ndarray], ws: List[np.ndarray]
    ) -> Dict[str, float]:
        tot_w = 0.0
        tot_wy = 0.0
        for y, w in zip(ys, ws, strict=False):
            tot_w += float(np.sum(w))
            tot_wy += float(np.sum(w * y))
        m = tot_wy / max(tot_w, 1e-12)

        # pooled weighted variance
        num = 0.0
        for y, w in zip(ys, ws, strict=False):
            num += float(np.sum(w * (y - m) ** 2))
        v = num / max(tot_w, 1e-12) if tot_w > 0 else max(1.0, m)

        if v > m + 1e-12:
            alpha = max(1e-8, m * m / (v - m))
        else:
            alpha = 1e6
        beta = max(1e-8, alpha / max(m, 1e-12))
        return {"alpha": float(alpha), "beta": float(beta)}

    @staticmethod
    def _pool_beta_binomial_hyper(
        tmpl: Union[BayesBreakBinomial, BayesBreakBernoulli],
        ys: List[np.ndarray],
        ws: List[np.ndarray],
        n_trials_scalar: float,
    ) -> Dict[str, float]:
        # mean on pooled successes/trials
        tot_w = 0.0
        tot_wy = 0.0
        for y, w in zip(ys, ws, strict=False):
            tot_w += float(np.sum(w))
            tot_wy += float(np.sum(w * y))
        T = tot_w * n_trials_scalar
        mu = tot_wy / max(T, 1e-12)

        # EB variance correction on proportions
        p_vals: List[np.ndarray] = []
        w_vals: List[np.ndarray] = []
        for y, w in zip(ys, ws, strict=False):
            p_vals.append(np.asarray(y, dtype=float) / max(n_trials_scalar, 1e-12))
            w_vals.append(w)
        p_all = np.concatenate(p_vals)
        w_all = np.concatenate(w_vals)
        wsum = float(np.sum(w_all))
        if wsum <= 0:
            w_all = np.ones_like(p_all)
            wsum = float(p_all.size)
        p_mean = float(np.sum(w_all * p_all) / wsum)
        var_p_obs = float(np.sum(w_all * (p_all - p_mean) ** 2) / wsum) if p_all.size > 1 else 1e-4
        noise = mu * (1.0 - mu) / max(n_trials_scalar, 1.0)
        var_p = max(var_p_obs - noise, 1e-12)

        tau = max(1e-8, mu * (1.0 - mu) / var_p - 1.0)
        alpha = mu * tau
        beta = (1.0 - mu) * tau
        return {"alpha": float(alpha), "beta": float(beta)}

    @staticmethod
    def _pool_beta_hyper(
        tmpl: BayesBreakBeta, ys: List[np.ndarray], ws: List[np.ndarray]
    ) -> Dict[str, float]:
        kappa = tmpl.concentration
        y_all = np.concatenate([np.asarray(y, dtype=float) for y in ys])
        w_all = np.concatenate([np.asarray(w, dtype=float) for w in ws])
        wsum = float(np.sum(w_all))
        if wsum <= 0:
            w_all = np.ones_like(y_all)
            wsum = float(y_all.size)
        mu = float(np.sum(w_all * y_all) / wsum)
        var_obs = float(np.sum(w_all * (y_all - mu) ** 2) / wsum) if y_all.size > 1 else 1e-4
        var_p = max(var_obs - mu * (1 - mu) / kappa, 1e-12)
        tau = max(1e-8, mu * (1 - mu) / var_p - 1.0)
        alpha = mu * tau
        beta = (1.0 - mu) * tau
        return {"alpha": float(alpha), "beta": float(beta)}

    # ---------------------------------------------------------------------
    # Build fixed-hyper estimator prototypes
    # ---------------------------------------------------------------------

    @staticmethod
    def _set_fixed_hyper(est: BayesBreakBase, hyper: Dict[str, float]) -> BayesBreakBase:
        est.estimate_hyper = False
        for k, v in hyper.items():
            # hyper parameters are exposed as attributes in our family classes
            if hasattr(est, k):
                setattr(est, k, float(v))
        return est

    def _build_group_estimator_prototype(
        self,
        hyper: Dict[str, float],
    ) -> BaseEstimator:
        if isinstance(self.base_estimator, BayesBreakMultivariate):
            raise RuntimeError("Internal error: multivariate prototypes are built separately.")
        est = clone(self.base_estimator)
        if not isinstance(est, BayesBreakBase):
            raise TypeError("base_estimator must be a BayesBreakBase or BayesBreakMultivariate.")
        return self._set_fixed_hyper(est, hyper)

    def _build_group_estimator_prototype_multivariate(
        self,
        hypers_per_channel: List[Dict[str, float]],
    ) -> BayesBreakMultivariate:
        if not isinstance(self.base_estimator, BayesBreakMultivariate):
            raise TypeError("base_estimator must be a BayesBreakMultivariate.")

        base = self.base_estimator
        channel_estimators: List[BayesBreakBase] = []
        for h in hypers_per_channel:
            ch_est = clone(base.base_estimator)
            if not isinstance(ch_est, BayesBreakBase):
                raise TypeError("BayesBreakMultivariate.base_estimator must be a BayesBreakBase.")
            channel_estimators.append(self._set_fixed_hyper(ch_est, h))

        # Use first channel estimator as prototype for multivariate wrapper
        # Note: this assumes all channels use the same estimator type
        return BayesBreakMultivariate(
            base_estimator=channel_estimators[0],
            combine=base.combine,
            k_max=base.k_max,
        )

    # ---------------------------------------------------------------------
    # sklearn API
    # ---------------------------------------------------------------------

    def fit(self, X: Any, y: Sequence[Hashable], sample_weight: Any = None) -> "BayesBreakGrouped":
        ys = _as_sequence_list(X)
        if len(ys) != len(y):
            raise ValueError("X and y must have the same number of sequences.")

        ws_in = _as_weight_list(sample_weight, ys)
        ws = [_normalize_weights_for_sequence(yy, ww) for yy, ww in zip(ys, ws_in, strict=False)]

        y_labels = np.asarray(list(y), dtype=object)
        classes, inv = np.unique(y_labels, return_inverse=True)
        self.classes_ = classes

        # priors
        if self.class_prior == "uniform":
            log_prior = np.full(len(classes), -math.log(len(classes)), dtype=float)
        else:
            counts = np.bincount(inv, minlength=len(classes)).astype(float)
            pri = counts / max(float(np.sum(counts)), 1.0)
            log_prior = np.log(np.maximum(pri, 1e-300))

        group_models: List[_GroupModel] = []

        for gi, g in enumerate(classes):
            idx = np.where(y_labels == g)[0]
            ys_g = [ys[i] for i in idx]
            ws_g = [ws[i] for i in idx]

            if isinstance(self.base_estimator, BayesBreakMultivariate):
                # pool per channel
                d = int(np.asarray(ys_g[0]).shape[1])
                hypers: List[Dict[str, float]] = []
                for ch in range(d):
                    y_ch = [np.asarray(seq)[:, ch].astype(float) for seq in ys_g]
                    w_ch = [np.asarray(w)[:, ch].astype(float) for w in ws_g]

                    tmpl = self.base_estimator.base_estimator
                    if not isinstance(tmpl, BayesBreakBase):
                        raise TypeError("base_estimator.base_estimator must be a BayesBreakBase.")

                    hyper_ch = self._pool_hyper_from_template(tmpl, y_ch, w_ch)
                    hypers.append(hyper_ch)

                proto = self._build_group_estimator_prototype_multivariate(hypers)
            else:
                if not isinstance(self.base_estimator, BayesBreakBase):
                    raise TypeError(
                        "base_estimator must be a BayesBreakBase or BayesBreakMultivariate."
                    )
                hyper = self._pool_hyper_from_template(self.base_estimator, ys_g, ws_g)
                proto = self._build_group_estimator_prototype(hyper)

            group_models.append(
                _GroupModel(label=g, prior_logprob=float(log_prior[gi]), estimator_prototype=proto)
            )

        self.group_models_ = group_models
        return self

    def _pool_hyper_from_template(
        self, tmpl: BayesBreakBase, ys: List[np.ndarray], ws: List[np.ndarray]
    ) -> Dict[str, float]:
        # If user provided fixed values, honour them.
        if not tmpl.estimate_hyper:
            # Let the family decide whether its attributes are set sufficiently.
            return tmpl._estimate_global_params(
                np.asarray(ys[0], dtype=float), np.asarray(ws[0], dtype=float)
            )

        if isinstance(tmpl, BayesBreakGaussian):
            return self._pool_gaussian_hyper(tmpl, ys, ws)
        if isinstance(tmpl, BayesBreakPoisson):
            return self._pool_poisson_hyper(tmpl, ys, ws)
        if isinstance(tmpl, BayesBreakBeta):
            return self._pool_beta_hyper(tmpl, ys, ws)
        if isinstance(tmpl, BayesBreakBinomial):
            if not np.isscalar(tmpl.n_trials):
                raise ValueError(
                    "Grouped hyper pooling for BayesBreakBinomial currently supports scalar n_trials only. "
                    "Provide a scalar n_trials or pre-specify alpha/beta and set estimate_hyper=False."
                )
            return self._pool_beta_binomial_hyper(tmpl, ys, ws, float(tmpl.n_trials))
        if isinstance(tmpl, BayesBreakBernoulli):
            return self._pool_beta_binomial_hyper(tmpl, ys, ws, 1.0)

        # Generic fallback: weighted average of per-sequence EB hypers
        weights = np.asarray([float(np.sum(w)) for w in ws], dtype=float)
        if float(np.sum(weights)) <= 0:
            weights = np.ones_like(weights)
        weights = weights / float(np.sum(weights))
        hypers = [
            tmpl._estimate_global_params(np.asarray(y, dtype=float), np.asarray(w, dtype=float))
            for y, w in zip(ys, ws, strict=False)
        ]
        keys = hypers[0].keys() if hypers else []
        out: Dict[str, float] = {}
        for k in keys:
            out[k] = float(sum(weights[i] * float(hypers[i][k]) for i in range(len(hypers))))
        return out

    # ------------------------------------------------------------------
    # Scoring / inference
    # ------------------------------------------------------------------

    def score_samples(self, X: Any, sample_weight: Any = None) -> np.ndarray:
        """Return log evidence under each group model.

        Returns
        -------
        log_evi : ndarray, shape (n_samples, n_groups)
            log P(y | group=g) for each sample and group.
        """
        require_fitted(self, ["group_models_", "classes_"])
        ys = _as_sequence_list(X)
        ws_in = _as_weight_list(sample_weight, ys)
        ws = [_normalize_weights_for_sequence(yy, ww) for yy, ww in zip(ys, ws_in, strict=False)]

        group_models = self.group_models_  # type: ignore[assignment]
        log_evi = np.empty((len(ys), len(group_models)), dtype=float)
        for i, (yy, ww) in enumerate(zip(ys, ws, strict=False)):
            for g, gm in enumerate(group_models):
                est = _fit_with_fixed_hyper(gm.estimator_prototype, yy, ww)
                log_evi[i, g] = float(est.log_evidence_)
        return log_evi

    def predict_proba(self, X: Any, sample_weight: Any = None) -> np.ndarray:
        require_fitted(self, ["group_models_", "classes_"])
        log_evi = self.score_samples(X, sample_weight=sample_weight)
        pri = np.asarray([gm.prior_logprob for gm in self.group_models_], dtype=float)  # type: ignore[arg-type]
        log_post = log_evi + pri[None, :]
        log_post -= logsumexp(log_post, axis=1)[:, None]
        return np.exp(log_post)

    def predict(self, X: Any, sample_weight: Any = None) -> np.ndarray:
        require_fitted(self, ["group_models_", "classes_"])
        proba = self.predict_proba(X, sample_weight=sample_weight)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]  # type: ignore[index]

    # ------------------------------------------------------------------
    # MAP signal evaluation
    # ------------------------------------------------------------------

    def map_signal(
        self,
        X: Any,
        sample_weight: Any = None,
        group: Optional[Union[Hashable, Sequence[Hashable]]] = None,
        return_boundaries: bool = False,
    ) -> Union[List[np.ndarray], Tuple[List[np.ndarray], List[List[int]]]]:
        """Return the MAP-like piecewise-constant reconstruction for each sequence.

        Parameters
        ----------
        X
            Sequences to evaluate.
        group
            If provided, evaluate each sequence under this group label (or a
            per-sequence list of labels). If omitted, uses the MAP group
            under the posterior group probabilities.
        return_boundaries
            If True, also return the MAP boundary list per sequence.
        """
        require_fitted(self, ["group_models_", "classes_"])
        ys = _as_sequence_list(X)
        ws_in = _as_weight_list(sample_weight, ys)
        ws = [_normalize_weights_for_sequence(yy, ww) for yy, ww in zip(ys, ws_in, strict=False)]

        if group is None:
            groups = list(self.predict(X, sample_weight=sample_weight))
        elif isinstance(group, (list, tuple, np.ndarray)):
            if len(group) != len(ys):
                raise ValueError(
                    "If group is a sequence, it must match the number of input sequences."
                )
            groups = list(group)
        else:
            groups = [group for _ in ys]

        gm_by_label = {gm.label: gm for gm in self.group_models_}  # type: ignore[union-attr]
        out: List[np.ndarray] = []
        bounds: List[List[int]] = []
        for yy, ww, glab in zip(ys, ws, groups, strict=False):
            if glab not in gm_by_label:
                raise ValueError(
                    f"Unknown group label: {glab!r}. Known: {list(gm_by_label.keys())}"
                )
            est = _fit_with_fixed_hyper(gm_by_label[glab].estimator_prototype, yy, ww)
            pc = est.pc_fit_
            out.append(np.asarray(pc))
            if return_boundaries:
                bounds.append(list(est.boundaries_))

        if return_boundaries:
            return out, bounds
        return out
