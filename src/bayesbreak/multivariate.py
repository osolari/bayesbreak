from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Tuple

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone

from .base import BayesBreakBase
from .utils import check_sample_weight, log_binom, logsumexp, require_fitted


def _normalize_multivariate_weights(
    sample_weight: Optional[np.ndarray], n: int, d: int
) -> np.ndarray:
    """Return a dense (n, d) weight matrix.

    Accepts:
      - None -> all ones
      - 1D (n,) -> broadcast to (n, d)
      - 2D (n, d) -> used as-is
      - scalar -> broadcast
    """

    if sample_weight is None or np.isscalar(sample_weight):
        w1 = check_sample_weight(sample_weight, n)
        return np.repeat(w1[:, None], d, axis=1)

    w = np.asarray(sample_weight, dtype=float)
    if w.ndim == 1:
        if w.shape[0] != n:
            raise ValueError(f"sample_weight has length {w.shape[0]}, expected {n}.")
        if np.any(~np.isfinite(w)) or np.any(w < 0):
            raise ValueError("sample_weight must be finite and non-negative.")
        return np.repeat(w[:, None], d, axis=1)
    if w.ndim == 2:
        if w.shape != (n, d):
            raise ValueError(f"sample_weight has shape {w.shape}, expected {(n, d)}.")
        if np.any(~np.isfinite(w)) or np.any(w < 0):
            raise ValueError("sample_weight must be finite and non-negative.")
        return w
    raise ValueError("sample_weight must be None, scalar, 1D (n,) or 2D (n,d).")


@dataclass
class _ChannelState:
    est: BayesBreakBase
    hyper: dict
    lA0: np.ndarray
    A1: np.ndarray


class BayesBreakMultivariate(BaseEstimator, RegressorMixin):
    """Multivariate wrapper for BayesBreak.

    This wrapper supports *shared-boundary* segmentation for vector-valued
    observations :math:`y_t \in \mathbb{R}^d` under a conditional independence
    assumption across channels given segment parameters.

    Two modes are supported:

    - ``combine='shared'``: a *single* segmentation is inferred using the joint
      block evidence :math:`\log \mathcal{L}_{ij} = \sum_{c=1}^d \log \mathcal{L}^{(c)}_{ij}`.
      Segment-wise posterior means are then computed per channel.
    - ``combine='independent'``: each channel is segmented independently by
      fitting a cloned estimator per channel.

    The base estimator must be a univariate :class:`~bayesbreak.base.BayesBreakBase`
    instance (Gaussian/Poisson/Binomial/Beta/Bernoulli, or a custom subclass).

    Notes
    -----
    - ``BayesBreakMultivariate`` implements the sklearn-style ``fit/predict/score``
      interface.
    - For ``combine='shared'``, the returned ``pc_fit_`` has shape ``(n, d)``.
    """

    def __init__(
        self,
        base_estimator: BayesBreakBase,
        *,
        combine: Literal["shared", "independent"] = "shared",
        k_max: Optional[int] = None,
    ):
        self.base_estimator = base_estimator
        self.combine = combine
        self.k_max = k_max

        # fitted attributes
        self.n_: Optional[int] = None
        self.d_: Optional[int] = None
        self.k_ml_: Optional[int] = None
        self.boundaries_: Optional[List[int]] = None
        self.boundary_post_: Optional[np.ndarray] = None
        self.pc_fit_: Optional[np.ndarray] = None
        self.brc_: Optional[np.ndarray] = None
        self.log_evidence_: Optional[float] = None
        self.channel_estimators_: Optional[List[BayesBreakBase]] = None

    # ---------------------------------------------------------------------
    # sklearn API
    # ---------------------------------------------------------------------

    def fit(self, X=None, y=None, sample_weight=None):
        # accept y or use X as y (mirrors BayesBreakBase)
        if y is None:
            if X is None:
                raise ValueError("Provide y (preferred) or X as a 2D array-like of shape (n, d).")
            y_arr = np.asarray(X, dtype=float)
        else:
            y_arr = np.asarray(y, dtype=float)
        if y_arr.ndim != 2:
            raise ValueError("Multivariate y must be 2D with shape (n, d).")
        n, d = y_arr.shape
        self.n_, self.d_ = int(n), int(d)

        # per-channel weights
        w_mat = _normalize_multivariate_weights(sample_weight, n, d)

        if self.combine == "independent":
            self.channel_estimators_ = []
            pc = np.zeros((n, d), dtype=float)
            brc = []
            loge = 0.0
            for c in range(d):
                est_c = clone(self.base_estimator)
                if self.k_max is not None:
                    est_c.k_max = int(self.k_max)
                est_c.fit(y_arr[:, c], sample_weight=w_mat[:, c])
                self.channel_estimators_.append(est_c)
                pc[:, c] = est_c.predict(None)
                if getattr(est_c, "brc_", None) is not None:
                    brc.append(est_c.get_regression_curve())
                loge += float(est_c.score())
            self.pc_fit_ = pc
            self.brc_ = np.column_stack(brc) if brc else None
            self.log_evidence_ = float(loge)
            # boundaries are per-channel in this mode
            self.boundaries_ = None
            self.boundary_post_ = None
            self.k_ml_ = None
            return self

        if self.combine != "shared":
            raise ValueError("combine must be either 'shared' or 'independent'.")

        # Shared-boundary: build channel states and combine block evidences.
        k_max = int(self.k_max) if self.k_max is not None else int(getattr(self.base_estimator, "k_max", 50))
        k_max = min(max(1, n), k_max)

        channel_states: List[_ChannelState] = []
        lA0_joint = None
        for c in range(d):
            est_c = clone(self.base_estimator)
            # We do not call est_c.fit() here to avoid running DP d times.
            # Instead we only build its block evidences.
            hyper_c = est_c._estimate_global_params(y_arr[:, c], w_mat[:, c])
            lA0_c, A1_c = est_c._compute_single_segment_stats(y_arr[:, c], hyper_c, w_mat[:, c])
            channel_states.append(_ChannelState(est=est_c, hyper=hyper_c, lA0=lA0_c, A1=A1_c))
            if lA0_joint is None:
                lA0_joint = lA0_c.copy()
            else:
                lA0_joint = lA0_joint + lA0_c

        assert lA0_joint is not None

        # Run DP on joint evidences (reuse BayesBreakBase internals)
        L, R = BayesBreakBase._compute_left_right_recursions(lA0_joint, n, k_max)
        logC, C, logE = BayesBreakBase._posterior_over_k(L, n, k_max)
        self.log_evidence_ = float(logE)

        # choose k_ml around E[k]
        ek = float(np.sum((np.arange(1, k_max + 1)) * C))
        valid = np.where(np.isfinite(logC))[0] + 1
        self.k_ml_ = int(valid[np.argmin((valid - ek) ** 2)])

        # boundary posteriors averaged over k
        d1 = BayesBreakBase._boundary_posteriors_marginal(L, R, logC, n, k_max)
        self.boundary_post_ = d1

        # MAP-like boundaries = top (k_ml-1) by d1
        boundaries = BayesBreakBase._select_boundaries_from_scores(d1, self.k_ml_, n)
        self.boundaries_ = boundaries

        # per-channel piecewise-constant fit
        pc = np.zeros((n, d), dtype=float)
        for a, b in zip(boundaries[:-1], boundaries[1:]):
            for c, st in enumerate(channel_states):
                mu = st.est._segment_posterior_mean(a, b, y_arr[:, c], st.hyper, w_mat[:, c])
                pc[a:b, c] = mu
        self.pc_fit_ = pc

        # optional Bayesian regression curve (per-channel)
        self.brc_ = None
        rc = getattr(self.base_estimator, "regression_curve", "none")
        if rc in {"fixed_k", "mix_k"}:
            brc = np.zeros((n, d), dtype=float)
            for c, st in enumerate(channel_states):
                A1_joint_c = self._make_channel_A1_joint(st, lA0_joint)
                if rc == "fixed_k":
                    brc[:, c] = self._bayes_regression_curve_fixed_k(L, R, A1_joint_c, n, self.k_ml_)
                else:
                    brc[:, c] = self._bayes_regression_curve_mixed_k(L, R, A1_joint_c, n, k_max, C)
            self.brc_ = brc

        # store fitted per-channel estimators in case callers want channel hyperparameters
        self.channel_estimators_ = [st.est for st in channel_states]
        return self

    def predict(self, X=None):
        require_fitted(self, ["pc_fit_"])
        return np.array(self.pc_fit_, copy=True)

    def score(self, X=None, y=None):
        require_fitted(self, ["log_evidence_"])
        return float(self.log_evidence_)

    # ---------------------------------------------------------------------
    # Convenience getters
    # ---------------------------------------------------------------------

    def get_boundaries(self) -> List[int]:
        require_fitted(self, ["boundaries_"])
        if self.boundaries_ is None:
            raise RuntimeError("Boundaries are undefined for combine='independent'.")
        return list(self.boundaries_)

    def get_boundary_posteriors(self) -> np.ndarray:
        require_fitted(self, ["boundary_post_"])
        if self.boundary_post_ is None:
            raise RuntimeError("Boundary posteriors are undefined for combine='independent'.")
        return np.array(self.boundary_post_, copy=True)

    def get_regression_curve(self) -> Optional[np.ndarray]:
        return None if self.brc_ is None else np.array(self.brc_, copy=True)

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _make_channel_A1_joint(st: _ChannelState, lA0_joint: np.ndarray) -> np.ndarray:
        """Construct A1 for one channel under the *joint* evidence.

        In the univariate algorithm we use ``A1[i,j] = A0[i,j] * E[mu | y_(i,j]]``.
        For the multivariate shared-boundary model, the DP is run using
        ``A0_joint[i,j] = prod_c A0_c[i,j]``. The posterior mean for a channel
        parameter depends only on the data in that channel, therefore we use
        ``A1_joint^{(c)}[i,j] = A0_joint[i,j] * E[mu_c | y_c(i,j]]``.

        We obtain ``E[mu_c | ...]`` via ``A1_c / A0_c`` where possible.

        This construction is numerically safe for the sizes used in the unit tests
        and example scripts. For large ``n`` and/or large ``d``, users may want a
        log-domain implementation.
        """

        lA0_c = st.lA0
        A1_c = st.A1

        # mu_hat = A1_c / exp(lA0_c) on the valid upper triangle.
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            mu_hat = A1_c * np.exp(-lA0_c)
            A1_joint = np.exp(lA0_joint) * mu_hat
        A1_joint[~np.isfinite(A1_joint)] = 0.0
        return A1_joint

    @staticmethod
    def _bayes_regression_curve_fixed_k(
        L: np.ndarray, R: np.ndarray, A1: np.ndarray, n: int, k: int
    ) -> np.ndarray:
        """Bayesian regression curve for fixed k.

        This mirrors :meth:`bayesbreak.base.BayesBreakBase._bayes_regression_curve_fixed_k`.
        """

        denom = L[k, n]
        diff = np.zeros(n + 1, dtype=float)
        for i in range(0, n):
            Li = L[0:k, i]
            for j in range(i + 1, n + 1):
                Rj = R[k - 1 :: -1, j]
                log_w_ij = float(logsumexp(Li + Rj) - denom)
                F1 = math.exp(log_w_ij) * float(A1[i, j])
                if F1 != 0.0:
                    diff[i] += F1
                    diff[j] -= F1
        mu = np.cumsum(diff)
        return mu[:n]

    @staticmethod
    def _bayes_regression_curve_mixed_k(
        L: np.ndarray, R: np.ndarray, A1: np.ndarray, n: int, k_max: int, C: np.ndarray
    ) -> np.ndarray:
        out = np.zeros(n, dtype=float)
        for k in range(1, k_max + 1):
            if C[k - 1] == 0.0 or not np.isfinite(L[k, n]):
                continue
            out += float(C[k - 1]) * BayesBreakMultivariate._bayes_regression_curve_fixed_k(L, R, A1, n, k)
        return out

