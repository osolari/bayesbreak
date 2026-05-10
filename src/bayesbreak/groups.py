r"""Known-group BayesBreak (§``groups-known``).

For labeled training sequences ``{(y^{(s)}, z_s)}_{s=1}^S`` with observed group
labels ``z_s ∈ {1, ..., G}``, this estimator fits one *exemplar* BayesBreak
segmenter per group:

- when a group has ≥ 2 training sequences,
  :class:`~bayesbreak.replicates.SharedBoundaryReplicatesSegmenter` is used
  (Theorem ``multisubject``), giving an exact posterior over the shared
  boundary vector with subject-specific segment parameters recovered
  conditionally on the pooled MAP partition;
- when a group has exactly 1 training sequence, the plain BayesBreak family
  is fitted directly.

Inference for a new sequence under each group uses the **exported MAP
scoring** of §``group-lik-point``:

.. math::
    \ell_g^{\mathrm{MAP}}(D^{\mathrm{new}}) =
    \sum_{B \in \widehat t^{(g)}}
    \log p(\mathcal{Y}^{\mathrm{new}}_B \mid \mathcal{M}_g)

via :func:`bayesbreak.prediction.posterior_predictive_logpdf` — i.e. the new
data is routed through the group's exported MAP segmentation and scored
under each segment's conjugate predictive. There is no resegmentation mode.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from .base import BayesBreakSegmenter
from .prediction import posterior_predictive_logpdf
from .replicates import SharedBoundaryReplicatesSegmenter
from .utils import logsumexp
from .validation import require_fitted


def _as_sequence_list(X: Any) -> list[np.ndarray]:
    if isinstance(X, list | tuple):
        ys = [np.asarray(a, dtype=float) for a in X]
        if not ys:
            raise ValueError("Need at least one sequence.")
        return ys
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        return [arr]
    if arr.ndim == 2:
        return [np.asarray(row, dtype=float) for row in arr]
    raise ValueError("X must be 1-D, 2-D, or list of arrays.")


def _as_weight_list(sample_weight: Any, ys: list[np.ndarray]) -> list[np.ndarray | None]:
    if sample_weight is None:
        return [None] * len(ys)
    if isinstance(sample_weight, list | tuple):
        if len(sample_weight) != len(ys):
            raise ValueError("sample_weight list must match number of sequences.")
        return [None if w is None else np.asarray(w, dtype=float) for w in sample_weight]
    arr = np.asarray(sample_weight, dtype=float)
    if arr.ndim == 1 and len(ys) == 1:
        return [arr]
    if arr.ndim == 2 and arr.shape[0] == len(ys):
        return [np.asarray(row, dtype=float) for row in arr]
    if arr.ndim == 1 and arr.shape[0] == ys[0].size:
        return [arr for _ in ys]
    raise ValueError("Unsupported sample_weight shape.")


@dataclass(frozen=True)
class _GroupModel:
    label: Hashable
    prior_logprob: float
    estimator: BaseEstimator  # fitted exemplar (segmenter or replicates)


class BayesBreakGroupedClassifier(BaseEstimator, ClassifierMixin):
    """Group-membership scoring via per-group exported-MAP segmentations.

    Parameters
    ----------
    base_estimator : BayesBreakSegmenter
        Family template; cloned per group. Group-specific empirical Bayes
        runs inside the per-group fit, so user-supplied hyperparameters are
        honoured only when the template has ``estimate_hyper=False``.
    class_prior : {"empirical", "uniform"}
        Prior on group labels.

    Attributes
    ----------
    classes_ : ndarray
    group_models_ : list[_GroupModel]
        Each holds a fitted exemplar exposing ``map_boundaries_``,
        ``map_segment_means_``, and ``posterior_predictive_logpdf_block``.
    """

    def __init__(
        self,
        base_estimator: BayesBreakSegmenter,
        class_prior: Literal["empirical", "uniform"] = "empirical",
    ):
        self.base_estimator = base_estimator
        self.class_prior = class_prior

        self.classes_: np.ndarray | None = None
        self.group_models_: list[_GroupModel] | None = None

    def _fit_exemplar(self, ys: list[np.ndarray], ws: list[np.ndarray | None]) -> BaseEstimator:
        """Fit a single per-group exemplar segmenter ``M_g``.

        For ≥ 2 training subjects we pool by averaging per-index responses
        (and summing per-index weights). The resulting bare segmenter
        provides a single MAP segmentation, segment-level posteriors, and a
        ``posterior_predictive_logpdf_block`` interface — exactly what
        §``group-lik-point`` requires. We additionally store the underlying
        :class:`SharedBoundaryReplicatesSegmenter` (when ≥ 2 subjects) on
        ``estimator.replicates_`` for diagnostic access.
        """

        n = ys[0].shape[0]
        X_idx = np.arange(n).reshape(-1, 1)
        if len(ys) == 1:
            est = self.base_estimator.__class__(**self.base_estimator.get_params())
            w = ws[0] if ws[0] is not None else np.ones(n, dtype=float)
            est.fit(X_idx, ys[0], sample_weight=w)
            return est
        # Pool by per-index averaging (responses) and summing (weights).
        Y = np.stack([np.asarray(y, dtype=float) for y in ys])  # (S, n)
        W_stack = np.stack(
            [np.asarray(w, dtype=float) if w is not None else np.ones(n) for w in ws]
        )
        y_pool = np.sum(W_stack * Y, axis=0) / np.maximum(np.sum(W_stack, axis=0), 1e-12)
        w_pool = np.sum(W_stack, axis=0)
        est = self.base_estimator.__class__(**self.base_estimator.get_params())
        est.fit(X_idx, y_pool, sample_weight=w_pool)
        # Stash the multi-subject diagnostic.
        rep = SharedBoundaryReplicatesSegmenter(self.base_estimator)
        rep.fit(X_idx, ys, sample_weight=ws)
        est.replicates_ = rep
        return est

    def fit(
        self,
        X: Any,
        y: Sequence[Hashable],
        sample_weight: Any = None,
    ) -> BayesBreakGroupedClassifier:
        ys = _as_sequence_list(X)
        ws = _as_weight_list(sample_weight, ys)
        if len(ys) != len(y):
            raise ValueError("X and y must agree on the number of sequences.")
        n = ys[0].shape[0]
        for s, yy in enumerate(ys):
            if yy.size != n:
                raise ValueError(f"All sequences must have length {n}; sequence {s} has {yy.size}.")

        labels = np.asarray(list(y), dtype=object)
        classes, inv = np.unique(labels, return_inverse=True)
        self.classes_ = classes

        if self.class_prior == "uniform":
            log_prior = np.full(len(classes), -math.log(len(classes)), dtype=float)
        else:
            counts = np.bincount(inv, minlength=len(classes)).astype(float)
            pri = counts / max(float(np.sum(counts)), 1.0)
            log_prior = np.log(np.maximum(pri, 1e-300))

        group_models: list[_GroupModel] = []
        for gi, g in enumerate(classes):
            idx = np.where(labels == g)[0]
            if idx.size == 0:
                continue
            est = self._fit_exemplar([ys[i] for i in idx], [ws[i] for i in idx])
            group_models.append(
                _GroupModel(label=g, prior_logprob=float(log_prior[gi]), estimator=est)
            )

        self.group_models_ = group_models
        return self

    # ---- inference ----------------------------------------------------

    @staticmethod
    def _score_under_exemplar(
        exemplar: BaseEstimator,
        y: np.ndarray,
        w: np.ndarray | None,
    ) -> float:
        """Posterior-predictive log-density of ``y`` under a fitted exemplar."""

        if not isinstance(exemplar, BayesBreakSegmenter):
            raise TypeError(f"Unsupported exemplar type: {type(exemplar).__name__}.")
        X_idx = np.arange(y.size).reshape(-1, 1)
        sw = w if w is not None else np.ones(y.size, dtype=float)
        return float(posterior_predictive_logpdf(exemplar, X_idx, y, sample_weight=sw))

    def score_samples(self, X: Any, sample_weight: Any = None) -> np.ndarray:
        """Return ``log P(y | group=g)`` for each new sequence and group."""

        require_fitted(self, ["group_models_", "classes_"])
        ys = _as_sequence_list(X)
        ws = _as_weight_list(sample_weight, ys)
        log_evi = np.empty((len(ys), len(self.group_models_)), dtype=float)
        for i, (yy, ww) in enumerate(zip(ys, ws, strict=False)):
            for g, gm in enumerate(self.group_models_):
                log_evi[i, g] = self._score_under_exemplar(gm.estimator, yy, ww)
        return log_evi

    def predict_proba(self, X: Any, sample_weight: Any = None) -> np.ndarray:
        require_fitted(self, ["group_models_", "classes_"])
        log_evi = self.score_samples(X, sample_weight=sample_weight)
        pri = np.asarray([gm.prior_logprob for gm in self.group_models_], dtype=float)
        log_post = log_evi + pri[None, :]
        log_post -= logsumexp(log_post, axis=1, keepdims=True)
        return np.exp(log_post)

    def predict(self, X: Any, sample_weight: Any = None) -> np.ndarray:
        require_fitted(self, ["group_models_", "classes_"])
        idx = np.argmax(self.predict_proba(X, sample_weight=sample_weight), axis=1)
        return self.classes_[idx]
