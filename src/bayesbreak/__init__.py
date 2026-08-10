"""Generalized hierarchical Bayesian segmentation via segment evidence and DP.

The public API follows strict scikit-learn conventions:

- Segmenters inherit :class:`sklearn.base.BaseEstimator` and
  :class:`sklearn.base.RegressorMixin`. ``fit(X, y)``, ``predict(X)``,
  ``score(X, y)``, ``transform(X)``.
- Classifier-style wrappers (known / latent groups) inherit
  :class:`sklearn.base.ClassifierMixin` and expose ``predict_proba``.
- All constructor arguments are stored untouched; validation happens in
  ``fit``.

See the accompanying manuscript under ``docs/manuscript/`` for the mathematical
background; this module is the reference implementation.
"""

from __future__ import annotations

from typing import Any

from ._version import __version__  # noqa: F401
from .base import BayesBreakSegmenter
from .diagnostics import (
    DiagnosticCheck,
    DiagnosticReport,
    run_dp_diagnostics,
    run_non_conjugate_diagnostics,
    run_prior_sensitivity,
    select_n_groups_by_holdout,
)
from .families import (
    BayesBreakBernoulli,
    BayesBreakBeta,
    BayesBreakBetaObs,
    BayesBreakBinomial,
    BayesBreakGaussian,
    BayesBreakLogisticNormal,
    BayesBreakNegBin,
    BayesBreakPoisson,
)
from .groups import BayesBreakGroupedClassifier
from .mixture import BayesBreakMixtureClassifier
from .multivariate import (
    IndependentMultivariateSegmenter,
    SharedBoundaryMultivariateSegmenter,
)
from .priors import PartitionPriorConfig
from .replicates import SharedBoundaryReplicatesSegmenter
from .sliding_window import SlidingWindowSegmenter

_FAMILY_REGISTRY = {
    ("gaussian", "normal"): BayesBreakGaussian,
    ("poisson", "count"): BayesBreakPoisson,
    ("binomial", "beta-binomial"): BayesBreakBinomial,
    ("beta", "fractional"): BayesBreakBeta,
    ("beta-obs", "beta_obs", "betaobservation", "beta-observation"): BayesBreakBetaObs,
    ("bernoulli", "binary"): BayesBreakBernoulli,
    (
        "logistic-normal",
        "logistic_normal",
        "logit-normal",
        "logit_normal",
    ): BayesBreakLogisticNormal,
    ("negbin", "negative-binomial", "negative_binomial", "nb"): BayesBreakNegBin,
}


def make_bayesbreak(family: str, **kwargs: Any) -> BayesBreakSegmenter:
    """Instantiate a BayesBreak segmenter by family name.

    Parameters
    ----------
    family : str
        One of ``{"gaussian", "poisson", "binomial", "beta", "beta-obs",
        "bernoulli", "logistic-normal"}`` (plus aliases; see source).
    **kwargs
        Forwarded to the estimator constructor.

    Raises
    ------
    ValueError
        If the family name is unknown.
    """

    key = family.strip().lower()
    for aliases, cls in _FAMILY_REGISTRY.items():
        if key in aliases:
            return cls(**kwargs)
    valid = sorted({a for aliases in _FAMILY_REGISTRY for a in aliases})
    raise ValueError(f"Unknown family={family!r}. Valid families: {valid}")


__all__ = [
    "BayesBreakSegmenter",
    "BayesBreakGaussian",
    "BayesBreakPoisson",
    "BayesBreakBinomial",
    "BayesBreakBeta",
    "BayesBreakBetaObs",
    "BayesBreakBernoulli",
    "BayesBreakLogisticNormal",
    "BayesBreakNegBin",
    "SharedBoundaryMultivariateSegmenter",
    "IndependentMultivariateSegmenter",
    "SharedBoundaryReplicatesSegmenter",
    "SlidingWindowSegmenter",
    "BayesBreakGroupedClassifier",
    "BayesBreakMixtureClassifier",
    "DiagnosticCheck",
    "DiagnosticReport",
    "PartitionPriorConfig",
    "run_dp_diagnostics",
    "run_non_conjugate_diagnostics",
    "run_prior_sensitivity",
    "select_n_groups_by_holdout",
    "make_bayesbreak",
    "__version__",
]
