"""BayesBreak: Bayesian piecewise-constant regression via dynamic programming.

Public API
----------
The package exposes a distribution-agnostic base class
:class:`bayesbreak.base.BayesBreakBase` and several conjugate families:

- :class:`bayesbreak.families.gaussian.BayesBreakGaussian`
- :class:`bayesbreak.families.poisson.BayesBreakPoisson`
- :class:`bayesbreak.families.binomial.BayesBreakBinomial`
- :class:`bayesbreak.families.beta.BayesBreakBeta`
- :class:`bayesbreak.families.bernoulli.BayesBreakBernoulli`

For convenience and backward compatibility, :class:`~bayesbreak.BayesBreak`
aliases :class:`~bayesbreak.families.gaussian.BayesBreakGaussian`.
"""

from __future__ import annotations

from typing import Any

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
from .mixture import BayesBreakMixture
from .multivariate import BayesBreakMultivariate

# Backward-compatible alias used in earlier drafts.
BayesBreak = BayesBreakGaussian


def make_bayesbreak(family: str, **kwargs: Any) -> BayesBreakBase:
    """Create a BayesBreak estimator by family name.

    Parameters
    ----------
    family:
        One of
        ``{'gaussian', 'poisson', 'binomial', 'beta', 'bernoulli', 'logistic-normal', 'beta-obs'}``.

    **kwargs:
        Passed to the corresponding estimator constructor.

    Returns
    -------
    BayesBreakBase
        Instantiated estimator.

    Raises
    ------
    ValueError
        If ``family`` is unrecognized.
    """

    key = family.strip().lower()
    if key in {"gaussian", "normal"}:
        return BayesBreakGaussian(**kwargs)
    if key in {"poisson", "count"}:
        return BayesBreakPoisson(**kwargs)
    if key in {"binomial", "beta-binomial"}:
        return BayesBreakBinomial(**kwargs)
    if key in {"beta", "fractional"}:
        return BayesBreakBeta(**kwargs)
    if key in {"beta-obs", "beta_obs", "betaobservation", "beta-observation"}:
        return BayesBreakBetaObs(**kwargs)
    if key in {"bernoulli", "binary", "logistic"}:
        # "logistic" is accepted as a common shorthand for binary sequences.
        return BayesBreakBernoulli(**kwargs)
    if key in {"logistic-normal", "logistic_normal", "logit-normal", "logit_normal"}:
        return BayesBreakLogisticNormal(**kwargs)
    raise ValueError(
        "Unknown family=%r. Expected one of: gaussian, poisson, binomial, beta, beta-obs, "
        "bernoulli, logistic-normal." % (family,)
    )


# Alias retained for readability in user code and unit tests.
make_model = make_bayesbreak


__all__ = [
    "BayesBreakBase",
    "BayesBreakGaussian",
    "BayesBreakPoisson",
    "BayesBreakBinomial",
    "BayesBreakBeta",
    "BayesBreakBetaObs",
    "BayesBreakBernoulli",
    "BayesBreakLogisticNormal",
    "BayesBreakMultivariate",
    "BayesBreakGrouped",
    "BayesBreakMixture",
    "BayesBreak",
    "make_bayesbreak",
    "make_model",
]
