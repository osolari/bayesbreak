"""BayesBreak: Bayesian piecewise-constant regression via dynamic programming.

Public API
----------
The package exposes a distribution-agnostic base class
:class:`bayesbreak.base.BayesBreakBase` and several conjugate families:

- :class:`bayesbreak.families.gaussian.BayesBreakGaussian`
- :class:`bayesbreak.families.poisson.BayesBreakPoisson`
- :class:`bayesbreak.families.binomial.BayesBreakBinomial`
- :class:`bayesbreak.families.beta.BayesBreakBeta`

For convenience and backward compatibility, :class:`~bayesbreak.BayesBreak`
aliases :class:`~bayesbreak.families.gaussian.BayesBreakGaussian`.
"""

from __future__ import annotations

from typing import Any, Dict

from .base import BayesBreakBase
from .families import BayesBreakBeta, BayesBreakBinomial, BayesBreakGaussian, BayesBreakPoisson

# Backward-compatible alias used in earlier drafts.
BayesBreak = BayesBreakGaussian


def make_bayesbreak(family: str, **kwargs: Any) -> BayesBreakBase:
    """Create a BayesBreak estimator by family name.

    Parameters
    ----------
    family:
        One of ``{'gaussian', 'poisson', 'binomial', 'beta'}``.

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
    raise ValueError(f"Unknown family={family!r}. Expected one of: gaussian, poisson, binomial, beta.")


# Alias retained for readability in user code and unit tests.
make_model = make_bayesbreak


__all__ = [
    "BayesBreakBase",
    "BayesBreakGaussian",
    "BayesBreakPoisson",
    "BayesBreakBinomial",
    "BayesBreakBeta",
    "BayesBreak",
    "make_bayesbreak",
    "make_model",
]
