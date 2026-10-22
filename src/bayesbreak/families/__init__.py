"""Likelihood families shipped with BayesBreak.

Families are implemented as subclasses of :class:`bayesbreak.base.BayesBreakBase`.
Most families provide closed-form (conjugate) block evidences; a small set of
non-conjugate blocks are provided via deterministic approximations and/or
numerical quadrature.
"""

from .bernoulli import BayesBreakBernoulli
from .beta import BayesBreakBeta
from .beta_obs import BayesBreakBetaObs
from .binomial import BayesBreakBinomial
from .gaussian import BayesBreakGaussian
from .logistic_normal import BayesBreakLogisticNormal
from .negative_binomial import BayesBreakNegBin
from .poisson import BayesBreakPoisson

__all__ = [
    "BayesBreakGaussian",
    "BayesBreakPoisson",
    "BayesBreakBinomial",
    "BayesBreakBeta",
    "BayesBreakBetaObs",
    "BayesBreakBernoulli",
    "BayesBreakLogisticNormal",
    "BayesBreakNegBin",
]
