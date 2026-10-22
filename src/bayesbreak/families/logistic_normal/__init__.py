"""Bernoulli data with a logistic-normal segment prior.

Subpackage layout — one file per approximation kernel plus a shared
utility module and the estimator façade:

- :mod:`._util` — sigmoid and MacKay's ``E[\\sigma(Z)]`` approximation.
- :mod:`._laplace` — 1-D Laplace expansion around the posterior mode.
- :mod:`._jj` — Jaakkola--Jordan variational bound (also serves ``pg_vb``).
- :mod:`._gh` — Gauss--Hermite quadrature reference.
- :mod:`._ep` — vectorised parallel EP with warm-started sweeps.
- :mod:`._estimator` — the ``BayesBreakLogisticNormal`` façade.
"""

from ._estimator import Approx, BayesBreakLogisticNormal

__all__ = ["Approx", "BayesBreakLogisticNormal"]
