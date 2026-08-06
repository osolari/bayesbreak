"""Interface-only BayesBreak skeleton.

No scientific algorithm is implemented in this module. See the canonical
coding handoff and the referenced CODE-BB task before adding behavior.
"""
from __future__ import annotations

def posterior_predictive_logpdf_block(*args: object, **kwargs: object) -> float:
    """Beta-observation predictive interface; Gaussian fallback is prohibited."""
    raise NotImplementedError("CODE-BB-006: Beta-observation posterior prediction is not implemented in the skeleton")
