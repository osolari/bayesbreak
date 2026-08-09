"""Interface-only BayesBreak skeleton.

No scientific algorithm is implemented in this module. See the canonical
coding handoff and the referenced CODE-BB task before adding behavior.
"""
from __future__ import annotations
from typing import Protocol
from bayesbreak.base import SegmentModel


class ConjugateSegmentFamily(SegmentModel, Protocol):
    """Marker protocol for analytically integrated segment families."""
