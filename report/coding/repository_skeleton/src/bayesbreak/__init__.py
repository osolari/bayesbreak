"""BayesBreak interface-only repository skeleton.

Importing the package exposes records and interfaces, not a completed
segmentation implementation.
"""
from ._status import CANONICAL_HANDOFF_ID, IMPLEMENTATION_STATUS, SCIENTIFIC_IMPLEMENTATION_COMPLETE

__all__ = [
    "CANONICAL_HANDOFF_ID",
    "IMPLEMENTATION_STATUS",
    "SCIENTIFIC_IMPLEMENTATION_COMPLETE",
]
