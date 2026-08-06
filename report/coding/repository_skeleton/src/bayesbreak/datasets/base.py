"""Interface-only BayesBreak skeleton.

No scientific algorithm is implemented in this module. See the canonical
coding handoff and the referenced CODE-BB task before adding behavior.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class DatasetCard:
    dataset_id: str
    source: str
    acquisition_date: str | None
    coordinate_definition: str
    observation_family: str
    n_observations: int
    n_sequences: int
    raw_hash: str
    processed_hash: str
    preprocessing: Sequence[str]
    limitations: Sequence[str]
    extra: Mapping[str, str]


def load_with_provenance(dataset_id: str, config: Mapping[str, object]) -> tuple[object, DatasetCard]:
    raise NotImplementedError("CODE-BB-011: dataset loading with provenance is not implemented in the skeleton")
