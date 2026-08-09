"""Interface-only BayesBreak skeleton.

No scientific algorithm is implemented in this module. See the canonical
coding handoff and the referenced CODE-BB task before adding behavior.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ResultRecord:
    result_id: str
    parent_result_id: str | None
    execution_status: str
    scientific_interpretation: str
    data_hash: str
    config_hash: str
    code_hash: str
    environment_hash: str
    metrics: Mapping[str, Any] = field(default_factory=dict)
    artifacts: Mapping[str, str] = field(default_factory=dict)


def validate_result_record(record: ResultRecord) -> None:
    raise NotImplementedError("CODE-BB-010: production result-record validation is not implemented in the skeleton")
