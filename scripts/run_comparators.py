"""Validate raw comparator inputs before algorithm dispatch or metric computation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bayesbreak.comparators import ComparatorInputSchema, TuningBudget


def build_comparator_request(
    values: np.ndarray,
    coordinate_axis: np.ndarray,
    *,
    task_type: str,
    parameter_evaluations: int,
    selection_rule: str,
    data_access: str,
    tuning_stratum: str,
    dataset: str,
) -> ComparatorInputSchema:
    """Construct a validated raw-observation comparator request."""

    return ComparatorInputSchema(
        values=values,
        coordinate_axis=coordinate_axis,
        task_type=task_type,
        tuning_budget=TuningBudget(
            parameter_evaluations=parameter_evaluations,
            selection_rule=selection_rule,
            data_access=data_access,
            tuning_stratum=tuning_stratum,
        ),
        metadata={"source_kind": "raw-observations", "dataset": dataset},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--values", type=Path, required=True)
    parser.add_argument("--coordinates", type=Path, required=True)
    parser.add_argument("--task-type", choices=("univariate", "multisequence"), required=True)
    parser.add_argument("--parameter-evaluations", type=int, required=True)
    parser.add_argument("--selection-rule", required=True)
    parser.add_argument("--data-access", required=True)
    parser.add_argument("--tuning-stratum", required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    request = build_comparator_request(
        np.load(args.values, allow_pickle=False),
        np.load(args.coordinates, allow_pickle=False),
        task_type=args.task_type,
        parameter_evaluations=args.parameter_evaluations,
        selection_rule=args.selection_rule,
        data_access=args.data_access,
        tuning_stratum=args.tuning_stratum,
        dataset=args.dataset,
    )
    values = np.asarray(request.values)
    print(
        json.dumps(
            {
                "dataset": request.metadata["dataset"],
                "source_kind": request.metadata["source_kind"],
                "task_type": request.task_type,
                "values_shape": list(values.shape),
                "coordinate_count": len(request.coordinate_axis),
                "tuning_stratum": request.tuning_budget.tuning_stratum,
                "validation_status": "validated-before-dispatch",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
