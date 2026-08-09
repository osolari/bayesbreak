#!/usr/bin/env python3
"""Validate repository-skeleton structure without claiming scientific completion."""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
REPORT = PROJECT / "revision_artifacts/phase6/REPOSITORY_SKELETON_VALIDATION.json"
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

REQUIRED_MODULES = [
    "bayesbreak",
    "bayesbreak.base",
    "bayesbreak.block_api",
    "bayesbreak.priors",
    "bayesbreak.design_prior",
    "bayesbreak.dp",
    "bayesbreak.map",
    "bayesbreak.posterior",
    "bayesbreak.replicates",
    "bayesbreak.groups",
    "bayesbreak.mixture",
    "bayesbreak.prediction",
    "bayesbreak.metrics",
    "bayesbreak.comparators",
    "bayesbreak.provenance",
    "bayesbreak.families.beta_obs",
    "bayesbreak.nonconjugate.error_bounds",
    "bayesbreak.datasets.base",
]
REQUIRED_SCHEMAS = [
    "result_sidecar.schema.json",
    "boundary_metric.schema.json",
    "approximation_error_record.schema.json",
]


def main() -> int:
    errors: list[str] = []
    for name in REQUIRED_MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"module import failed: {name}: {exc}")

    status = json.loads((ROOT / "IMPLEMENTATION_STATUS.json").read_text())
    if status.get("scientific_implementation_complete") is not False:
        errors.append("scientific_implementation_complete must be false")
    task_ids = [item["task_id"] for item in status.get("tasks", [])]
    expected_tasks = [f"CODE-BB-{index:03d}" for index in range(1, 17)]
    if task_ids != expected_tasks:
        errors.append(f"task IDs differ: {task_ids}")

    for name in REQUIRED_SCHEMAS:
        path = ROOT / "schemas" / name
        if not path.exists():
            errors.append(f"missing schema: {name}")
        else:
            json.loads(path.read_text())

    registry = json.loads((ROOT / "experiments/registry.json").read_text())
    experiment_ids = [item["id"] for item in registry.get("experiments", [])]
    expected_experiments = [f"EPR-BB-{index:03d}" for index in range(1, 16)]
    if experiment_ids != expected_experiments:
        errors.append("experiment registry IDs are incomplete or out of order")

    report = {
        "passed": not errors,
        "errors": errors,
        "module_count": len(REQUIRED_MODULES),
        "task_count": len(task_ids),
        "experiment_count": len(experiment_ids),
        "schema_count": len(REQUIRED_SCHEMAS),
        "scientific_implementation_complete": status.get("scientific_implementation_complete"),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")

    if errors:
        print("SKELETON CHECK FAILED")
        for error in errors:
            print("-", error)
        return 1
    print("SKELETON CHECK PASSED")
    print(
        f"modules={len(REQUIRED_MODULES)} tasks={len(task_ids)} "
        f"experiments={len(experiment_ids)} schemas={len(REQUIRED_SCHEMAS)}"
    )
    print("scientific_implementation_complete=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
