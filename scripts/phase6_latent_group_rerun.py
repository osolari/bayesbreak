"""Pilot and execute the authorized EPR-BB-005 corrected latent-group rerun."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import resource
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from bayesbreak import BayesBreakGaussian, BayesBreakMixtureClassifier

RESULT_ID = "RES-BB-SYN-005"
PARENT_RESULT_ID = "RES-BB-SYN-002"
PROTOCOL_ID = "EPR-BB-005"


@dataclass(frozen=True)
class StressCell:
    name: str
    n_sequences: int = 24
    n: int = 80
    sigma: float = 1.0
    separation: float = 1.0
    group0_fraction: float = 0.5
    n_groups_fit: int = 2
    duplicate_templates: bool = False


STRESS_CELLS = (
    StressCell("archived"),
    StressCell("fewer-sequences", n_sequences=12),
    StressCell("shorter-sequences", n=40),
    StressCell("higher-noise", sigma=1.5),
    StressCell("lower-separation", separation=0.5),
    StressCell("imbalanced", group0_fraction=0.75),
    StressCell("overspecified-g", n_groups_fit=3),
    StressCell("duplicate-templates", duplicate_templates=True),
)


def generate_dataset(cell: StressCell, seed: int) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    rng = np.random.default_rng(seed)
    boundaries0 = [0, cell.n // 3, 2 * cell.n // 3, cell.n]
    boundaries1 = (
        list(boundaries0) if cell.duplicate_templates else [0, cell.n // 4, 3 * cell.n // 4, cell.n]
    )
    levels0 = np.asarray([0.0, 1.0, -0.5]) * cell.separation
    levels1 = (
        levels0.copy()
        if cell.duplicate_templates
        else np.asarray([0.5, -1.0, 0.8]) * cell.separation
    )
    mean0 = piecewise_mean(cell.n, boundaries0, levels0)
    mean1 = piecewise_mean(cell.n, boundaries1, levels1)
    n_group0 = min(
        cell.n_sequences - 1,
        max(1, int(round(cell.n_sequences * cell.group0_fraction))),
    )
    labels = np.asarray(
        [0] * n_group0 + [1] * (cell.n_sequences - n_group0),
        dtype=int,
    )
    rng.shuffle(labels)
    values = np.stack(
        [
            (mean0 if label == 0 else mean1) + cell.sigma * rng.standard_normal(cell.n)
            for label in labels
        ]
    )
    return values, labels, [boundaries0, boundaries1]


def piecewise_mean(n: int, boundaries: list[int], levels: np.ndarray) -> np.ndarray:
    output = np.empty(n, dtype=float)
    for start, stop, level in zip(boundaries[:-1], boundaries[1:], levels, strict=True):
        output[start:stop] = level
    return output


def run_dataset(cell: StressCell, data_seed: int, optimizer_seed: int) -> dict[str, object]:
    values, labels, true_templates = generate_dataset(cell, data_seed)
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    estimator = BayesBreakMixtureClassifier(
        BayesBreakGaussian(k_max=min(8, cell.n)),
        n_groups=cell.n_groups_fit,
        k_max=min(8, cell.n),
        max_iter=8,
        tol=1e-4,
        random_state=optimizer_seed,
        n_restarts=3,
    ).fit(values)
    elapsed_wall = time.perf_counter() - start_wall
    elapsed_cpu = time.process_time() - start_cpu
    predicted = np.argmax(estimator.responsibilities_, axis=1)
    ari = float(adjusted_rand_score(labels, predicted))
    hard_accuracy = label_invariant_accuracy(labels, predicted)
    template_distance, matched_templates = match_template_distance(
        [state.template for state in estimator.group_states_],
        true_templates,
    )
    effective_counts = estimator.responsibilities_.sum(axis=0)
    collapsed = bool(
        np.any(effective_counts < 1.0)
        or len({tuple(state.template) for state in estimator.group_states_})
        < len(estimator.group_states_)
    )
    restart_records = [asdict(record) for record in estimator.restart_diagnostics_]
    objective_trace = [float(value) for value in estimator.objective_trace_]
    monotone = all(
        current >= previous - max(1e-10, estimator.tol * max(1.0, abs(previous)))
        for previous, current in zip(objective_trace, objective_trace[1:], strict=False)
    )
    state_payload = {
        "templates": [state.template for state in estimator.group_states_],
        "pi": estimator.pi_.tolist(),
        "responsibilities": estimator.responsibilities_.tolist(),
    }
    return {
        "cell": cell.name,
        "data_seed": data_seed,
        "optimizer_seed": optimizer_seed,
        "ari": ari,
        "hard_accuracy": hard_accuracy,
        "template_distance": template_distance,
        "matched_templates": matched_templates,
        "collapsed": collapsed,
        "objective_monotone": monotone,
        "objective_trace": objective_trace,
        "final_objective": float(estimator.final_objective_),
        "selected_restart": int(estimator.selected_restart_),
        "restart_diagnostics": restart_records,
        "effective_group_counts": effective_counts.tolist(),
        "elapsed_wall_seconds": elapsed_wall,
        "elapsed_cpu_seconds": elapsed_cpu,
        "state_hash": sha256_json(state_payload),
        "data_hash": sha256_arrays(values, labels),
    }


def label_invariant_accuracy(truth: np.ndarray, predicted: np.ndarray) -> float:
    truth_labels = np.unique(truth)
    predicted_labels = np.unique(predicted)
    counts = np.zeros((truth_labels.size, predicted_labels.size), dtype=int)
    for row, truth_label in enumerate(truth_labels):
        for column, predicted_label in enumerate(predicted_labels):
            counts[row, column] = int(
                np.sum((truth == truth_label) & (predicted == predicted_label))
            )
    rows, columns = linear_sum_assignment(-counts)
    return float(counts[rows, columns].sum() / truth.size)


def match_template_distance(
    fitted: list[list[int]],
    truth: list[list[int]],
) -> tuple[float | None, list[dict[str, object]]]:
    if not fitted or not truth:
        return None, []
    costs = np.zeros((len(fitted), len(truth)), dtype=float)
    for row, fitted_template in enumerate(fitted):
        fitted_inner = np.asarray(fitted_template[1:-1], dtype=float)
        for column, true_template in enumerate(truth):
            true_inner = np.asarray(true_template[1:-1], dtype=float)
            if fitted_inner.size == 0 or true_inner.size == 0:
                costs[row, column] = float(max(fitted_inner.size, true_inner.size))
            else:
                forward = np.mean(
                    np.min(np.abs(fitted_inner[:, None] - true_inner[None, :]), axis=1)
                )
                reverse = np.mean(
                    np.min(np.abs(true_inner[:, None] - fitted_inner[None, :]), axis=1)
                )
                count_penalty = abs(fitted_inner.size - true_inner.size)
                costs[row, column] = float(0.5 * (forward + reverse) + count_penalty)
    rows, columns = linear_sum_assignment(costs)
    matches = [
        {
            "fitted_group": int(row),
            "truth_group": int(column),
            "distance": float(costs[row, column]),
        }
        for row, column in zip(rows, columns, strict=True)
    ]
    return float(np.mean([record["distance"] for record in matches])), matches


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    cells: dict[str, dict[str, object]] = {}
    for cell in STRESS_CELLS:
        subset = [record for record in records if record["cell"] == cell.name]
        if not subset:
            continue
        cells[cell.name] = {
            "n_runs": len(subset),
            "ari": interval_summary([float(record["ari"]) for record in subset]),
            "hard_accuracy": interval_summary(
                [float(record["hard_accuracy"]) for record in subset]
            ),
            "template_distance": interval_summary(
                [float(record["template_distance"]) for record in subset]
            ),
            "collapse_rate": float(np.mean([bool(record["collapsed"]) for record in subset])),
            "objective_monotone_rate": float(
                np.mean([bool(record["objective_monotone"]) for record in subset])
            ),
            "valid_restart_rate": float(
                np.mean(
                    [
                        diagnostic["status"] == "valid"
                        for record in subset
                        for diagnostic in record["restart_diagnostics"]
                    ]
                )
            ),
            "wall_seconds": interval_summary(
                [float(record["elapsed_wall_seconds"]) for record in subset]
            ),
        }
    return {"cells": cells}


def interval_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    standard_error = float(np.std(array, ddof=1) / math.sqrt(array.size)) if array.size > 1 else 0.0
    return {
        "mean": mean,
        "standard_error": standard_error,
        "ci95_lower": mean - 1.96 * standard_error,
        "ci95_upper": mean + 1.96 * standard_error,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def sha256_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def current_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def peak_rss() -> dict[str, object]:
    return {
        "value": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "units": "bytes" if platform.system() == "Darwin" else "KiB",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--seed", type=int, default=260805)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")

    cells = STRESS_CELLS[:1] if args.mode == "pilot" else STRESS_CELLS
    repetitions = 1 if args.mode == "pilot" else args.repetitions
    run_start = time.perf_counter()
    records: list[dict[str, object]] = []
    for cell_index, cell in enumerate(cells):
        for repetition in range(repetitions):
            data_seed = args.seed + 10_000 * cell_index + repetition
            optimizer_seed = args.seed + 1_000_000 + 10_000 * cell_index + repetition
            records.append(run_dataset(cell, data_seed, optimizer_seed))
    elapsed = time.perf_counter() - run_start
    projected_runs = len(STRESS_CELLS) * args.repetitions
    output = {
        "schema_version": "1.0.0",
        "mode": args.mode,
        "result_id": RESULT_ID,
        "parent_result_id": PARENT_RESULT_ID,
        "protocol_id": PROTOCOL_ID,
        "code_commit": current_commit(),
        "seed_base": args.seed,
        "repetitions_per_cell": repetitions,
        "cells": [asdict(cell) for cell in cells],
        "records": records,
        "summary": summarize(records),
        "resources": {
            "elapsed_wall_seconds": elapsed,
            "peak_rss": peak_rss(),
            "output_bytes": 0,
            "projected_full_runs": projected_runs,
            "projected_full_wall_seconds": elapsed / len(records) * projected_runs,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    output["resources"]["output_bytes"] = args.output.stat().st_size
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **output["resources"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
