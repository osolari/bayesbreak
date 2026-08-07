"""Pilot and execute the authorized EPR-BB-010/EPR-BB-013 CGH rerun."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import resource
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

import bayesbreak
from bayesbreak import BayesBreakGaussian, SharedBoundaryReplicatesSegmenter
from bayesbreak.baselines import run_binseg, run_dynp, run_pelt, run_wbs
from bayesbreak.baselines._types import BaselineResult
from bayesbreak.comparators import (
    FAILURE_ID_AXIS_MISMATCH,
    ComparatorInputSchema,
    TuningBudget,
)
from bayesbreak.datasets.base import DatasetCard, load_with_provenance
from bayesbreak.metrics import boundary_metrics

RESULT_ID = "RES-BB-CMP-003"
PARENT_RESULT_ID = "RES-BB-CMP-002"
PROTOCOL_IDS = ("EPR-BB-010", "EPR-BB-013")
SOURCE_URI = "https://github.com/cran/ecp/raw/master/data/ACGH.RData"
EXPECTED_SOURCE_SHA256 = "b82da97ffe6b5c431a60c3f811ee5c339708562126ed4a3d0b0344f2f2e09a63"
EXPECTED_MATRIX_SHA256 = "1551547d50564227ac020c196e56cae0b29c76ccf76eb4140d47628019cdb9a8"
EXPECTED_CARD_DATA_HASH = "9c368d425ad3a2246f7a7ea10e976894c840b232c3c51ff29153da1b67024787"
EXPECTED_DESCRIPTOR_HASH = "b47230bd390a41413b0e57ae4742aa247a9a85594884d2fa877ca6c6b6fbbc90"
EXPECTED_SHAPE = (2215, 43)
EXPECTED_SUBJECT_IDS = (
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    14,
    15,
    16,
    17,
    18,
    19,
    21,
    22,
    24,
    26,
    28,
    30,
    31,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    53,
    54,
    57,
)
PELT_MULTIPLIERS = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def sha256_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_source(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Array-CGH source does not exist: {path}")
    source_hash = sha256_file(path)
    if source_hash != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            f"{FAILURE_ID_AXIS_MISMATCH}: source hash {source_hash} does not match "
            f"the authorized ACGH source {EXPECTED_SOURCE_SHA256}"
        )
    return source_hash


def parse_source(path: Path) -> tuple[np.ndarray, tuple[int, ...]]:
    import rdata

    converted = rdata.conversion.convert(rdata.parser.parse_file(path))
    acgh = converted.get("ACGH")
    if not isinstance(acgh, dict) or "data" not in acgh or "individual" not in acgh:
        raise RuntimeError(f"{FAILURE_ID_AXIS_MISMATCH}: ACGH source fields are missing")
    data_object = acgh["data"]
    matrix = np.asarray(getattr(data_object, "values", data_object), dtype=float)
    subject_ids = tuple(int(value) for value in np.asarray(acgh["individual"]).ravel())
    validate_exact_matrix(matrix, subject_ids)
    return np.ascontiguousarray(matrix), subject_ids


def validate_exact_matrix(matrix: np.ndarray, subject_ids: tuple[int, ...]) -> None:
    if matrix.shape != EXPECTED_SHAPE:
        raise RuntimeError(
            f"{FAILURE_ID_AXIS_MISMATCH}: expected probes-by-subjects shape "
            f"{EXPECTED_SHAPE}, got {matrix.shape}"
        )
    if subject_ids != EXPECTED_SUBJECT_IDS:
        raise RuntimeError(f"{FAILURE_ID_AXIS_MISMATCH}: subject identifiers changed")
    if not np.all(np.isfinite(matrix)):
        raise RuntimeError(f"{FAILURE_ID_AXIS_MISMATCH}: raw ACGH matrix is nonfinite")
    matrix_hash = sha256_array(matrix)
    if matrix_hash != EXPECTED_MATRIX_SHA256:
        raise RuntimeError(
            f"{FAILURE_ID_AXIS_MISMATCH}: parsed matrix hash {matrix_hash} does not match "
            f"{EXPECTED_MATRIX_SHA256}"
        )


def validate_loader(
    source_matrix: np.ndarray,
) -> tuple[np.ndarray, DatasetCard]:
    bundle, card = load_with_provenance("cgh")
    loaded_matrix = np.asarray(bundle.y, dtype=float)
    weights = np.asarray(bundle.sample_weight, dtype=float)
    if bundle.source != "downloaded" or loaded_matrix.shape != EXPECTED_SHAPE:
        raise RuntimeError(f"{FAILURE_ID_AXIS_MISMATCH}: production loader did not return raw ACGH")
    if not np.array_equal(loaded_matrix, source_matrix):
        raise RuntimeError(f"{FAILURE_ID_AXIS_MISMATCH}: production loader changed ACGH values")
    if weights.shape != EXPECTED_SHAPE or not np.all(np.isfinite(weights)):
        raise RuntimeError(f"{FAILURE_ID_AXIS_MISMATCH}: Gaussian precision shape is invalid")
    if np.any(weights <= 0):
        raise RuntimeError(f"{FAILURE_ID_AXIS_MISMATCH}: Gaussian precisions must be positive")
    if card.data_hash != EXPECTED_CARD_DATA_HASH:
        raise RuntimeError(f"{FAILURE_ID_AXIS_MISMATCH}: dataset-card hash changed")
    if card.descriptor_hash != EXPECTED_DESCRIPTOR_HASH:
        raise RuntimeError(f"{FAILURE_ID_AXIS_MISMATCH}: precision hash changed")
    return np.ascontiguousarray(weights), card


def build_raw_request(
    probe_by_subject: np.ndarray,
    probe_axis: np.ndarray,
) -> ComparatorInputSchema:
    matrix = np.asarray(probe_by_subject, dtype=float)
    axis = np.asarray(probe_axis, dtype=float)
    return ComparatorInputSchema(
        values=np.ascontiguousarray(matrix.T),
        coordinate_axis=axis,
        task_type="multisequence",
        tuning_budget=TuningBudget(
            parameter_evaluations=len(PELT_MULTIPLIERS),
            selection_rule=(
                "matched-k diagnostic: fixed n_bkps for Dynp/Binseg/WBS; PELT grid selected "
                "only by absolute distance from the shared BayesBreak MAP count"
            ),
            data_access="full raw matrix for descriptive matched-k agreement",
            tuning_stratum="matched-k-agreement-only",
        ),
        metadata={"source_kind": "raw-observations", "dataset": "cgh"},
    )


def algorithm_signal(request: ComparatorInputSchema) -> np.ndarray:
    subject_by_probe = np.asarray(request.values, dtype=float)
    signal = np.ascontiguousarray(subject_by_probe.T)
    if signal.shape[0] != len(request.coordinate_axis):
        raise RuntimeError(f"{FAILURE_ID_AXIS_MISMATCH}: algorithm signal axis changed")
    return signal


def fit_bayesbreak_reference(
    request: ComparatorInputSchema,
    subject_by_probe_weights: np.ndarray,
    *,
    k_max: int,
) -> dict[str, Any]:
    values = np.asarray(request.values, dtype=float)
    weights = np.asarray(subject_by_probe_weights, dtype=float)
    axis = np.asarray(request.coordinate_axis, dtype=float).reshape(-1, 1)
    if weights.shape != values.shape:
        raise ValueError(f"weights shape {weights.shape} does not match values {values.shape}")

    shared_start = time.perf_counter()
    shared = SharedBoundaryReplicatesSegmenter(
        BayesBreakGaussian(k_max=k_max, regression_curve="none")
    ).fit(axis, values, sample_weight=weights)
    shared_wall = time.perf_counter() - shared_start

    independent_start = time.perf_counter()
    independent: list[dict[str, Any]] = []
    for subject_index, (subject_values, subject_weights) in enumerate(
        zip(values, weights, strict=True)
    ):
        estimator = BayesBreakGaussian(k_max=k_max).fit(
            axis,
            subject_values,
            sample_weight=subject_weights,
        )
        independent.append(
            {
                "subject_index": subject_index,
                "k_map": int(estimator.k_map_),
                "boundaries": [int(value) for value in estimator.map_boundaries_[1:-1]],
                "log_evidence": float(estimator.log_evidence_),
            }
        )
    independent_wall = time.perf_counter() - independent_start
    return {
        "shared": {
            "estimator": "SharedBoundaryReplicatesSegmenter[BayesBreakGaussian]",
            "k_max": k_max,
            "k_map": int(shared.k_map_),
            "boundaries": [int(value) for value in shared.map_boundaries_[1:-1]],
            "log_evidence": float(shared.log_evidence_),
            "wall_seconds": shared_wall,
        },
        "independent": independent,
        "independent_wall_seconds": independent_wall,
    }


def exact_boundary_jaccard(predicted: list[int], reference: list[int]) -> float:
    predicted_set = set(predicted)
    reference_set = set(reference)
    union = predicted_set | reference_set
    return len(predicted_set & reference_set) / len(union) if union else 1.0


def comparator_record(
    result: BaselineResult,
    reference: list[int],
    *,
    runtime: float,
    tuning: dict[str, Any],
) -> dict[str, Any]:
    predicted = [int(value) for value in result.boundaries]
    metric = boundary_metrics(
        predicted,
        reference,
        tolerance=3,
        reference_type="model-derived-map",
        prediction_axis="probe-index",
        reference_axis="probe-index",
    )
    return {
        "algorithm": result.algorithm,
        "package": result.package,
        "package_version": result.package_version,
        "n": result.n,
        "k_hat": result.k,
        "boundaries": predicted,
        "runtime_seconds": runtime,
        "tuning": tuning,
        "exact_boundary_jaccard": exact_boundary_jaccard(predicted, reference),
        "boundary_metrics_tau3": metric.to_dict(),
    }


def choose_pelt_candidate(candidates: list[dict[str, Any]], target_n_bkps: int) -> dict[str, Any]:
    if not candidates:
        raise ValueError("At least one PELT candidate is required")
    return min(
        candidates,
        key=lambda candidate: (
            abs(int(candidate["n_bkps"]) - target_n_bkps),
            abs(math.log2(float(candidate["multiplier"]))),
            float(candidate["penalty"]),
        ),
    )


def run_comparators(
    request: ComparatorInputSchema,
    reference: list[int],
    *,
    random_seed: int,
) -> list[dict[str, Any]]:
    signal = algorithm_signal(request)
    n_bkps = len(reference)
    common = {"cost_model": "l2", "min_size": 2, "jump": 1}
    records: list[dict[str, Any]] = []

    pelt_scale = max(
        float(np.var(signal, axis=0, ddof=1).sum() * math.log(signal.shape[0])),
        np.finfo(float).eps,
    )
    pelt_candidates: list[dict[str, Any]] = []
    pelt_results: list[BaselineResult] = []
    for multiplier in PELT_MULTIPLIERS:
        penalty = pelt_scale * multiplier
        start = time.perf_counter()
        result = run_pelt(signal, penalty=penalty, **common)
        runtime = time.perf_counter() - start
        pelt_results.append(result)
        pelt_candidates.append(
            {
                "multiplier": multiplier,
                "penalty": penalty,
                "n_bkps": int(result.boundaries.size),
                "runtime_seconds": runtime,
            }
        )
    selected = choose_pelt_candidate(pelt_candidates, n_bkps)
    selected_index = pelt_candidates.index(selected)
    records.append(
        comparator_record(
            pelt_results[selected_index],
            reference,
            runtime=float(selected["runtime_seconds"]),
            tuning={
                "selection_rule": "minimum absolute boundary-count distance; no F1 access",
                "target_n_bkps": n_bkps,
                "scale": pelt_scale,
                "candidates": pelt_candidates,
                "selected_index": selected_index,
                **common,
            },
        )
    )

    fixed_runs = (
        ("optimal_partitioning", lambda: run_dynp(signal, n_bkps=n_bkps, **common)),
        ("binary_segmentation", lambda: run_binseg(signal, n_bkps=n_bkps, **common)),
        (
            "wild_binary_segmentation",
            lambda: run_wbs(
                signal,
                n_bkps=n_bkps,
                n_random_windows=100,
                random_state=random_seed,
                **common,
            ),
        ),
    )
    for algorithm, runner in fixed_runs:
        start = time.perf_counter()
        result = runner()
        runtime = time.perf_counter() - start
        tuning: dict[str, Any] = {
            "selection_rule": "fixed to shared BayesBreak MAP boundary count",
            "target_n_bkps": n_bkps,
            **common,
        }
        if algorithm == "wild_binary_segmentation":
            tuning.update({"n_random_windows": 100, "random_state": random_seed})
        records.append(comparator_record(result, reference, runtime=runtime, tuning=tuning))
    return records


def code_revision() -> dict[str, Any]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    relevant_paths = (
        "pyproject.toml",
        "scripts/phase6_cgh_comparator_rerun.py",
        "src/bayesbreak",
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", *relevant_paths],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("CGH rerun code must be committed before scientific execution")
    return {
        "commit": commit,
        "commit_sha256": hashlib.sha256(commit.encode("ascii")).hexdigest(),
        "relevant_paths_clean": True,
    }


def environment_record() -> dict[str, Any]:
    packages = {
        name: importlib.metadata.version(name)
        for name in ("bayesbreak", "numpy", "rdata", "ruptures", "scipy")
    }
    if packages["bayesbreak"] != bayesbreak.__version__:
        raise RuntimeError(
            "Installed BayesBreak distribution metadata does not match the imported module: "
            f"{packages['bayesbreak']} != {bayesbreak.__version__}"
        )
    record = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "bayesbreak_module_version": bayesbreak.__version__,
        "packages": packages,
    }
    return {**record, "sha256": sha256_json(record)}


def peak_rss() -> dict[str, Any]:
    return {
        "value": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "units": "bytes" if platform.system() == "Darwin" else "KiB",
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["resources"]["output_bytes"] = 0
    for _ in range(3):
        path.write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        size = path.stat().st_size
        if payload["resources"]["output_bytes"] == size:
            break
        payload["resources"]["output_bytes"] = size


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path.home() / ".cache" / "bayesbreak" / "ACGH.RData",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=261003)
    parser.add_argument("--pilot-probes", type=int, default=300)
    parser.add_argument("--pilot-subjects", type=int, default=4)
    args = parser.parse_args()
    if not 20 <= args.pilot_probes <= EXPECTED_SHAPE[0]:
        parser.error("--pilot-probes must be between 20 and 2215")
    if not 2 <= args.pilot_subjects <= EXPECTED_SHAPE[1]:
        parser.error("--pilot-subjects must be between 2 and 43")

    revision = code_revision()
    started = time.perf_counter()
    source_path = args.source.expanduser().resolve()
    source_hash = verify_source(source_path)
    full_matrix, subject_ids = parse_source(source_path)
    full_weights, card = validate_loader(full_matrix)

    if args.mode == "pilot":
        n_probes = args.pilot_probes
        n_subjects = args.pilot_subjects
        k_max = min(6, n_probes)
    else:
        n_probes, n_subjects = EXPECTED_SHAPE
        k_max = 15
    matrix = np.ascontiguousarray(full_matrix[:n_probes, :n_subjects])
    weights = np.ascontiguousarray(full_weights[:n_probes, :n_subjects])
    axis = np.arange(n_probes, dtype=float)
    request = build_raw_request(matrix, axis)
    reference = fit_bayesbreak_reference(request, weights.T, k_max=k_max)
    reference_boundaries = list(reference["shared"]["boundaries"])
    comparators = run_comparators(request, reference_boundaries, random_seed=args.seed)
    elapsed = time.perf_counter() - started

    config = {
        "mode": args.mode,
        "n_probes": n_probes,
        "n_subjects": n_subjects,
        "subject_ids": list(subject_ids[:n_subjects]),
        "k_max": k_max,
        "metric_tolerance": 3,
        "metric_reference_type": "model-derived-map",
        "cost_model": "l2",
        "min_size": 2,
        "jump": 1,
        "pelt_multipliers": list(PELT_MULTIPLIERS),
        "wbs_random_windows": 100,
        "random_seed": args.seed,
        "stratum": "matched-k-agreement-only",
    }
    work_ratio = (
        (EXPECTED_SHAPE[0] / n_probes) ** 2 * (EXPECTED_SHAPE[1] / n_subjects) * (15 / k_max)
    )
    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_role": "resource-pilot" if args.mode == "pilot" else "corrected-execution",
        "result_id": RESULT_ID,
        "parent_result_id": PARENT_RESULT_ID,
        "protocol_ids": list(PROTOCOL_IDS),
        "execution_status": "executed",
        "scientific_interpretation": (
            "implementation-verification" if args.mode == "pilot" else "pending-independent-review"
        ),
        "source": {
            "uri": SOURCE_URI,
            "cache_path": "$HOME/.cache/bayesbreak/ACGH.RData",
            "sha256": source_hash,
            "matrix_sha256": sha256_array(full_matrix),
            "matrix_shape": list(full_matrix.shape),
            "subject_ids": list(subject_ids),
        },
        "dataset_card": card.to_dict(),
        "coordinate_metadata": {
            "raw_storage_orientation": "probe-by-subject",
            "validated_request_orientation": "subject-by-probe",
            "algorithm_orientation": "probe-by-subject",
            "prediction_axis": "probe-index",
            "reference_axis": "probe-index",
            "reference_type": "model-derived-map",
            "external_annotations": "none-independently-verified",
        },
        "config": config,
        "config_sha256": sha256_json(config),
        "code": revision,
        "environment": environment_record(),
        "bayesbreak": reference,
        "comparators": comparators,
        "resources": {
            "elapsed_wall_seconds": elapsed,
            "peak_rss": peak_rss(),
            "output_bytes": 0,
            "full_work_projection_formula": "pilot_wall*(2215/n)^2*(43/S)*(15/k_max)",
            "full_work_ratio": work_ratio,
            "projected_full_wall_seconds": elapsed * work_ratio,
        },
        "limitations": [
            "Matched-k agreement uses the BayesBreak MAP count and is not independent tuning.",
            "The model-derived BayesBreak MAP is not external biological ground truth.",
            "No external-accuracy or predictive-superiority claim is supported by this stratum.",
        ],
    }
    write_json(args.output, payload)
    print(json.dumps({"output": str(args.output), **payload["resources"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
