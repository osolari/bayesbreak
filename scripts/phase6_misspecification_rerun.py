"""Pilot and execute the authorized EPR-BB-015 failure-boundary suite."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import pickle
import platform
import resource
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t

import bayesbreak
from bayesbreak import (
    BayesBreakGaussian,
    BayesBreakLogisticNormal,
    BayesBreakPoisson,
    SharedBoundaryReplicatesSegmenter,
)
from bayesbreak.diagnostics import run_non_conjugate_diagnostics
from bayesbreak.metrics import boundary_metrics
from bayesbreak.priors import PartitionPriorConfig

RESULT_ID = "RES-BB-SYN-006"
PROTOCOL_ID = "EPR-BB-015"
PLAN_PATH = Path("provenance/epr-bb-015-plan.json")


def load_plan(root: Path) -> dict[str, Any]:
    payload = json.loads((root / PLAN_PATH).read_text(encoding="utf-8"))
    if payload["protocol_id"] != PROTOCOL_ID or payload["planned_result_id"] != RESULT_ID:
        raise ValueError("Unexpected EPR-BB-015 plan identity")
    return payload


def piecewise_mean(lengths: list[int], means: list[float]) -> tuple[np.ndarray, list[int]]:
    if len(lengths) != len(means) or any(length < 1 for length in lengths):
        raise ValueError("Positive segment lengths must match segment means")
    values = np.concatenate(
        [np.full(length, mean) for length, mean in zip(lengths, means, strict=True)]
    )
    boundaries = [0]
    for length in lengths:
        boundaries.append(boundaries[-1] + length)
    return values, boundaries


def generate_standard_cell(cell_id: str, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    if cell_id == "null-gaussian":
        latent, boundaries = piecewise_mean([120], [0.0])
        observed = latent + rng.normal(0.0, 1.0, latent.size)
        return standard_payload(cell_id, observed, boundaries, "gaussian", k_max=8)
    if cell_id == "heavy-tail-gaussian":
        latent, boundaries = piecewise_mean([40, 40, 40], [0.0, 2.0, -1.0])
        noise = rng.standard_t(df=3, size=latent.size) / math.sqrt(3.0)
        observed = latent + 0.6 * noise
        return standard_payload(cell_id, observed, boundaries, "gaussian", k_max=8)
    if cell_id == "zero-inflated-poisson":
        rates, boundaries = piecewise_mean([40, 40, 40], [2.0, 12.0, 3.0])
        observed = rng.poisson(rates).astype(float)
        observed[rng.random(rates.size) < 0.35] = 0.0
        return standard_payload(cell_id, observed, boundaries, "poisson", k_max=8)
    if cell_id == "dense-gaussian":
        lengths = [10] * 12
        latent, boundaries = piecewise_mean(lengths, [0.0, 2.0] * 6)
        observed = latent + rng.normal(0.0, 0.5, latent.size)
        return standard_payload(cell_id, observed, boundaries, "gaussian", k_max=15)
    if cell_id == "short-segment-gaussian":
        latent, boundaries = piecewise_mean([48, 4, 48], [0.0, 4.0, 0.0])
        observed = latent + rng.normal(0.0, 0.4, latent.size)
        return standard_payload(cell_id, observed, boundaries, "gaussian", k_max=8)
    if cell_id == "prior-conflict-gaussian":
        latent, boundaries = piecewise_mean([40, 40, 40], [0.0, 2.0, -1.0])
        observed = latent + rng.normal(0.0, 0.4, latent.size)
        payload = standard_payload(cell_id, observed, boundaries, "gaussian", k_max=8)
        payload["partition_prior"] = PartitionPriorConfig(parameters={"min_segment_length": 50.0})
        return payload
    raise ValueError(f"Unknown standard cell: {cell_id}")


def standard_payload(
    cell_id: str,
    observed: np.ndarray,
    boundaries: list[int],
    family: str,
    *,
    k_max: int,
) -> dict[str, Any]:
    return {
        "cell": cell_id,
        "values": np.asarray(observed, dtype=float),
        "true_boundaries": boundaries,
        "family": family,
        "k_max": k_max,
        "partition_prior": None,
    }


def generate_shared_cell(seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = 120
    common_mean, common_boundaries = piecewise_mean([40, 40, 40], [0.0, 2.0, -1.0])
    noise_scales = np.array([0.25, 0.25, 0.5, 0.5, 1.0, 1.0, 1.5, 1.5])
    sequences: list[np.ndarray] = []
    subject_boundaries: list[list[int]] = []
    for subject, sigma in enumerate(noise_scales):
        mean = common_mean.copy()
        boundaries = list(common_boundaries)
        if subject < 2:
            mean[60:80] += 1.5
            boundaries = [0, 40, 60, 80, n]
        sequences.append(mean + rng.normal(0.0, sigma, n))
        subject_boundaries.append(boundaries)
    weights = [np.full(n, 1.0 / (sigma * sigma)) for sigma in noise_scales]
    return {
        "cell": "shared-boundary-heterogeneity",
        "sequences": sequences,
        "weights": weights,
        "common_boundaries": common_boundaries,
        "subject_boundaries": subject_boundaries,
        "noise_scales": noise_scales,
    }


def generate_logistic_cell(seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    probabilities, boundaries = piecewise_mean([40, 40], [0.02, 0.98])
    observed = rng.binomial(1, probabilities).astype(float)
    return {
        "cell": "logistic-approximation-failure",
        "values": observed,
        "true_boundaries": boundaries,
    }


def run_standard_cell(cell_id: str, seed: int) -> dict[str, Any]:
    generated = generate_standard_cell(cell_id, seed)
    values = generated["values"]
    coordinates = np.arange(values.size, dtype=float).reshape(-1, 1)
    family = generated["family"]
    kwargs = {
        "k_max": generated["k_max"],
        "partition_prior": generated["partition_prior"],
    }
    estimator = (
        BayesBreakGaussian(**kwargs) if family == "gaussian" else BayesBreakPoisson(**kwargs)
    )
    started = time.perf_counter()
    estimator.fit(coordinates, values)
    elapsed = time.perf_counter() - started
    truth = generated["true_boundaries"][1:-1]
    predicted = estimator.map_boundaries_[1:-1]
    metrics = boundary_metrics(
        predicted,
        truth,
        tolerance=3,
        reference_type="simulated-truth",
        prediction_axis="observation-index",
        reference_axis="observation-index",
    )
    posterior = np.asarray(estimator.k_posterior_, dtype=float)
    entropy = float(-np.sum(posterior[posterior > 0] * np.log(posterior[posterior > 0])))
    return {
        "status": "executed",
        "cell": cell_id,
        "seed": seed,
        "family": family,
        "n": int(values.size),
        "true_boundaries": generated["true_boundaries"],
        "predicted_boundaries": estimator.map_boundaries_,
        "k_max": int(generated["k_max"]),
        "k_map": int(estimator.k_map_),
        "k_error": int(estimator.k_map_ - len(generated["true_boundaries"]) + 1),
        "posterior_mass_at_k_max": float(estimator.k_posterior_[-1]),
        "map_at_k_max": bool(estimator.k_map_ == generated["k_max"]),
        "posterior_k_entropy": entropy,
        "boundary_metrics": metrics.to_dict(),
        "false_discovery_count": len(predicted) if not truth else None,
        "false_positive_dataset": bool(predicted) if not truth else None,
        "missed_change_count": len(truth) - len(metrics.matches),
        "complete_boundary_recovery": len(metrics.matches) == len(truth),
        "log_evidence": float(estimator.log_evidence_),
        "wall_seconds": elapsed,
        "data_hash": sha256_arrays(values),
    }


def run_shared_cell(seed: int) -> dict[str, Any]:
    generated = generate_shared_cell(seed)
    sequences = generated["sequences"]
    weights = generated["weights"]
    coordinates = np.arange(120, dtype=float).reshape(-1, 1)
    started = time.perf_counter()
    shared = SharedBoundaryReplicatesSegmenter(
        BayesBreakGaussian(k_max=8),
        k_max=8,
    ).fit(coordinates, sequences, sample_weight=weights)
    shared_elapsed = time.perf_counter() - started
    common_truth = generated["common_boundaries"][1:-1]
    shared_metrics = boundary_metrics(
        shared.map_boundaries_[1:-1],
        common_truth,
        tolerance=3,
        reference_type="simulated-common-truth",
        prediction_axis="observation-index",
        reference_axis="observation-index",
    )
    shared_subject_records: list[dict[str, Any]] = []
    for subject, truth in enumerate(generated["subject_boundaries"]):
        metrics = boundary_metrics(
            shared.map_boundaries_[1:-1],
            truth[1:-1],
            tolerance=3,
            reference_type="simulated-subject-truth",
            prediction_axis="observation-index",
            reference_axis="observation-index",
        )
        shared_subject_records.append(
            {
                "subject": subject,
                "truth_boundaries": truth,
                "predicted_boundaries": shared.map_boundaries_,
                "metrics": metrics.to_dict(),
            }
        )
    independent_records: list[dict[str, Any]] = []
    independent_started = time.perf_counter()
    for subject, (values, sample_weight, truth) in enumerate(
        zip(sequences, weights, generated["subject_boundaries"], strict=True)
    ):
        estimator = BayesBreakGaussian(k_max=8).fit(
            coordinates,
            values,
            sample_weight=sample_weight,
        )
        metrics = boundary_metrics(
            estimator.map_boundaries_[1:-1],
            truth[1:-1],
            tolerance=3,
            reference_type="simulated-subject-truth",
            prediction_axis="observation-index",
            reference_axis="observation-index",
        )
        independent_records.append(
            {
                "subject": subject,
                "k_map": int(estimator.k_map_),
                "truth_boundaries": truth,
                "predicted_boundaries": estimator.map_boundaries_,
                "metrics": metrics.to_dict(),
            }
        )
    independent_elapsed = time.perf_counter() - independent_started
    shared_internal = np.asarray(shared.map_boundaries_[1:-1], dtype=float)
    deviation_detected = bool(np.any(np.abs(shared_internal - 60.0) <= 3.0))
    return {
        "status": "executed",
        "cell": generated["cell"],
        "seed": seed,
        "n": 120,
        "n_sequences": len(sequences),
        "noise_scales": generated["noise_scales"].tolist(),
        "common_boundaries": generated["common_boundaries"],
        "shared_boundaries": shared.map_boundaries_,
        "shared_k_map": int(shared.k_map_),
        "shared_metrics": shared_metrics.to_dict(),
        "shared_subject_metrics": shared_subject_records,
        "shared_mean_subject_f1": float(
            np.mean([record["metrics"]["f1"] for record in shared_subject_records])
        ),
        "subject_specific_boundary_60_selected_as_shared": deviation_detected,
        "independent": independent_records,
        "independent_mean_f1": float(
            np.mean([record["metrics"]["f1"] for record in independent_records])
        ),
        "shared_wall_seconds": shared_elapsed,
        "independent_wall_seconds": independent_elapsed,
        "wall_seconds": shared_elapsed + independent_elapsed,
        "data_hash": sha256_arrays(*sequences, *weights),
    }


def run_ep_worker(input_path: Path, output_path: Path) -> None:
    values = np.load(input_path, allow_pickle=False)
    coordinates = np.arange(values.size, dtype=float).reshape(-1, 1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        started = time.perf_counter()
        estimator = BayesBreakLogisticNormal(k_max=4, approx="ep", max_iter=20).fit(
            coordinates, values
        )
        fit_wall_seconds = time.perf_counter() - started
    worker_payload = {
        "estimator": estimator,
        "fit_wall_seconds": fit_wall_seconds,
        "data_hash": sha256_arrays(values),
        "warnings": [
            {
                "category": warning.category.__name__,
                "message": str(warning.message),
                "source": Path(warning.filename).name,
                "line": warning.lineno,
            }
            for warning in caught
        ],
        "peak_rss": process_peak_rss(resource.RUSAGE_SELF),
    }
    output_path.write_bytes(pickle.dumps(worker_payload, protocol=pickle.HIGHEST_PROTOCOL))


def run_ep_bounded(
    values: np.ndarray,
    reference: BayesBreakLogisticNormal,
    truth: list[int],
    timeout_seconds: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="bayesbreak-ep-") as temporary:
        directory = Path(temporary)
        input_path = directory / "values.npy"
        output_path = directory / "worker.pkl"
        np.save(input_path, values, allow_pickle=False)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--ep-worker",
            "--input-path",
            str(input_path),
            "--worker-output",
            str(output_path),
        ]
        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return {
                "status": "timed-out",
                "timeout_seconds": timeout_seconds,
                "wall_seconds": time.perf_counter() - started,
            }
        except subprocess.CalledProcessError as exc:
            return {
                "status": "failed",
                "wall_seconds": time.perf_counter() - started,
                "returncode": exc.returncode,
                "stderr": exc.stderr,
            }
        worker_payload = pickle.loads(output_path.read_bytes())
    if worker_payload["data_hash"] != sha256_arrays(values):
        raise RuntimeError("EP worker input hash does not match parent data")
    estimator = worker_payload["estimator"]
    diagnostics = run_non_conjugate_diagnostics(estimator, reference)
    truth_metrics = boundary_metrics(
        estimator.map_boundaries_[1:-1],
        truth,
        tolerance=3,
        reference_type="simulated-truth",
        prediction_axis="observation-index",
        reference_axis="observation-index",
    )
    record = {
        "status": "executed",
        "timeout_seconds": timeout_seconds,
        "wall_seconds": time.perf_counter() - started,
        "fit_wall_seconds": worker_payload["fit_wall_seconds"],
        "k_map": int(estimator.k_map_),
        "boundaries": estimator.map_boundaries_,
        "diagnostics": diagnostics.to_dict(),
        "truth_metrics": truth_metrics.to_dict(),
        "warnings": worker_payload["warnings"],
        "peak_rss": worker_payload["peak_rss"],
    }
    if completed.stderr:
        record["stderr"] = completed.stderr
    return record


def run_logistic_cell(seed: int, ep_timeout_seconds: int) -> dict[str, Any]:
    generated = generate_logistic_cell(seed)
    values = generated["values"]
    coordinates = np.arange(values.size, dtype=float).reshape(-1, 1)
    started = time.perf_counter()
    reference = BayesBreakLogisticNormal(
        k_max=4,
        approx="quadrature",
        gh_points=120,
    ).fit(coordinates, values)
    truth = generated["true_boundaries"][1:-1]
    methods = {
        "quadrature-40": BayesBreakLogisticNormal(k_max=4, approx="quadrature", gh_points=40),
        "laplace": BayesBreakLogisticNormal(k_max=4, approx="laplace"),
    }
    method_records: dict[str, Any] = {}
    for name, estimator in methods.items():
        method_started = time.perf_counter()
        estimator.fit(coordinates, values)
        diagnostics = run_non_conjugate_diagnostics(estimator, reference)
        truth_metrics = boundary_metrics(
            estimator.map_boundaries_[1:-1],
            truth,
            tolerance=3,
            reference_type="simulated-truth",
            prediction_axis="observation-index",
            reference_axis="observation-index",
        )
        method_records[name] = {
            "status": "executed",
            "wall_seconds": time.perf_counter() - method_started,
            "k_map": int(estimator.k_map_),
            "boundaries": estimator.map_boundaries_,
            "diagnostics": diagnostics.to_dict(),
            "truth_metrics": truth_metrics.to_dict(),
        }
    method_records["ep"] = run_ep_bounded(values, reference, truth, ep_timeout_seconds)
    reference_metrics = boundary_metrics(
        reference.map_boundaries_[1:-1],
        truth,
        tolerance=3,
        reference_type="simulated-truth",
        prediction_axis="observation-index",
        reference_axis="observation-index",
    )
    return {
        "status": "executed",
        "cell": generated["cell"],
        "seed": seed,
        "n": int(values.size),
        "true_boundaries": generated["true_boundaries"],
        "reference_k_map": int(reference.k_map_),
        "reference_boundaries": reference.map_boundaries_,
        "reference_metrics": reference_metrics.to_dict(),
        "methods": method_records,
        "wall_seconds": time.perf_counter() - started,
        "data_hash": sha256_arrays(values),
    }


def cell_input_hashes(cell_id: str, seed: int) -> dict[str, str | None]:
    if cell_id == "shared-boundary-heterogeneity":
        generated = generate_shared_cell(seed)
        return {
            "data_hash": sha256_arrays(*generated["sequences"]),
            "truth_hash": sha256_json(
                {
                    "common": generated["common_boundaries"],
                    "subjects": generated["subject_boundaries"],
                }
            ),
            "weights_hash": sha256_arrays(*generated["weights"]),
            "effective_config_hash": sha256_json(
                {"family": "gaussian-shared-boundary", "k_max": 8, "n_sequences": 8}
            ),
        }
    if cell_id == "logistic-approximation-failure":
        generated = generate_logistic_cell(seed)
        return {
            "data_hash": sha256_arrays(generated["values"]),
            "truth_hash": sha256_json(generated["true_boundaries"]),
            "weights_hash": None,
            "effective_config_hash": sha256_json(
                {
                    "family": "logistic-normal",
                    "k_max": 4,
                    "reference": "quadrature-120",
                }
            ),
        }
    generated = generate_standard_cell(cell_id, seed)
    prior = generated["partition_prior"]
    prior_payload = None
    if prior is not None:
        prior_payload = {
            "segment_cohesion": prior.segment_cohesion,
            "boundary_hazard": prior.boundary_hazard,
            "parameters": dict(prior.parameters),
        }
    return {
        "data_hash": sha256_arrays(generated["values"]),
        "truth_hash": sha256_json(generated["true_boundaries"]),
        "weights_hash": None,
        "effective_config_hash": sha256_json(
            {
                "family": generated["family"],
                "k_max": generated["k_max"],
                "partition_prior": prior_payload,
            }
        ),
    }


def run_cell(cell_id: str, seed: int, ep_timeout_seconds: int) -> dict[str, Any]:
    started = time.perf_counter()
    input_hashes = cell_input_hashes(cell_id, seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            if cell_id == "shared-boundary-heterogeneity":
                record = run_shared_cell(seed)
            elif cell_id == "logistic-approximation-failure":
                record = run_logistic_cell(seed, ep_timeout_seconds)
            else:
                record = run_standard_cell(cell_id, seed)
        except Exception as exc:
            record = {
                "status": "failed",
                "cell": cell_id,
                "seed": seed,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(),
                "wall_seconds": time.perf_counter() - started,
            }
        record["warnings"] = [
            {
                "category": warning.category.__name__,
                "message": str(warning.message),
                "source": Path(warning.filename).name,
                "line": warning.lineno,
            }
            for warning in caught
        ]
        record.update(input_hashes)
    return record


def summarize(records: list[dict[str, Any]], cell_ids: list[str]) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    for cell_id in cell_ids:
        subset = [record for record in records if record["cell"] == cell_id]
        executed = [record for record in subset if record["status"] == "executed"]
        cell_summary: dict[str, Any] = {
            "n_runs": len(subset),
            "n_executed": len(executed),
            "n_failed": len(subset) - len(executed),
            "failure_rate": (len(subset) - len(executed)) / len(subset),
            "failure_rate_interval": interval_summary(
                [float(record["status"] != "executed") for record in subset]
            ),
            "wall_seconds": interval_summary([float(record["wall_seconds"]) for record in subset]),
        }
        if executed and "boundary_metrics" in executed[0]:
            cell_summary["boundary_f1"] = interval_summary(
                [float(record["boundary_metrics"]["f1"]) for record in executed]
            )
            cell_summary["k_error"] = interval_summary(
                [float(record["k_error"]) for record in executed]
            )
            cell_summary["posterior_k_entropy"] = interval_summary(
                [float(record["posterior_k_entropy"]) for record in executed]
            )
            cell_summary["missed_change_count"] = interval_summary(
                [float(record["missed_change_count"]) for record in executed]
            )
            cell_summary["posterior_mass_at_k_max"] = interval_summary(
                [float(record["posterior_mass_at_k_max"]) for record in executed]
            )
            cell_summary["map_saturation_rate"] = interval_summary(
                [float(record["map_at_k_max"]) for record in executed]
            )
            cell_summary["complete_boundary_recovery_rate"] = interval_summary(
                [float(record["complete_boundary_recovery"]) for record in executed]
            )
        if executed and cell_id == "null-gaussian":
            cell_summary["false_discovery_count"] = interval_summary(
                [float(record["false_discovery_count"]) for record in executed]
            )
            cell_summary["false_positive_dataset_rate"] = interval_summary(
                [float(record["false_positive_dataset"]) for record in executed]
            )
        if executed and cell_id == "shared-boundary-heterogeneity":
            cell_summary["shared_common_truth_f1"] = interval_summary(
                [float(record["shared_metrics"]["f1"]) for record in executed]
            )
            cell_summary["shared_mean_subject_f1"] = interval_summary(
                [float(record["shared_mean_subject_f1"]) for record in executed]
            )
            cell_summary["independent_mean_f1"] = interval_summary(
                [float(record["independent_mean_f1"]) for record in executed]
            )
            cell_summary["subject_deviation_selected_rate"] = interval_summary(
                [
                    float(record["subject_specific_boundary_60_selected_as_shared"])
                    for record in executed
                ]
            )
        if executed and cell_id == "logistic-approximation-failure":
            for method in ("quadrature-40", "laplace", "ep"):
                method_executed = [
                    record["methods"][method]
                    for record in executed
                    if record["methods"][method]["status"] == "executed"
                ]
                cell_summary[f"{method}_execution_rate"] = len(method_executed) / len(executed)
                cell_summary[f"{method}_timeout_rate"] = interval_summary(
                    [
                        float(record["methods"][method]["status"] == "timed-out")
                        for record in executed
                    ]
                )
                cell_summary[f"{method}_max_block_error"] = interval_summary(
                    [
                        float(record["diagnostics"]["extra"]["block_error_max"])
                        for record in method_executed
                    ]
                )
                cell_summary[f"{method}_empirical_tv"] = interval_summary(
                    [
                        float(record["diagnostics"]["extra"]["pk_tv_empirical"])
                        for record in method_executed
                    ]
                )
                cell_summary[f"{method}_conditional_tv_bound"] = interval_summary(
                    [
                        float(
                            record["diagnostics"]["extra"]["conditional_partition_bounds"][
                                "tv_upper_bound"
                            ]
                        )
                        for record in method_executed
                        if record["diagnostics"]["extra"]["conditional_partition_bounds"]
                        is not None
                    ]
                )
                cell_summary[f"{method}_map_jaccard"] = interval_summary(
                    [
                        float(record["diagnostics"]["extra"]["map_path_jaccard"])
                        for record in method_executed
                    ]
                )
                cell_summary[f"{method}_truth_f1"] = interval_summary(
                    [float(record["truth_metrics"]["f1"]) for record in method_executed]
                )
                cell_summary[f"{method}_verified_convergence_rate"] = interval_summary(
                    [
                        float(
                            record["diagnostics"]["extra"]["segment_error_record"][
                                "convergence_status"
                            ]
                            == "verified"
                        )
                        for record in method_executed
                    ]
                )
        cells[cell_id] = cell_summary
    return {"cells": cells}


def interval_summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    if array.size > 1:
        standard_error = float(np.std(array, ddof=1) / math.sqrt(array.size))
        critical = float(student_t.ppf(0.975, df=array.size - 1))
    else:
        standard_error = 0.0
        critical = 0.0
    return {
        "mean": mean,
        "standard_error": standard_error,
        "ci95_lower": mean - critical * standard_error,
        "ci95_upper": mean + critical * standard_error,
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
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def code_revision(root: Path) -> dict[str, Any]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    relevant = (
        "provenance/epr-bb-015-plan.json",
        "scripts/phase6_misspecification_rerun.py",
        "src/bayesbreak",
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", *relevant],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("EPR-BB-015 code and plan must be committed before execution")
    return {
        "commit": commit,
        "commit_sha256": hashlib.sha256(commit.encode("ascii")).hexdigest(),
        "relevant_paths_clean": True,
    }


def environment_record() -> dict[str, Any]:
    packages = {
        name: importlib.metadata.version(name)
        for name in ("bayesbreak", "numpy", "scipy", "scikit-learn")
    }
    if packages["bayesbreak"] != bayesbreak.__version__:
        raise RuntimeError("Installed and imported BayesBreak versions disagree")
    record = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "bayesbreak_module_version": bayesbreak.__version__,
        "packages": packages,
    }
    return {**record, "sha256": sha256_json(record)}


def process_peak_rss(who: int) -> dict[str, Any]:
    return {
        "value": resource.getrusage(who).ru_maxrss,
        "units": "bytes" if platform.system() == "Darwin" else "KiB",
    }


def peak_rss() -> dict[str, Any]:
    self_usage = process_peak_rss(resource.RUSAGE_SELF)
    child_usage = process_peak_rss(resource.RUSAGE_CHILDREN)
    return {
        "value": max(self_usage["value"], child_usage["value"]),
        "self_value": self_usage["value"],
        "children_value": child_usage["value"],
        "units": self_usage["units"],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["resources"]["output_bytes"] = 0
    for _ in range(3):
        path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        size = path.stat().st_size
        if payload["resources"]["output_bytes"] == size:
            break
        payload["resources"]["output_bytes"] = size


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pilot", "full"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ep-worker", action="store_true")
    parser.add_argument("--input-path", type=Path)
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args()

    if args.ep_worker:
        if args.input_path is None or args.worker_output is None:
            parser.error("--ep-worker requires --input-path and --worker-output")
        run_ep_worker(args.input_path, args.worker_output)
        return 0
    if args.mode is None or args.output is None:
        parser.error("--mode and --output are required for suite execution")

    root = Path(__file__).resolve().parents[1]
    plan = load_plan(root)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {args.output}")
    if args.mode == "full" and plan.get("full_execution_approved") is not True:
        raise RuntimeError("Full EPR-BB-015 execution is not approved in the frozen plan")
    revision = code_revision(root)
    cell_ids = [cell["id"] for cell in plan["cells"]]
    repetitions = (
        int(plan["pilot_repetitions_per_cell"])
        if args.mode == "pilot"
        else int(plan["full_repetitions_per_cell"])
    )
    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    for cell_index, cell_id in enumerate(cell_ids):
        for repetition in range(repetitions):
            seed = int(plan["seed_base"]) + 10_000 * cell_index + repetition
            records.append(run_cell(cell_id, seed, int(plan["ep_timeout_seconds"])))
    elapsed = time.perf_counter() - started
    projected_full_runs = len(cell_ids) * int(plan["full_repetitions_per_cell"])
    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_role": "resource-pilot" if args.mode == "pilot" else "original-execution",
        "result_id": RESULT_ID,
        "parent_result_id": None,
        "protocol_id": PROTOCOL_ID,
        "execution_status": "executed",
        "scientific_interpretation": (
            "implementation-verification" if args.mode == "pilot" else "pending-independent-review"
        ),
        "mode": args.mode,
        "plan_sha256": hashlib.sha256((root / PLAN_PATH).read_bytes()).hexdigest(),
        "config_sha256": sha256_json(plan),
        "seed_base": plan["seed_base"],
        "seed_schedule": plan["seed_schedule"],
        "repetitions_per_cell": repetitions,
        "full_execution_approved": plan["full_execution_approved"],
        "cell_ids": cell_ids,
        "code": revision,
        "environment": environment_record(),
        "records": records,
        "summary": summarize(records, cell_ids),
        "resources": {
            "elapsed_wall_seconds": elapsed,
            "peak_rss": peak_rss(),
            "output_bytes": 0,
            "projected_full_runs": projected_full_runs,
            "projected_full_wall_seconds": (
                elapsed
                * int(plan["full_repetitions_per_cell"])
                / int(plan["pilot_repetitions_per_cell"])
                if args.mode == "pilot"
                else elapsed
            ),
        },
        "interpretation_limit": (
            "This suite maps declared failure regimes. It is not a universal robustness claim "
            "and retains failed, null, reversed, nonconverged, and timed-out outcomes."
        ),
    }
    write_json(args.output, payload)
    print(json.dumps({"output": str(args.output), **payload["resources"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
