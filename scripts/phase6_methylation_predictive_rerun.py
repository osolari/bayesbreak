"""Pilot and execute the authorized EPR-BB-012 methylation predictive rerun."""

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
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t

import bayesbreak
from bayesbreak import BayesBreakBetaObs
from bayesbreak.datasets.base import DatasetCard, load_with_provenance
from bayesbreak.datasets.methylation import _parse_methylkit_table
from bayesbreak.metrics import boundary_metrics
from bayesbreak.prediction import posterior_predictive_logpdf

RESULT_ID = "RES-BB-RD-008Q"
PARENT_RESULT_ID = "RES-BB-RD-007Q"
PROTOCOL_ID = "EPR-BB-012"
SOURCE_URI = "https://github.com/al2na/methylKit/raw/master/inst/extdata/test1.myCpG.txt"
EXPECTED_SOURCE_SHA256 = "f823f0eebd6ec44994c28882c1b7d16ea21eaf32ee49c93a1a149c5096b5b54e"
EXPECTED_DATA_HASH = "822cd6e347fa777308e9bb6b9e398a6499f1aa16cc2e7340ad0d0884119c40fb"
EXPECTED_DESCRIPTOR_HASH = "0237720765d5ff3b7e2364f32def43f454cc41ca3acaff3e6aa2177c310c775c"
EXPECTED_N = 1904
N_SPLITS = 10
BLOCK_SIZE = 152
SEED_BASE = 261201


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_source(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"MethylKit source does not exist: {path}")
    source_hash = sha256_file(path)
    if source_hash != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            f"Unexpected methylKit source hash {source_hash}; expected {EXPECTED_SOURCE_SHA256}"
        )
    return source_hash


def load_exact_source(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, DatasetCard]:
    positions, values, precision = _parse_methylkit_table(path.read_text(encoding="utf-8"))
    bundle, card = load_with_provenance("methylation")
    loaded_positions = np.asarray(bundle.X[:, 0], dtype=float)
    loaded_values = np.asarray(bundle.y, dtype=float)
    loaded_precision = np.asarray(bundle.sample_weight, dtype=float)
    if bundle.source != "downloaded":
        raise RuntimeError("Production loader did not return the exact methylKit source")
    if positions.size != EXPECTED_N or not np.all(np.diff(positions) > 0):
        raise RuntimeError(f"Expected {EXPECTED_N} strictly ordered CpG coordinates")
    if not np.array_equal(positions, loaded_positions):
        raise RuntimeError("Production loader changed methylKit coordinates")
    if not np.array_equal(values, loaded_values):
        raise RuntimeError("Production loader changed methylKit fractions")
    if not np.array_equal(precision, loaded_precision):
        raise RuntimeError("Production loader changed methylKit coverage")
    if np.any(~np.isfinite(values)) or np.any((values <= 0) | (values >= 1)):
        raise RuntimeError("Methylation fractions must be finite and in the open unit interval")
    if np.any(~np.isfinite(precision)) or np.any(precision <= 0):
        raise RuntimeError("Per-CpG Beta precisions must be finite and positive")
    if card.data_hash != EXPECTED_DATA_HASH:
        raise RuntimeError("Methylation dataset-card hash changed")
    if card.descriptor_hash != EXPECTED_DESCRIPTOR_HASH:
        raise RuntimeError("Methylation precision hash changed")
    return positions, values, precision, card


def build_splits(
    n: int = EXPECTED_N,
    *,
    n_splits: int = N_SPLITS,
    block_size: int = BLOCK_SIZE,
    seed_base: int = SEED_BASE,
) -> list[dict[str, Any]]:
    if n < 4 or n_splits < 1 or block_size < 1:
        raise ValueError("n, n_splits, and block_size must define nonempty interior blocks")
    edges = np.linspace(1, n - 1, n_splits + 1, dtype=int)
    splits: list[dict[str, Any]] = []
    occupied: set[int] = set()
    for split_index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        if upper - lower < block_size:
            raise ValueError("Each ordered stratum must be at least block_size observations")
        seed = seed_base + split_index
        rng = np.random.default_rng(seed)
        start = int(rng.integers(lower, upper - block_size + 1))
        stop = start + block_size
        test_indices = np.arange(start, stop, dtype=np.intp)
        if occupied.intersection(test_indices.tolist()):
            raise RuntimeError("Stratified methylation holdout blocks overlap")
        occupied.update(test_indices.tolist())
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_indices] = False
        train_indices = np.flatnonzero(train_mask).astype(np.intp, copy=False)
        if train_indices[0] != 0 or train_indices[-1] != n - 1:
            raise RuntimeError("Every split must retain both global support endpoints")
        splits.append(
            {
                "split_index": split_index,
                "seed": seed,
                "stratum": [int(lower), int(upper)],
                "start": start,
                "stop": stop,
                "test_indices": test_indices,
                "train_indices": train_indices,
            }
        )
    return splits


def map_boundaries_to_original(
    train_indices: np.ndarray,
    boundaries: list[int],
) -> list[int]:
    internal = boundaries[1:-1]
    return [int(train_indices[index]) for index in internal]


def fit_estimator(
    positions: np.ndarray,
    values: np.ndarray,
    precision: np.ndarray,
) -> tuple[BayesBreakBetaObs, float]:
    start = time.perf_counter()
    estimator = BayesBreakBetaObs(
        k_max=15,
        phi=precision,
        quadrature_points=32,
        regression_curve="none",
    ).fit(positions.reshape(-1, 1), values)
    return estimator, time.perf_counter() - start


def run_split(
    split: dict[str, Any],
    positions: np.ndarray,
    values: np.ndarray,
    precision: np.ndarray,
    full_boundaries: list[int],
) -> dict[str, Any]:
    train_indices = split["train_indices"]
    test_indices = split["test_indices"]
    if not np.all(
        (positions[test_indices] >= positions[train_indices[0]])
        & (positions[test_indices] <= positions[train_indices[-1]])
    ):
        raise RuntimeError("A methylation holdout coordinate is outside fitted support")
    estimator, fit_wall = fit_estimator(
        positions[train_indices],
        values[train_indices],
        precision[train_indices],
    )
    score_start = time.perf_counter()
    scores = posterior_predictive_logpdf(
        estimator,
        positions[test_indices].reshape(-1, 1),
        values[test_indices],
        sample_weight=precision[test_indices],
        per_sample=True,
        extrapolation="error",
    )
    score_wall = time.perf_counter() - score_start
    if not isinstance(scores, np.ndarray) or not np.all(np.isfinite(scores)):
        raise RuntimeError("Beta-observation posterior-predictive scores must be finite")
    mapped_boundaries = map_boundaries_to_original(train_indices, estimator.map_boundaries_)
    stability = boundary_metrics(
        mapped_boundaries,
        full_boundaries,
        tolerance=3,
        reference_type="model-derived-full-data-map",
        prediction_axis="original-CpG-index",
        reference_axis="original-CpG-index",
    )
    total = float(np.sum(scores))
    return {
        "split_index": split["split_index"],
        "seed": split["seed"],
        "stratum": split["stratum"],
        "test_start_index": split["start"],
        "test_stop_index_exclusive": split["stop"],
        "test_coordinate_support": [
            float(positions[test_indices[0]]),
            float(positions[test_indices[-1]]),
        ],
        "training_coordinate_support": [
            float(positions[train_indices[0]]),
            float(positions[train_indices[-1]]),
        ],
        "n_train": int(train_indices.size),
        "n_test": int(test_indices.size),
        "test_indices_hash": sha256_array(test_indices),
        "phi_train_hash": sha256_array(precision[train_indices]),
        "phi_new_hash": sha256_array(precision[test_indices]),
        "phi_new_min": float(np.min(precision[test_indices])),
        "phi_new_max": float(np.max(precision[test_indices])),
        "extrapolation_policy": "error",
        "prediction_metadata": dict(estimator.prediction_metadata_),
        "per_sample_log_predictive": scores.tolist(),
        "total_log_predictive": total,
        "mean_log_predictive": total / scores.size,
        "k_map": int(estimator.k_map_),
        "boundaries_original_index": mapped_boundaries,
        "boundary_stability_tau3": stability.to_dict(),
        "fit_wall_seconds": fit_wall,
        "score_wall_seconds": score_wall,
    }


def interval_summary(values: list[float]) -> dict[str, float]:
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


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = float(sum(record["total_log_predictive"] for record in records))
    denominator = int(sum(record["n_test"] for record in records))
    return {
        "n_splits": len(records),
        "total_log_predictive": total,
        "total_denominator": denominator,
        "pooled_mean_log_predictive": total / denominator,
        "split_mean_log_predictive": interval_summary(
            [record["mean_log_predictive"] for record in records]
        ),
        "k_map": interval_summary([float(record["k_map"]) for record in records]),
        "boundary_stability_f1_tau3": interval_summary(
            [record["boundary_stability_tau3"]["f1"] for record in records]
        ),
        "calibration_status": (
            "not-computed: BayesBreakBetaObs has no certified PIT CDF helper; "
            "no calibration statistic is defined for this execution"
        ),
    }


def code_revision() -> dict[str, Any]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    relevant_paths = (
        "pyproject.toml",
        "scripts/phase6_methylation_predictive_rerun.py",
        "src/bayesbreak",
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", *relevant_paths],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("Methylation rerun code must be committed before scientific execution")
    return {
        "commit": commit,
        "commit_sha256": hashlib.sha256(commit.encode("ascii")).hexdigest(),
        "relevant_paths_clean": True,
    }


def environment_record() -> dict[str, Any]:
    packages = {name: importlib.metadata.version(name) for name in ("bayesbreak", "numpy", "scipy")}
    if packages["bayesbreak"] != bayesbreak.__version__:
        raise RuntimeError("Installed and imported BayesBreak versions disagree")
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
        path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
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
        default=Path.home() / ".cache" / "bayesbreak" / "methylkit_test1.myCpG.txt",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    revision = code_revision()
    started = time.perf_counter()
    source_path = args.source.expanduser().resolve()
    source_hash = verify_source(source_path)
    positions, values, precision, card = load_exact_source(source_path)
    splits = build_splits()
    selected_splits = splits[:1] if args.mode == "pilot" else splits
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("default")
        full_estimator, full_fit_wall = fit_estimator(positions, values, precision)
        full_boundaries = [int(value) for value in full_estimator.map_boundaries_[1:-1]]
        records = [
            run_split(split, positions, values, precision, full_boundaries)
            for split in selected_splits
        ]
    elapsed = time.perf_counter() - started
    split_manifest = [
        {
            "split_index": split["split_index"],
            "seed": split["seed"],
            "stratum": split["stratum"],
            "start": split["start"],
            "stop": split["stop"],
            "test_indices_hash": sha256_array(split["test_indices"]),
        }
        for split in splits
    ]
    config = {
        "mode": args.mode,
        "observation_family": "beta-observation",
        "k_max": 15,
        "quadrature_points": 32,
        "training_likelihood_power_weights": "unit weights",
        "training_phi": "per-CpG coverage",
        "prediction_phi_new": "held-out per-CpG coverage",
        "extrapolation_policy": "error",
        "split_design": "ten stratified seeded contiguous interior CpG blocks",
        "n_splits": N_SPLITS,
        "block_size": BLOCK_SIZE,
        "seed_base": SEED_BASE,
        "boundary_stability_tolerance_CpGs": 3,
        "split_manifest": split_manifest,
    }
    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_role": "resource-pilot" if args.mode == "pilot" else "corrected-execution",
        "result_id": RESULT_ID,
        "parent_result_id": PARENT_RESULT_ID,
        "protocol_id": PROTOCOL_ID,
        "execution_status": "executed",
        "scientific_interpretation": (
            "implementation-verification" if args.mode == "pilot" else "pending-independent-review"
        ),
        "source": {
            "uri": SOURCE_URI,
            "cache_path": "$HOME/.cache/bayesbreak/methylkit_test1.myCpG.txt",
            "sha256": source_hash,
            "n_CpGs": int(values.size),
            "position_hash": sha256_array(positions),
            "response_hash": sha256_array(values),
            "phi_hash": sha256_array(precision),
            "preprocessing": "freqC/100 clipped to [0.001, 0.999] by the loader",
        },
        "dataset_card": card.to_dict(),
        "coordinate_metadata": {
            "prediction_axis": "CpG-genomic-coordinate",
            "reference_axis": "original-CpG-index",
            "reference_type": "model-derived-full-data-map",
            "fitted_support": [float(positions[0]), float(positions[-1])],
            "all_holdouts_in_support": True,
            "external_annotations": "none-independently-verified",
        },
        "config": config,
        "config_sha256": sha256_json(config),
        "split_sha256": sha256_json(split_manifest),
        "code": revision,
        "environment": environment_record(),
        "full_data_reference": {
            "k_map": int(full_estimator.k_map_),
            "boundaries_original_index": full_boundaries,
            "log_evidence": float(full_estimator.log_evidence_),
            "fit_wall_seconds": full_fit_wall,
        },
        "records": records,
        "summary": summarize(records),
        "warnings": [
            {
                "category": warning.category.__name__,
                "message": str(warning.message),
                "source": Path(warning.filename).name,
                "line": warning.lineno,
            }
            for warning in caught_warnings
        ],
        "resources": {
            "elapsed_wall_seconds": elapsed,
            "peak_rss": peak_rss(),
            "output_bytes": 0,
            "projected_full_wall_seconds": elapsed * (N_SPLITS + 1) / 2,
            "projected_full_fits": N_SPLITS + 1,
        },
        "limitations": [
            "Repeated blocks are regions of one chromosome, not independent biological samples.",
            "Boundary stability is agreement with the model-derived full-data MAP, not truth.",
            "No certified Beta-observation PIT helper is available, so calibration is not reported.",
            "This execution does not add a second region, cell type, or external atlas annotation.",
        ],
    }
    write_json(args.output, payload)
    print(json.dumps({"output": str(args.output), **payload["resources"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
