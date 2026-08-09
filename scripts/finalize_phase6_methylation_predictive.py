"""Finalize artifacts and provenance for corrected result RES-BB-RD-008Q."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from bayesbreak.provenance import (
    InterpretationStatus,
    LineageStatus,
    ResultRecord,
    ResultStatus,
    write_sidecar,
)

RESULT_ID = "RES-BB-RD-008Q"
PARENT_RESULT_ID = "RES-BB-RD-007Q"
PROTOCOL_ID = "EPR-BB-012"
EXPECTED_SOURCE_SHA256 = "f823f0eebd6ec44994c28882c1b7d16ea21eaf32ee49c93a1a149c5096b5b54e"
EXPECTED_DATA_HASH = "822cd6e347fa777308e9bb6b9e398a6499f1aa16cc2e7340ad0d0884119c40fb"
EXPECTED_DESCRIPTOR_HASH = "0237720765d5ff3b7e2364f32def43f454cc41ca3acaff3e6aa2177c310c775c"


def finalize(results_path: Path) -> dict[str, Path]:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    validate_payload(payload)

    output_dir = results_path.parent
    summary_csv = output_dir / "split_summary.csv"
    summary_md = output_dir / "SUMMARY.md"
    figure_path = output_dir / "predictive_summary.png"
    sidecar_path = output_dir / "result_sidecar.json"
    rows = split_rows(payload)
    write_summary_csv(rows, summary_csv)
    write_summary_markdown(payload, rows, summary_md)
    write_summary_figure(payload, rows, figure_path)

    output_hashes = {
        "results": sha256_file(results_path),
        "summary_table": sha256_file(summary_csv),
        "summary_report": sha256_file(summary_md),
        "summary_figure": sha256_file(figure_path),
    }
    summary = payload["summary"]
    record = ResultRecord(
        result_id=RESULT_ID,
        execution_status=ResultStatus.EXECUTED,
        scientific_interpretation=InterpretationStatus.VALID_FOR_STATED_INTERPRETATION,
        lineage_status=LineageStatus.CORRECTED,
        parent_result_id=PARENT_RESULT_ID,
        data_hash=payload["dataset_card"]["data_hash"],
        weights_hash=payload["dataset_card"]["descriptor_hash"],
        split_hash=payload["split_sha256"],
        config_hash=payload["config_sha256"],
        code_hash=sha256_git_tree(root, payload["code"]["commit"]),
        environment_hash=payload["environment"]["sha256"],
        coordinate_metadata={
            "prediction_axis": "CpG-genomic-coordinate",
            "reference_axis": "original-CpG-index",
            "reference_type": "model-derived-full-data-map",
            "fitted_support": payload["coordinate_metadata"]["fitted_support"],
            "extrapolation_policy": "error",
            "all_holdouts_in_support": True,
        },
        metrics={
            "protocol_id": PROTOCOL_ID,
            "observation_family": "beta-observation",
            "n_splits": summary["n_splits"],
            "total_log_predictive": summary["total_log_predictive"],
            "total_denominator": summary["total_denominator"],
            "pooled_mean_log_predictive": summary["pooled_mean_log_predictive"],
            "split_mean_log_predictive": summary["split_mean_log_predictive"],
            "k_map": summary["k_map"],
            "boundary_stability_f1_tau3": summary["boundary_stability_f1_tau3"],
            "calibration_status": summary["calibration_status"],
            "full_data_reference": payload["full_data_reference"],
            "split_rows": rows,
            "elapsed_wall_seconds": payload["resources"]["elapsed_wall_seconds"],
            "peak_rss": payload["resources"]["peak_rss"],
            "captured_warnings": payload["warnings"],
            "interpretation_limit": (
                "Scores are pointwise Beta-observation log predictive densities over ten "
                "predeclared in-support chromosome blocks with observed coverage as phi_new. "
                "Blocks are regions of one chromosome, not independent biological samples; "
                "no certified PIT calibration or external biological truth is reported."
            ),
        },
        artifacts={
            "results": relative(root, results_path),
            "summary_table": relative(root, summary_csv),
            "summary_report": relative(root, summary_md),
            "summary_figure": relative(root, figure_path),
        },
        output_hashes=output_hashes,
    )
    write_sidecar(sidecar_path, record)
    return {
        "results": results_path,
        "summary_table": summary_csv,
        "summary_report": summary_md,
        "summary_figure": figure_path,
        "sidecar": sidecar_path,
    }


def validate_payload(payload: dict[str, Any]) -> None:
    if payload.get("record_role") != "corrected-execution":
        raise ValueError("Expected a full corrected-execution record")
    if payload.get("result_id") != RESULT_ID or payload.get("parent_result_id") != PARENT_RESULT_ID:
        raise ValueError("Unexpected result lineage")
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Unexpected experiment protocol")
    source = payload.get("source", {})
    if source.get("sha256") != EXPECTED_SOURCE_SHA256 or source.get("n_CpGs") != 1904:
        raise ValueError("Unexpected methylKit source identity")
    card = payload.get("dataset_card", {})
    if card.get("data_hash") != EXPECTED_DATA_HASH:
        raise ValueError("Unexpected methylation data hash")
    if card.get("descriptor_hash") != EXPECTED_DESCRIPTOR_HASH:
        raise ValueError("Unexpected methylation precision hash")
    config = payload.get("config", {})
    required_config = {
        "mode": "full",
        "observation_family": "beta-observation",
        "training_likelihood_power_weights": "unit weights",
        "training_phi": "per-CpG coverage",
        "prediction_phi_new": "held-out per-CpG coverage",
        "extrapolation_policy": "error",
        "n_splits": 10,
        "block_size": 152,
    }
    for name, expected in required_config.items():
        if config.get(name) != expected:
            raise ValueError(f"Unexpected methylation config field {name}")
    axes = payload.get("coordinate_metadata", {})
    if axes.get("prediction_axis") != "CpG-genomic-coordinate":
        raise ValueError("Unexpected predictive coordinate axis")
    if axes.get("all_holdouts_in_support") is not True:
        raise ValueError("Every holdout must be inside fitted support")
    if axes.get("external_annotations") != "none-independently-verified":
        raise ValueError("External annotation status changed")
    reference = payload.get("full_data_reference", {})
    if reference.get("k_map") != 15 or len(reference.get("boundaries_original_index", ())) != 14:
        raise ValueError("Expected the reproduced full-data 15-segment reference")
    if not math.isclose(reference.get("log_evidence", math.nan), -9518.667508691196, abs_tol=1e-8):
        raise ValueError("Full-data reference log evidence changed")

    records = payload.get("records", ())
    if len(records) != 10:
        raise ValueError("Expected ten methylation split records")
    test_hashes: set[str] = set()
    for record in records:
        if record.get("n_train") != 1752 or record.get("n_test") != 152:
            raise ValueError("Unexpected methylation split dimensions")
        if record.get("extrapolation_policy") != "error":
            raise ValueError("Coordinate clipping or endpoint assignment is prohibited")
        if record.get("prediction_metadata", {}).get("extrapolation") != "error":
            raise ValueError("Prediction provenance does not record extrapolation='error'")
        if record.get("training_coordinate_support") != axes.get("fitted_support"):
            raise ValueError("A split did not retain both fitted support endpoints")
        if record.get("phi_new_min", 0) <= 0 or not math.isfinite(
            record.get("phi_new_max", math.nan)
        ):
            raise ValueError("Held-out Beta precision must be finite and positive")
        scores = record.get("per_sample_log_predictive", ())
        if len(scores) != 152 or not all(math.isfinite(float(score)) for score in scores):
            raise ValueError("Every held-out score must be finite")
        if not math.isclose(
            sum(scores), record.get("total_log_predictive", math.nan), abs_tol=1e-9
        ):
            raise ValueError("Split total does not equal the per-sample score sum")
        test_hash = record.get("test_indices_hash")
        if not isinstance(test_hash, str) or test_hash in test_hashes:
            raise ValueError("Methylation test split hashes must be unique")
        test_hashes.add(test_hash)

    summary = payload.get("summary", {})
    if summary.get("n_splits") != 10 or summary.get("total_denominator") != 1520:
        raise ValueError("Unexpected predictive score denominator")
    total = sum(record["total_log_predictive"] for record in records)
    if not math.isclose(total, summary.get("total_log_predictive", math.nan), abs_tol=1e-9):
        raise ValueError("Summary total does not equal split totals")
    if not math.isclose(
        total / 1520, summary.get("pooled_mean_log_predictive", math.nan), abs_tol=1e-12
    ):
        raise ValueError("Summary mean does not equal total divided by denominator")
    if not str(summary.get("calibration_status", "")).startswith("not-computed"):
        raise ValueError("Uncertified Beta-observation calibration must remain unreported")
    resources = payload.get("resources", {})
    if resources.get("projected_full_wall_seconds") != resources.get("elapsed_wall_seconds"):
        raise ValueError("Full execution must report observed full runtime")


def split_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in payload["records"]:
        stability = record["boundary_stability_tau3"]
        rows.append(
            {
                "split": int(record["split_index"]) + 1,
                "seed": record["seed"],
                "test_start_index": record["test_start_index"],
                "test_stop_index_exclusive": record["test_stop_index_exclusive"],
                "n_train": record["n_train"],
                "n_test": record["n_test"],
                "total_log_predictive": record["total_log_predictive"],
                "mean_log_predictive": record["mean_log_predictive"],
                "k_map": record["k_map"],
                "boundary_stability_f1_tau3": stability["f1"],
                "boundary_stability_mae_tau3": stability["mae_or_na"],
                "phi_new_min": record["phi_new_min"],
                "phi_new_max": record["phi_new_max"],
                "fit_wall_seconds": record["fit_wall_seconds"],
                "score_wall_seconds": record["score_wall_seconds"],
            }
        )
    return rows


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = tuple(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_summary_markdown(
    payload: dict[str, Any],
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    summary = payload["summary"]
    score_interval = summary["split_mean_log_predictive"]
    stability_interval = summary["boundary_stability_f1_tau3"]
    lines = [
        "# RES-BB-RD-008Q corrected methylation posterior prediction",
        "",
        f"Parent: `{PARENT_RESULT_ID}`. Protocol: `{PROTOCOL_ID}`.",
        f"Scientific execution commit: `{payload['code']['commit']}`.",
        "",
        "The exact hashed methylKit chromosome-21 source contains 1,904 ordered CpGs. "
        "Ten predeclared, disjoint, stratified interior blocks each hold out 152 CpGs while "
        "retaining both global fitted-support endpoints. The fitted family is "
        "`BayesBreakBetaObs`; training coverage is `phi`, held-out coverage is positive "
        "`phi_new`, and every prediction uses `extrapolation=error`.",
        "",
        f"Across 1,520 held-out CpGs, total log predictive score was "
        f"{summary['total_log_predictive']:.3f} and pooled mean score was "
        f"{summary['pooled_mean_log_predictive']:.3f}. The mean of the ten split means was "
        f"{score_interval['mean']:.3f} (95% t interval {score_interval['ci95_lower']:.3f} "
        f"to {score_interval['ci95_upper']:.3f}). Mean boundary-stability F1@3 against the "
        f"model-derived full-data MAP was {stability_interval['mean']:.3f} "
        f"(95% t interval {stability_interval['ci95_lower']:.3f} to "
        f"{stability_interval['ci95_upper']:.3f}). All split fits selected 15 segments.",
        "",
        "| Split | Test indices | Mean log predictive | MAP segments | Stability F1@3 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['split']} | {row['test_start_index']}:{row['test_stop_index_exclusive']} "
            f"| {row['mean_log_predictive']:.3f} | {row['k_map']} | "
            f"{row['boundary_stability_f1_tau3']:.3f} |"
        )
    lines.extend(
        [
            "",
            "This corrected result is not numerically comparable to the excluded parent score: "
            "the observation-family predictive distribution and split definition both changed. "
            "The ten blocks are regions of one chromosome, not independent biological samples. "
            "No certified Beta-observation PIT helper or external biological changepoint truth "
            "is available, so calibration and external accuracy are not reported.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_figure(
    payload: dict[str, Any],
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    x = np.arange(1, len(rows) + 1)
    split_means = np.asarray([row["mean_log_predictive"] for row in rows])
    stability = np.asarray([row["boundary_stability_f1_tau3"] for row in rows])
    summary = payload["summary"]
    score_interval = summary["split_mean_log_predictive"]
    stability_interval = summary["boundary_stability_f1_tau3"]

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.2), sharex=True, layout="constrained")
    axes[0].bar(x, split_means, color="#16697A")
    axes[0].axhline(score_interval["mean"], color="#C23B22", linewidth=1.5, label="Mean")
    axes[0].axhspan(
        score_interval["ci95_lower"],
        score_interval["ci95_upper"],
        color="#C23B22",
        alpha=0.14,
        label="95% t interval",
    )
    axes[0].set_ylabel("Mean log predictive score")
    axes[0].set_title("Family-correct in-support methylation prediction")
    axes[0].legend(frameon=False, ncol=2)

    axes[1].plot(x, stability, marker="o", color="#4C956C", linewidth=1.5)
    axes[1].axhline(stability_interval["mean"], color="#7A5195", linewidth=1.5)
    axes[1].axhspan(
        stability_interval["ci95_lower"],
        stability_interval["ci95_upper"],
        color="#7A5195",
        alpha=0.14,
    )
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Boundary stability F1@3")
    axes[1].set_xlabel("Stratified chromosome block")
    axes[1].set_xticks(x)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_git_tree(root: Path, commit: str) -> str:
    listing = subprocess.check_output(
        ["git", "ls-tree", "-r", commit, "--", "src", "scripts", "tests"],
        cwd=root,
    )
    return hashlib.sha256(listing).hexdigest()


def relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    artifacts = finalize(args.results)
    print(json.dumps({name: str(path) for name, path in artifacts.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
