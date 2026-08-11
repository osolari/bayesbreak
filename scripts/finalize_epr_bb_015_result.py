"""Finalize pending-review artifacts for RES-BB-SYN-006."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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

RESULT_ID = "RES-BB-SYN-006"
PROTOCOL_ID = "EPR-BB-015"

PRIMARY_FAILURE_METRICS = {
    "null-gaussian": ("false_positive_dataset_rate", False),
    "heavy-tail-gaussian": ("complete_boundary_recovery_rate", True),
    "zero-inflated-poisson": ("map_saturation_rate", False),
    "dense-gaussian": ("map_saturation_rate", False),
    "short-segment-gaussian": ("complete_boundary_recovery_rate", True),
    "prior-conflict-gaussian": ("missed_change_rate", False),
    "shared-boundary-heterogeneity": ("subject_deviation_selected_rate", False),
    "logistic-approximation-failure": ("ep_timeout_rate", False),
}


def finalize(results_path: Path) -> dict[str, Path]:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    validate_payload(payload)

    output_dir = results_path.parent
    summary_csv = output_dir / "failure_summary.csv"
    summary_md = output_dir / "SUMMARY.md"
    figure_path = output_dir / "failure_map.png"
    sidecar_path = output_dir / "result_sidecar.json"

    rows = failure_rows(payload)
    write_summary_csv(rows, summary_csv)
    write_summary_markdown(payload, rows, summary_md)
    write_failure_map(payload, rows, figure_path)

    output_hashes = {
        "results": sha256_file(results_path),
        "summary_table": sha256_file(summary_csv),
        "summary_report": sha256_file(summary_md),
        "summary_figure": sha256_file(figure_path),
    }
    record = ResultRecord(
        result_id=RESULT_ID,
        execution_status=ResultStatus.EXECUTED,
        scientific_interpretation=InterpretationStatus.PENDING,
        lineage_status=LineageStatus.ORIGINAL,
        parent_result_id=None,
        data_hash=aggregate_input_hash(payload),
        config_hash=payload["config_sha256"],
        code_hash=payload["code"]["commit_sha256"],
        environment_hash=payload["environment"]["sha256"],
        coordinate_metadata={
            "prediction_axis": "cell-specific observation-index or subject partition",
            "reference_axis": (
                "predeclared simulated truth or high-accuracy approximation reference"
            ),
            "reference_type": "simulated-truth and approximation-reference",
        },
        metrics={
            "protocol_id": PROTOCOL_ID,
            "n_cells": len(payload["cell_ids"]),
            "n_records": len(payload["records"]),
            "repetitions_per_cell": payload["repetitions_per_cell"],
            "all_top_level_records_executed": True,
            "ep_fit_timeout_count": sum(
                record["methods"]["ep"]["status"] == "timed-out"
                for record in payload["records"]
                if record["cell"] == "logistic-approximation-failure"
            ),
            "elapsed_wall_seconds": payload["resources"]["elapsed_wall_seconds"],
            "peak_rss": payload["resources"]["peak_rss"],
            "primary_failure_rates": {row["cell"]: row["primary_failure_rate"] for row in rows},
            "interpretation_limit": (
                "This suite maps declared failure regimes. It is not a universal robustness "
                "claim; scientific interpretation remains pending independent review."
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
    if payload.get("result_id") != RESULT_ID or payload.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Unexpected EPR-BB-015 result identity")
    if payload.get("mode") != "full" or payload.get("repetitions_per_cell") != 50:
        raise ValueError("Expected the full 50-repetition EPR-BB-015 result")
    if len(payload.get("records", [])) != 400:
        raise ValueError("Expected all 400 EPR-BB-015 records")
    if any(record["status"] != "executed" for record in payload["records"]):
        raise ValueError("Every top-level outcome must be retained as executed")
    for cell_id in payload["cell_ids"]:
        records = [record for record in payload["records"] if record["cell"] == cell_id]
        if len(records) != 50:
            raise ValueError(f"Expected 50 retained records for {cell_id}")
    logistic = [
        record
        for record in payload["records"]
        if record["cell"] == "logistic-approximation-failure"
    ]
    if any(record["methods"]["ep"]["status"] != "timed-out" for record in logistic):
        raise ValueError("Every EP timeout must remain explicit")
    if any(
        key in record["methods"]["ep"]
        for record in logistic
        for key in ("diagnostics", "truth_metrics")
    ):
        raise ValueError("Timed-out EP records cannot contain imputed diagnostics")


def failure_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summaries = payload["summary"]["cells"]
    for cell_id in payload["cell_ids"]:
        summary = summaries[cell_id]
        metric_name, reverse = PRIMARY_FAILURE_METRICS[cell_id]
        interval = summary[metric_name]
        if interval is None:
            raise ValueError(f"Missing primary failure metric for {cell_id}")
        if reverse:
            rate = 1.0 - interval["mean"]
            lower = 1.0 - interval["ci95_upper"]
            upper = 1.0 - interval["ci95_lower"]
            display_name = f"one_minus_{metric_name}"
        else:
            rate = interval["mean"]
            lower = interval["ci95_lower"]
            upper = interval["ci95_upper"]
            display_name = metric_name
        rows.append(
            {
                "cell": cell_id,
                "n_runs": summary["n_runs"],
                "n_failed": summary["n_failed"],
                "primary_failure_metric": display_name,
                "primary_failure_rate": rate,
                "primary_failure_ci95_lower": max(0.0, lower),
                "primary_failure_ci95_upper": min(1.0, upper),
                "boundary_f1_mean": value(summary, "boundary_f1"),
                "complete_recovery_rate": value(summary, "complete_boundary_recovery_rate"),
                "map_saturation_rate": value(summary, "map_saturation_rate"),
                "missed_change_rate": value(summary, "missed_change_rate"),
                "wall_seconds_mean": summary["wall_seconds"]["mean"],
            }
        )
    return rows


def value(summary: dict[str, Any], key: str) -> float | None:
    interval = summary.get(key)
    return None if interval is None else float(interval["mean"])


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_summary_markdown(payload: dict[str, Any], rows: list[dict[str, Any]], path: Path) -> None:
    summaries = payload["summary"]["cells"]
    shared = summaries["shared-boundary-heterogeneity"]
    logistic = summaries["logistic-approximation-failure"]
    row_map = {row["cell"]: row for row in rows}
    lines = [
        "# RES-BB-SYN-006 misspecification and negative-control suite",
        "",
        f"Protocol: `{PROTOCOL_ID}`. Code commit: `{payload['code']['commit']}`.",
        "Scientific interpretation: **pending independent review**.",
        "",
        "All 400 predeclared datasets were retained: 50 in each of eight cells. No "
        "top-level cell failed. All 50 EP fits reached the predeclared 20-second "
        "fit-only timeout; no EP diagnostics were imputed.",
        "",
        "## Predeclared failure indicators",
        "",
        "| Cell | Failure indicator | Rate | 95% interval |",
        "|---|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['cell']}` | `{row['primary_failure_metric']}` | "
            f"{row['primary_failure_rate']:.3f} | "
            f"{row['primary_failure_ci95_lower']:.3f} to "
            f"{row['primary_failure_ci95_upper']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Key observations",
            "",
            f"- Null Gaussian false-positive dataset rate: "
            f"{row_map['null-gaussian']['primary_failure_rate']:.3f}.",
            f"- Zero-inflated Poisson and dense Gaussian MAP saturation rates: "
            f"{row_map['zero-inflated-poisson']['primary_failure_rate']:.3f} and "
            f"{row_map['dense-gaussian']['primary_failure_rate']:.3f}.",
            f"- Short-segment exact recovery rate: "
            f"{summaries['short-segment-gaussian']['complete_boundary_recovery_rate']['mean']:.3f}.",
            "- Prior-conflict fits assigned zero posterior mass to unsupported segment counts "
            "and missed the truth-compatible boundaries in every dataset.",
            f"- Shared mean subject F1 was {shared['shared_mean_subject_f1']['mean']:.3f}; "
            f"mean independent F1 was {shared['independent_mean_f1']['mean']:.3f}.",
            f"- Mean empirical posterior TV was "
            f"{logistic['quadrature-40_empirical_tv']['mean']:.3f} for quadrature-40 and "
            f"{logistic['laplace_empirical_tv']['mean']:.3f} for Laplace. Large maximum "
            "block errors and smaller posterior TV must be reported together.",
            "",
            "These outcomes map the declared failure regimes. They are not evidence of "
            "universal robustness, model superiority, or external-truth accuracy. Acceptance "
            "for manuscript conclusions remains pending independent scientific review.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_failure_map(payload: dict[str, Any], rows: list[dict[str, Any]], path: Path) -> None:
    summaries = payload["summary"]["cells"]
    names = [row["cell"].replace("-gaussian", "").replace("-", " ") for row in rows]
    rates = np.asarray([row["primary_failure_rate"] for row in rows])
    lower = np.asarray([row["primary_failure_ci95_lower"] for row in rows])
    upper = np.asarray([row["primary_failure_ci95_upper"] for row in rows])
    x = np.arange(len(rows))

    fig, axes = plt.subplots(2, 1, figsize=(11.0, 7.2), layout="constrained")
    axes[0].errorbar(
        x,
        rates,
        yerr=np.vstack((np.maximum(0.0, rates - lower), np.maximum(0.0, upper - rates))),
        fmt="o",
        color="#B23A48",
        capsize=4,
    )
    axes[0].set_ylabel("Predeclared failure-indicator rate")
    axes[0].set_ylim(-0.03, 1.05)
    axes[0].set_xticks(x, names, rotation=25, ha="right")
    axes[0].set_title("Failure indicators are cell-specific and not directly comparable")

    shared = summaries["shared-boundary-heterogeneity"]
    logistic = summaries["logistic-approximation-failure"]
    labels = ["Shared F1", "Independent F1", "Quadrature TV", "Laplace TV"]
    values = [
        shared["shared_mean_subject_f1"]["mean"],
        shared["independent_mean_f1"]["mean"],
        logistic["quadrature-40_empirical_tv"]["mean"],
        logistic["laplace_empirical_tv"]["mean"],
    ]
    axes[1].bar(labels, values, color=["#125E75", "#3A7D44", "#D49F32", "#8F5D9F"])
    axes[1].set_ylim(0, 1.0)
    axes[1].set_ylabel("Mean recorded metric")
    axes[1].set_title("Same-truth F1 contrast and executed approximation sensitivity")
    fig.suptitle("RES-BB-SYN-006 pending-review failure map")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def aggregate_input_hash(payload: dict[str, Any]) -> str:
    identity = [
        {
            "cell": record["cell"],
            "seed": record["seed"],
            "data_hash": record["data_hash"],
            "truth_hash": record["truth_hash"],
            "effective_config_hash": record["effective_config_hash"],
            "weights_hash": record.get("weights_hash"),
        }
        for record in payload["records"]
    ]
    return sha256_json(identity)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
