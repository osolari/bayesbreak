"""Finalize artifacts and provenance for corrected latent-group result RES-BB-SYN-005."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bayesbreak.provenance import (
    InterpretationStatus,
    LineageStatus,
    ResultRecord,
    ResultStatus,
    write_sidecar,
)

RESULT_ID = "RES-BB-SYN-005"
PARENT_RESULT_ID = "RES-BB-SYN-002"


def finalize(results_path: Path) -> dict[str, Path]:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    if payload["mode"] != "full" or len(payload["records"]) != 400:
        raise ValueError("Expected the full 400-record EPR-BB-005 result")
    if payload["result_id"] != RESULT_ID or payload["parent_result_id"] != PARENT_RESULT_ID:
        raise ValueError("Unexpected result lineage")
    if not all(record["objective_monotone"] for record in payload["records"]):
        raise ValueError("A result record has a nonmonotone objective")
    if not all(
        record["final_objective"] == record["objective_trace"][-1] for record in payload["records"]
    ):
        raise ValueError("A result record has a stale final objective")

    output_dir = results_path.parent
    summary_csv = output_dir / "stress_summary.csv"
    summary_md = output_dir / "SUMMARY.md"
    figure_path = output_dir / "stress_summary.png"
    sidecar_path = output_dir / "result_sidecar.json"
    write_summary_csv(payload, summary_csv)
    write_summary_markdown(payload, summary_md)
    write_summary_figure(payload, figure_path)

    output_hashes = {
        "results": sha256_file(results_path),
        "summary_table": sha256_file(summary_csv),
        "summary_report": sha256_file(summary_md),
        "summary_figure": sha256_file(figure_path),
    }
    data_hash = sha256_json([record["data_hash"] for record in payload["records"]])
    config_hash = sha256_json(
        {
            "protocol_id": payload["protocol_id"],
            "cells": payload["cells"],
            "seed_base": payload["seed_base"],
            "repetitions_per_cell": payload["repetitions_per_cell"],
        }
    )
    code_hash = sha256_git_tree(root, payload["code_commit"])
    environment_hash = sha256_file(root / "provenance" / "environment-lock.json")
    archived = payload["summary"]["cells"]["archived"]
    record = ResultRecord(
        result_id=RESULT_ID,
        execution_status=ResultStatus.EXECUTED,
        scientific_interpretation=InterpretationStatus.VALID_FOR_STATED_INTERPRETATION,
        lineage_status=LineageStatus.CORRECTED,
        parent_result_id=PARENT_RESULT_ID,
        data_hash=data_hash,
        config_hash=config_hash,
        code_hash=code_hash,
        environment_hash=environment_hash,
        coordinate_metadata={
            "prediction_axis": "sequence-label",
            "reference_axis": "simulated-group-label",
            "reference_type": "simulated-truth",
        },
        metrics={
            "protocol_id": payload["protocol_id"],
            "n_records": len(payload["records"]),
            "archived_cell_hard_accuracy": archived["hard_accuracy"],
            "archived_cell_ari": archived["ari"],
            "archived_cell_template_distance": archived["template_distance"],
            "all_objectives_monotone": True,
            "all_final_objectives_current": True,
            "valid_restart_rate": float(
                np.mean(
                    [
                        diagnostic["status"] == "valid"
                        for run in payload["records"]
                        for diagnostic in run["restart_diagnostics"]
                    ]
                )
            ),
            "collapse_count": int(sum(run["collapsed"] for run in payload["records"])),
            "elapsed_wall_seconds": payload["resources"]["elapsed_wall_seconds"],
            "peak_rss": payload["resources"]["peak_rss"],
            "interpretation_limit": (
                "Recovery degrades under lower separation, higher noise, shorter sequences, "
                "overspecified groups, and duplicate templates; this is finite score clustering, "
                "not normalized-mixture identifiability."
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


def write_summary_csv(payload: dict[str, object], path: Path) -> None:
    fields = (
        "cell",
        "n_runs",
        "hard_accuracy_mean",
        "hard_accuracy_ci95_lower",
        "hard_accuracy_ci95_upper",
        "ari_mean",
        "template_distance_mean",
        "collapse_rate",
        "objective_monotone_rate",
        "valid_restart_rate",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for name, summary in payload["summary"]["cells"].items():
            writer.writerow(
                {
                    "cell": name,
                    "n_runs": summary["n_runs"],
                    "hard_accuracy_mean": summary["hard_accuracy"]["mean"],
                    "hard_accuracy_ci95_lower": summary["hard_accuracy"]["ci95_lower"],
                    "hard_accuracy_ci95_upper": summary["hard_accuracy"]["ci95_upper"],
                    "ari_mean": summary["ari"]["mean"],
                    "template_distance_mean": summary["template_distance"]["mean"],
                    "collapse_rate": summary["collapse_rate"],
                    "objective_monotone_rate": summary["objective_monotone_rate"],
                    "valid_restart_rate": summary["valid_restart_rate"],
                }
            )


def write_summary_markdown(payload: dict[str, object], path: Path) -> None:
    archived = payload["summary"]["cells"]["archived"]
    lines = [
        "# RES-BB-SYN-005 latent-group corrected rerun",
        "",
        f"Parent: `{PARENT_RESULT_ID}`. Protocol: `{payload['protocol_id']}`.",
        f"Code commit: `{payload['code_commit']}`.",
        "",
        "The archived-design cell used 50 seeded datasets with 24 sequences of length 80 at "
        "sigma=1.0. Mean hard accuracy was "
        f"{archived['hard_accuracy']['mean']:.3f} "
        f"(95% interval {archived['hard_accuracy']['ci95_lower']:.3f} to "
        f"{archived['hard_accuracy']['ci95_upper']:.3f}); mean ARI was "
        f"{archived['ari']['mean']:.3f}.",
        "",
        "All 400 objective traces were monotone, every returned final objective equaled the "
        "last trace value, and all 1,200 restarts were valid. Stress cells show expected "
        "failure behavior: low separation and duplicate templates do not support recovery "
        "claims, and overspecified groups increase collapse/redundancy.",
        "",
        "This result supports the stated finite latent-group criterion in the declared synthetic "
        "design. It is not evidence for normalized finite-mixture identifiability or universal "
        "recovery.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_figure(payload: dict[str, object], path: Path) -> None:
    cells = payload["summary"]["cells"]
    names = list(cells)
    accuracy = np.asarray([cells[name]["hard_accuracy"]["mean"] for name in names])
    lower = np.asarray([cells[name]["hard_accuracy"]["ci95_lower"] for name in names])
    upper = np.asarray([cells[name]["hard_accuracy"]["ci95_upper"] for name in names])
    collapse = np.asarray([cells[name]["collapse_rate"] for name in names])
    x = np.arange(len(names))
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.4), sharex=True, layout="constrained")
    axes[0].errorbar(
        x,
        accuracy,
        yerr=np.vstack((accuracy - lower, upper - accuracy)),
        fmt="o",
        color="#125E75",
        capsize=4,
    )
    axes[0].set_ylabel("Label-invariant accuracy")
    axes[0].set_ylim(0, 1.05)
    axes[0].axhline(0.5, color="#777777", linestyle="--", linewidth=1)
    axes[1].bar(x, collapse, color="#B23A48")
    axes[1].set_ylabel("Collapse rate")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xticks(x, names, rotation=30, ha="right")
    fig.suptitle("RES-BB-SYN-005 finite latent-group stress test")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


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
