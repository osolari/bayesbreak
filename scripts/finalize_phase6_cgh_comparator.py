"""Finalize artifacts and provenance for corrected result RES-BB-CMP-003."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from bayesbreak.provenance import (
    InterpretationStatus,
    LineageStatus,
    ResultRecord,
    ResultStatus,
    write_sidecar,
)

RESULT_ID = "RES-BB-CMP-003"
PARENT_RESULT_ID = "RES-BB-CMP-002"
PROTOCOL_IDS = ("EPR-BB-010", "EPR-BB-013")
EXPECTED_SOURCE_SHA256 = "b82da97ffe6b5c431a60c3f811ee5c339708562126ed4a3d0b0344f2f2e09a63"
EXPECTED_MATRIX_SHA256 = "1551547d50564227ac020c196e56cae0b29c76ccf76eb4140d47628019cdb9a8"
EXPECTED_ALGORITHMS = {
    "pelt",
    "optimal_partitioning",
    "binary_segmentation",
    "wild_binary_segmentation",
}
DISPLAY_NAMES = {
    "pelt": "PELT",
    "optimal_partitioning": "Optimal partitioning",
    "binary_segmentation": "Binary segmentation",
    "wild_binary_segmentation": "Wild binary segmentation",
}


def finalize(results_path: Path) -> dict[str, Path]:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    validate_payload(payload)

    output_dir = results_path.parent
    summary_csv = output_dir / "comparator_summary.csv"
    summary_md = output_dir / "SUMMARY.md"
    figure_path = output_dir / "boundary_agreement.png"
    sidecar_path = output_dir / "result_sidecar.json"
    rows = comparator_rows(payload)
    write_summary_csv(rows, summary_csv)
    write_summary_markdown(payload, rows, summary_md)
    write_summary_figure(payload, rows, figure_path)

    output_hashes = {
        "results": sha256_file(results_path),
        "summary_table": sha256_file(summary_csv),
        "summary_report": sha256_file(summary_md),
        "summary_figure": sha256_file(figure_path),
    }
    shared = payload["bayesbreak"]["shared"]
    pelt = next(row for row in rows if row["algorithm"] == "pelt")
    record = ResultRecord(
        result_id=RESULT_ID,
        execution_status=ResultStatus.EXECUTED,
        scientific_interpretation=InterpretationStatus.VALID_FOR_STATED_INTERPRETATION,
        lineage_status=LineageStatus.CORRECTED,
        parent_result_id=PARENT_RESULT_ID,
        data_hash=payload["dataset_card"]["data_hash"],
        weights_hash=payload["dataset_card"]["descriptor_hash"],
        config_hash=payload["config_sha256"],
        code_hash=sha256_git_tree(root, payload["code"]["commit"]),
        environment_hash=payload["environment"]["sha256"],
        coordinate_metadata={
            "prediction_axis": "probe-index",
            "reference_axis": "probe-index",
            "reference_type": "model-derived-map",
            "raw_storage_orientation": "probe-by-subject",
            "comparator_request_orientation": "subject-by-probe",
            "n_probes": 2215,
            "n_subjects": 43,
        },
        metrics={
            "protocol_ids": list(PROTOCOL_IDS),
            "shared_k_map": shared["k_map"],
            "shared_boundaries": shared["boundaries"],
            "shared_log_evidence": shared["log_evidence"],
            "independent_log_evidence_sum": sum(
                record["log_evidence"] for record in payload["bayesbreak"]["independent"]
            ),
            "comparator_rows": rows,
            "pelt_grid_attained_target_count": pelt["count_status"] == "exact-matched-k",
            "elapsed_wall_seconds": payload["resources"]["elapsed_wall_seconds"],
            "peak_rss": payload["resources"]["peak_rss"],
            "captured_warnings": payload["warnings"],
            "interpretation_limit": (
                "Agreement is measured against the model-derived BayesBreak MAP on the common "
                "probe axis, not external biological truth. PELT reports the closest result in "
                "the predeclared eight-penalty grid because that grid did not attain 14 boundaries."
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
    if tuple(payload.get("protocol_ids", ())) != PROTOCOL_IDS:
        raise ValueError("Unexpected experiment protocols")
    source = payload.get("source", {})
    if source.get("sha256") != EXPECTED_SOURCE_SHA256:
        raise ValueError("Unexpected raw source hash")
    if source.get("matrix_sha256") != EXPECTED_MATRIX_SHA256:
        raise ValueError("Unexpected parsed matrix hash")
    if source.get("matrix_shape") != [2215, 43]:
        raise ValueError("Expected the full 2215-probe by 43-subject matrix")
    config = payload.get("config", {})
    if config.get("n_probes") != 2215 or config.get("n_subjects") != 43:
        raise ValueError("Expected a full-data configuration")
    axes = payload.get("coordinate_metadata", {})
    if axes.get("prediction_axis") != "probe-index" or axes.get("reference_axis") != "probe-index":
        raise ValueError("Comparator and reference axes must both be probe-index")
    if axes.get("reference_type") != "model-derived-map":
        raise ValueError("Expected a model-derived MAP reference")
    if axes.get("external_annotations") != "none-independently-verified":
        raise ValueError("External annotation status changed")
    shared = payload.get("bayesbreak", {}).get("shared", {})
    if shared.get("k_map") != 15 or len(shared.get("boundaries", ())) != 14:
        raise ValueError("Expected the reproduced 15-segment shared reference")
    independent = payload.get("bayesbreak", {}).get("independent", ())
    if len(independent) != 43:
        raise ValueError("Expected all 43 independent subject fits")
    comparators = payload.get("comparators", ())
    if {record.get("algorithm") for record in comparators} != EXPECTED_ALGORITHMS:
        raise ValueError("Comparator set is incomplete")
    for record in comparators:
        if record.get("n") != 2215:
            raise ValueError("Comparator result is not on the full probe axis")
        metric = record.get("boundary_metrics_tau3", {})
        if metric.get("prediction_axis") != "probe-index":
            raise ValueError("Comparator metric prediction axis changed")
        if metric.get("reference_axis") != "probe-index":
            raise ValueError("Comparator metric reference axis changed")
        if metric.get("reference_type") != "model-derived-map":
            raise ValueError("Comparator metric reference type changed")
    fixed_count = [record for record in comparators if record["algorithm"] != "pelt"]
    if not all(len(record["boundaries"]) == 14 for record in fixed_count):
        raise ValueError("A fixed-count comparator did not return 14 boundaries")
    pelt = next(record for record in comparators if record["algorithm"] == "pelt")
    candidates = pelt["tuning"]["candidates"]
    selected = candidates[pelt["tuning"]["selected_index"]]
    target = pelt["tuning"]["target_n_bkps"]
    selected_key = (abs(selected["n_bkps"] - target), abs(_log2(selected["multiplier"])))
    if selected_key != min(
        (abs(candidate["n_bkps"] - target), abs(_log2(candidate["multiplier"])))
        for candidate in candidates
    ):
        raise ValueError("PELT selection does not follow the predeclared count rule")


def comparator_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in payload["comparators"]:
        metric = record["boundary_metrics_tau3"]
        target = record["tuning"]["target_n_bkps"]
        n_bkps = len(record["boundaries"])
        candidates = record["tuning"].get("candidates", ())
        rows.append(
            {
                "algorithm": record["algorithm"],
                "package_version": record["package_version"],
                "target_n_bkps": target,
                "n_bkps": n_bkps,
                "count_status": (
                    "exact-matched-k" if n_bkps == target else "closest-grid-count-mismatch"
                ),
                "parameter_evaluations": len(candidates) if candidates else 1,
                "selected_runtime_seconds": record["runtime_seconds"],
                "tuning_runtime_seconds": (
                    sum(candidate["runtime_seconds"] for candidate in candidates)
                    if candidates
                    else record["runtime_seconds"]
                ),
                "exact_boundary_jaccard": record["exact_boundary_jaccard"],
                "f1_tau3": metric["f1"],
                "matched_mae_tau3": metric["mae_or_na"],
                "matched_count_tau3": len(metric["matched_pairs"]),
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
    lines = [
        "# RES-BB-CMP-003 corrected array-CGH comparator",
        "",
        f"Parent: `{PARENT_RESULT_ID}`. Protocols: `EPR-BB-010`, `EPR-BB-013`.",
        f"Scientific execution commit: `{payload['code']['commit']}`.",
        "",
        "The exact hashed CRAN ecp ACGH matrix contains 2,215 probes and 43 subjects. "
        "The shared BayesBreak fit reproduced 15 MAP segments and the archived pooled log "
        f"evidence ({payload['bayesbreak']['shared']['log_evidence']:.10f}). Comparators ran "
        "on the unflattened raw matrix and were scored on the same probe-index axis.",
        "",
        "| Algorithm | Boundaries (target 14) | Count status | F1@3 | Matched MAE@3 | "
        "Exact Jaccard |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        mae = "NA" if row["matched_mae_tau3"] is None else f"{row['matched_mae_tau3']:.3f}"
        lines.append(
            f"| {DISPLAY_NAMES[row['algorithm']]} | {row['n_bkps']} | "
            f"{row['count_status']} | "
            f"{row['f1_tau3']:.3f} | {mae} | {row['exact_boundary_jaccard']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Dynp, binary segmentation, and WBS use the shared BayesBreak MAP boundary count. "
            "The predeclared eight-value PELT penalty grid did not attain 14 boundaries; its "
            "closest candidate returned 11 and is reported as a count mismatch without "
            "post-hoc retuning.",
            "",
            "These are agreement diagnostics against a model-derived MAP reference, not "
            "external biological accuracy or evidence of predictive superiority. No "
            "independently verified external changepoint annotations are available.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_figure(
    payload: dict[str, Any],
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    reference = payload["bayesbreak"]["shared"]["boundaries"]
    records = {record["algorithm"]: record for record in payload["comparators"]}
    algorithm_order = [row["algorithm"] for row in rows]
    labels = ["BayesBreak MAP", *[DISPLAY_NAMES[algorithm] for algorithm in algorithm_order]]
    colors = ["#111111", "#C23B22", "#16697A", "#4C956C", "#7A5195"]
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 6.0), layout="constrained")
    for index, (algorithm, color) in enumerate(zip([None, *algorithm_order], colors, strict=True)):
        boundaries = reference if algorithm is None else records[algorithm]["boundaries"]
        axes[0].vlines(boundaries, index - 0.35, index + 0.35, color=color, linewidth=1.5)
    axes[0].set_xlim(0, 2215)
    axes[0].set_yticks(range(len(labels)), labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Probe index")
    axes[0].set_title("Common-axis boundary locations")

    algorithms = [DISPLAY_NAMES[row["algorithm"]] for row in rows]
    positions = range(len(rows))
    width = 0.36
    axes[1].bar(
        [position - width / 2 for position in positions],
        [row["f1_tau3"] for row in rows],
        width,
        label="F1@3",
        color="#16697A",
    )
    axes[1].bar(
        [position + width / 2 for position in positions],
        [row["exact_boundary_jaccard"] for row in rows],
        width,
        label="Exact Jaccard",
        color="#C23B22",
    )
    axes[1].set_xticks(list(positions), algorithms, rotation=18, ha="right")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Agreement")
    axes[1].legend(frameon=False)
    axes[1].set_title("Agreement with model-derived BayesBreak MAP")
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


def _log2(value: float) -> float:
    import math

    return math.log2(float(value))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    artifacts = finalize(args.results)
    print(json.dumps({name: str(path) for name, path in artifacts.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
