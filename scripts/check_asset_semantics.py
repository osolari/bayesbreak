"""Generate and validate semantic metadata for immutable archived result assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs" / "manuscript"
HASH_BASELINE = REPORT / "revision_artifacts" / "phase5" / "PHASE4R_READONLY_HASHES.json"
RESULT_REGISTRY = REPORT / "shared" / "metadata" / "result_interpretation.json"
VISUAL_QA = REPORT / "revision_artifacts" / "phase6" / "VISUAL_QA.json"
MANIFEST = ROOT / "provenance" / "archived-asset-manifest.json"

_LINKS = {
    "fig1_synthetic_gaussian": (["RES-BB-SYN-001"], "fig:single_synth", "simulated-truth"),
    "fig2_family_showcase": (["RES-BB-SYN-004"], "fig:family_showcase", "simulated-truth"),
    "fig3_boundary_calibration": (["RES-BB-SYN-001"], "fig:calibration", "simulated-truth"),
    "fig4_latent_groups": (["RES-BB-SYN-002"], "fig:latent_groups", "simulated-truth"),
    "fig4_latent_groups_cropped": (["RES-BB-SYN-002"], "fig:latent_groups", "simulated-truth"),
    "fig5_runtime_scaling": (["RES-BB-SYN-003"], "fig:runtime", "not-applicable"),
    "fig6_welllog": (["RES-BB-RD-001", "RES-BB-RD-002"], "fig:welllog", "fitted-map"),
    "fig7_cgh": (["RES-BB-RD-003", "RES-BB-RD-004"], "fig:cgh", "fitted-map"),
    "fig8_spx": (["RES-BB-RD-005", "RES-BB-RD-006"], "fig:spx", "fitted-map"),
    "fig9_methylation": (["RES-BB-RD-007"], "fig:methylation", "fitted-map"),
    "baselines_metrics": (
        ["RES-BB-CMP-001", "RES-BB-CMP-002"],
        "paper-app:comparison-tables",
        "agreement-diagnostic",
    ),
    "realdata_metrics": (
        ["RES-BB-RD-001", "RES-BB-RD-003", "RES-BB-RD-005", "RES-BB-RD-007", "RES-BB-RD-007Q"],
        "tab:real-summary-paper",
        "mixed-diagnostic",
    ),
    "table1_runtime_scaling": (["RES-BB-SYN-003"], "tab:runtime_scaling", "not-applicable"),
    "table2_posterior_summary": (["RES-BB-SYN-001"], "tab:posterior_summary", "simulated-truth"),
    "table3_conjugate_summary": (["RES-BB-SYN-004"], "tab:single_quant", "simulated-truth"),
    "table4_nonconj_tradeoff": (["RES-BB-SYN-004"], "tab:nonconj_tradeoff", "simulated-truth"),
    "table0_metrics_overview": ([], "paper:metric-definitions", "not-applicable"),
}


def build_manifest() -> dict[str, object]:
    baseline = json.loads(HASH_BASELINE.read_text(encoding="utf-8"))["files"]
    result_records = {
        item["result_id"]: item
        for item in json.loads(RESULT_REGISTRY.read_text(encoding="utf-8"))["results"]
    }
    assets: list[dict[str, object]] = []
    for relative_path, expected_hash in sorted(baseline.items()):
        path = REPORT / relative_path
        stem = path.stem
        result_ids, caption_anchor, marker_role = _LINKS.get(
            stem,
            ([], "not-in-active-report", "not-declared"),
        )
        interpretations = {
            result_id: result_records[result_id]["interpretation_status"]
            for result_id in result_ids
        }
        assets.append(
            {
                "path": f"docs/manuscript/{relative_path}",
                "sha256": expected_hash,
                "status": "archived-read-only",
                "result_ids": result_ids,
                "interpretation_status": interpretations,
                "caption_anchor": caption_anchor,
                "marker_role": marker_role,
                "external_annotation_source": None,
            }
        )
    return {
        "schema_version": "1.0.0",
        "generated_from": [
            "docs/manuscript/revision_artifacts/phase5/PHASE4R_READONLY_HASHES.json",
            "docs/manuscript/shared/metadata/result_interpretation.json",
        ],
        "asset_count": len(assets),
        "assets": assets,
    }


def validate_manifest(manifest: dict[str, object] | None = None) -> dict[str, object]:
    current = manifest or build_manifest()
    assets = current["assets"]
    hash_mismatches: list[str] = []
    missing_assets: list[str] = []
    missing_semantics: list[str] = []
    for record in assets:
        path = ROOT / record["path"]
        if not path.exists():
            missing_assets.append(record["path"])
            continue
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != record["sha256"]:
            hash_mismatches.append(record["path"])
        if not all(
            key in record
            for key in (
                "status",
                "result_ids",
                "interpretation_status",
                "caption_anchor",
                "marker_role",
                "external_annotation_source",
            )
        ):
            missing_semantics.append(record["path"])

    sources = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in (
            REPORT / "book" / "chapters" / "14-real-data.tex",
            REPORT / "paper" / "sections" / "09-results.tex",
            REPORT / "paper" / "sections" / "06-approximation-prediction.tex",
            REPORT / "executive" / "sections" / "04-limitations.tex",
        )
    )
    exclusion_checks = {
        "RES-BB-CMP-002": "RES-BB-CMP-002" in sources and "excluded" in sources.lower(),
        "RES-BB-RD-007Q": "RES-BB-RD-007Q" in sources and "excluded" in sources.lower(),
    }
    marker_semantics = all(
        text in sources
        for text in (
            "fitted MAP boundaries",
            "not independently annotated macro events",
            "no external boundary annotation is asserted",
        )
    )
    visual_qa = json.loads(VISUAL_QA.read_text(encoding="utf-8"))
    passed = bool(
        current["asset_count"] == len(assets) == 53
        and not hash_mismatches
        and not missing_assets
        and not missing_semantics
        and all(exclusion_checks.values())
        and marker_semantics
        and visual_qa.get("status") == "pass"
        and not visual_qa.get("final_findings", {}).get("reported_defects")
    )
    return {
        "passed": passed,
        "asset_count": len(assets),
        "hash_mismatches": hash_mismatches,
        "missing_assets": missing_assets,
        "missing_semantics": missing_semantics,
        "exclusion_checks": exclusion_checks,
        "marker_semantics": marker_semantics,
        "visual_qa_status": visual_qa.get("status"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    manifest = build_manifest()
    if args.write:
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    result = validate_manifest(manifest)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
