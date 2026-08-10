#!/usr/bin/env python3
"""Check synchronization of the canonical BayesBreak coding handoff."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "revision_artifacts/phase6/HANDOFF_SYNC_VALIDATION.json"


def load_renderer():
    path = ROOT / "scripts/render_canonical_handoff.py"
    spec = importlib.util.spec_from_file_location("render_canonical_handoff", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load renderer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def unique_ids(records: list[dict], key: str) -> bool:
    values = [r[key] for r in records]
    return len(values) == len(set(values))


def main() -> int:
    renderer = load_renderer()
    data = json.loads(renderer.SRC.read_text())
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}

    expected = {
        renderer.TEX_DETAIL: renderer.render_detail(data),
        renderer.TEX_SUMMARY: renderer.render_summary(data),
        renderer.SYNC_MANIFEST: renderer.render_sync_manifest(data),
    }
    for path, text in expected.items():
        checks[f"generated:{path.relative_to(ROOT)}"] = path.exists() and path.read_text() == text

    checks["unique_task_ids"] = unique_ids(data["tasks"], "id")
    checks["unique_experiment_ids"] = unique_ids(data["experiments"], "id")
    checks["unique_claim_ids"] = unique_ids(data["claims"], "claim_id")
    checks["unique_result_ids"] = unique_ids(data["results"], "result_id")
    checks["unique_failure_ids"] = unique_ids(data["failure_states"], "id")

    claim_doc = json.loads((ROOT / "shared/metadata/claim_traceability.json").read_text())
    exp_doc = json.loads((ROOT / "shared/metadata/experiment_protocols.json").read_text())
    result_doc = json.loads((ROOT / "shared/metadata/result_interpretation.json").read_text())
    checks["claims_match_metadata"] = data["claims"] == claim_doc["claims"]
    checks["experiments_match_metadata"] = data["experiments"] == exp_doc["protocols"]
    checks["results_match_metadata"] = data["results"] == result_doc["results"]

    appendix = (ROOT / "book/appendices/coding_agent_handoff.tex").read_text()
    checks["book_appendix_uses_generated_source"] = "shared/handoffs/coding_agent_handoff.tex" in appendix

    details["counts"] = {
        "tasks": len(data["tasks"]),
        "experiments": len(data["experiments"]),
        "claims": len(data["claims"]),
        "results": len(data["results"]),
        "failure_states": len(data["failure_states"]),
        "stages": len(data["stage_sequence"]),
    }
    report = {"passed": all(checks.values()), "checks": checks, "details": details}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    if report["passed"]:
        print("HANDOFF SYNC CHECK PASSED")
        print(json.dumps(details["counts"], sort_keys=True))
        return 0
    print("HANDOFF SYNC CHECK FAILED")
    for name, ok in checks.items():
        if not ok:
            print("-", name)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
