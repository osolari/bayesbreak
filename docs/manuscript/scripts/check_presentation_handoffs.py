#!/usr/bin/env python3
"""Validate final presentation-source handoffs; no slides are generated."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "presentation_handoffs"
REPORT = ROOT / "revision_artifacts/phase6/PRESENTATION_HANDOFF_VALIDATION.json"
TITLE = "Generalized Hierarchical Bayesian Segmentation with Irregular Designs, Multi-Sequence Hierarchies, and Grouped/Latent-Group Designs"
ROADMAPS = [
    "shared/figures/tikz/roadmaps/executive_implementation_sequence.tex",
    "shared/figures/tikz/roadmaps/executive_release_decisions.tex",
]
REQUIRED = [
    "shared/PRESENTATION_SOURCE_OF_TRUTH.md",
    "shared/EVIDENCE_STATUS_LEDGER.md",
    "shared/ASSET_AND_DIAGRAM_INDEX.md",
    "shared/CITATION_MAP.md",
    "shared/NOTATION_AND_TERMINOLOGY.md",
    "shared/AVAILABLE_TEMPLATES_AND_BUILD_TARGETS.md",
    "shared/PRESENTATION_DO_NOT_CHANGE.md",
    "technical/TECHNICAL_DECK_BRIEF.md",
    "technical/REQUIRED_THEORY_AND_RESULTS.md",
    "technical/TECHNICAL_NARRATIVE_BOUNDARY.md",
    "technical/MAIN_ROUTE_AND_APPENDIX_GUIDANCE.md",
    "executive/EXECUTIVE_DECK_BRIEF.md",
    "executive/DECISION_AND_VALUE_SUMMARY.md",
    "executive/EXECUTIVE_NARRATIVE_BOUNDARY.md",
    "executive/EXECUTIVE_DO_NOT_CHANGE.md",
]
BANNED = [r"\bauditable\b", r"\bauditability\b", r"\bcontracts?\b", r"local-to-global", r"evidence architecture", r"positive-score", r"\bquarantin(?:e|ed|ing)\b"]


def main() -> int:
    checks: dict[str, bool] = {}
    texts: dict[str, str] = {}
    for rel in REQUIRED:
        path = BASE / rel
        checks[f"exists:{rel}"] = path.is_file()
        texts[rel] = path.read_text() if path.is_file() else ""

    source = texts["shared/PRESENTATION_SOURCE_OF_TRUTH.md"]
    checks["exact_title"] = TITLE in source and TITLE in texts["shared/PRESENTATION_DO_NOT_CHANGE.md"]
    checks["test_state_present"] = all(token in source for token in ["RES-BB-QA-003", "179", "173 passed", "five skipped", "one EP logistic-normal timeout", "zero failed", "RES-BB-QA-002"])
    checks["excluded_results_present"] = all(token in source for token in ["RES-BB-CMP-002", "RES-BB-RD-007Q"])

    tech_text = "\n".join(texts[r] for r in REQUIRED if r.startswith("technical/"))
    exec_text = "\n".join(texts[r] for r in REQUIRED if r.startswith("executive/"))
    checks["technical_has_no_roadmap_sources"] = "shared/figures/tikz/roadmaps/" not in tech_text
    checks["technical_explicit_no_roadmap_rule"] = "must not contain" in texts["technical/TECHNICAL_DECK_BRIEF.md"] and "roadmap" in texts["technical/TECHNICAL_DECK_BRIEF.md"]
    checks["executive_has_exact_approved_roadmaps"] = all(path in exec_text for path in ROADMAPS) and exec_text.count("shared/figures/tikz/roadmaps/") == 2

    slide_paths = [p for p in ROOT.rglob("*") if p.is_file() and any(part.lower().startswith("slide") for part in p.parts)]
    checks["no_slide_sources"] = not slide_paths

    scientific_text = "\n".join(texts.values()).lower()
    checks["no_rejected_branding"] = not any(re.search(pattern, scientific_text, flags=re.I) for pattern in BANNED)

    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "required_file_count": len(REQUIRED),
        "approved_roadmaps": ROADMAPS,
        "slide_source_paths": [str(p.relative_to(ROOT)) for p in slide_paths],
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    if report["passed"]:
        print("PRESENTATION HANDOFF CHECK PASSED")
        print(f"files={len(REQUIRED)} roadmaps={len(ROADMAPS)} slides=0")
        return 0
    print("PRESENTATION HANDOFF CHECK FAILED")
    for name, ok in checks.items():
        if not ok:
            print("-", name)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
