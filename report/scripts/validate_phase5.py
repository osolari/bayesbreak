#!/usr/bin/env python3
"""Mechanical validation for the BayesBreak Phase 5 release."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PHASE_DIR = ROOT / "revision_artifacts" / "phase5"
OUT = PHASE_DIR / "BayesBreak_PHASE_5_VALIDATION.json"
EXPECTATIONS = PHASE_DIR / "RELEASE_EXPECTATIONS.json"
READONLY_HASHES = PHASE_DIR / "PHASE4R_READONLY_HASHES.json"

TITLE = (
    "Generalized Hierarchical Bayesian Segmentation with Irregular Designs, "
    "Multi-Sequence Hierarchies, and Grouped/Latent-Group Designs"
)
OLD_TITLE = "BayesBreak: Exact and Auditable Bayesian Segmentation from Local Block Evidence"

BANNED_SCIENTIFIC_LANGUAGE: dict[str, str] = {
    "auditable_or_auditability": r"\baudit(?:able|ability)\b",
    "generic_contract": r"\bcontracts?\b",
    "evidence_architecture": r"\bevidence architecture\b",
    "evidence_gate": r"\bevidence gates?\b",
    "local_to_global_branding": r"\blocal[- ]to[- ]global\b",
    "positive_score_branding": r"\bpositive[- ]score\b",
    "quarantine_branding": r"\bquarantin\w*\b",
    "protected_method_identity": r"\bprotected method identity\b",
}

FORMAL_ENVS = ("theorem", "proposition", "lemma", "corollary")

TARGETS: dict[str, dict[str, str]] = {
    "technical_book": {
        "pdf": "build/bayesbreak-technical-book.pdf",
        "log": "build/bayesbreak-technical-book.log",
        "page_size": "595.276 x 841.89 pts (A4)",
    },
    "main_paper_two_column": {
        "pdf": "build/paper/bayesbreak-main-paper.pdf",
        "log": "build/paper/bayesbreak-main-paper.log",
        "page_size": "612 x 792 pts (letter)",
    },
    "main_paper_single_column": {
        "pdf": "build/paper-single/bayesbreak-main-paper-single.pdf",
        "log": "build/paper-single/bayesbreak-main-paper-single.log",
        "page_size": "612 x 792 pts (letter)",
    },
    "executive_summary": {
        "pdf": "build/executive/bayesbreak-executive-summary.pdf",
        "log": "build/executive/bayesbreak-executive-summary.log",
        "page_size": "612 x 792 pts (letter)",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(*args: str) -> str:
    return subprocess.check_output(args, text=True, errors="replace")


def normalize_space(text: str) -> str:
    return " ".join(text.split())


def pdf_info(path: Path) -> dict[str, str]:
    record: dict[str, str] = {}
    for line in run("pdfinfo", str(path)).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            record[key.strip()] = value.strip()
    return record


def pdf_text(path: Path) -> str:
    return run("pdftotext", "-layout", str(path), "-")


def page_lengths(text: str) -> list[int]:
    pages = text.split("\f")
    if pages and not pages[-1].strip():
        pages.pop()
    return [len(normalize_space(page)) for page in pages]


def font_embedding(path: Path) -> tuple[bool, int, list[str]]:
    rows = [line for line in run("pdffonts", str(path)).splitlines()[2:] if line.strip()]
    failures: list[str] = []
    pattern = re.compile(r"\s+(yes|no)\s+(yes|no)\s+(yes|no)\s+\d+\s+\d+\s*$", re.I)
    for row in rows:
        match = pattern.search(row)
        if not match or match.group(1).lower() != "yes":
            failures.append(row)
    return not failures, len(rows), failures


def latex_diagnostics(log_text: str) -> dict[str, Any]:
    overfull_h = [float(value) for value in re.findall(r"Overfull \\hbox \(([-0-9.]+)pt too wide\)", log_text)]
    return {
        "undefined_reference_warnings": len(
            re.findall(r"LaTeX Warning: Reference .* undefined|There were undefined references", log_text, re.I)
        ),
        "undefined_citation_warnings": len(
            re.findall(r"Citation .* undefined|There were undefined citations", log_text, re.I)
        ),
        "multiply_defined_label_warnings": len(re.findall(r"multiply defined", log_text, re.I)),
        "missing_graphic_warnings": len(
            re.findall(r"LaTeX Error: File .* not found|File .* not found", log_text, re.I)
        ),
        "missing_character_warnings": len(re.findall(r"Missing character:", log_text)),
        "overfull_hboxes": len(overfull_h),
        "largest_overfull_hbox_pt": max(overfull_h) if overfull_h else 0.0,
        "overfull_vboxes": len(re.findall(r"Overfull \\vbox", log_text)),
        "underfull_hboxes": len(re.findall(r"Underfull \\hbox", log_text)),
        "underfull_vboxes": len(re.findall(r"Underfull \\vbox", log_text)),
        "fatal_errors": len(re.findall(r"^! |Emergency stop|Fatal error", log_text, re.M | re.I)),
    }


def strip_tex_comments(text: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in text.splitlines())


def formal_result_record(paths: list[Path]) -> dict[str, Any]:
    statements = 0
    proofs = 0
    proof_obligations = 0
    gaps: list[dict[str, str]] = []
    for path in paths:
        source = strip_tex_comments(path.read_text(errors="replace"))
        statements += sum(len(re.findall(rf"\\begin\{{{env}\}}", source)) for env in FORMAL_ENVS)
        proofs += len(re.findall(r"\\begin\{proof\}", source))
        proof_obligations += len(re.findall(r"\\begin\{proofobligation\}", source))
        end_pattern = re.compile(r"\\end\{(" + "|".join(FORMAL_ENVS) + r")\}")
        for match in end_pattern.finditer(source):
            tail = re.sub(r"^\s*", "", source[match.end():])
            if not tail.startswith(r"\begin{proof}"):
                gaps.append(
                    {
                        "file": path.relative_to(ROOT).as_posix(),
                        "environment": match.group(1),
                        "following_text": normalize_space(tail[:180]),
                    }
                )
    return {
        "established_statements": statements,
        "proofs": proofs,
        "proof_obligations": proof_obligations,
        "immediate_proof_gaps": gaps,
    }


def bibliography_record() -> dict[str, Any]:
    bib_path = ROOT / "shared/bibliography/references.bib"
    manifest_path = ROOT / "shared/bibliography/annotation_manifest.json"
    bib_text = bib_path.read_text(errors="replace")
    keys = re.findall(r"@[A-Za-z]+\s*\{\s*([^,\s]+)", bib_text)
    annotations = sorted((ROOT / "shared/bibliography/annotated_entries").glob("*.tex"))
    manifest = json.loads(manifest_path.read_text())
    entries = manifest.get("entries", [])
    manifest_keys = [entry["bib_key"] for entry in entries]
    missing_files: list[str] = []
    hash_mismatches: list[str] = []
    missing_relationship: list[str] = []
    missing_verification: list[str] = []
    for entry in entries:
        path = ROOT / entry["path"]
        if not path.exists():
            missing_files.append(entry["path"])
            continue
        if sha256(path) != entry["sha256"]:
            hash_mismatches.append(entry["path"])
        text = path.read_text(errors="replace")
        lowered = text.lower()
        if not entry.get("explicit_project_relationship") or not (
            "bayesbreak" in lowered or "this project" in lowered
        ):
            missing_relationship.append(entry["path"])
        if "verification status:" not in lowered:
            missing_verification.append(entry["path"])
    return {
        "bibliography_key_count": len(keys),
        "unique_bibliography_key_count": len(set(keys)),
        "annotation_file_count": len(annotations),
        "manifest_entry_count": len(manifest_keys),
        "one_to_one_complete": set(keys) == set(manifest_keys) == {path.stem for path in annotations},
        "duplicate_bibliography_keys": sorted(key for key, count in Counter(keys).items() if count > 1),
        "duplicate_manifest_keys": sorted(key for key, count in Counter(manifest_keys).items() if count > 1),
        "missing_annotation_files": missing_files,
        "annotation_hash_mismatches": hash_mismatches,
        "annotations_without_project_relationship": missing_relationship,
        "annotations_without_verification_status": missing_verification,
    }


def scientific_source_record() -> dict[str, Any]:
    roots = [
        ROOT / "book",
        ROOT / "paper",
        ROOT / "executive",
        ROOT / "shared/components",
        ROOT / "shared/figures/tikz",
        ROOT / "shared/bibliography/annotated_entries",
        ROOT / "shared/handoffs",
        ROOT / "shared/metadata",
        ROOT / "coding/CODING_AGENT_HANDOFF.md",
        ROOT / "presentation_handoffs",
    ]
    files: list[Path] = [ROOT / "shared/metadata.tex", ROOT / "README.md", ROOT / "STATUS.md"]
    suffixes = {".tex", ".md", ".json", ".sty", ".cls"}
    for root in roots:
        if root.is_file():
            files.append(root)
        elif root.exists():
            files.extend(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)
    files = sorted(set(files))
    hits: dict[str, list[dict[str, Any]]] = {name: [] for name in BANNED_SCIENTIFIC_LANGUAGE}
    old_title_hits: list[str] = []
    for path in files:
        text = path.read_text(errors="replace")
        if OLD_TITLE.lower() in text.lower():
            old_title_hits.append(path.relative_to(ROOT).as_posix())
        for name, pattern in BANNED_SCIENTIFIC_LANGUAGE.items():
            for line_no, line in enumerate(text.splitlines(), 1):
                if re.search(pattern, line, re.I):
                    hits[name].append(
                        {
                            "file": path.relative_to(ROOT).as_posix(),
                            "line": line_no,
                            "text": line.strip()[:240],
                        }
                    )
    return {"files_checked": len(files), "old_title_hits": old_title_hits, "banned_language_hits": hits}


def target_record(name: str, spec: dict[str, str], expected: dict[str, Any]) -> dict[str, Any]:
    pdf = ROOT / spec["pdf"]
    log = ROOT / spec["log"]
    if not pdf.exists() or not log.exists():
        return {
            "exists": False,
            "missing": [str(path.relative_to(ROOT)) for path in (pdf, log) if not path.exists()],
        }
    info = pdf_info(pdf)
    text = pdf_text(pdf)
    normalized = normalize_space(text)
    lowered = text.lower()
    lengths = page_lengths(text)
    fonts_ok, font_count, font_failures = font_embedding(pdf)
    language_hits = {
        label: len(re.findall(pattern, lowered, re.I))
        for label, pattern in BANNED_SCIENTIFIC_LANGUAGE.items()
    }
    pages = int(info.get("Pages", "0"))
    return {
        "exists": True,
        "path": pdf.relative_to(ROOT).as_posix(),
        "sha256": sha256(pdf),
        "metadata_title": info.get("Title", ""),
        "title_exact_in_metadata": info.get("Title", "") == TITLE,
        "title_present_in_extracted_text": TITLE in normalized,
        "old_title_present": OLD_TITLE.lower() in lowered,
        "banned_language_hits": language_hits,
        "pages": pages,
        "expected_pages": expected.get("pages"),
        "page_count_matches": pages == expected.get("pages"),
        "page_size": info.get("Page size", ""),
        "expected_page_size": spec["page_size"],
        "page_size_matches": info.get("Page size", "") == spec["page_size"],
        "minimum_page_text_characters": min(lengths) if lengths else 0,
        "short_text_pages_lt_30": [index + 1 for index, value in enumerate(lengths) if value < 30],
        "literal_double_question_marks": len(re.findall(r"\?\?", text)),
        "fonts_embedded": fonts_ok,
        "font_count": font_count,
        "font_embedding_failures": font_failures,
        "latex": latex_diagnostics(log.read_text(errors="replace")),
    }


def result_hash_record() -> dict[str, Any]:
    data = json.loads(READONLY_HASHES.read_text())
    expected = data.get("files", {})
    missing: list[str] = []
    mismatches: list[dict[str, str]] = []
    for rel, digest in expected.items():
        path = ROOT / rel
        if not path.exists():
            missing.append(rel)
        else:
            actual = sha256(path)
            if actual != digest:
                mismatches.append({"path": rel, "expected": digest, "actual": actual})
    return {
        "file_count": len(expected),
        "missing": missing,
        "hash_mismatches": mismatches,
        "all_unchanged": not missing and not mismatches,
    }


def roadmap_record() -> dict[str, Any]:
    paths = [path for path in ROOT.rglob("*") if path.is_file() and path.suffix.lower() in {".tex", ".md"}]
    references: dict[str, list[str]] = {}
    for path in paths:
        text = path.read_text(errors="replace")
        if "shared/figures/tikz/roadmaps/" in text:
            references[path.relative_to(ROOT).as_posix()] = sorted(
                set(re.findall(r"shared/figures/tikz/roadmaps/[A-Za-z0-9_./-]+\.tex", text))
            )
    paper_refs = {path: refs for path, refs in references.items() if path.startswith("paper/")}
    technical_refs = {
        path: refs for path, refs in references.items() if path.startswith("presentation_handoffs/technical/")
    }
    executive_refs = {path: refs for path, refs in references.items() if path.startswith("executive/")}
    approved = {
        "shared/figures/tikz/roadmaps/executive_implementation_sequence.tex",
        "shared/figures/tikz/roadmaps/executive_release_decisions.tex",
    }
    used_exec = {item for refs in executive_refs.values() for item in refs}
    return {
        "references": references,
        "main_paper_has_no_roadmap_references": not paper_refs,
        "technical_handoff_has_no_roadmap_references": not technical_refs,
        "executive_uses_exact_approved_roadmaps": used_exec == approved,
        "executive_roadmaps": sorted(used_exec),
    }


def report_passed(path: Path) -> bool:
    if not path.exists():
        return False
    return bool(json.loads(path.read_text()).get("passed"))


def main() -> None:
    if not EXPECTATIONS.exists():
        raise SystemExit(f"Missing release expectations: {EXPECTATIONS}")
    expectations = json.loads(EXPECTATIONS.read_text())
    expected_targets = expectations.get("targets", {})

    targets = {
        name: target_record(name, spec, expected_targets.get(name, {}))
        for name, spec in TARGETS.items()
    }

    source = scientific_source_record()
    bibliography = bibliography_record()
    readonly_results = result_hash_record()
    roadmap = roadmap_record()

    chapter_files = sorted((ROOT / "book/chapters").glob("*.tex"))
    appendix_files = sorted((ROOT / "book/appendices").glob("*.tex"))
    paper_files = sorted((ROOT / "paper/sections").glob("*.tex"))
    book_formal = formal_result_record(chapter_files + appendix_files)
    paper_formal = formal_result_record(paper_files)

    required_paths = [
        ROOT / "book/main.tex",
        ROOT / "book/saim-book.sty",
        ROOT / "paper/main.tex",
        ROOT / "paper/main-two-column.tex",
        ROOT / "paper/main-single-column.tex",
        ROOT / "paper/saim-paper.cls",
        ROOT / "executive/main.tex",
        ROOT / "executive/sections/00-summary.tex",
        ROOT / "executive/sections/01-objective.tex",
        ROOT / "executive/sections/02-method.tex",
        ROOT / "executive/sections/03-evidence.tex",
        ROOT / "executive/sections/04-limitations.tex",
        ROOT / "executive/sections/05-implementation.tex",
        ROOT / "executive/sections/06-decisions.tex",
        ROOT / "executive/sections/07-conclusion.tex",
        ROOT / "shared/saim",
        ROOT / "shared/figures/tikz/roadmaps/executive_implementation_sequence.tex",
        ROOT / "shared/figures/tikz/roadmaps/executive_release_decisions.tex",
        ROOT / "shared/handoffs/coding_agent_handoff.json",
        ROOT / "shared/handoffs/coding_agent_handoff.tex",
        ROOT / "coding/CODING_AGENT_HANDOFF.md",
        ROOT / "coding/repository_skeleton/IMPLEMENTATION_STATUS.json",
        ROOT / "presentation_handoffs/shared/PRESENTATION_SOURCE_OF_TRUTH.md",
        ROOT / "presentation_handoffs/technical/TECHNICAL_DECK_BRIEF.md",
        ROOT / "presentation_handoffs/executive/EXECUTIVE_DECK_BRIEF.md",
    ]
    missing_required = [path.relative_to(ROOT).as_posix() for path in required_paths if not path.exists()]

    target_checks: dict[str, bool] = {}
    for name, record in targets.items():
        latex = record.get("latex", {})
        target_checks[name] = bool(
            record.get("exists")
            and record.get("title_exact_in_metadata")
            and record.get("title_present_in_extracted_text")
            and not record.get("old_title_present")
            and all(value == 0 for value in record.get("banned_language_hits", {}).values())
            and record.get("page_count_matches")
            and record.get("page_size_matches")
            and not record.get("short_text_pages_lt_30")
            and record.get("literal_double_question_marks") == 0
            and record.get("fonts_embedded")
            and latex.get("undefined_reference_warnings") == 0
            and latex.get("undefined_citation_warnings") == 0
            and latex.get("multiply_defined_label_warnings") == 0
            and latex.get("missing_graphic_warnings") == 0
            and latex.get("missing_character_warnings") == 0
            and latex.get("overfull_hboxes") == 0
            and latex.get("overfull_vboxes") == 0
            and latex.get("fatal_errors") == 0
        )

    handoff_status = json.loads((ROOT / "coding/repository_skeleton/IMPLEMENTATION_STATUS.json").read_text())
    visual_qa = json.loads((PHASE_DIR / "VISUAL_QA.json").read_text())
    slide_paths = [
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and any(part.lower().startswith("slide") for part in path.parts)
    ]
    active_hits_empty = all(not entries for entries in source["banned_language_hits"].values())

    checks = {
        "required_paths_present": not missing_required,
        "saim_template_applied_to_all_paper_form_artifacts": all(
            (ROOT / rel).exists()
            for rel in ("book/saim-book.sty", "paper/saim-paper.cls", "shared/saim", "executive/main.tex")
        ),
        "exact_title_in_shared_metadata": TITLE in (ROOT / "shared/metadata.tex").read_text(errors="replace"),
        "old_replacement_title_absent_from_active_source": not source["old_title_hits"],
        "rejected_branding_absent_from_active_source": active_hits_empty,
        "technical_book_target_passes": target_checks["technical_book"],
        "two_column_paper_target_passes": target_checks["main_paper_two_column"],
        "single_column_paper_target_passes": target_checks["main_paper_single_column"],
        "executive_summary_target_passes": target_checks["executive_summary"],
        "book_has_17_chapter_source_files": len(chapter_files) == 17,
        "book_has_five_required_appendices": len(appendix_files) == 5,
        "book_formal_results_have_immediate_proofs": book_formal["established_statements"] == 28
        and book_formal["proofs"] == 28
        and not book_formal["immediate_proof_gaps"],
        "book_has_one_proof_obligation": book_formal["proof_obligations"] == 1,
        "paper_formal_results_have_immediate_proofs": paper_formal["established_statements"] == 12
        and paper_formal["proofs"] == 12
        and not paper_formal["immediate_proof_gaps"],
        "paper_has_one_proof_obligation": paper_formal["proof_obligations"] == 1,
        "bibliography_annotations_complete": bibliography["bibliography_key_count"] == 38
        and bibliography["one_to_one_complete"]
        and not bibliography["duplicate_bibliography_keys"]
        and not bibliography["duplicate_manifest_keys"]
        and not bibliography["missing_annotation_files"]
        and not bibliography["annotation_hash_mismatches"]
        and not bibliography["annotations_without_project_relationship"]
        and not bibliography["annotations_without_verification_status"],
        "canonical_handoff_sync_check_passes": report_passed(PHASE_DIR / "HANDOFF_SYNC_VALIDATION.json"),
        "presentation_handoff_check_passes": report_passed(PHASE_DIR / "PRESENTATION_HANDOFF_VALIDATION.json"),
        "repository_skeleton_check_passes": report_passed(PHASE_DIR / "REPOSITORY_SKELETON_VALIDATION.json"),
        "repository_is_explicitly_incomplete": handoff_status.get("scientific_implementation_complete") is False,
        "visual_qa_complete": visual_qa.get("total_pages_inspected") == 254
        and visual_qa.get("total_contact_sheets_inspected") == 22
        and not visual_qa.get("final_findings", {}).get("reported_defects"),
        "archived_result_assets_are_unchanged": readonly_results["all_unchanged"],
        "roadmap_location_policy_passes": roadmap["main_paper_has_no_roadmap_references"]
        and roadmap["technical_handoff_has_no_roadmap_references"]
        and roadmap["executive_uses_exact_approved_roadmaps"],
        "no_slide_sources_generated": not slide_paths,
        "executive_has_eight_section_sources": len(list((ROOT / "executive/sections").glob("*.tex"))) == 8,
        "excluded_result_ids_preserved": all(
            token in "\n".join(
                path.read_text(errors="replace")
                for path in chapter_files + appendix_files + paper_files + sorted((ROOT / "executive/sections").glob("*.tex"))
            )
            for token in ("RES-BB-CMP-002", "RES-BB-RD-007Q", "-387.50040013308154", "95,245")
        ),
        "implementation_test_state_preserved": all(
            token in "\n".join(
                path.read_text(errors="replace")
                for path in sorted((ROOT / "executive/sections").glob("*.tex"))
            )
            for token in ("179", "157", "five", "17")
        ),
    }

    validation = {
        "phase": "5",
        "title": TITLE,
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "missing_required_paths": missing_required,
        "targets": targets,
        "source_language": source,
        "formal_results": {"book": book_formal, "paper": paper_formal},
        "bibliography": bibliography,
        "readonly_result_assets": readonly_results,
        "roadmaps": roadmap,
        "slide_source_paths": slide_paths,
        "implementation_status": {
            "scientific_implementation_complete": handoff_status.get("scientific_implementation_complete"),
            "task_count": len(handoff_status.get("tasks", [])),
        },
        "visual_qa": visual_qa,
        "real_result_policy": {
            "populated_archived_values_changed": False,
            "research_experiments_rerun": False,
            "scientific_package_code_changed": False,
            "RES-BB-CMP-002": "executed historical result; excluded from comparator conclusions because the coordinate axes are incompatible",
            "RES-BB-RD-007Q": "executed historical result; excluded from posterior-predictive conclusions because the observation-family routine and coordinate rule were invalid",
        },
    }
    PHASE_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps({"status": validation["status"], "checks": checks}, indent=2))
    if validation["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
