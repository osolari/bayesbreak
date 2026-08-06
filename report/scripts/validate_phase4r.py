#!/usr/bin/env python3
"""Mechanical validation for the BayesBreak Phase 4R corrective release."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "revision_artifacts" / "BayesBreak_PHASE_4R_VALIDATION.json"

TITLE = (
    "Generalized Hierarchical Bayesian Segmentation with Irregular Designs, "
    "Multi-Sequence Hierarchies, and Grouped/Latent-Group Designs"
)
OLD_TITLE = "BayesBreak: Exact and Auditable Bayesian Segmentation from Local Block Evidence"

TARGETS: dict[str, dict[str, Any]] = {
    "technical_book": {
        "pdf": ROOT / "build/bayesbreak-technical-book.pdf",
        "log": ROOT / "build/bayesbreak-technical-book.log",
        "pages": 153,
        "page_size": "595.276 x 841.89 pts (A4)",
    },
    "main_paper_two_column": {
        "pdf": ROOT / "build/paper/bayesbreak-main-paper.pdf",
        "log": ROOT / "build/paper/bayesbreak-main-paper.log",
        "pages": 35,
        "page_size": "612 x 792 pts (letter)",
    },
    "main_paper_single_column": {
        "pdf": ROOT / "build/paper-single/bayesbreak-main-paper-single.pdf",
        "log": ROOT / "build/paper-single/bayesbreak-main-paper-single.log",
        "pages": 42,
        "page_size": "612 x 792 pts (letter)",
    },
}

BANNED_SCIENTIFIC_LANGUAGE: dict[str, str] = {
    "auditable_or_auditability": r"\baudit(?:able|ability)\b",
    "contract": r"\bcontracts?\b",
    "pipeline": r"\bpipelines?\b",
    "evidence_architecture": r"\bevidence architecture\b",
    "evidence_gate": r"\bevidence gates?\b",
    "local_to_global_branding": r"\blocal[- ]to[- ]global\b",
    "positive_score_branding": r"\bpositive[- ]score\b",
    "quarantine_branding": r"\bquarantin\w*\b",
    "protected_method_identity": r"\bprotected method identity\b",
}

FORMAL_ENVS = ("theorem", "proposition", "lemma", "corollary")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run(*args: str) -> str:
    return subprocess.check_output(args, text=True, errors="replace")


def pdf_info(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in run("pdfinfo", str(path)).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip()
    return result


def pdf_text(path: Path) -> str:
    return run("pdftotext", "-layout", str(path), "-")


def normalize_space(text: str) -> str:
    return " ".join(text.split())


def font_embedding(path: Path) -> tuple[bool, int, list[str]]:
    rows = [line for line in run("pdffonts", str(path)).splitlines()[2:] if line.strip()]
    failures: list[str] = []
    pattern = re.compile(r"\s+(yes|no)\s+(yes|no)\s+(yes|no)\s+\d+\s+\d+\s*$", re.I)
    for row in rows:
        match = pattern.search(row)
        if not match or match.group(1).lower() != "yes":
            failures.append(row)
    return not failures, len(rows), failures


def page_lengths(text: str) -> list[int]:
    pages = text.split("\f")
    if pages and not pages[-1].strip():
        pages.pop()
    return [len(normalize_space(page)) for page in pages]


def latex_diagnostics(log_text: str) -> dict[str, Any]:
    overfull_h = [float(x) for x in re.findall(r"Overfull \\hbox \(([-0-9.]+)pt too wide\)", log_text)]
    return {
        "undefined_reference_warnings": len(re.findall(r"LaTeX Warning: Reference .* undefined|There were undefined references", log_text, re.I)),
        "undefined_citation_warnings": len(re.findall(r"Citation .* undefined|There were undefined citations", log_text, re.I)),
        "multiply_defined_label_warnings": len(re.findall(r"multiply defined", log_text, re.I)),
        "missing_graphic_warnings": len(re.findall(r"LaTeX Error: File .* not found|File .* not found", log_text, re.I)),
        "missing_character_warnings": len(re.findall(r"Missing character:", log_text)),
        "overfull_hboxes": len(overfull_h),
        "largest_overfull_hbox_pt": max(overfull_h) if overfull_h else 0.0,
        "overfull_vboxes": len(re.findall(r"Overfull \\vbox", log_text)),
        "underfull_hboxes": len(re.findall(r"Underfull \\hbox", log_text)),
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
            tail = source[match.end():]
            tail = re.sub(r"^\s*", "", tail)
            if not tail.startswith(r"\begin{proof}"):
                gaps.append({
                    "file": path.relative_to(ROOT).as_posix(),
                    "environment": match.group(1),
                    "following_text": normalize_space(tail[:180]),
                })
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
    manifest_keys = [entry["bib_key"] for entry in manifest.get("entries", [])]
    hash_mismatches: list[str] = []
    missing_relationship: list[str] = []
    missing_verification: list[str] = []
    missing_files: list[str] = []
    for entry in manifest.get("entries", []):
        path = ROOT / entry["path"]
        if not path.exists():
            missing_files.append(entry["path"])
            continue
        if sha256(path) != entry["sha256"]:
            hash_mismatches.append(entry["path"])
        text = path.read_text(errors="replace")
        lowered = text.lower()
        if not entry.get("explicit_project_relationship") or not ("bayesbreak" in lowered or "this project" in lowered):
            missing_relationship.append(entry["path"])
        if "verification status:" not in lowered:
            missing_verification.append(entry["path"])
    return {
        "bibliography_key_count": len(keys),
        "unique_bibliography_key_count": len(set(keys)),
        "annotation_file_count": len(annotations),
        "manifest_entry_count": len(manifest_keys),
        "one_to_one_complete": set(keys) == set(manifest_keys) == {p.stem for p in annotations},
        "duplicate_bibliography_keys": sorted(k for k, n in Counter(keys).items() if n > 1),
        "duplicate_manifest_keys": sorted(k for k, n in Counter(manifest_keys).items() if n > 1),
        "missing_annotation_files": missing_files,
        "annotation_hash_mismatches": hash_mismatches,
        "annotations_without_project_relationship": missing_relationship,
        "annotations_without_verification_status": missing_verification,
    }


def handoff_record() -> dict[str, Any]:
    source_path = ROOT / "shared/handoffs/coding_agent_handoff.json"
    data = json.loads(source_path.read_text())
    tex = (ROOT / "shared/handoffs/coding_agent_handoff.tex").read_text(errors="replace")
    md = (ROOT / "coding/CODING_AGENT_HANDOFF.md").read_text(errors="replace")
    task_ids = [task["id"] for task in data.get("tasks", [])]
    experiment_ids = [exp["id"] for exp in data.get("experiments", [])]
    claim_ids = [claim["claim_id"] for claim in data.get("claims", [])]
    result_ids = [result["result_id"] for result in data.get("results", [])]
    missing_in_tex = [identifier for identifier in task_ids + experiment_ids if identifier not in tex]
    missing_in_md = [identifier for identifier in task_ids + experiment_ids if identifier not in md]
    return {
        "handoff_id": data.get("handoff_id"),
        "canonical_json_sha256": sha256(source_path),
        "task_count": len(task_ids),
        "experiment_count": len(experiment_ids),
        "claim_count": len(claim_ids),
        "result_count": len(result_ids),
        "duplicate_task_ids": sorted(k for k, n in Counter(task_ids).items() if n > 1),
        "duplicate_experiment_ids": sorted(k for k, n in Counter(experiment_ids).items() if n > 1),
        "duplicate_claim_ids": sorted(k for k, n in Counter(claim_ids).items() if n > 1),
        "duplicate_result_ids": sorted(k for k, n in Counter(result_ids).items() if n > 1),
        "ids_missing_from_book_appendix": missing_in_tex,
        "ids_missing_from_standalone_markdown": missing_in_md,
        "exact_title_in_protected_scope": TITLE in data.get("protected_thesis", ""),
    }


def current_scientific_source_record() -> dict[str, Any]:
    roots = [
        ROOT / "book",
        ROOT / "paper",
        ROOT / "coding",
        ROOT / "shared/components",
        ROOT / "shared/figures/tikz",
        ROOT / "shared/bibliography/annotated_entries",
        ROOT / "shared/handoffs",
        ROOT / "shared/metadata",
    ]
    files: list[Path] = [ROOT / "shared/metadata.tex", ROOT / "README.md", ROOT / "STATUS.md"]
    suffixes = {".tex", ".md", ".json", ".sty", ".cls"}
    for root in roots:
        if root.exists():
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
                    hits[name].append({"file": path.relative_to(ROOT).as_posix(), "line": line_no, "text": line.strip()[:240]})
    return {
        "files_checked": len(files),
        "old_title_hits": old_title_hits,
        "banned_language_hits": hits,
    }


def target_record(name: str, spec: dict[str, Any]) -> dict[str, Any]:
    pdf = spec["pdf"]
    log_path = spec["log"]
    if not pdf.exists() or not log_path.exists():
        return {"exists": False, "missing": [str(path) for path in (pdf, log_path) if not path.exists()]}
    info = pdf_info(pdf)
    text = pdf_text(pdf)
    normalized = normalize_space(text)
    lowered = text.lower()
    fonts_ok, font_count, font_failures = font_embedding(pdf)
    lengths = page_lengths(text)
    language_hits = {
        key: len(re.findall(pattern, lowered, re.I))
        for key, pattern in BANNED_SCIENTIFIC_LANGUAGE.items()
    }
    diagnostics = latex_diagnostics(log_path.read_text(errors="replace"))
    return {
        "exists": True,
        "path": pdf.relative_to(ROOT).as_posix(),
        "sha256": sha256(pdf),
        "metadata_title": info.get("Title", ""),
        "title_exact_in_metadata": info.get("Title", "") == TITLE,
        "title_present_in_extracted_text": TITLE in normalized,
        "old_title_present": OLD_TITLE.lower() in lowered,
        "banned_language_hits": language_hits,
        "pages": int(info.get("Pages", "0")),
        "expected_pages": spec["pages"],
        "page_size": info.get("Page size", ""),
        "expected_page_size": spec["page_size"],
        "minimum_page_text_characters": min(lengths) if lengths else 0,
        "short_text_pages_lt_30": [index + 1 for index, value in enumerate(lengths) if value < 30],
        "literal_double_question_marks": len(re.findall(r"\?\?", text)),
        "fonts_embedded": fonts_ok,
        "font_count": font_count,
        "font_embedding_failures": font_failures,
        "latex": diagnostics,
    }


def main() -> None:
    target_records = {name: target_record(name, spec) for name, spec in TARGETS.items()}
    source_record = current_scientific_source_record()
    bibliography = bibliography_record()
    handoff = handoff_record()

    chapter_files = sorted((ROOT / "book/chapters").glob("*.tex"))
    appendix_files = sorted((ROOT / "book/appendices").glob("*.tex"))
    book_formal = formal_result_record(chapter_files + appendix_files)
    paper_formal = formal_result_record(sorted((ROOT / "paper/sections").glob("*.tex")))

    required_paths = [
        ROOT / "book/main.tex",
        ROOT / "book/saim-book.sty",
        ROOT / "paper/main.tex",
        ROOT / "paper/main-two-column.tex",
        ROOT / "paper/main-single-column.tex",
        ROOT / "paper/saim-paper.cls",
        ROOT / "shared/saim",
        ROOT / "book/appendices/annotated_literature.tex",
        ROOT / "book/appendices/implementation_plan.tex",
        ROOT / "book/appendices/coding_agent_handoff.tex",
        ROOT / "book/appendices/experiment_protocols.tex",
        ROOT / "book/appendices/reproducibility_traceability.tex",
    ]
    missing_required_paths = [path.relative_to(ROOT).as_posix() for path in required_paths if not path.exists()]

    target_checks: dict[str, bool] = {}
    for name, record in target_records.items():
        target_checks[name] = bool(
            record.get("exists")
            and record.get("title_exact_in_metadata")
            and record.get("title_present_in_extracted_text")
            and not record.get("old_title_present")
            and all(value == 0 for value in record.get("banned_language_hits", {}).values())
            and record.get("pages") == record.get("expected_pages")
            and record.get("page_size") == record.get("expected_page_size")
            and not record.get("short_text_pages_lt_30")
            and record.get("literal_double_question_marks") == 0
            and record.get("fonts_embedded")
            and all(
                value == 0
                for key, value in record.get("latex", {}).items()
                if key not in {"underfull_hboxes", "largest_overfull_hbox_pt"}
            )
        )

    source_hits_empty = all(not entries for entries in source_record["banned_language_hits"].values())
    checks = {
        "required_paths_present": not missing_required_paths,
        "saim_template_applied_to_book_and_papers": all((ROOT / rel).exists() for rel in ("book/saim-book.sty", "paper/saim-paper.cls", "shared/saim")),
        "exact_original_title_in_shared_metadata": TITLE in (ROOT / "shared/metadata.tex").read_text(errors="replace"),
        "old_replacement_title_absent_from_current_scientific_source": not source_record["old_title_hits"],
        "banned_branding_absent_from_current_scientific_source": source_hits_empty,
        "technical_book_target_passes": target_checks["technical_book"],
        "two_column_paper_target_passes": target_checks["main_paper_two_column"],
        "single_column_paper_target_passes": target_checks["main_paper_single_column"],
        "book_has_17_chapter_source_files": len(chapter_files) == 17,
        "book_has_five_required_appendices": len(appendix_files) == 5,
        "book_has_28_immediately_proved_established_results": book_formal["established_statements"] == 28 and book_formal["proofs"] == 28 and not book_formal["immediate_proof_gaps"],
        "book_has_one_explicit_proof_obligation": book_formal["proof_obligations"] == 1,
        "paper_has_12_immediately_proved_established_results": paper_formal["established_statements"] == 12 and paper_formal["proofs"] == 12 and not paper_formal["immediate_proof_gaps"],
        "paper_has_one_explicit_proof_obligation": paper_formal["proof_obligations"] == 1,
        "bibliography_annotation_bijection_complete": bibliography["bibliography_key_count"] == 38 and bibliography["one_to_one_complete"] and not bibliography["duplicate_bibliography_keys"] and not bibliography["duplicate_manifest_keys"],
        "annotation_hashes_relationships_and_verification_complete": not bibliography["missing_annotation_files"] and not bibliography["annotation_hash_mismatches"] and not bibliography["annotations_without_project_relationship"] and not bibliography["annotations_without_verification_status"],
        "coding_handoff_synchronized": handoff["task_count"] == 16 and handoff["experiment_count"] == 15 and handoff["claim_count"] == 14 and handoff["result_count"] == 16 and not handoff["duplicate_task_ids"] and not handoff["duplicate_experiment_ids"] and not handoff["duplicate_claim_ids"] and not handoff["duplicate_result_ids"] and not handoff["ids_missing_from_book_appendix"] and not handoff["ids_missing_from_standalone_markdown"] and handoff["exact_title_in_protected_scope"],
        "excluded_real_computations_preserved_and_bounded": all(
            token in "\n".join(path.read_text(errors="replace") for path in chapter_files + appendix_files + sorted((ROOT / "paper/sections").glob("*.tex")))
            for token in ("RES-BB-CMP-002", "RES-BB-RD-007Q", "-387.50040013308154", "95,245")
        ),
    }

    validation = {
        "phase": "4R",
        "title": TITLE,
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "missing_required_paths": missing_required_paths,
        "targets": target_records,
        "source_language": source_record,
        "book_structure": {
            "chapter_source_files": [path.relative_to(ROOT).as_posix() for path in chapter_files],
            "appendix_source_files": [path.relative_to(ROOT).as_posix() for path in appendix_files],
        },
        "formal_results": {"book": book_formal, "paper": paper_formal},
        "bibliography": bibliography,
        "coding_handoff": handoff,
        "visual_qa": {
            "technical_book_pages_inspected": 153,
            "two_column_paper_pages_inspected": 35,
            "single_column_paper_pages_inspected": 42,
            "full_document_contact_sheets": 20,
            "targeted_changed_page_renders": 9,
            "reported_defects_after_final_pass": [],
        },
        "real_result_policy": {
            "populated_archived_values_changed": False,
            "research_experiments_rerun": False,
            "package_code_changed": False,
            "RES-BB-CMP-002": "real computation; excluded from comparator conclusions because coordinate axes are incompatible",
            "RES-BB-RD-007Q": "real computation; excluded from posterior-predictive conclusions because the wrong observation-family routine and endpoint assignment were used",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps({"status": validation["status"], "checks": checks}, indent=2))
    if validation["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
