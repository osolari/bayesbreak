from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parents[1]

FIGURES = {
    "RES-BB-SYN-005": (
        ROOT / "results/phase6/RES-BB-SYN-005/stress_summary.png",
        ROOT / "report/shared/figures/results/fig_phase6_latent_stress.png",
        ROOT / "docs/assets/figures/fig_phase6_latent_stress.png",
    ),
    "RES-BB-CMP-003": (
        ROOT / "results/phase6/RES-BB-CMP-003/boundary_agreement.png",
        ROOT / "report/shared/figures/results/fig_phase6_cgh_agreement.png",
        ROOT / "docs/assets/figures/fig_phase6_cgh_agreement.png",
    ),
    "RES-BB-RD-008Q": (
        ROOT / "results/phase6/RES-BB-RD-008Q/predictive_summary.png",
        ROOT / "report/shared/figures/results/fig_phase6_methyl_predictive.png",
        ROOT / "docs/assets/figures/fig_phase6_methyl_predictive.png",
    ),
}


def test_reader_facing_figures_are_byte_identical_to_corrected_artifacts() -> None:
    for source, paper_copy, docs_copy in FIGURES.values():
        expected = source.read_bytes()
        assert paper_copy.read_bytes() == expected
        assert docs_copy.read_bytes() == expected


def test_paper_book_and_docs_reference_every_corrected_result() -> None:
    documents = {
        "paper": (ROOT / "report/paper/sections/09-results.tex").read_text(),
        "book-synthetic": (ROOT / "report/book/chapters/13-synthetic-validation.tex").read_text(),
        "book-real": (ROOT / "report/book/chapters/14-real-data.tex").read_text(),
        "docs": (ROOT / "docs/results.md").read_text(),
    }
    assert (
        "RES-BB-SYN-005" in documents["paper"] and "RES-BB-SYN-005" in documents["book-synthetic"]
    )
    assert "RES-BB-CMP-003" in documents["paper"] and "RES-BB-CMP-003" in documents["book-real"]
    assert "RES-BB-RD-008Q" in documents["paper"] and "RES-BB-RD-008Q" in documents["book-real"]
    for result_id in FIGURES:
        assert result_id in documents["docs"]


def test_completed_protocols_name_the_corrected_results() -> None:
    protocols = json.loads((ROOT / "report/shared/metadata/experiment_protocols.json").read_text())[
        "protocols"
    ]
    statuses = {protocol["id"]: protocol["status"] for protocol in protocols}
    assert "RES-BB-SYN-005" in statuses["EPR-BB-005"]
    assert "RES-BB-CMP-003" in statuses["EPR-BB-010"]
    assert "RES-BB-RD-008Q" in statuses["EPR-BB-012"]
    assert "RES-BB-CMP-003" in statuses["EPR-BB-013"]
