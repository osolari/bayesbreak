#!/usr/bin/env python3
"""Structural validator for BayesBreak experiment plan and manuscript slots.

Only validates prospective protocol schemas and scientific text markers.
Does not claim any experiment was run or any numerical metric was checked.
"""
from __future__ import annotations
import json
from pathlib import Path
import re
import sys


def main() -> int:
    skill = Path(__file__).resolve().parents[1]
    root = skill.parents[2]
    paper = root / "docs/manuscripts/paper"
    payload = json.loads((skill / "protocols/experiments.json").read_text())
    exp = payload["experiments"]
    assert payload["status"] == "planned_not_executed"
    ids = [item["id"] for item in exp]
    assert ids == [f"E{i:02d}" for i in range(11)], ids
    abstract = (paper / "sections/00-abstract.tex").read_text()
    shared = root / "docs/manuscripts/shared"
    section_files = list((paper / "sections").glob("*.tex"))
    fig_defs = list((shared / "figures/defs").glob("*.tex"))
    tab_defs = list((shared / "tables/defs").glob("*.tex"))
    tab_results = list((shared / "tables/results").glob("*.tex"))
    tex = "\n".join(
        p.read_text() for p in section_files + fig_defs + tab_defs + tab_results
    )
    assert "\\documentclass[11pt,onecolumn]" in (paper / "main.tex").read_text()
    assert "\\section{Problem formulation and notation}" in tex
    assert "\\section{Experimental protocols}" in tex
    assert "TBD" in tex and "Planned Figure" in tex
    assert "\\EstablishedTag" not in tex and "\\ConditionalTag" not in tex
    # Em-dash and triple-dash checks apply only to prose section files, not table bodies.
    sections_tex = "\n".join(p.read_text() for p in section_files)
    assert "\u2014" not in sections_tex and "---" not in sections_tex
    assert "\u2014" not in abstract
    for item in exp:
        assert item["type"]
        for key in ("paper_figure_label", "paper_table_label"):
            label = item.get(key)
            if label is not None:
                assert f"\\label{{{label}}}" in tex, (item["id"], label)
        assert item["id"] in tex, item["id"]
    for p in (paper / "sections").glob("*.tex"):
        data = p.read_text()
        for env in ("theorem", "proposition", "lemma", "corollary"):
            pat = re.compile(r"\\begin\{" + env + r"\}(?:\[[^\]]*\])?.*?\\end\{" + env + r"\}",re.S)
            for match in pat.finditer(data):
                rest = data[match.end():].lstrip()
                assert rest.startswith("\\begin{proof}"), (p.name, env)
    print(f"OK: {len(ids)} prospective experiment protocols; paper slots, proof adjacency, one-column and no-status-tag checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
