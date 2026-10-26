#!/usr/bin/env python3
"""validate_inventory.py

Verify that every active data-driven figure and numerical table in the
BayesBreak manuscript is fully covered by the two-stage rendering architecture:

  Stage A (prepare_data.py)  -- runs experiments, writes plot-ready CSVs
  Stage B (render.py)        -- reads CSVs only, writes PDF + PNG

The validator:
  1. Recursively follows \\input{} references from paper/main.tex to discover
     all active figures and tables.
  2. For each data-driven figure: checks CSV inputs, JSON config, PDF output,
     PNG output, and render-function registration in render.py.
  3. For each generated table: checks CSV input, JSON config, .tex output, and
     render-function registration.
  4. Warns on assets with UNVERIFIED provenance rather than failing.
  5. Exits 0 when all checks pass; exits 1 on any hard failure.

Usage:
    python scripts/figures/validate_inventory.py
    python scripts/figures/validate_inventory.py --verbose
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import NamedTuple

_ROOT = Path(__file__).resolve().parents[2]
_MANUSCRIPTS = _ROOT / "docs" / "manuscripts"
_SHARED = _MANUSCRIPTS / "shared"
_FIG_DATA = _SHARED / "figures" / "data"
_FIG_RESULTS = _MANUSCRIPTS / "paper" / "figures"   # canonical output directory
_TAB_DATA = _SHARED / "tables" / "data"
_TAB_RESULTS = _SHARED / "tables" / "results"
_RENDER_PY = _ROOT / "scripts" / "figures" / "render.py"


# ---------------------------------------------------------------------------
# Asset manifest
# ---------------------------------------------------------------------------
# Maps stem → description of required artifacts.
#
# CATEGORY:
#   "data_driven"  CSV + JSON + PDF + PNG + render key all required
#   "unverified"   JSON only required (data gap); PNG expected but soft warning
#   "tikz"         TikZ source (no data files); just verify the .tex source
#   "static_table" Hardcoded table body; verify .tex exists in results
#   "generated_table" CSV + JSON + .tex + render key required

class _Asset(NamedTuple):
    category: str
    csvs: tuple[str, ...]        # filenames in _FIG_DATA (figures) or _TAB_DATA (tables)
    json: str | None             # filename in _FIG_DATA / _TAB_DATA
    outputs: tuple[str, ...]     # expected outputs in _FIG_RESULTS / _TAB_RESULTS
    render_key: str | None       # key in render.py _REGISTRY (None = no render fn needed)
    note: str = ""


_FIGURE_MANIFEST: dict[str, _Asset] = {
    # --- showcase figures (paper + book) ---
    "fig3_boundary_calibration": _Asset(
        "data_driven",
        ("fig3_boundary_calibration.csv",),
        "fig3_boundary_calibration.json",
        ("fig3_boundary_calibration.pdf", "fig3_boundary_calibration.png"),
        "fig3",
        note="Referenced in book chapter 13 only; no paper defs wrapper",
    ),
    "fig1_synthetic_gaussian": _Asset(
        "data_driven",
        ("fig1_synthetic_gaussian.csv",),
        "fig1_synthetic_gaussian.json",
        ("fig1_synthetic_gaussian.pdf", "fig1_synthetic_gaussian.png"),
        "fig1",
    ),
    "fig2_family_showcase": _Asset(
        "data_driven",
        ("fig2_family_showcase.csv",),
        "fig2_family_showcase.json",
        ("fig2_family_showcase.pdf", "fig2_family_showcase.png"),
        "fig2",
    ),
    "fig4_latent_groups_cropped": _Asset(
        "data_driven",
        ("fig4_assignments.csv", "fig4_boundary.csv", "fig4_timeseries.csv"),
        "fig4_latent_groups.json",
        ("fig4_latent_groups_cropped.pdf", "fig4_latent_groups_cropped.png"),
        "fig4",
    ),
    "fig5_runtime_scaling": _Asset(
        "data_driven",
        ("fig5_runtime_scaling_data.csv",),
        "fig5_runtime_scaling.json",
        ("fig5_runtime_scaling.pdf", "fig5_runtime_scaling.png"),
        "fig5",
    ),
    "fig6_welllog": _Asset(
        "data_driven",
        ("fig6_welllog.csv",),
        "fig6_welllog.json",
        ("fig6_welllog.pdf", "fig6_welllog.png"),
        "fig6",
    ),
    "fig7_cgh": _Asset(
        "data_driven",
        ("fig7_cgh.csv",),
        "fig7_cgh.json",
        ("fig7_cgh.pdf", "fig7_cgh.png"),
        "fig7",
    ),
    "fig9_methylation": _Asset(
        "data_driven",
        ("fig9_methylation.csv",),
        "fig9_methylation.json",
        ("fig9_methylation.pdf", "fig9_methylation.png"),
        "fig9",
    ),
    # --- experiment figures ---
    "fig_E01_calibration": _Asset(
        "data_driven",
        ("fig_E01_calibration.csv",),
        "fig_E01_calibration.json",
        ("fig_E01_calibration.pdf", "fig_E01_calibration.png"),
        "fig_E01_calibration",
    ),
    "fig_E01_recovery": _Asset(
        "data_driven",
        ("fig_E01_data.csv",),
        "fig_E01_recovery.json",
        ("fig_E01_recovery.pdf", "fig_E01_recovery.png"),
        "fig_E01_recovery",
    ),
    "fig_E02_recovery": _Asset(
        "data_driven",
        ("fig_E02_data.csv",),
        "fig_E02_recovery.json",
        ("fig_E02_recovery.pdf", "fig_E02_recovery.png"),
        "fig_E02_recovery",
    ),
    "fig_E03_gap_true": _Asset(
        "data_driven",
        ("fig_E03_data.csv",),
        "fig_E03_gap_true.json",
        ("fig_E03_gap_true.pdf", "fig_E03_gap_true.png"),
        "fig_E03_gap_true",
    ),
    "fig_E03_gap_false": _Asset(
        "data_driven",
        ("fig_E03_data.csv",),
        "fig_E03_gap_false.json",
        ("fig_E03_gap_false.pdf", "fig_E03_gap_false.png"),
        "fig_E03_gap_false",
    ),
    "fig_E04_pooled": _Asset(
        "data_driven",
        ("fig_E04_data.csv",),
        "fig_E04_pooled.json",
        ("fig_E04_pooled.pdf", "fig_E04_pooled.png"),
        "fig_E04_pooled",
    ),
    "fig_E05_templates": _Asset(
        "data_driven",
        ("fig_E05_data.csv",),
        "fig_E05_templates.json",
        ("fig_E05_templates.pdf", "fig_E05_templates.png"),
        "fig_E05_templates",
    ),
    "fig_E06_local_error": _Asset(
        "data_driven",
        ("fig_E06_local_error.csv",),
        "fig_E06_local_error.json",
        ("fig_E06_local_error.pdf", "fig_E06_local_error.png"),
        "fig_E06_local_error",
    ),
    "fig_E06_global_error": _Asset(
        "data_driven",
        ("fig_E06_global_error.csv",),
        "fig_E06_global_error.json",
        ("fig_E06_global_error.pdf", "fig_E06_global_error.png"),
        "fig_E06_global_error",
    ),
    "fig_E07_robustness": _Asset(
        "data_driven",
        ("fig_E07_robustness.csv",),
        "fig_E07_robustness.json",
        ("fig_E07_robustness.pdf", "fig_E07_robustness.png"),
        "fig_E07_robustness",
    ),
    "fig_E10_scaling": _Asset(
        "data_driven",
        ("fig_E10_data.csv",),
        "fig_E10_scaling.json",
        ("fig_E10_scaling.pdf", "fig_E10_scaling.png"),
        "fig_E10_scaling",
    ),
    "fig_E10_stage_breakdown": _Asset(
        "data_driven",
        ("fig_E10_data.csv",),
        "fig_E10_stage_breakdown.json",
        ("fig_E10_stage_breakdown.pdf", "fig_E10_stage_breakdown.png"),
        "fig_E10_stage_breakdown",
        note="Stage-breakdown panel; shares CSV with fig_E10_scaling",
    ),
    "fig_latent_template_stress": _Asset(
        "data_driven",
        ("fig_latent_template_stress.csv",),
        "fig_latent_template_stress.json",
        ("fig_latent_template_stress.pdf", "fig_latent_template_stress.png"),
        "fig_latent_stress",
    ),
    # --- unverified / static committed assets ---
    "fig8_spx": _Asset(
        "unverified",
        (),
        "fig8_spx.json",
        ("fig8_spx.pdf", "fig8_spx.png"),
        None,
        note="Simulated SPX placeholder (status=PLANNED); no panel-data CSV. "
             "Referenced in book chapter 14 only.",
    ),
    "fig_methylation_heldout": _Asset(
        "unverified",
        (),
        "fig_methylation_heldout.json",
        ("fig_methylation_heldout.png",),
        None,
        note="Requires E09 held-out analysis; data unavailable (provenance_status=UNVERIFIED)",
    ),
    "fig_cgh_probe_agreement": _Asset(
        "unverified",
        (),
        "fig_cgh_probe_agreement.json",
        ("fig_cgh_probe_agreement.png",),
        None,
        note="Requires E09 held-out analysis; data unavailable (provenance_status=UNVERIFIED)",
    ),
    # --- TikZ / diagram figures (no data required) ---
    "fig_semiring_inference_routes": _Asset(
        "tikz",
        (),
        None,
        (),
        None,
        note="TikZ diagram; source in shared/figures/tikz/",
    ),
    "fig_cohesion_boundary_hazard": _Asset(
        "tikz",
        (),
        None,
        (),
        None,
        note="TikZ diagram",
    ),
    "fig_shared_boundary_plate": _Asset(
        "tikz",
        (),
        None,
        (),
        None,
        note="TikZ plate diagram",
    ),
}

_TABLE_MANIFEST: dict[str, _Asset] = {
    # Generated tables (render function writes .tex from CSV)
    "tab_E01_recovery": _Asset(
        "generated_table",
        ("tab_E01_recovery.csv",),
        "tab_E01_recovery.json",
        ("tab_E01_recovery.tex",),
        "tab_E01_recovery",
    ),
    "tab_E02_recovery": _Asset(
        "generated_table",
        ("tab_E02_recovery.csv",),
        "tab_E02_recovery.json",
        ("tab_E02_recovery.tex",),
        "tab_E02_recovery",
    ),
    "tab_E03_prior_comparison": _Asset(
        "generated_table",
        ("tab_E03_prior_comparison.csv",),
        "tab_E03_prior_comparison.json",
        ("tab_E03_prior_comparison.tex",),
        None,
        note="Rendered by prepare_tab_extra; no separate render key (copied in Stage A)",
    ),
    "tab_E04_comparison": _Asset(
        "generated_table",
        ("tab_E04_comparison.csv",),
        "tab_E04_comparison.json",
        ("tab_E04_comparison.tex",),
        None,
        note="Rendered by prepare_tab_extra; no separate render key",
    ),
    "tab_E05_latent": _Asset(
        "generated_table",
        ("tab_E05_latent.csv",),
        "tab_E05_latent.json",
        ("tab_E05_latent.tex",),
        "tab_E05_latent",
    ),
    "tab_E07_robustness": _Asset(
        "generated_table",
        ("tab_E07_robustness.csv",),
        "tab_E07_robustness.json",
        ("tab_E07_robustness.tex",),
        "tab_E07_robustness",
    ),
    "tab_E10_scaling": _Asset(
        "generated_table",
        ("tab_E10_scaling.csv",),
        "tab_E10_scaling.json",
        ("tab_E10_scaling.tex",),
        "tab_E10_scaling",
    ),
    # Static tables (hardcoded LaTeX; only verify .tex output exists)
    "table1_runtime_scaling": _Asset(
        "static_table",
        (),
        None,
        ("table1_runtime_scaling.tex",),
        None,
        note="Static table; authored manually",
    ),
    "table2_posterior_summary": _Asset(
        "static_table",
        (),
        None,
        ("table2_posterior_summary.tex",),
        None,
        note="Static table",
    ),
    "table3_conjugate_summary": _Asset(
        "static_table",
        (),
        None,
        ("table3_conjugate_summary.tex",),
        None,
        note="Static table",
    ),
    "table4_nonconj_tradeoff": _Asset(
        "static_table",
        (),
        None,
        ("table4_nonconj_tradeoff.tex",),
        None,
        note="Static table",
    ),
}


# ---------------------------------------------------------------------------
# Manuscript parser
# ---------------------------------------------------------------------------

def _resolve_input(path_str: str, base: Path) -> Path | None:
    """Resolve a LaTeX \\input{} path (relative to docs/manuscripts/).

    Returns None when the resolved path does not exist.
    """
    # Strip surrounding braces if somehow included
    path_str = path_str.strip("{} \t")
    # Ensure .tex extension
    if not path_str.endswith(".tex"):
        path_str += ".tex"
    resolved = _MANUSCRIPTS / path_str
    return resolved if resolved.exists() else None


def _collect_inputs(tex_path: Path, visited: set[Path]) -> list[Path]:
    """Return all .tex files reachable via \\input{} from tex_path."""
    if tex_path in visited:
        return []
    visited.add(tex_path)
    collected = [tex_path]
    try:
        src = tex_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return collected
    for m in re.finditer(r"\\input\{([^}]+)\}", src):
        target = _resolve_input(m.group(1), tex_path.parent)
        if target is not None:
            collected.extend(_collect_inputs(target, visited))
    return collected


def _discover_figures(all_tex: list[Path]) -> dict[str, str]:
    """Return {stem: source_tex} for every \\includegraphics pointing to paper/figures/."""
    found: dict[str, str] = {}
    # Match the canonical output path paper/figures/ used after Phase 2 migration
    pattern = re.compile(
        r"\\includegraphics[^{]*\{paper/figures/([^}]+)\}"
    )
    for tex in all_tex:
        try:
            src = tex.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for m in pattern.finditer(src):
            path_part = m.group(1)
            # Strip extension to get stem
            stem = re.sub(r"\.(pdf|png|jpg)$", "", path_part)
            if stem not in found:
                found[stem] = str(tex.relative_to(_ROOT))
    return found


def _discover_generated_tables(all_tex: list[Path]) -> dict[str, str]:
    """Return {stem: source_tex} for \\input pointing to shared/tables/results/*.tex."""
    found: dict[str, str] = {}
    pattern = re.compile(
        r"\\input\{shared/tables/results/([^}]+\.tex)\}"
    )
    for tex in all_tex:
        try:
            src = tex.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for m in pattern.finditer(src):
            name = m.group(1)
            stem = re.sub(r"\.tex$", "", name)
            if stem not in found:
                found[stem] = str(tex.relative_to(_ROOT))
    return found


# ---------------------------------------------------------------------------
# render.py registry reader
# ---------------------------------------------------------------------------

def _load_render_registry() -> set[str]:
    """Extract the set of keys registered in render.py _REGISTRY."""
    try:
        src = _RENDER_PY.read_text(encoding="utf-8")
    except OSError:
        return set()
    # Find the _REGISTRY dict body (values have no nested braces, so [^}]+ is safe)
    m = re.search(r"_REGISTRY[^=]*=[^{]*\{([^}]+)\}", src, re.DOTALL)
    if not m:
        return set()
    body = m.group(1)
    return set(re.findall(r'"([^"]+)":', body))


# ---------------------------------------------------------------------------
# Checkers
# ---------------------------------------------------------------------------

class _Check:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []
        self.passes: list[str] = []

    def ok(self, msg: str) -> None:
        self.passes.append(msg)

    def fail(self, msg: str) -> None:
        self.failures.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


def _check_figure(
    stem: str,
    asset: _Asset,
    render_keys: set[str],
    check: _Check,
    verbose: bool,
) -> None:
    label = f"[fig:{stem}]"

    if asset.category == "tikz":
        # Just verify the defs wrapper exists
        defs_path = _SHARED / "figures" / "defs" / f"fig_{stem}.tex"
        alt_path = _SHARED / "figures" / "defs" / f"{stem}.tex"
        if defs_path.exists() or alt_path.exists():
            if verbose:
                check.ok(f"{label} TikZ source def exists")
        else:
            check.warn(f"{label} TikZ def file not found (non-critical for rendering)")
        return

    if asset.category == "unverified":
        # Soft checks only
        if asset.json:
            p = _FIG_DATA / asset.json
            if p.exists():
                import json
                with open(p) as f:
                    j = json.load(f)
                status = j.get("provenance_status", "")
                if verbose:
                    check.ok(f"{label} provenance JSON exists (status={status})")
            else:
                check.warn(f"{label} provenance JSON missing: {asset.json}")
        for out in asset.outputs:
            p = _FIG_RESULTS / out
            if p.exists():
                if verbose:
                    check.ok(f"{label} output exists: {out}")
            else:
                check.warn(
                    f"{label} output missing: {out}  "
                    f"(UNVERIFIED - data gap, not a hard failure)"
                )
        return

    # data_driven
    for csv_name in asset.csvs:
        p = _FIG_DATA / csv_name
        if p.exists():
            if verbose:
                check.ok(f"{label} CSV: {csv_name}")
        else:
            check.fail(f"{label} MISSING CSV: {csv_name}")

    if asset.json:
        p = _FIG_DATA / asset.json
        if p.exists():
            if verbose:
                check.ok(f"{label} JSON: {asset.json}")
        else:
            check.fail(f"{label} MISSING JSON: {asset.json}")

    for out in asset.outputs:
        p = _FIG_RESULTS / out
        if p.exists():
            if verbose:
                check.ok(f"{label} output: {out}")
        else:
            check.fail(f"{label} MISSING output: {out}")

    if asset.render_key is not None:
        if asset.render_key in render_keys:
            if verbose:
                check.ok(f"{label} render key '{asset.render_key}' registered")
        else:
            check.fail(
                f"{label} render key '{asset.render_key}' NOT in render.py _REGISTRY"
            )


def _check_table(
    stem: str,
    asset: _Asset,
    render_keys: set[str],
    check: _Check,
    verbose: bool,
) -> None:
    label = f"[tab:{stem}]"

    if asset.category == "static_table":
        for out in asset.outputs:
            p = _TAB_RESULTS / out
            if p.exists():
                if verbose:
                    check.ok(f"{label} static .tex exists: {out}")
            else:
                check.fail(f"{label} MISSING static .tex: {out}")
        return

    # generated_table
    for csv_name in asset.csvs:
        p = _TAB_DATA / csv_name
        if p.exists():
            if verbose:
                check.ok(f"{label} CSV: {csv_name}")
        else:
            check.fail(f"{label} MISSING CSV: {csv_name}")

    if asset.json:
        p = _TAB_DATA / asset.json
        if p.exists():
            if verbose:
                check.ok(f"{label} JSON: {asset.json}")
        else:
            check.fail(f"{label} MISSING JSON: {asset.json}")

    for out in asset.outputs:
        p = _TAB_RESULTS / out
        if p.exists():
            if verbose:
                check.ok(f"{label} .tex: {out}")
        else:
            check.fail(f"{label} MISSING .tex: {out}")

    if asset.render_key is not None:
        if asset.render_key in render_keys:
            if verbose:
                check.ok(f"{label} render key '{asset.render_key}' registered")
        else:
            check.fail(
                f"{label} render key '{asset.render_key}' NOT in render.py _REGISTRY"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verbose", "-v", action="store_true", help="Print passing checks too")
    args = ap.parse_args(argv)

    verbose: bool = args.verbose
    check = _Check()

    # --- discover manuscript assets (paper + book) ---
    main_tex = _MANUSCRIPTS / "paper" / "main.tex"
    book_tex = _MANUSCRIPTS / "book" / "main.tex"
    if not main_tex.exists():
        print(f"ERROR: main.tex not found at {main_tex}", file=sys.stderr)
        return 1

    visited: set[Path] = set()
    all_tex = _collect_inputs(main_tex, visited)
    if book_tex.exists():
        all_tex += _collect_inputs(book_tex, visited)

    discovered_figures = _discover_figures(all_tex)
    discovered_tables = _discover_generated_tables(all_tex)

    render_keys = _load_render_registry()

    # --- check for stale references to old output path ---
    old_path_pattern = re.compile(r"shared/figures/results/")
    stale_refs: list[str] = []
    for tex in all_tex:
        try:
            src = tex.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for m in old_path_pattern.finditer(src):
            stale_refs.append(str(tex.relative_to(_ROOT)))
            break  # one report per file is enough
    if stale_refs:
        for ref in stale_refs:
            check.fail(
                f"[path] stale reference to 'shared/figures/results/' in {ref} "
                f"-- update to 'paper/figures/'"
            )

    # --- check figures ---
    print(f"Discovered {len(discovered_figures)} figure asset(s) in paper + book")
    for stem, source in sorted(discovered_figures.items()):
        if stem in _FIGURE_MANIFEST:
            _check_figure(stem, _FIGURE_MANIFEST[stem], render_keys, check, verbose)
        else:
            check.fail(
                f"[fig:{stem}] discovered in manuscript ({source}) "
                f"but NOT in validator manifest -- add it to _FIGURE_MANIFEST"
            )

    # --- check discovered tables ---
    print(f"Discovered {len(discovered_tables)} generated table(s) in manuscript")
    for stem, source in sorted(discovered_tables.items()):
        if stem in _TABLE_MANIFEST:
            _check_table(stem, _TABLE_MANIFEST[stem], render_keys, check, verbose)
        else:
            check.fail(
                f"[tab:{stem}] discovered in manuscript ({source}) "
                f"but NOT in validator manifest -- add it to _TABLE_MANIFEST"
            )

    # --- also verify static tables referenced via defs ---
    # These are tables whose defs/ wrapper hardcodes the tabular (no \input to results/)
    # but whose results/ .tex must still exist (because they were authored and committed).
    # Discover by scanning defs/ for tables NOT covered by discovered_tables.
    defs_dir = _SHARED / "tables" / "defs"
    for def_file in sorted(defs_dir.glob("*.tex")):
        try:
            src = def_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for m in re.finditer(r"\\input\{shared/tables/results/([^}]+\.tex)\}", src):
            name = m.group(1)
            stem = re.sub(r"\.tex$", "", name)
            if stem not in discovered_tables and stem in _TABLE_MANIFEST:
                _check_table(stem, _TABLE_MANIFEST[stem], render_keys, check, verbose)

    # Static table .tex files that are not discovered via \input but need to exist
    static_stems = [k for k, v in _TABLE_MANIFEST.items() if v.category == "static_table"]
    for stem in static_stems:
        _check_table(stem, _TABLE_MANIFEST[stem], render_keys, check, verbose)

    # --- report ---
    print()
    if check.warnings:
        print(f"WARNINGS ({len(check.warnings)}):")
        for w in check.warnings:
            print(f"  WARN  {w}")
        print()

    if check.failures:
        print(f"FAILURES ({len(check.failures)}):")
        for f in check.failures:
            print(f"  FAIL  {f}")
        print()
        n_pass = len(check.passes)
        n_warn = len(check.warnings)
        n_fail = len(check.failures)
        print(f"Result: {n_pass} passed, {n_warn} warned, {n_fail} FAILED")
        return 1

    n_pass = len(check.passes)
    n_warn = len(check.warnings)
    print(f"Result: {n_pass} passed, {n_warn} warned, 0 failed -- inventory complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
