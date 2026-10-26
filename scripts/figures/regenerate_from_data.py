"""Regenerate all E01-E07 experiment figures from canonical figure-input data.

Input data lives in docs/manuscripts/shared/figures/data/ (CSV files committed
to the repo). Outputs go to docs/manuscripts/shared/figures/results/.

This script does not re-run any experiments. It only redraws figures from the
committed numerical data, so style changes can be applied quickly.

Usage:
    python scripts/figures/regenerate_from_data.py [--experiments E01 E03 ...]
"""

from __future__ import annotations

import argparse
import ast
import csv
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts" / "figures"))
sys.path.insert(0, str(_ROOT / "scripts"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

_DATA_DIR = _ROOT / "docs" / "manuscripts" / "shared" / "figures" / "data"
_OUT_DIR = _ROOT / "docs" / "manuscripts" / "shared" / "figures" / "results"


def _load_csv(path: Path) -> list[dict]:
    """Load CSV and coerce numeric strings to float where possible."""
    with open(path) as f:
        rows = []
        for row in csv.DictReader(f):
            d: dict = {}
            for k, v in row.items():
                try:
                    d[k] = float(v)
                except (ValueError, TypeError):
                    d[k] = v
            rows.append(d)
    return rows


def regen_e01() -> None:
    from run_e01_gaussian_calibration import regenerate_figures
    log.info("E01: calling regenerate_figures()")
    regenerate_figures(manuscript_dest=_OUT_DIR)


def regen_e02() -> None:
    from run_e02_family_portability import regenerate_figures
    log.info("E02: calling regenerate_figures()")
    regenerate_figures(manuscript_dest=_OUT_DIR)


def regen_e03() -> None:
    from run_e03_irregular_design import _make_figures
    rows = _load_csv(_DATA_DIR / "fig_E03_data.csv")
    log.info("E03: %d rows loaded", len(rows))
    _make_figures(rows, _OUT_DIR)


def regen_e04() -> None:
    from run_e04_shared_boundary import _make_figures
    rows = _load_csv(_DATA_DIR / "fig_E04_data.csv")
    log.info("E04: %d rows loaded", len(rows))
    _make_figures(rows, _OUT_DIR)


def regen_e05() -> None:
    import run_e05_latent_templates as _e05_mod
    rows = _load_csv(_DATA_DIR / "fig_E05_data.csv")
    log.info("E05: %d rows loaded", len(rows))
    orig = _e05_mod.FIGURES_DIR
    _e05_mod.FIGURES_DIR = _OUT_DIR
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        _e05_mod._generate_figures(rows)
    finally:
        _e05_mod.FIGURES_DIR = orig


def regen_e06() -> None:
    from run_e06_nonconjugate import (
        BlockMetric, SegResult, _write_figures as _e06_write_figures
    )

    bm_rows = _load_csv(_DATA_DIR / "fig_E06_metrics.csv")
    block_metrics = [
        BlockMetric(
            B=int(r["B"]), p=float(r["p"]), replicate=int(r["replicate"]),
            S=float(r["S"]), ref_lml=float(r["ref_lml"]),
            la_lml=float(r["la_lml"]), jj_lml=float(r["jj_lml"]),
            ep_lml=float(r["ep_lml"]),
            la_signed_err=float(r["la_signed_err"]),
            jj_signed_err=float(r["jj_signed_err"]),
            ep_signed_err=float(r["ep_signed_err"]),
            la_abs_err=float(r["la_abs_err"]),
            jj_abs_err=float(r["jj_abs_err"]),
            ep_abs_err=float(r["ep_abs_err"]),
            jj_lb_ok=str(r["jj_lb_ok"]).lower() in ("true", "1"),
            ep_convergence=str(r["ep_convergence"]).lower() in ("true", "1"),
            la_runtime=float(r["la_runtime"]),
            jj_runtime=float(r["jj_runtime"]),
            ep_runtime=float(r["ep_runtime"]),
            ref_runtime=float(r["ref_runtime"]),
        )
        for r in bm_rows
    ]

    seg_rows = _load_csv(_DATA_DIR / "fig_E06_seg_results.csv")
    seg_results = []
    for r in seg_rows:
        def _parse_list(s: str) -> list:
            try:
                return ast.literal_eval(str(s))
            except Exception:
                return []
        seg_results.append(SegResult(
            replicate=int(r["replicate"]),
            method=str(r["method"]),
            map_boundaries=_parse_list(r.get("map_boundaries", "[]")),
            f1_at_3=float(r.get("f1_at_3", 0.0)),
            pk=_parse_list(r.get("pk", "[]")),
            bm_at_true=_parse_list(r.get("bm_at_true", "[]")),
        ))

    log.info("E06: %d block metrics, %d seg results", len(block_metrics), len(seg_results))
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    _e06_write_figures(_OUT_DIR, block_metrics, seg_results)


def regen_e07() -> None:
    from run_e07_misspecification import _plot_results
    rows = _load_csv(_DATA_DIR / "fig_E07_data.csv")
    bool_cols = {"fit_failed", "is_correct_spec"}
    for row in rows:
        for col in bool_cols:
            if col in row:
                row[col] = str(row[col]).lower() in ("true", "1")
    log.info("E07: %d rows loaded", len(rows))
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    _plot_results(rows, _OUT_DIR)


_REGEN_MAP: dict[str, callable] = {
    "E01": regen_e01,
    "E02": regen_e02,
    "E03": regen_e03,
    "E04": regen_e04,
    "E05": regen_e05,
    "E06": regen_e06,
    "E07": regen_e07,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--experiments", nargs="*", default=list(_REGEN_MAP.keys()),
        help="Which experiments to regenerate (default: all).",
    )
    args = ap.parse_args()

    requested = [e.upper() for e in args.experiments]
    unknown = [e for e in requested if e not in _REGEN_MAP]
    if unknown:
        ap.error(f"Unknown experiments: {unknown}. Choose from {list(_REGEN_MAP)}")

    for name in requested:
        log.info("=== Regenerating %s figures ===", name)
        try:
            _REGEN_MAP[name]()
            log.info("%s done", name)
        except Exception as exc:
            log.error("%s FAILED: %s", name, exc, exc_info=True)


if __name__ == "__main__":
    main()
