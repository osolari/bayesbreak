"""Figure 7: array-CGH copy-number change-point recovery via shared-boundary
multi-subject pooling (Theorem ``multisubject``).

Real path: ``bayesbreak.datasets.load_cgh`` returns a
``(n_probes, n_subjects)`` bundle from the ``cran/ecp`` ``ACGH.RData``
mirror. We pass the multi-subject array to
:class:`SharedBoundaryReplicatesSegmenter` (heteroscedastic Gaussian per
subject, weights from the loader's rolling-MAD precisions), which yields
one MAP boundary vector shared across subjects together with subject-
specific MAP segment means. Falls back to the simulated single-subject
analog when the download is unavailable.

Outputs
-------
- results/figures/fig7_cgh.{png,pdf}
- results/figures/fig7_cgh.json (run record + DP diagnostics)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts" / "figures"))

from _realdata import make_realdata_figure  # noqa: E402
from _style import (  # noqa: E402
    COLOR_CYCLE,
    COLORS,
    add_panel_label,
    save_figure,
    setup_style,
)

from bayesbreak import (  # noqa: E402
    BayesBreakGaussian,
    SharedBoundaryReplicatesSegmenter,
    run_dp_diagnostics,
)
from bayesbreak.datasets import load_cgh  # noqa: E402
from bayesbreak.experiments._fitcache import cache_key, fit_or_load  # noqa: E402
from bayesbreak.experiments._placeholder import hash_array, write_run_record  # noqa: E402


def _multi_subject_figure(
    bundle: Any,
    outdir: Path,
    fig_name: str = "fig7_cgh",
    *,
    n_show: int = 4,
) -> None:
    """Render the CGH multi-subject pooled segmentation figure."""

    setup_style(font_scale=0.95)

    n, S = bundle.y.shape
    # Subset to the first ``n_show`` subjects for visualization, but pool over
    # *all* available subjects in the fit.
    X = bundle.X[:, 0] if bundle.X.ndim == 2 else bundle.X
    rep_template = SharedBoundaryReplicatesSegmenter(
        BayesBreakGaussian(k_max=15, regression_curve="none")
    )
    key = cache_key(
        y=bundle.y,
        sample_weight=bundle.sample_weight,
        params={"replicates": True, **rep_template.base_estimator.get_params()},
        extra=fig_name,
    )

    def _fit() -> dict[str, Any]:
        rep_template.fit(
            X.reshape(-1, 1),
            [bundle.y[:, s] for s in range(S)],
            sample_weight=[bundle.sample_weight[:, s] for s in range(S)],
        )
        # Per-subject single-subject log evidences for panel D.
        per_subj_logE = []
        for s in range(S):
            e = BayesBreakGaussian(k_max=15).fit(
                X.reshape(-1, 1), bundle.y[:, s], sample_weight=bundle.sample_weight[:, s]
            )
            per_subj_logE.append(float(e.log_evidence_))
        return {"rep": rep_template, "per_subj_logE": per_subj_logE}

    fit_payload = fit_or_load(outdir / f"{fig_name}.fit.pkl", key, _fit)
    rep = fit_payload["rep"]
    per_subj_logE = fit_payload["per_subj_logE"]

    idx = np.arange(n, dtype=float)
    fig = plt.figure(figsize=(8.0, 9.2))
    gs = fig.add_gridspec(4, 1, height_ratios=[2.4, 1.0, 1.0, 1.1], hspace=0.34)
    axA = fig.add_subplot(gs[0])
    axB = fig.add_subplot(gs[1], sharex=axA)
    axC = fig.add_subplot(gs[2])
    axD = fig.add_subplot(gs[3])

    # Panel A: representative subjects with shared MAP boundaries.
    pal = COLOR_CYCLE
    show = list(range(min(n_show, S)))
    for i, s in enumerate(show):
        offset = i * 0.6  # vertical stacking for legibility
        axA.scatter(
            idx,
            bundle.y[:, s] + offset,
            s=3,
            alpha=0.32,
            color=pal[i % len(pal)],
            edgecolors="none",
            label=f"subj {s}",
        )
        axA.plot(idx, rep.map_curve_[s] + offset, lw=1.4, color=pal[i % len(pal)])
    for b in rep.map_boundaries_[1:-1]:
        axA.axvline(float(b), color="k", ls="--", lw=0.5, alpha=0.55)
    axA.set_ylabel(r"$\log_2$ ratio (offset per subject)")
    axA.legend(loc="best", ncol=min(4, len(show)), fontsize=10, frameon=False)
    add_panel_label(axA, "A", f"Array-CGH ({bundle.source}, n_subj={S})")
    plt.setp(axA.get_xticklabels(), visible=False)

    # Panel B: pooled boundary marginal.
    bm = rep.boundary_marginals_
    xb = idx[1:]
    axB.fill_between(xb, 0.0, bm, color=COLORS["blue"], alpha=0.4, linewidth=0)
    axB.plot(xb, bm, color=COLORS["blue"], lw=1.0)
    for b in rep.map_boundaries_[1:-1]:
        axB.axvline(float(b), color="k", ls="--", lw=0.5, alpha=0.55)
    axB.set_ylim(0.0, 1.05)
    axB.set_ylabel(r"$P(b_i = 1 \mid y, k_{\mathrm{map}})$")
    axB.set_xlabel("probe index")
    add_panel_label(axB, "B")

    # Panel C: P(k|y).
    k_vals = np.arange(1, rep.k_posterior_.size + 1)
    axC.bar(k_vals, rep.k_posterior_, color=COLORS["blue"], edgecolor="none", width=0.7)
    axC.axvline(
        rep.k_map_, color=COLORS["red"], ls="--", lw=1.0, label=rf"$k_{{MAP}} = {rep.k_map_}$"
    )
    axC.set_xlim(0.3, rep.k_posterior_.size + 0.7)
    axC.set_xlabel("k")
    axC.set_ylabel(r"$P(k \mid y)$")
    axC.legend(loc="best", fontsize=10, frameon=False)
    add_panel_label(axC, "C")

    # Panel D: per-subject log evidence (loaded from the fit cache above).
    axD.bar(np.arange(S), per_subj_logE, color=COLORS["grey"], edgecolor="none")
    axD.axhline(
        rep.log_evidence_ / max(1, S),
        color=COLORS["red"],
        ls="--",
        lw=1.0,
        label=f"pooled log E / S = {rep.log_evidence_ / max(1, S):.2f}",
    )
    axD.set_xlabel("subject")
    axD.set_ylabel("per-subject log evidence")
    axD.legend(loc="best", fontsize=10, frameon=False)
    add_panel_label(axD, "D")

    save_figure(fig, outdir / fig_name, formats=("png", "pdf"))

    # Sidecar JSON: provenance + DP diagnostics.
    diag = run_dp_diagnostics(rep)
    record_extra: dict[str, Any] = {
        "source": bundle.source,
        "description": bundle.description,
        "y_hash": hash_array(bundle.y),
        "sample_weight_hash": hash_array(bundle.sample_weight),
        "n_probes": int(n),
        "n_subjects": int(S),
        "k_map": int(rep.k_map_),
        "log_evidence_pooled": float(rep.log_evidence_),
        "per_subject_log_evidence": per_subj_logE,
        "map_boundaries": list(rep.map_boundaries_),
        "dp_diagnostics": diag.to_dict(),
        "estimator": "SharedBoundaryReplicatesSegmenter",
    }
    write_run_record(
        outdir / f"{fig_name}.pdf",
        dataset=bundle.name,
        source=bundle.source,
        extra=record_extra,
    )


def main(outdir: Path, simulated: bool) -> None:
    bundle = load_cgh(simulated=simulated)
    outdir.mkdir(parents=True, exist_ok=True)

    if bundle.y.ndim == 2:
        _multi_subject_figure(bundle, outdir)
        return

    # Single-subject fallback (simulated path).
    est = BayesBreakGaussian(k_max=15, regression_curve="mix_k")
    make_realdata_figure(
        estimator=est,
        bundle=bundle,
        outdir=outdir,
        fig_name="fig7_cgh",
        y_label=r"$\log_2$ ratio",
        title=f"Array-CGH ({bundle.source})",
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=Path("results/figures"))
    ap.add_argument("--simulated", action="store_true")
    args = ap.parse_args()
    main(outdir=args.outdir, simulated=args.simulated)
