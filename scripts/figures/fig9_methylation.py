"""Figure 9: CpG methylation change-point recovery.

Real-data showcase on a methylation fraction sequence (via
:func:`bayesbreak.datasets.load_methylation`). Pass ``--csv-path`` to use a
local CSV; otherwise the deterministic simulated analog is used.

Outputs
-------
- results/figures/fig9_methylation.{png,pdf}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts" / "figures"))

import numpy as np  # noqa: E402
from _realdata import make_realdata_figure  # noqa: E402

from bayesbreak import BayesBreakBeta  # noqa: E402
from bayesbreak.datasets import load_methylation  # noqa: E402


def main(outdir: Path, simulated: bool, csv_path: str | None) -> None:
    bundle = load_methylation(simulated=simulated, csv_path=csv_path)

    # If the loader supplies per-CpG coverage as sample_weight, route it
    # into BayesBreakBetaObs as φ_t (the report's known-precision Beta-obs
    # parametrisation §sec:realdata-methylation). Drop the weight from the
    # bundle so it is not also applied as a power-likelihood weight.
    from bayesbreak import BayesBreakBetaObs as _BetaObs

    if bundle.sample_weight is not None and bundle.sample_weight.shape == bundle.y.shape:
        phi = np.asarray(bundle.sample_weight, dtype=float)
        # Floor coverage at 1.0 to avoid degenerate Beta(0, 0).
        phi = np.maximum(phi, 1.0)
        est = _BetaObs(k_max=15, phi=phi, regression_curve="mix_k")
        bundle.sample_weight = None
    else:
        est = BayesBreakBeta(k_max=15, concentration=70.0, regression_curve="mix_k")
    outdir.mkdir(parents=True, exist_ok=True)
    make_realdata_figure(
        estimator=est,
        bundle=bundle,
        outdir=outdir,
        fig_name="fig9_methylation",
        y_label="methylation fraction",
        title=f"CpG methylation ({bundle.source})",
        show_null_baseline=False,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=Path("results/figures"))
    ap.add_argument("--simulated", action="store_true")
    ap.add_argument("--csv-path", type=str, default=None)
    args = ap.parse_args()
    main(outdir=args.outdir, simulated=args.simulated, csv_path=args.csv_path)
