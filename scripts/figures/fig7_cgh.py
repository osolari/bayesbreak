"""Figure 7: array-CGH copy-number change-point recovery.

Real-data showcase on the Lai et al. (2005) CGH dataset (via
:func:`bayesbreak.datasets.load_cgh`). Falls back to the simulated CGH analog
when the download is unavailable. Heteroscedastic probe quality is passed
through ``sample_weight``.

Outputs
-------
- docs/report/figures/fig7_cgh.{png,pdf}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts" / "figures"))

from _realdata import make_realdata_figure  # noqa: E402

from bayesbreak import BayesBreakGaussian  # noqa: E402
from bayesbreak.datasets import load_cgh  # noqa: E402


def main(outdir: Path, simulated: bool) -> None:
    bundle = load_cgh(simulated=simulated)
    est = BayesBreakGaussian(k_max=15, regression_curve="mix_k")
    outdir.mkdir(parents=True, exist_ok=True)
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
    ap.add_argument("--outdir", type=Path, default=Path("docs/report/figures"))
    ap.add_argument("--simulated", action="store_true")
    args = ap.parse_args()
    main(outdir=args.outdir, simulated=args.simulated)
