"""Figure 6: well-log change-point recovery.

Real-data showcase on the Ó Ruanaidh well-log NMR signal (via
:func:`bayesbreak.datasets.load_welllog`). Falls back to a deterministic
simulated analog when the download is unavailable.

Outputs
-------
- docs/report/figures/fig6_welllog.{png,pdf}
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
from bayesbreak.datasets import load_welllog  # noqa: E402


def main(outdir: Path, simulated: bool, subsample: int) -> None:
    bundle = load_welllog(simulated=simulated)
    if subsample > 1:
        bundle.X = bundle.X[::subsample]
        bundle.y = bundle.y[::subsample]
        if bundle.true_boundaries:
            bundle.true_boundaries = [b // subsample for b in bundle.true_boundaries]

    # Well-log is ~4050 pts; raw values in the 10^5 range make the diff-array
    # Bayes curve numerically unstable at the edges. MAP alone tells the story.
    est = BayesBreakGaussian(k_max=40, regression_curve="none")
    outdir.mkdir(parents=True, exist_ok=True)
    make_realdata_figure(
        estimator=est,
        bundle=bundle,
        outdir=outdir,
        fig_name="fig6_welllog",
        y_label="NMR response",
        title=f"Well-log ({bundle.source})",
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=Path("docs/report/figures"))
    ap.add_argument("--simulated", action="store_true")
    # Default subsample keeps runtime reasonable for n ~ 4k.
    ap.add_argument("--subsample", type=int, default=8)
    args = ap.parse_args()
    main(outdir=args.outdir, simulated=args.simulated, subsample=args.subsample)
