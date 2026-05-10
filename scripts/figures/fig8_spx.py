"""Figure 8: S&P 500 volatility-regime recovery.

Real-data showcase on daily ^GSPC log-squared returns (via
:func:`bayesbreak.datasets.load_spx`; requires ``yfinance`` for the live
download, otherwise falls back to a deterministic GARCH-like regime analog).

Outputs
-------
- docs/report/figures/fig8_spx.{png,pdf}
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
from bayesbreak.datasets import load_spx  # noqa: E402


def main(outdir: Path, simulated: bool, subsample: int, verified: bool = False) -> None:
    bundle = load_spx(simulated=simulated)
    if subsample > 1:
        bundle.X = bundle.X[::subsample]
        bundle.y = bundle.y[::subsample]
        if bundle.true_boundaries:
            bundle.true_boundaries = [b // subsample for b in bundle.true_boundaries]

    # SPX over ~2263 daily bars: many small volatility regimes => large k_max.
    est = BayesBreakGaussian(k_max=50, regression_curve="mix_k")
    outdir.mkdir(parents=True, exist_ok=True)
    make_realdata_figure(
        estimator=est,
        bundle=bundle,
        outdir=outdir,
        fig_name="fig8_spx",
        y_label=r"$\log r_t^2$",
        title=f"S&P 500 volatility regimes ({bundle.source})",
        verified=verified,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=Path("docs/report/figures"))
    ap.add_argument("--simulated", action="store_true")
    ap.add_argument("--subsample", type=int, default=4)
    ap.add_argument(
        "--verified",
        action="store_true",
        help="Author-approved finalized run (skip the placeholder watermark).",
    )
    args = ap.parse_args()
    main(
        outdir=args.outdir,
        simulated=args.simulated,
        subsample=args.subsample,
        verified=args.verified,
    )
