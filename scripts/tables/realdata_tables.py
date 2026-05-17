"""Populate the four real-data metric tables in §6 from the on-disk
``.cache/fitcache/`` artifacts produced by the figure scripts.

This script does **not** refit on simulated fallback data. On a machine
where the upstream loaders (TCPD welllog, cran/ecp ACGH, yfinance SPX,
methylKit chr21) are not reachable, the populator instead reads the
pickled estimators left behind by the prior network-enabled run of the
``fig6_welllog`` / ``fig7_cgh`` / ``fig8_spx`` / ``fig9_methylation``
figure scripts. Those caches were produced from real data; their
``k_map_``, ``log_evidence_``, ``log_joint_map_``, and
``boundary_marginals_`` attributes are the manuscript-quality numbers
that should go into ``tab:real_welllog`` / ``tab:real_cgh`` /
``tab:real_spx`` / ``tab:real_methylation``.

Outputs::

    docs/report/figures/realdata_metrics.json
    docs/report/figures/realdata_metrics.txt
"""

from __future__ import annotations

import argparse
import json
import pickle
import platform
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np  # noqa: E402

import bayesbreak  # noqa: E402

FITCACHE = _ROOT / ".cache" / "fitcache"


def _packages() -> dict[str, str]:
    import numpy as _np
    import scipy as _sp

    return {
        "numpy": _np.__version__,
        "scipy": _sp.__version__,
        "bayesbreak": bayesbreak.__version__,
        "python": platform.python_version(),
    }


def _load_pkl(name: str) -> Any:
    path = FITCACHE / f"{name}.fit.pkl"
    if not path.exists():
        return None
    with path.open("rb") as fh:
        obj = pickle.load(fh)
    return obj.get("fit") if isinstance(obj, dict) else obj


def _load_runtime(name: str) -> float | None:
    """Read the cached fit's wall-clock runtime (if recorded). Old caches
    that pre-date the timing extension return None."""
    path = FITCACHE / f"{name}.fit.pkl"
    if not path.exists():
        return None
    try:
        with path.open("rb") as fh:
            obj = pickle.load(fh)
    except Exception:
        return None
    if isinstance(obj, dict):
        rt = obj.get("runtime_s")
        return float(rt) if rt is not None else None
    return None


def _boundary_entropy(p: np.ndarray | None) -> float | None:
    if p is None:
        return None
    eps = 1e-12
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return float(-(p * np.log(p) + (1.0 - p) * np.log1p(-p)).sum())


def _row_from_estimator(est: Any, *, config: str, runtime_s: float | None = None) -> dict[str, Any]:
    return {
        "config": config,
        "k_hat": int(est.k_map_) if hasattr(est, "k_map_") else None,
        "log_evidence": float(est.log_evidence_)
        if getattr(est, "log_evidence_", None) is not None
        else None,
        "log_joint_map": float(est.log_joint_map_)
        if getattr(est, "log_joint_map_", None) is not None
        else None,
        "n": int(est.n_) if hasattr(est, "n_") else None,
        "boundary_entropy_nats": _boundary_entropy(getattr(est, "boundary_marginals_", None)),
        "runtime_s": runtime_s,
    }


def welllog_rows() -> dict[str, Any]:
    est = _load_pkl("fig6_welllog")
    runtime_s = _load_runtime("fig6_welllog")
    rows: list[dict[str, Any]] = []
    if est is not None:
        rows.append(
            _row_from_estimator(est, config="Index-uniform prior, g ≡ 1", runtime_s=runtime_s)
        )
        # Refit the length-aware-prior variant on the cached training data
        # so the prior-sensitivity row can be populated without network.
        try:
            from bayesbreak import BayesBreakGaussian

            y = np.asarray(est._y_train_, dtype=float)
            X = np.asarray(est.x_design_, dtype=float).reshape(-1, 1)
            w = est.sample_weight_
            t0 = time.perf_counter()
            est_len = BayesBreakGaussian(
                k_max=int(est.k_max),
                length_prior=lambda d: float(d),
                regression_curve="none",
            ).fit(X, y, sample_weight=w)
            rt_len = time.perf_counter() - t0
            rows.append(
                _row_from_estimator(
                    est_len, config="Length-aware prior, g(ℓ) ∝ ℓ", runtime_s=rt_len
                )
            )
        except Exception as exc:
            rows.append(
                {
                    "config": "Length-aware prior, g(ℓ) ∝ ℓ",
                    "k_hat": None,
                    "log_evidence": None,
                    "needs_refit": f"length-aware refit failed: {exc!r}",
                }
            )
    else:
        rows.append({"config": "(no cache available)", "needs_refit": "fig6_welllog cache missing"})
    return {
        "dataset": "welllog",
        "rows": rows,
        "notes": (
            "ECE (boundary) requires external ground-truth boundaries; left as "
            "`---` until verified annotations are loaded."
        ),
    }


def cgh_rows() -> dict[str, Any]:
    obj = _load_pkl("fig7_cgh")
    runtime_s = _load_runtime("fig7_cgh")
    rows: list[dict[str, Any]] = []
    if obj is not None and isinstance(obj, dict):
        rep = obj.get("rep")
        per_le = obj.get("per_subj_logE", [])
        if rep is not None:
            rows.append(
                _row_from_estimator(
                    rep, config="Shared boundaries, subject-specific μ", runtime_s=runtime_s
                )
            )
        if per_le:
            rows.append(
                {
                    "config": "Independent per-subject (no pooling)",
                    "k_hat": None,  # union over 43 subject fits, not a single k
                    "log_evidence": float(sum(per_le)),
                    "log_joint_map": None,
                    "n": rep.n_ if rep is not None else None,
                    "n_subjects": len(per_le),
                    "note": (
                        "log_evidence is the sum of per-subject log A^0_{0,n} under "
                        "independent BayesBreakGaussian(k_max=15) fits; no single k_hat is reported."
                    ),
                }
            )
    elif obj is not None:
        rows.append(_row_from_estimator(obj, config="Shared boundaries, subject-specific μ"))
    else:
        rows.append({"config": "(no cache available)", "needs_refit": "fig7_cgh cache missing"})
    return {
        "dataset": "cgh",
        "rows": rows,
        "notes": (
            "Boundary F1 / MAE require external Snijders-2001 annotations; left as "
            "`---` until those annotations are bundled or verified at run time."
        ),
    }


def spx_rows() -> dict[str, Any]:
    est = _load_pkl("fig8_spx")
    runtime_s = _load_runtime("fig8_spx")
    rows: list[dict[str, Any]] = []
    if est is not None:
        rows.append(_row_from_estimator(est, config="Gaussian on log r_t^2", runtime_s=runtime_s))
        # Refit the Bernoulli-on-threshold-crossings variant on the
        # cached SPX training data so the secondary specification row
        # can be populated without yfinance.
        try:
            from bayesbreak import BayesBreakBernoulli

            y = np.asarray(est._y_train_, dtype=float)
            X = np.asarray(est.x_design_, dtype=float).reshape(-1, 1)
            thresh = float(np.quantile(y, 0.95))
            crossings = (y > thresh).astype(float)
            t0 = time.perf_counter()
            est_be = BayesBreakBernoulli(k_max=int(est.k_max)).fit(X, crossings)
            rt_be = time.perf_counter() - t0
            row = _row_from_estimator(
                est_be,
                config="Bernoulli on threshold crossings (95th pct)",
                runtime_s=rt_be,
            )
            row["threshold_quantile"] = 0.95
            rows.append(row)
        except Exception as exc:
            rows.append(
                {
                    "config": "Bernoulli on threshold crossings (95th pct)",
                    "k_hat": None,
                    "log_evidence": None,
                    "needs_refit": f"threshold-crossings refit failed: {exc!r}",
                }
            )
    else:
        rows.append({"config": "(no cache available)", "needs_refit": "fig8_spx cache missing"})
    return {
        "dataset": "spx",
        "rows": rows,
        "notes": (
            "Visual-alignment column is a qualitative description in §6 prose; "
            "no automated metric is reported here."
        ),
    }


def methylation_rows(*, do_holdout: bool = True) -> dict[str, Any]:
    est = _load_pkl("fig9_methylation")
    runtime_s = _load_runtime("fig9_methylation")
    rows: list[dict[str, Any]] = []
    if est is not None:
        row = _row_from_estimator(
            est,
            config="chr21 region A (methylKit test1.myCpG, n=1904)",
            runtime_s=runtime_s,
        )
        rows.append(row)

        if do_holdout:
            # Re-fit on the first 80% and score the last 20% to report a
            # held-out log-predictive number on the same real-data sequence.
            try:
                from bayesbreak import BayesBreakBetaObs
                from bayesbreak.datasets import load_methylation
                from bayesbreak.prediction import posterior_predictive_logpdf

                bundle = load_methylation()
                if bundle.source == "downloaded":
                    n = int(bundle.y.size)
                    cut = int(0.8 * n)
                    phi = bundle.sample_weight if bundle.sample_weight is not None else 50.0
                    phi_tr = phi[:cut] if hasattr(phi, "__len__") else phi
                    t0 = time.perf_counter()
                    est_tr = BayesBreakBetaObs(k_max=15, phi=phi_tr).fit(
                        bundle.X[:cut], bundle.y[:cut]
                    )
                    rt = time.perf_counter() - t0
                    per = posterior_predictive_logpdf(
                        est_tr, bundle.X[cut:], bundle.y[cut:], per_sample=True
                    )
                    rows[-1].update(
                        {
                            "held_out_loglik": float(np.sum(per)),
                            "held_out_n": int(n - cut),
                            "holdout_train_runtime_s": float(rt),
                            "held_out_train_k_hat": int(est_tr.k_map_),
                        }
                    )
            except Exception as exc:
                rows[-1]["holdout_error"] = repr(exc)

        rows.append(
            {
                "config": "Second cell type / second region",
                "k_hat": None,
                "log_evidence": None,
                "log_joint_map": None,
                "needs_refit": (
                    "the methylKit test1.myCpG file is a single chr21 region; a second "
                    "row requires the Loyfer 2023 atlas pipeline (GEO GSE186458 + "
                    "wgbs_tools / UXM_deconv) which is not yet wired in this repo."
                ),
            }
        )
    else:
        rows.append(
            {"config": "(no cache available)", "needs_refit": "fig9_methylation cache missing"}
        )
    return {
        "dataset": "methylation",
        "rows": rows,
        "notes": (
            "Boundary F1 vs. atlas requires verified annotations from the "
            "loyfer2023atlas pipeline; left as `---` until those annotations are bundled."
        ),
    }


def main(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "packages": _packages(),
        "source": (
            "loaded from .cache/fitcache/ (produced by fig6/7/8/9 scripts under real-data network access)"
        ),
        "welllog": welllog_rows(),
        "cgh": cgh_rows(),
        "spx": spx_rows(),
        "methylation": methylation_rows(),
    }
    (outdir / "realdata_metrics.json").write_text(json.dumps(report, indent=2, default=str))

    lines: list[str] = []
    for ds in ("welllog", "cgh", "spx", "methylation"):
        block = report[ds]
        lines.append(f"## {ds}")
        if "notes" in block:
            lines.append(f"   {block['notes']}")
        for row in block["rows"]:
            cfg = row.get("config", "?")
            k = row.get("k_hat")
            le = row.get("log_evidence")
            ljm = row.get("log_joint_map")
            line = f"  - {cfg}: k_hat={k}, log_evidence={le}, log_joint_map={ljm}"
            if "needs_refit" in row:
                line += f"\n      needs_refit: {row['needs_refit']}"
            if "note" in row:
                line += f"\n      note: {row['note']}"
            if "held_out_loglik" in row:
                line += (
                    f"\n      held_out_loglik={row['held_out_loglik']}, "
                    f"held_out_n={row['held_out_n']}, "
                    f"holdout_train_k_hat={row.get('held_out_train_k_hat')}, "
                    f"holdout_train_runtime_s={row.get('holdout_train_runtime_s')}"
                )
            if "n_subjects" in row:
                line += f"\n      n_subjects={row['n_subjects']}"
            lines.append(line)
        lines.append("")
    (outdir / "realdata_metrics.txt").write_text("\n".join(lines), encoding="utf-8")
    for line in lines:
        print(line)
    print(f"\nwrote {outdir / 'realdata_metrics.json'}")
    print(f"wrote {outdir / 'realdata_metrics.txt'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--outdir", type=Path, default=Path("docs/report/figures"), help="output directory"
    )
    args = ap.parse_args()
    main(outdir=args.outdir)
