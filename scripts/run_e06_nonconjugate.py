"""E06: nonconjugate block-evidence approximations and segmentation propagation.

Implements the frozen E06 protocol from experiment_protocols.md.

Block oracle experiment:
- B in {4, 8, 16, 32, 64, 128}, p in {0.05, 0.5, 0.95}, 100 replicates per cell (1800 blocks).
- Methods: quadrature (reference), Laplace, JJ, EP.
- pg_vb is an alias of JJ; it is NOT benchmarked as an independent method.
- Seed: SeedSequence([20261021, 6, cell_index, replicate]).
- Reference: mode-centered adaptive quadrature (epsabs=1e-12); GH-200 centered on the prior
  is inaccurate for large near-balanced blocks where the posterior is sharply peaked.

Segmentation propagation:
- n=96, k_true=3, boundaries (32, 64), theta levels logit((0.2, 0.8, 0.2)).
- 100 independent Bernoulli sequences.
- SeedSequence([20261021, 6, 1000 + replicate]).
- Methods: quadrature reference, Laplace, JJ, EP.
- Metrics: p(k|y), boundary marginals, MAP partition, F1@3.

Outputs in: results/confirmatory/E06_nonconjugate/<config_sha256[:16]>/

Usage:
    uv run python scripts/run_e06_nonconjugate.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import scipy.integrate
import scipy.optimize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("run_e06")

# ---------------------------------------------------------------------------
# Frozen E06 configuration
# ---------------------------------------------------------------------------
E06_CONFIG: dict[str, Any] = {
    "experiment_id": "E06",
    "block_sizes": [4, 8, 16, 32, 64, 128],
    "true_success_probs": [0.05, 0.5, 0.95],
    "n_replicates_block": 100,
    "prior_nu": 0.0,
    "prior_rho2": 4.0,
    "methods": ["quadrature", "laplace", "jj", "ep"],
    # Reference: mode-centered adaptive quadrature (not GH-200).
    # GH-200 centered on the prior is inaccurate for large B near-balanced blocks
    # where the posterior is sharply peaked far from the prior center.
    # Mode-centered adaptive quadrature (epsabs=1e-12) provides an accurate reference.
    "reference_method": "mode_centered_quad",
    "reference_epsabs": 1e-12,
    "reference_epsrel": 1e-11,
    # GH points for the segmentation propagation experiment (approximation quality).
    "seg_gh_points": 80,
    "ep_tol_kl": 1e-7,
    "ep_damping": 0.7,
    "ep_max_iter": 300,
    "ep_n_gh": 80,
    "laplace_max_iter": 50,
    "jj_max_iter": 500,
    "jj_tol": 1e-12,
    "seed_base": [20261021, 6],
    "gate_jj_lb_rate": 0.95,
    "gate_jj_lb_tol": 1e-8,
    "seg_n": 96,
    "seg_k_true": 3,
    "seg_boundaries": [32, 64],
    "seg_p_levels": [0.2, 0.8, 0.2],
    "seg_n_replicates": 100,
    "seg_k_max": 8,
    "seg_seed_offset": 1000,
    "protocol_version": "1.0.1",
    "note_pg_vb": "pg_vb is an alias of jj in the estimator code; not benchmarked independently",
}


def _config_sha256() -> str:
    s = json.dumps(E06_CONFIG, sort_keys=True).encode("utf-8")
    return hashlib.sha256(s).hexdigest()


# ---------------------------------------------------------------------------
# Reference: mode-centered adaptive quadrature
# ---------------------------------------------------------------------------

def _mode_centered_quad(
    S: float,
    N: float,
    nu: float,
    rho2: float,
    epsabs: float = 1e-12,
    epsrel: float = 1e-11,
) -> float:
    """Mode-centered adaptive quadrature log marginal likelihood.

    GH quadrature centered on the prior N(nu, rho2) is inaccurate for large
    blocks where the posterior is sharply peaked away from the prior center.
    This function finds the posterior mode, centers the integration there,
    and uses scipy adaptive quadrature for an accurate result.
    """

    def log_f(theta: float) -> float:
        if theta <= 0:
            ll = S * theta - N * math.log1p(math.exp(theta))
        else:
            ll = S * theta - N * (theta + math.log1p(math.exp(-theta)))
        lp = -0.5 * (math.log(2 * math.pi * rho2) + (theta - nu) ** 2 / rho2)
        return ll + lp

    opt = scipy.optimize.minimize_scalar(
        lambda t: -log_f(t),
        bounds=(-20.0, 20.0),
        method="bounded",
        options={"xatol": 1e-12},
    )
    theta_mode = float(opt.x)
    log_max = log_f(theta_mode)

    def sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    sig = sigmoid(theta_mode)
    h = -(N * sig * (1.0 - sig) + 1.0 / rho2)
    half_width = max(10.0 * math.sqrt(-1.0 / h) if h < 0 else 5.0, 5.0)

    def shifted(t: float) -> float:
        val = log_f(theta_mode + t) - log_max
        return math.exp(val) if val > -500 else 0.0

    res, _err = scipy.integrate.quad(
        shifted, -half_width, half_width, epsabs=epsabs, epsrel=epsrel, limit=500
    )
    if res <= 0.0:
        return -math.inf
    return log_max + math.log(res)


# ---------------------------------------------------------------------------
# Block metric computation
# ---------------------------------------------------------------------------

@dataclass
class BlockMetric:
    B: int
    p: float
    replicate: int
    S: float
    ref_lml: float
    la_lml: float
    jj_lml: float
    ep_lml: float
    la_signed_err: float
    jj_signed_err: float
    ep_signed_err: float
    la_abs_err: float
    jj_abs_err: float
    ep_abs_err: float
    jj_lb_ok: bool
    ep_convergence: bool
    la_runtime: float
    jj_runtime: float
    ep_runtime: float
    ref_runtime: float


def compute_block_metrics(
    B: int,
    p: float,
    cell_index: int,
    replicate: int,
    cfg: dict[str, Any],
) -> BlockMetric:
    """Draw one block and compute metrics for all methods."""
    from bayesbreak.families.logistic_normal._ep import ep_single_block
    from bayesbreak.families.logistic_normal._jj import jj_block
    from bayesbreak.families.logistic_normal._laplace import laplace_block

    nu = cfg["prior_nu"]
    rho2 = cfg["prior_rho2"]

    # Draw block: E06 seed scheme
    ss = np.random.SeedSequence(cfg["seed_base"] + [cell_index, replicate])
    child = ss.spawn(1)[0]
    rng = np.random.default_rng(child)
    y = rng.binomial(1, p, size=B).astype(float)
    S = float(y.sum())
    N = float(B)

    S_arr = np.array([S])
    N_arr = np.array([N])

    # Reference: mode-centered adaptive quadrature.
    t0 = time.perf_counter()
    ref = _mode_centered_quad(
        S, N, nu, rho2,
        epsabs=cfg["reference_epsabs"],
        epsrel=cfg["reference_epsrel"],
    )
    ref_runtime = time.perf_counter() - t0

    # Laplace
    t0 = time.perf_counter()
    la_arr, _m, _v, _g = laplace_block(
        S_arr, N_arr, nu=nu, rho2=rho2, max_iter=cfg["laplace_max_iter"]
    )
    la = float(la_arr[0])
    la_runtime = time.perf_counter() - t0

    # JJ
    t0 = time.perf_counter()
    jj_arr, _m, _v = jj_block(
        S_arr, N_arr, nu=nu, rho2=rho2,
        max_iter=cfg["jj_max_iter"], tol=cfg["jj_tol"],
    )
    jj = float(jj_arr[0])
    jj_runtime = time.perf_counter() - t0

    # EP: use the actual drawn block (not just sufficient statistics)
    t0 = time.perf_counter()
    ep_logA0, _m, _v, _pm, ep_ok = ep_single_block(
        y, np.ones(B),
        nu=nu, rho2=rho2,
        max_iter=cfg["ep_max_iter"],
        tol_kl=cfg["ep_tol_kl"],
        damping=cfg["ep_damping"],
        n_gh=cfg["ep_n_gh"],
    )
    ep = float(ep_logA0)
    ep_runtime = time.perf_counter() - t0

    jj_lb_ok = jj <= ref + cfg["gate_jj_lb_tol"]

    return BlockMetric(
        B=B, p=p, replicate=replicate, S=S,
        ref_lml=ref, la_lml=la, jj_lml=jj, ep_lml=ep,
        la_signed_err=la - ref, jj_signed_err=jj - ref, ep_signed_err=ep - ref,
        la_abs_err=abs(la - ref), jj_abs_err=abs(jj - ref), ep_abs_err=abs(ep - ref),
        jj_lb_ok=jj_lb_ok, ep_convergence=bool(ep_ok),
        la_runtime=la_runtime, jj_runtime=jj_runtime,
        ep_runtime=ep_runtime, ref_runtime=ref_runtime,
    )


# ---------------------------------------------------------------------------
# Segmentation propagation
# ---------------------------------------------------------------------------

@dataclass
class SegResult:
    replicate: int
    method: str
    map_boundaries: list[int] = field(default_factory=list)
    f1_at_3: float = 0.0
    pk: list[float] = field(default_factory=list)
    bm_at_true: list[float] = field(default_factory=list)


def _f1_at_tau(pred: list[int], true: list[int], tau: int = 3) -> float:
    if not true:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    matched_p: set[int] = set()
    matched_t: set[int] = set()
    for tb in true:
        best_i: int | None = None
        best_d = tau + 1
        for i, pb in enumerate(pred):
            if i in matched_p:
                continue
            d = abs(pb - tb)
            if d <= tau and d < best_d:
                best_i, best_d = i, d
        if best_i is not None:
            matched_p.add(best_i)
            matched_t.add(tb)
    tp = len(matched_p)
    prec = tp / len(pred)
    rec = tp / len(true)
    return 2.0 * prec * rec / (prec + rec) if prec + rec > 0.0 else 0.0


def run_segmentation_propagation(
    cfg: dict[str, Any],
    output_dir: Path,
) -> list[SegResult]:
    from bayesbreak.families.logistic_normal import BayesBreakLogisticNormal

    n = cfg["seg_n"]
    boundaries = cfg["seg_boundaries"]
    p_levels = cfg["seg_p_levels"]
    k_max = cfg["seg_k_max"]
    nu = cfg["prior_nu"]
    rho2 = cfg["prior_rho2"]
    n_reps = cfg["seg_n_replicates"]
    methods = cfg["methods"]

    theta_levels = [math.log(p) - math.log(1 - p) for p in p_levels]
    X_design = np.arange(n, dtype=float).reshape(-1, 1)

    seg_results: list[SegResult] = []

    for rep in range(n_reps):
        ss = np.random.SeedSequence(cfg["seed_base"] + [cfg["seg_seed_offset"] + rep])
        child = ss.spawn(1)[0]
        rng = np.random.default_rng(child)

        y = np.zeros(n, dtype=float)
        segs = list(zip([0] + boundaries, boundaries + [n]))
        for seg_i, (s, e) in enumerate(segs):
            p_seg = 1.0 / (1.0 + math.exp(-theta_levels[seg_i]))
            y[s:e] = rng.binomial(1, p_seg, size=e - s).astype(float)

        ref_bm: np.ndarray | None = None
        ref_pk: np.ndarray | None = None

        for method in methods:
            est = BayesBreakLogisticNormal(
                k_max=k_max,
                approx=method,
                nu=nu,
                rho2=rho2,
                estimate_hyper=False,
                max_iter=100,
                ep_tol_kl=cfg["ep_tol_kl"],
                ep_damping=cfg["ep_damping"],
                gh_points=cfg["seg_gh_points"],
            )
            est.fit(X_design, y)

            pk = np.asarray(est.k_posterior_, dtype=float)
            bm = np.asarray(est.boundary_marginals_, dtype=float)
            map_b = list(est.map_boundaries_)
            f1 = _f1_at_tau(map_b, boundaries, tau=3)

            if method == "quadrature":
                ref_bm = bm
                ref_pk = pk

            # Boundary marginals at true boundary positions
            bm_at_true = []
            for b in boundaries:
                idx = b - 1  # boundary_marginals_[i] = p(b_{i+1}=1|y) or similar
                if 0 <= idx < len(bm):
                    bm_at_true.append(float(bm[idx]))
                else:
                    bm_at_true.append(float("nan"))

            seg_results.append(SegResult(
                replicate=rep,
                method=method,
                map_boundaries=sorted(map_b),
                f1_at_3=f1,
                pk=pk.tolist(),
                bm_at_true=bm_at_true,
            ))

        if rep % 20 == 0:
            logger.info("Segmentation rep %d/%d done", rep + 1, n_reps)

    return seg_results


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _write_manifest(output_dir: Path, cfg: dict[str, Any], cfg_sha: str) -> None:
    env: dict[str, str] = {
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "utc_start": datetime.now(timezone.utc).isoformat(),
    }
    manifest = {
        "experiment_id": "E06",
        "config_sha256": cfg_sha,
        "config": cfg,
        "environment": env,
        "protocol_version": cfg["protocol_version"],
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest written to %s", output_dir / "manifest.json")


def _write_metrics(output_dir: Path, block_metrics: list[BlockMetric]) -> None:
    fields = list(BlockMetric.__dataclass_fields__.keys())
    with open(output_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for m in block_metrics:
            writer.writerow(asdict(m))
    logger.info("Block metrics written (%d rows)", len(block_metrics))


def _write_oracle_checks(output_dir: Path, block_metrics: list[BlockMetric]) -> None:
    # Degenerate blocks: all-success and all-failure per B value
    from bayesbreak.families.logistic_normal._ep import ep_single_block
    from bayesbreak.families.logistic_normal._jj import jj_block
    from bayesbreak.families.logistic_normal._laplace import laplace_block

    nu, rho2 = E06_CONFIG["prior_nu"], E06_CONFIG["prior_rho2"]
    checks: list[dict[str, Any]] = []

    for B in [4, 8, 16, 32, 64]:
        for label, S in [("all-failure", 0.0), ("all-success", float(B))]:
            mc_ref = _mode_centered_quad(
                S, float(B), nu, rho2,
                epsabs=E06_CONFIG["reference_epsabs"],
                epsrel=E06_CONFIG["reference_epsrel"],
            )
            ref = mc_ref  # reference = mode-centered quad
            S_arr = np.array([S])
            N_arr = np.array([float(B)])
            la_arr, _, _, _ = laplace_block(S_arr, N_arr, nu=nu, rho2=rho2)
            jj_arr, _, _ = jj_block(S_arr, N_arr, nu=nu, rho2=rho2, max_iter=500, tol=1e-12)
            y_degen = np.zeros(B)
            if S == float(B):
                y_degen[:] = 1.0
            ep_logA0, _, _, _, ep_ok = ep_single_block(
                y_degen, np.ones(B), nu=nu, rho2=rho2,
                max_iter=300, tol_kl=1e-7, damping=0.7, n_gh=80,
            )
            checks.append({
                "B": B, "label": label, "S": S,
                "ref_mc_quad": mc_ref,
                "la": float(la_arr[0]), "jj": float(jj_arr[0]),
                "ep": float(ep_logA0), "ep_ok": bool(ep_ok),
                "la_err": float(la_arr[0]) - ref,
                "jj_err": float(jj_arr[0]) - ref,
                "ep_err": float(ep_logA0) - ref,
                "jj_lb_ok": float(jj_arr[0]) <= ref + 1e-8,
            })
            logger.info(
                "Oracle %s B=%d: mc_ref=%.4f la=%.4f jj=%.4f ep=%.4f | "
                "JJ<=%.5f EP_ok=%s",
                label, B, mc_ref,
                float(la_arr[0]), float(jj_arr[0]), float(ep_logA0),
                float(jj_arr[0]) - ref, ep_ok,
            )

    with open(output_dir / "oracle_checks.json", "w") as f:
        json.dump(checks, f, indent=2)
    logger.info("Oracle checks written")


def _write_summary_metrics(
    output_dir: Path,
    block_metrics: list[BlockMetric],
    seg_results: list[SegResult],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Compute and write summary metrics.json."""
    jj_lb_total = len(block_metrics)
    jj_lb_ok = sum(1 for m in block_metrics if m.jj_lb_ok)
    jj_lb_rate = jj_lb_ok / jj_lb_total if jj_lb_total else 0.0

    la_abs = [m.la_abs_err for m in block_metrics]
    jj_abs = [m.jj_abs_err for m in block_metrics]
    ep_abs = [m.ep_abs_err for m in block_metrics]

    n_nan_la = sum(1 for m in block_metrics if not math.isfinite(m.la_lml))
    n_nan_jj = sum(1 for m in block_metrics if not math.isfinite(m.jj_lml))
    n_nan_ep = sum(1 for m in block_metrics if not math.isfinite(m.ep_lml))

    methods_non_ref = ["laplace", "jj", "ep"]

    # Segmentation summaries
    seg_by_method: dict[str, list[SegResult]] = {}
    for r in seg_results:
        seg_by_method.setdefault(r.method, []).append(r)
    ref_by_rep: dict[int, SegResult] = {
        r.replicate: r for r in seg_results if r.method == "quadrature"
    }

    seg_summary: dict[str, dict[str, float]] = {}
    for method in ["laplace", "jj", "ep"]:
        method_results = seg_by_method.get(method, [])
        pk_diffs: list[float] = []
        bm_diffs: list[float] = []
        map_mismatches = 0
        f1_vals: list[float] = []
        for r in method_results:
            ref_r = ref_by_rep.get(r.replicate)
            if ref_r is None:
                continue
            pk_a = np.array(r.pk)
            pk_r = np.array(ref_r.pk)
            n_k = min(len(pk_a), len(pk_r))
            if n_k > 0:
                pk_diffs.append(float(np.max(np.abs(pk_a[:n_k] - pk_r[:n_k]))))
            bm_a = np.array(r.bm_at_true)
            bm_r = np.array(ref_r.bm_at_true)
            if len(bm_a) > 0:
                bm_diffs.append(float(np.max(np.abs(bm_a - bm_r))))
            if sorted(r.map_boundaries) != sorted(ref_r.map_boundaries):
                map_mismatches += 1
            f1_vals.append(r.f1_at_3)
        seg_summary[method] = {
            "mean_max_pk_diff": float(np.mean(pk_diffs)) if pk_diffs else float("nan"),
            "max_pk_diff": float(max(pk_diffs)) if pk_diffs else float("nan"),
            "mean_max_bm_diff": float(np.mean(bm_diffs)) if bm_diffs else float("nan"),
            "max_bm_diff": float(max(bm_diffs)) if bm_diffs else float("nan"),
            "map_mismatches": map_mismatches,
            "n_seqs": len(method_results),
            "mean_f1_at_3": float(np.mean(f1_vals)) if f1_vals else float("nan"),
        }

    summary = {
        "experiment_id": "E06",
        "n_blocks_total": jj_lb_total,
        "jj_lb_ok_count": jj_lb_ok,
        "jj_lb_rate": jj_lb_rate,
        "gate_jj_lb_rate": cfg["gate_jj_lb_rate"],
        "gate_jj_lb_pass": jj_lb_rate >= cfg["gate_jj_lb_rate"],
        "n_nan_laplace": n_nan_la,
        "n_nan_jj": n_nan_jj,
        "n_nan_ep": n_nan_ep,
        "gate_no_nan_pass": n_nan_la == 0 and n_nan_jj == 0,
        "block_metrics": {
            "laplace": {
                "median_abs_err": float(np.median(la_abs)),
                "max_abs_err": float(max(la_abs)) if la_abs else float("nan"),
                "median_signed_err": float(np.median([m.la_signed_err for m in block_metrics])),
                "median_runtime_s": float(np.median([m.la_runtime for m in block_metrics])),
            },
            "jj": {
                "median_abs_err": float(np.median(jj_abs)),
                "max_abs_err": float(max(jj_abs)) if jj_abs else float("nan"),
                "median_signed_err": float(np.median([m.jj_signed_err for m in block_metrics])),
                "median_runtime_s": float(np.median([m.jj_runtime for m in block_metrics])),
            },
            "ep": {
                "median_abs_err": float(np.median(ep_abs)),
                "max_abs_err": float(max(ep_abs)) if ep_abs else float("nan"),
                "median_signed_err": float(np.median([m.ep_signed_err for m in block_metrics])),
                "median_runtime_s": float(np.median([m.ep_runtime for m in block_metrics])),
                "n_above_reference": sum(1 for m in block_metrics if m.ep_signed_err > 1e-6),
                "n_convergence_ok": sum(1 for m in block_metrics if m.ep_convergence),
            },
        },
        "segmentation_propagation": seg_summary,
        "note_pg_vb": (
            "pg_vb aliases jj in the estimator code path; it is NOT benchmarked "
            "as an independent method. No distinct Polya-Gamma VB is implemented."
        ),
        "utc_complete": datetime.now(timezone.utc).isoformat(),
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary metrics written")
    return summary


def _write_table(output_dir: Path, block_metrics: list[BlockMetric]) -> None:
    """Write LaTeX table tab_E06_approx.tex."""
    from collections import defaultdict

    methods_info = [
        ("laplace", "Laplace",
         lambda m: m.la_abs_err, lambda m: m.la_runtime, lambda m: True),
        ("jj", "JJ (= pg\\_vb alias)",
         lambda m: m.jj_abs_err, lambda m: m.jj_runtime, lambda m: m.jj_lb_ok),
        ("ep", "EP",
         lambda m: m.ep_abs_err, lambda m: m.ep_runtime, lambda m: True),
    ]

    # Compute per-method stats
    rows = []
    for key, name, err_fn, rt_fn, lb_fn in methods_info:
        abs_errs = [err_fn(m) for m in block_metrics if math.isfinite(err_fn(m))]
        rts = [rt_fn(m) for m in block_metrics]
        lb_viol = sum(1 for m in block_metrics if not lb_fn(m)) if key == "jj" else 0
        n_fail = sum(
            1 for m in block_metrics
            if key == "ep" and not m.ep_convergence
        )
        med_err = np.median(abs_errs) if abs_errs else float("nan")
        max_err = max(abs_errs) if abs_errs else float("nan")
        med_rt = np.median(rts) * 1000  # ms

        jj_viol_str = f"{lb_viol}/{len(block_metrics)}" if key == "jj" else "N/A"
        fail_str = f"{n_fail}/{len(block_metrics)}" if key == "ep" else "---"

        rows.append(
            f"  {name} & {med_err:.4f} & {max_err:.4f} & "
            f"{jj_viol_str} & {fail_str} & {med_rt:.3f}ms \\\\"
        )

    tex = r"""\begin{table}[ht]
\centering
\caption{E06: nonconjugate block-evidence approximations. Block sizes $B\in\{4,8,16,32,64,128\}$,
$p\in\{0.05,0.5,0.95\}$, 100 independent replicates per cell (1800 blocks total).
Prior $\theta\sim\mathcal{N}(0,4)$. Reference: mode-centred adaptive quadrature (\texttt{epsabs}$=10^{-12}$).
\texttt{pg\_vb} is an alias of JJ and is not listed separately.
JJ LB rate: fraction of blocks where JJ $\le$ reference $+$ \num{1e-8}.
EP is \emph{not} a guaranteed lower bound.}
\label{tab:planned-nonconjugate-paper}
\begin{tabular}{lrrrrr}
\toprule
Method & Median $|\Delta\log Z|$ & Max $|\Delta\log Z|$ & JJ LB violations & EP failures & Median runtime \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    with open(output_dir / "tables" / "tab_E06_approx.tex", "w") as f:
        f.write(tex)
    logger.info("Table written")


def _write_failed_runs(output_dir: Path, block_metrics: list[BlockMetric]) -> None:
    failed = [m for m in block_metrics if not math.isfinite(m.ref_lml)]
    if not failed:
        logger.info("No failed runs (reference NaN/inf)")
        return
    with open(output_dir / "failed_runs.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["B", "p", "replicate", "S", "ref_lml"])
        writer.writeheader()
        for m in failed:
            writer.writerow({"B": m.B, "p": m.p, "replicate": m.replicate, "S": m.S, "ref_lml": m.ref_lml})
    logger.warning("%d failed runs written to failed_runs.csv", len(failed))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _write_figures(
    output_dir: Path,
    block_metrics: list[BlockMetric],
    seg_results: list[SegResult],
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from _style import setup_style, METHOD_COLORS, METHOD_MARKERS, METHOD_LINESTYLES
        setup_style()
    except ImportError:
        logger.warning("matplotlib not available; figures skipped")
        return

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: absolute log-evidence error by B and p, for each method.
    B_values = sorted(set(m.B for m in block_metrics))
    p_values = sorted(set(m.p for m in block_metrics))
    methods_plot = [
        ("laplace", "Laplace", lambda m: m.la_abs_err),
        ("jj", "JJ", lambda m: m.jj_abs_err),
        ("ep", "EP", lambda m: m.ep_abs_err),
    ]

    fig, axes = plt.subplots(1, len(p_values), figsize=(10, 3.5), sharey=False)
    for ax_i, p in enumerate(p_values):
        ax = axes[ax_i]
        for key, label, err_fn in methods_plot:
            meds = []
            for B in B_values:
                errs = [err_fn(m) for m in block_metrics if m.B == B and abs(m.p - p) < 0.01]
                meds.append(np.median(errs) if errs else float("nan"))
            ax.semilogy(
                B_values, meds,
                marker=METHOD_MARKERS.get(key, "o"),
                linestyle=METHOD_LINESTYLES.get(key, "-"),
                color=METHOD_COLORS.get(key, "#444444"),
                label=label,
            )
        ax.set_xlabel("Block size $B$")
        ax.annotate(f"$p = {p}$", xy=(0.97, 0.96), xycoords='axes fraction', fontsize=8, va='top', ha='right')
        ax.legend(fontsize=8)
        if ax_i == 0:
            ax.set_ylabel("Median absolute log-evidence error")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_E06_local_error.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "fig_E06_local_error.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Figure fig_E06_local_error written")

    # Figure 2: p(k|y) and boundary-marginal discrepancies by method.
    ref_by_rep = {r.replicate: r for r in seg_results if r.method == "quadrature"}
    seg_methods = ["laplace", "jj", "ep"]
    seg_labels = ["Laplace", "JJ", "EP"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for method, label in zip(seg_methods, seg_labels):
        pk_diffs: list[float] = []
        bm_diffs: list[float] = []
        for r in seg_results:
            if r.method != method:
                continue
            ref = ref_by_rep.get(r.replicate)
            if ref is None:
                continue
            pk_a = np.array(r.pk)
            pk_r = np.array(ref.pk)
            n_k = min(len(pk_a), len(pk_r))
            if n_k > 0:
                pk_diffs.append(float(np.max(np.abs(pk_a[:n_k] - pk_r[:n_k]))))
            bm_a = np.array(r.bm_at_true)
            bm_r = np.array(ref.bm_at_true)
            if len(bm_a) > 0 and np.all(np.isfinite(bm_a)) and np.all(np.isfinite(bm_r)):
                bm_diffs.append(float(np.max(np.abs(bm_a - bm_r))))

        ax1.violinplot([pk_diffs], positions=[seg_labels.index(label)],
                       showmedians=True, widths=0.6)
        ax2.violinplot([bm_diffs], positions=[seg_labels.index(label)],
                       showmedians=True, widths=0.6)

    ax1.set_xticks(range(len(seg_labels)))
    ax1.set_xticklabels(seg_labels)
    ax1.set_ylabel("Max |p(k|y) difference| vs quadrature reference")

    ax2.set_xticks(range(len(seg_labels)))
    ax2.set_xticklabels(seg_labels)
    ax2.set_ylabel("Max |boundary marginal difference| at true boundaries vs reference")

    fig.tight_layout()
    fig.savefig(fig_dir / "fig_E06_global_error.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "fig_E06_global_error.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Figure fig_E06_global_error written")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = E06_CONFIG
    cfg_sha = _config_sha256()
    output_dir = Path("results/confirmatory/E06_nonconjugate") / cfg_sha[:16]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    (output_dir / "tables").mkdir(exist_ok=True)

    log_path = output_dir / "run_log.txt"
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logging.getLogger().addHandler(file_handler)

    logger.info("=" * 70)
    logger.info("E06: nonconjugate block-evidence approximations")
    logger.info("Config SHA256: %s", cfg_sha)
    logger.info("Output directory: %s", output_dir)
    logger.info("=" * 70)

    _write_manifest(output_dir, cfg, cfg_sha)

    # ---------- Block oracle experiment ----------
    B_values = cfg["block_sizes"]
    p_values = cfg["true_success_probs"]
    n_reps = cfg["n_replicates_block"]

    block_metrics: list[BlockMetric] = []
    cell_index = 0
    for B in B_values:
        for p in p_values:
            logger.info("Block oracle: B=%d, p=%.2f (%d replicates)", B, p, n_reps)
            cell_start = time.time()
            for rep in range(n_reps):
                try:
                    m = compute_block_metrics(B, p, cell_index, rep, cfg)
                    block_metrics.append(m)
                except Exception as exc:
                    logger.error("Block B=%d p=%.2f rep=%d FAILED: %s", B, p, rep, exc)
            cell_elapsed = time.time() - cell_start
            jj_viol = sum(1 for m in block_metrics[-n_reps:] if not m.jj_lb_ok)
            la_med = np.median([m.la_abs_err for m in block_metrics[-n_reps:]])
            jj_med = np.median([m.jj_abs_err for m in block_metrics[-n_reps:]])
            ep_med = np.median([m.ep_abs_err for m in block_metrics[-n_reps:]])
            logger.info(
                "  Done in %.1fs | la_med=%.4f jj_med=%.4f ep_med=%.4f jj_lb_viol=%d/%d",
                cell_elapsed, la_med, jj_med, ep_med, jj_viol, n_reps,
            )
            cell_index += 1

    _write_metrics(output_dir, block_metrics)
    _write_oracle_checks(output_dir, block_metrics)
    _write_failed_runs(output_dir, block_metrics)

    # ---------- Segmentation propagation ----------
    logger.info("Running segmentation propagation experiment (%d sequences)...", cfg["seg_n_replicates"])
    seg_results = run_segmentation_propagation(cfg, output_dir)

    # Save segmentation results
    seg_rows = []
    for r in seg_results:
        seg_rows.append({
            "replicate": r.replicate,
            "method": r.method,
            "map_boundaries": json.dumps(r.map_boundaries),
            "f1_at_3": r.f1_at_3,
            "pk_k1": r.pk[0] if len(r.pk) > 0 else float("nan"),
            "pk_k2": r.pk[1] if len(r.pk) > 1 else float("nan"),
            "pk_k3": r.pk[2] if len(r.pk) > 2 else float("nan"),
        })
    with open(output_dir / "seg_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=seg_rows[0].keys())
        writer.writeheader()
        writer.writerows(seg_rows)

    # ---------- Summary, table, figures ----------
    summary = _write_summary_metrics(output_dir, block_metrics, seg_results, cfg)
    _write_table(output_dir, block_metrics)
    _write_figures(output_dir, block_metrics, seg_results)

    # ---------- Gate evaluation ----------
    logger.info("")
    logger.info("=" * 70)
    logger.info("E06 GATE EVALUATION")
    logger.info("=" * 70)
    gate_pass = True

    # Gate 1: all block-oracle cells complete
    n_complete = len(block_metrics)
    n_expected = len(B_values) * len(p_values) * n_reps
    g1 = n_complete == n_expected
    logger.info("Gate 1: All cells complete (%d/%d): %s", n_complete, n_expected, "PASS" if g1 else "FAIL")
    gate_pass &= g1

    # Gate 2: JJ lower-bound
    jj_lb_rate = summary["jj_lb_rate"]
    g2 = jj_lb_rate >= cfg["gate_jj_lb_rate"]
    logger.info(
        "Gate 2: JJ lower-bound >=%.0f%% blocks (%.1f%%): %s",
        100 * cfg["gate_jj_lb_rate"], 100 * jj_lb_rate, "PASS" if g2 else "FAIL",
    )
    gate_pass &= g2

    # Gate 3: no NaN on non-degenerate blocks (Laplace, JJ)
    g3 = summary["gate_no_nan_pass"]
    logger.info("Gate 3: No NaN in Laplace/JJ: %s", "PASS" if g3 else "FAIL")
    gate_pass &= g3

    # Gate 4: degenerate blocks tested
    g4 = (output_dir / "oracle_checks.json").exists()
    logger.info("Gate 4: Degenerate block checks exist: %s", "PASS" if g4 else "FAIL")
    gate_pass &= g4

    logger.info("")
    if gate_pass:
        logger.info("E06 PASS")
    else:
        logger.warning("E06 FAIL")

    # Update manifest with completion info
    with open(output_dir / "manifest.json") as f:
        manifest = json.load(f)
    manifest["utc_complete"] = datetime.now(timezone.utc).isoformat()
    manifest["gate_pass"] = gate_pass
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Output directory: %s", output_dir)


if __name__ == "__main__":
    main()
