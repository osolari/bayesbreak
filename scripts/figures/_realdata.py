"""Shared four-panel template for the real-data figures (fig6-fig9).

Layout:

A. raw y + MAP piecewise-constant fit (+ Bayes curve when enabled); MAP
   boundaries overlaid as vertical red dashes.
B. P(b_i = 1 | y) boundary marginals on the same index axis as A.
C. P(k | y) posterior over segment count (independent x-axis in k-space).
D. cumulative held-out posterior-predictive log-density vs a k=1 null
   (independent x-axis in index-space).

A and B share one x-axis (index / design space). C and D have independent
x-axes so C's k-range doesn't compress A and B.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from _style import COLORS, add_panel_label, save_figure, setup_style  # noqa: E402

from bayesbreak import BayesBreakSegmenter, run_dp_diagnostics
from bayesbreak.datasets import DatasetBundle
from bayesbreak.experiments._fitcache import cache_key, fit_or_load
from bayesbreak.experiments._placeholder import hash_array, write_run_record
from bayesbreak.prediction import posterior_predictive_logpdf


def _held_out_trace(estimator: BayesBreakSegmenter, bundle: DatasetBundle) -> np.ndarray:
    per = posterior_predictive_logpdf(estimator, bundle.X, bundle.y, per_sample=True)
    assert isinstance(per, np.ndarray)
    return np.cumsum(per)


def _null_trace(estimator: BayesBreakSegmenter, bundle: DatasetBundle) -> np.ndarray:
    """Cumulative posterior-predictive under a forced one-segment null model."""

    null = estimator.__class__(**{k: v for k, v in estimator.get_params().items() if k != "k_max"})
    null.set_params(k_max=1)
    null.fit(bundle.X, bundle.y, sample_weight=bundle.sample_weight)
    per = posterior_predictive_logpdf(null, bundle.X, bundle.y, per_sample=True)
    assert isinstance(per, np.ndarray)
    return np.cumsum(per)


def planned_caption(text: str, *, planned: bool) -> str:
    """Prepend ``(PLANNED) `` to a caption/title when status is not finalized.

    Mirrors the manuscript's status convention (§6 IDs 6-E1/6-E2; new-zip
    CHANGELOG entry CG-E1/E2): every real-data figure/table caption begins
    with ``(PLANNED)`` until the corresponding finalized pipeline (verified
    source, recorded hashes/seeds, observed numeric values) replaces the
    placeholder. ``planned=False`` is reserved for the rare case where the
    figure was regenerated from a fully verified pipeline.
    """

    prefix = "(PLANNED) "
    if not planned:
        return text
    if text.startswith(prefix):
        return text
    return prefix + text


def make_realdata_figure(
    *,
    estimator: BayesBreakSegmenter,
    bundle: DatasetBundle,
    outdir: Path,
    fig_name: str,
    y_label: str,
    title: str,
    show_null_baseline: bool = True,
    extra_kwargs: dict[str, Any] | None = None,
    planned: bool | None = None,
) -> None:
    """Fit ``estimator`` on ``bundle`` and render the four-panel figure.

    When ``planned`` is ``None`` the status is inferred from
    ``bundle.source``: simulated/fallback bundles render with a
    ``(PLANNED)`` prefix on the title; bundles loaded from a verified
    download render with no prefix. Pass ``planned=True`` or
    ``planned=False`` to force the status. Figures populated from real
    sources are reported as observed results and must not carry the
    ``(PLANNED)`` tag.
    """

    extra_kwargs = extra_kwargs or {}
    setup_style(font_scale=0.95)
    if planned is None:
        planned = bundle.source != "downloaded"
    title = planned_caption(title, planned=planned)

    # Fit cache: skip refit when the data + estimator params are unchanged.
    key = cache_key(
        y=bundle.y,
        sample_weight=bundle.sample_weight,
        params=estimator.get_params(),
        extra=fig_name,
    )

    def _fit() -> BayesBreakSegmenter:
        estimator.fit(bundle.X, bundle.y, sample_weight=bundle.sample_weight, **extra_kwargs)
        return estimator

    estimator = fit_or_load(outdir / f"{fig_name}.fit.pkl", key, _fit)

    n = bundle.y.size
    # Plot everything in index space so panels A and B share a clean axis.
    idx = np.arange(n, dtype=float)

    fig = plt.figure(figsize=(7.8, 8.4))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.5, 1.0, 1.0, 1.1], hspace=0.32)
    axA = fig.add_subplot(gs[0])
    axB = fig.add_subplot(gs[1], sharex=axA)  # A and B only
    axC = fig.add_subplot(gs[2])
    axD = fig.add_subplot(gs[3])

    # ---- Panel A: raw y + MAP fit --------------------------------------------
    axA.scatter(
        idx,
        bundle.y,
        s=4,
        alpha=0.35,
        color=COLORS["grey"],
        edgecolors="none",
        label="observed",
        zorder=1,
    )
    axA.plot(idx, estimator.predict(bundle.X), lw=1.6, color=COLORS["blue"], label="MAP", zorder=3)
    if estimator.bayes_curve_mean_ is not None:
        axA.plot(
            idx,
            estimator.bayes_curve_mean_,
            lw=1.2,
            color=COLORS["red"],
            alpha=0.9,
            label="Bayes curve",
            zorder=2,
        )
    # MAP-boundary verticals (skip endpoints 0 and n).
    for b in estimator.map_boundaries_[1:-1]:
        axA.axvline(float(b), color=COLORS["red"], ls="--", lw=0.6, alpha=0.7, zorder=4)
    # Ground-truth boundaries (only when known, e.g. simulated bundles).
    for b in bundle.true_boundaries[1:-1]:
        axA.axvline(float(b), color=COLORS["green"], ls=":", lw=0.9, alpha=0.7, zorder=4)
    axA.set_ylabel(y_label)
    axA.set_xlim(0, n)
    # Tighten y-axis to the data range with a 5% pad.
    y_min, y_max = float(np.nanmin(bundle.y)), float(np.nanmax(bundle.y))
    pad = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
    axA.set_ylim(y_min - pad, y_max + pad)
    axA.legend(loc="best", ncol=3, fontsize=10, frameon=False)
    add_panel_label(axA, "A", title)
    plt.setp(axA.get_xticklabels(), visible=False)

    # ---- Panel B: boundary marginals -----------------------------------------
    if estimator.boundary_marginals_ is not None:
        xb = idx[1:]
        axB.fill_between(
            xb, 0, estimator.boundary_marginals_, color=COLORS["blue"], alpha=0.35, linewidth=0
        )
        axB.plot(xb, estimator.boundary_marginals_, color=COLORS["blue"], lw=1.0)
    for b in estimator.map_boundaries_[1:-1]:
        axB.axvline(float(b), color=COLORS["red"], ls="--", lw=0.6, alpha=0.8)
    for b in bundle.true_boundaries[1:-1]:
        axB.axvline(float(b), color=COLORS["green"], ls=":", lw=0.9, alpha=0.6)
    axB.set_ylim(0, 1.05)
    axB.set_xlim(0, n)
    axB.set_xlabel("index")
    axB.set_ylabel(r"$P(b_i = 1 \mid y)$")
    add_panel_label(axB, "B")

    # ---- Panel C: posterior over k (independent x-axis) ----------------------
    k_vals = np.arange(1, estimator.k_posterior_.size + 1)
    axC.bar(k_vals, estimator.k_posterior_, color=COLORS["blue"], edgecolor="none", width=0.75)
    axC.axvline(
        estimator.k_map_,
        color=COLORS["red"],
        ls="--",
        lw=1.0,
        label=rf"$k_{{MAP}} = {estimator.k_map_}$",
    )
    axC.set_xlim(0.3, estimator.k_posterior_.size + 0.7)
    axC.set_xlabel("k")
    axC.set_ylabel(r"$P(k \mid y)$")
    axC.legend(loc="best", fontsize=10, frameon=False)
    add_panel_label(axC, "C")

    # ---- Panel D: cumulative held-out log-likelihood -------------------------
    trace = _held_out_trace(estimator, bundle)
    axD.plot(idx, trace, lw=1.4, color=COLORS["blue"], label="BayesBreak MAP")
    if show_null_baseline:
        try:
            null_tr = _null_trace(estimator, bundle)
            axD.plot(idx, null_tr, lw=1.0, color=COLORS["grey"], ls=":", label="k=1 null")
        except Exception:
            pass
    axD.set_xlim(0, n)
    axD.set_xlabel("index")
    axD.set_ylabel(r"$\sum_{t' \leq t}\log p(y_{t'}\mid \mathcal{M})$")
    axD.legend(loc="best", fontsize=10, frameon=False)
    add_panel_label(axD, "D")

    save_figure(fig, outdir / fig_name, formats=("png", "pdf"))

    # Panel-level data sidecar: numerical inputs for each panel so figure
    # style can be changed without refitting the model.
    _panel_data: dict[str, Any] = {
        "source": bundle.source,
        "n": int(n),
        "y": bundle.y.tolist(),
        "map_prediction": estimator.predict(bundle.X).tolist(),
        "map_boundaries": list(estimator.map_boundaries_),
        "true_boundaries": list(bundle.true_boundaries),
        "boundary_marginals": (
            estimator.boundary_marginals_.tolist()
            if estimator.boundary_marginals_ is not None else None
        ),
        "bayes_curve_mean": (
            estimator.bayes_curve_mean_.tolist()
            if estimator.bayes_curve_mean_ is not None else None
        ),
        "k_posterior": estimator.k_posterior_.tolist(),
        "k_map": int(estimator.k_map_),
        "held_out_trace": trace.tolist(),
    }
    data_dest = outdir / f"{fig_name}_panel_data.json"
    with open(data_dest, "w") as _f:
        import json as _json
        _json.dump(_panel_data, _f)

    # Write canonical CSV + JSON to shared data dir so render.py can work from cache.
    import csv as _csv
    _canonical = Path(__file__).resolve().parents[2] / "docs" / "manuscripts" / "shared" / "figures" / "data"
    _canonical.mkdir(parents=True, exist_ok=True)
    with open(_canonical / f"{fig_name}_panel_data.json", "w") as _fj:
        _json.dump(_panel_data, _fj)
    _n_pd = _panel_data["n"]
    _bm_list = _panel_data.get("boundary_marginals") or []
    _bc_list = _panel_data.get("bayes_curve_mean") or []
    _ht_list = _panel_data.get("held_out_trace") or []
    _mb_set = set(list(_panel_data.get("map_boundaries", []))[1:-1])
    _tb_set = set(list(_panel_data.get("true_boundaries", []))[1:-1])
    _ts_rows = []
    for _x in range(_n_pd):
        _ts_rows.append({
            "x": _x,
            "y": _panel_data["y"][_x] if _x < len(_panel_data["y"]) else float("nan"),
            "map_prediction": _panel_data["map_prediction"][_x] if _x < len(_panel_data["map_prediction"]) else float("nan"),
            "boundary_marginal": _bm_list[_x - 1] if _x > 0 and _bm_list else float("nan"),
            "bayes_curve_mean": _bc_list[_x] if _bc_list and _x < len(_bc_list) else float("nan"),
            "held_out_trace": _ht_list[_x] if _ht_list and _x < len(_ht_list) else float("nan"),
            "null_trace": float("nan"),
            "is_map_boundary": 1 if _x in _mb_set else 0,
            "is_true_boundary": 1 if _x in _tb_set else 0,
        })
    _csv_stem = fig_name.replace("_panel_data", "")
    _csv_name = f"{_csv_stem}.csv"
    with open(_canonical / _csv_name, "w", newline="") as _cf:
        _w = _csv.DictWriter(_cf, fieldnames=list(_ts_rows[0].keys()))
        _w.writeheader()
        _w.writerows(_ts_rows)
    _kp = _panel_data.get("k_posterior", [])
    if _kp:
        with open(_canonical / f"{_csv_stem}_kposterior.csv", "w", newline="") as _kf:
            _kw = _csv.DictWriter(_kf, fieldnames=["k", "prob"])
            _kw.writeheader()
            _kw.writerows([{"k": _k + 1, "prob": _p} for _k, _p in enumerate(_kp)])

    # Sidecar JSON: provenance + DP diagnostics.
    diag = run_dp_diagnostics(estimator)
    record_extra: dict[str, Any] = {
        "source": bundle.source,
        "status": "PLANNED" if planned else "FINAL",
        "description": bundle.description,
        "y_hash": hash_array(bundle.y),
        "sample_weight_hash": hash_array(bundle.sample_weight),
        "n": int(bundle.y.shape[0]),
        "k_map": int(estimator.k_map_),
        "log_evidence": float(estimator.log_evidence_),
        "map_boundaries": list(estimator.map_boundaries_),
        "dp_diagnostics": diag.to_dict(),
        "estimator": type(estimator).__name__,
        "estimator_params": {
            k: (v.tolist() if isinstance(v, np.ndarray) else (repr(v) if callable(v) else v))
            for k, v in estimator.get_params().items()
        },
    }
    write_run_record(
        outdir / f"{fig_name}.pdf",
        dataset=bundle.name,
        source=bundle.source,
        extra=record_extra,
    )
