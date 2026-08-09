"""CpG methylation loader with **per-CpG precision** ``φ_t``.

Real source: the ``test1.myCpG.txt`` example file shipped with the
``methylKit`` R/Bioconductor package
(``https://github.com/al2na/methylKit/raw/master/inst/extdata/test1.myCpG.txt``).
It is a tab-separated table of ~1900 CpG calls on chr21 with columns
``chrBase, chr, base, strand, coverage, freqC, freqT``. We treat
``freqC / 100`` as the methylation fraction ``y_t ∈ (0, 1)`` and the
``coverage`` column as the per-CpG precision ``φ_t`` (consistent with the
report's §``sec:realdata-methylation`` Beta-response specification).

The manuscript's appendix recipe (§``app:real-data-methylation``) points to
the Loyfer et al. 2023 atlas distributed via the GitHub repository
``nloyfer/meth_atlas`` and via NCBI GEO accession ``GSE186458``; the
present Python loader is a lightweight fallback for that pipeline.

When the network is unavailable, or the user provides a local CSV with the
same column schema, the loader gracefully falls back to the deterministic
simulated analog or the user-provided file.

Caveat (for future maintainers, verified May 2026): the manuscript
appendix recipe points to ``nloyfer/meth_atlas`` as the companion code
for Loyfer et al. 2023. This is **factually incorrect**:
``nloyfer/meth_atlas`` is the code base for the older Moss et al. 2018
array-deconvolution method (Nature Communications 9:5068), **not** the
2023 atlas. The verified companion software for ``loyfer2023atlas`` is
split between ``nloyfer/wgbs_tools`` (the ``wgbstools`` suite used to
build the atlas) and ``nloyfer/UXM_deconv`` (the UXM fragment-level
deconvolution tool). The Nature 2023 paper's data-availability statement
itself names these repositories. The verified GEO accession remains
**GSE186458** (data in bigWig and beta format compatible with
wgbstools). A finalized atlas pipeline that fails to locate the per-CpG
``coverage.tsv.gz`` matrix at the ``meth_atlas`` URL should switch to
``wgbs_tools`` / ``UXM_deconv`` under the same GitHub owner.
"""

from __future__ import annotations

import io
import pathlib

import numpy as np

from . import DatasetBundle
from ._cache import banner, cache_dir, describe_fallback
from ._simulate import simulate_methylation

_METHYLKIT_URL = "https://github.com/al2na/methylKit/raw/master/inst/extdata/test1.myCpG.txt"


def _parse_methylkit_table(text: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse a methylKit-style table; return (positions, freqC/100, coverage).

    The file is tab-separated with a header line ``chrBase chr base strand
    coverage freqC freqT``.
    """

    rows: list[tuple[int, float, float]] = []
    header_seen = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split("\t")
        if not header_seen:
            header_seen = True
            continue
        if len(parts) < 7:
            continue
        try:
            base = int(parts[2])
            coverage = float(parts[4])
            freqC = float(parts[5])
        except ValueError:
            continue
        if not np.isfinite(coverage) or coverage <= 0:
            continue
        if not np.isfinite(freqC):
            continue
        rows.append((base, freqC / 100.0, coverage))
    if not rows:
        raise ValueError("No parseable rows found in methylKit table.")
    rows.sort(key=lambda r: r[0])
    pos = np.array([r[0] for r in rows], dtype=float)
    y = np.array([r[1] for r in rows], dtype=float)
    cov = np.array([r[2] for r in rows], dtype=float)
    # Clamp y away from {0, 1} for the Beta likelihood.
    eps = 1e-3
    y = np.clip(y, eps, 1.0 - eps)
    return pos, y, cov


def _load_methylkit_real() -> DatasetBundle | None:
    try:
        import requests  # type: ignore[import-untyped]
    except ImportError:
        return None
    cache = cache_dir() / "methylkit_test1.myCpG.txt"
    try:
        if cache.exists():
            text = cache.read_text(encoding="utf-8")
        else:
            r = requests.get(_METHYLKIT_URL, timeout=30)
            if r.status_code != 200 or len(r.content) < 1000:
                return None
            text = r.text
            cache.write_text(text, encoding="utf-8")
        positions, y, cov = _parse_methylkit_table(text)
    except Exception:  # pragma: no cover - network / parse edge cases
        return None
    if y.size < 50:
        return None
    banner(f"methylation: loaded n_CpGs={y.size} (chr21, methylKit test1.myCpG).")
    return DatasetBundle(
        X=positions.reshape(-1, 1),
        y=y,
        sample_weight=cov,  # per-CpG read coverage ⇒ φ_t
        true_boundaries=[],
        name="methylation",
        source="downloaded",
        description=(
            f"methylKit chr21 CpG methylation fractions (n={y.size}); "
            "sample_weight is per-CpG read coverage."
        ),
        metadata={"url": _METHYLKIT_URL},
    )


def _load_methylkit_csv(path: pathlib.Path) -> DatasetBundle | None:
    """Try to parse a user-supplied CSV.

    Two schemas are accepted:
    - methylKit-style: ``chrBase, chr, base, strand, coverage, freqC, freqT``.
    - Single-column: one methylation fraction per line, ``y_t ∈ (0, 1)``.
    """

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        describe_fallback("methylation", f"read failed: {exc}")
        return None

    # Detect schema by sniffing the header.
    head = text.splitlines()[:3]
    if head and "coverage" in head[0].lower() and "freqc" in head[0].lower():
        try:
            positions, y, cov = _parse_methylkit_table(text)
        except ValueError:
            describe_fallback("methylation", "methylKit-style parse failed")
            return None
        banner(f"methylation: loaded {y.size} CpGs from {path}")
        return DatasetBundle(
            X=positions.reshape(-1, 1),
            y=y,
            sample_weight=cov,
            true_boundaries=[],
            name="methylation",
            source="downloaded",
            description=f"User-provided methylKit-style table (n={y.size}) from {path}.",
            metadata={"csv_path": str(path)},
        )

    # Single-column fallback.
    try:
        raw = np.loadtxt(io.StringIO(text), delimiter=",", ndmin=1)
    except Exception as exc:
        describe_fallback("methylation", f"parse failed: {exc}")
        return None
    y = np.asarray(raw, dtype=float).ravel()
    y = y[np.isfinite(y) & (y > 0.0) & (y < 1.0)]
    if y.size < 50:
        describe_fallback("methylation", f"only {y.size} valid rows after filtering")
        return None
    banner(f"methylation: loaded {y.size} fractions from {path}")
    return DatasetBundle(
        X=np.arange(y.size, dtype=float).reshape(-1, 1),
        y=y,
        sample_weight=None,
        true_boundaries=[],
        name="methylation",
        source="downloaded",
        description=f"User-provided methylation fractions (n={y.size}) from {path}.",
        metadata={"csv_path": str(path)},
    )


def load_methylation(
    *,
    simulated: bool = False,
    csv_path: str | pathlib.Path | None = None,
) -> DatasetBundle:
    """Load a CpG methylation sequence ``y_i ∈ (0, 1)`` with per-CpG coverage.

    Parameters
    ----------
    simulated : bool, default False
        Force the deterministic simulated analog and ignore ``csv_path``.
    csv_path : str or Path or None
        Optional path to a methylKit-style table or a single-column CSV of
        methylation fractions.

    Returns
    -------
    DatasetBundle
        ``y`` is the methylation fraction in ``(0, 1)``;
        ``sample_weight`` is the per-CpG read coverage when available
        (used as ``φ_t`` in :class:`~bayesbreak.BayesBreakBetaObs`).
    """

    if simulated:
        return DatasetBundle.from_dict(simulate_methylation())

    if csv_path is not None:
        path = pathlib.Path(csv_path).expanduser()
        if not path.exists():
            describe_fallback("methylation", f"{path} not found")
        else:
            bundle = _load_methylkit_csv(path)
            if bundle is not None:
                return bundle

    bundle = _load_methylkit_real()
    if bundle is not None:
        return bundle

    describe_fallback("methylation", "real download unavailable")
    return DatasetBundle.from_dict(simulate_methylation())
