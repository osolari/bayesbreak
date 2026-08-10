"""Well-log (NMR tool-response) loader.

Real source: the TCPD mirror of the classic Ó Ruanaidh well-log signal,
the well-log NMR dataset of Fearnhead (2006) — see
``\\citet{fearnhead2006exact}`` in the report's §``app:real-data-welllog``.
A copy ships with the R package ``changepoint`` (Killick & Eckley); the
manuscript's R recipe uses ``data(Lai2005fig4)`` (or ``data(Wellog)`` /
``data(wavenumber)`` as fallbacks).

We prefer the ``well_log.txt`` file (full n=4050 series) and fall back to
the ``well_log.json`` subset if that is unavailable. When neither can be
fetched we return the deterministic simulated analog.

Caveat (for future maintainers, verified May 2026): the manuscript
appendix recipe uses ``data(Lai2005fig4)`` from the ``changepoint``
package, but this is **factually incorrect** — ``Lai2005fig4`` is the
array-CGH example of Lai et al. (2005), not the well-log NMR series.
The verified 4050-vector NMR series of Fearnhead & Clifford (2003)
ships as the ``welldata`` object in the R package
``changepoint.influence`` (CRAN). The CRAN page itself documents it as
"a vector of length 4050. The data described and provided in
Fearnhead and Clifford (2003)." This caveat is reproduced in the
canonical coding-handoff author-verification list; a finalized
real-data run that hits the ``Lai2005fig4`` label-mismatch should
switch to ``changepoint.influence::welldata``.
"""

from __future__ import annotations

import json

import numpy as np

from . import DatasetBundle
from ._cache import describe_fallback, try_fetch
from ._simulate import simulate_welllog

_WELLLOG_TXT = (
    "https://raw.githubusercontent.com/alan-turing-institute/TCPD/master/"
    "datasets/well_log/well_log.txt"
)
_WELLLOG_JSON = (
    "https://raw.githubusercontent.com/alan-turing-institute/TCPD/master/"
    "datasets/well_log/well_log.json"
)


def _parse_txt(path) -> np.ndarray:
    # TCPD .txt format: one numeric value per line.
    y = np.loadtxt(path, dtype=float)
    return y[np.isfinite(y)]


def _parse_json(path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    raw = payload.get("series") or payload.get("data") or []
    values: list[float] = []
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        values = [float(v) for v in raw[0].get("raw", [])]
    elif isinstance(raw, list):
        values = [float(v) for v in raw]
    y = np.asarray(values, dtype=float)
    return y[np.isfinite(y)]


def load_welllog(*, simulated: bool = False) -> DatasetBundle:
    """Load the well-log change-point dataset (Ó Ruanaidh & Fitzgerald 1996,
    as bundled with the R package ``changepoint``).

    Parameters
    ----------
    simulated : bool, default False
        Force the deterministic simulated analog regardless of whether the
        real download is available.
    """

    if simulated:
        return DatasetBundle.from_dict(simulate_welllog())

    for url, fname, parser in (
        (_WELLLOG_TXT, "well_log.txt", _parse_txt),
        (_WELLLOG_JSON, "well_log.json", _parse_json),
    ):
        path = try_fetch(url=url, known_hash=None, fname=fname)
        if path is None:
            continue
        try:
            y = parser(path)
            if y.size < 50:
                continue
        except Exception:
            continue
        return DatasetBundle(
            X=np.arange(y.size, dtype=float).reshape(-1, 1),
            y=y,
            sample_weight=None,
            true_boundaries=[],
            name="welllog",
            source="downloaded",
            description=f"Ó Ruanaidh well-log NMR tool response (n={y.size}) via TCPD.",
            metadata={"url": url},
        )

    describe_fallback("welllog", "download unavailable")
    return DatasetBundle.from_dict(simulate_welllog())
