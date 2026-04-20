"""Well-log (NMR tool-response) loader.

Real source: the TCPD mirror of the classic Ó Ruanaidh well-log signal. We
prefer the ``well_log.txt`` file (full n=4050 series) and fall back to the
``well_log.json`` subset if that is unavailable. When neither can be fetched
we return the deterministic simulated analog.
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
    """Load the well-log change-point dataset (Ó Ruanaidh & Fitzgerald 1996).

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
