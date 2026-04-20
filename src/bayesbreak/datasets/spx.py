"""S&P-500 volatility-regime loader.

Real path: ``yfinance`` (opt-in ``bayesbreak[datasets-live]`` extra). We fetch
daily ``^GSPC`` closes over a fixed window, build log-squared returns, and
return them as the response. When ``yfinance`` is missing or the download
fails, we fall back to the deterministic simulated GARCH-like regime analog.
"""

from __future__ import annotations

import numpy as np

from . import DatasetBundle
from ._cache import describe_fallback
from ._simulate import simulate_spx


def _yfinance_fetch(start: str, end: str) -> np.ndarray | None:
    try:
        import yfinance as yf
    except ImportError:
        return None
    try:
        df = yf.download(
            "^GSPC",
            start=start,
            end=end,
            progress=False,
            auto_adjust=True,
        )
    except Exception:  # pragma: no cover - network
        return None
    if df is None or df.empty or "Close" not in df.columns:
        return None
    close = np.asarray(df["Close"].to_numpy(), dtype=float)
    close = close[np.isfinite(close)]
    if close.size < 100:
        return None
    return close


def load_spx(
    *,
    simulated: bool = False,
    start: str = "2015-01-01",
    end: str = "2023-12-31",
) -> DatasetBundle:
    """Load S&P-500 volatility-regime observations.

    The response is ``log(r_t^2)`` where ``r_t`` is the daily log-return.
    Under a piecewise-constant volatility model, ``log(r_t^2)`` is (up to an
    additive constant) Gaussian within each regime and shifts across regimes.

    Parameters
    ----------
    simulated : bool, default False
        Force the deterministic simulated analog regardless of ``yfinance``
        availability.
    start, end : str
        ``YYYY-MM-DD`` date window passed to ``yfinance``.
    """

    if simulated:
        return DatasetBundle.from_dict(simulate_spx())

    close = _yfinance_fetch(start=start, end=end)
    if close is None:
        describe_fallback("spx", "yfinance unavailable")
        return DatasetBundle.from_dict(simulate_spx())

    returns = np.diff(np.log(close))
    y = np.log(returns**2 + 1e-12)
    return DatasetBundle(
        X=np.arange(y.size, dtype=float).reshape(-1, 1),
        y=y,
        sample_weight=None,
        true_boundaries=[],
        name="spx",
        source="downloaded",
        description=(f"^GSPC daily log-squared returns ({start} \u2192 {end}, n={y.size})."),
        metadata={"ticker": "^GSPC", "start": start, "end": end},
    )
