"""Publication-quality matplotlib + seaborn styling for BayesBreak figures.

Every figure script calls :func:`setup_style` once at the top. This module
enforces consistent typography, colour palette, line weights, and figure sizes
across all figures shipped with the report.

Aesthetic conventions (matching the report):

- **Full spines** on all four sides of every axes (no despining).
- **Large fonts** — body text ≈ 14 pt, ticks ≈ 12 pt, panel labels ≈ 16 pt.
- **Muted, colour-blind-safe palette** — Paul Tol's "muted" set.
- Tighter linewidths than seaborn defaults to keep dense panels readable.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module="fontTools")
warnings.filterwarnings("ignore", message=".*timestamp.*")
warnings.filterwarnings("ignore", message=".*extra bytes.*")
logging.getLogger("fontTools").setLevel(logging.ERROR)

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

# Column widths (inches) following a ~170 mm text width.
SINGLE_COL_WIDTH = 3.5
DOUBLE_COL_WIDTH = 7.0
FULL_PAGE_WIDTH = 7.0

DPI_SCREEN = 150
DPI_PRINT = 300


# Paul Tol's "muted" palette — colour-blind safe and visually softer than
# "bright". Order tuned for the figures in the report (blue, red, green, ...).
_MUTED_PALETTE = [
    "#4477AA",  # blue (Tol "muted")
    "#CC6677",  # rose
    "#44AA99",  # teal
    "#DDCC77",  # sand
    "#88CCEE",  # cyan
    "#AA4499",  # purple
    "#117733",  # green
    "#999933",  # olive
    "#882255",  # wine
]

COLORS = {
    "blue": _MUTED_PALETTE[0],
    "red": _MUTED_PALETTE[1],
    "green": _MUTED_PALETTE[2],
    "yellow": _MUTED_PALETTE[3],
    "cyan": _MUTED_PALETTE[4],
    "purple": _MUTED_PALETTE[5],
    "grey": "#888888",
    "black": "#222222",
    "lightgrey": "#DDDDDD",
    "orange": "#EE7733",
    # Saim brand indigo — reserved for accents that should read as "BayesBreak".
    "saim": "#32127A",
    "saim_accent": "#5B36B7",
    "saim_soft": "#D6CFE4",
}

COLOR_CYCLE = _MUTED_PALETTE[:7]


def setup_style(
    *,
    font_scale: float = 1.0,
    use_tex: bool = False,
    context: str = "paper",
) -> None:
    """Set matplotlib + seaborn defaults for publication figures.

    Parameters
    ----------
    font_scale : float
        Multiplier applied to all font sizes (default 1.0 → ~14 pt body).
    use_tex : bool
        If True, render text via LaTeX (slower, crisper math).
    context : str
        Seaborn context ("paper", "notebook", "talk", "poster").
    """

    sns.set_theme(
        style="ticks",
        context=context,
        palette=_MUTED_PALETTE,
        font_scale=font_scale,
        rc={
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        },
    )

    base = 13.0 * font_scale
    params: dict = {
        "figure.dpi": DPI_SCREEN,
        "figure.facecolor": "white",
        "figure.constrained_layout.use": True,
        "font.size": base,
        "axes.labelsize": base + 0.5,
        "axes.titlesize": base + 1.5,
        "axes.titleweight": "medium",
        "axes.titlepad": 12.0,
        "axes.labelweight": "regular",
        "axes.linewidth": 1.0,
        "axes.edgecolor": "#222222",
        # Full spines on every side.
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.grid": False,
        "axes.prop_cycle": mpl.cycler(color=COLOR_CYCLE),
        "xtick.labelsize": base - 1.0,
        "ytick.labelsize": base - 1.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.minor.width": 0.7,
        "ytick.minor.width": 0.7,
        "xtick.minor.size": 2.5,
        "ytick.minor.size": 2.5,
        "lines.linewidth": 1.7,
        "lines.markersize": 4.5,
        "legend.fontsize": base - 1.0,
        "legend.frameon": False,
        "legend.borderaxespad": 0.3,
        "legend.handlelength": 1.5,
        "savefig.dpi": DPI_PRINT,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "savefig.transparent": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "pdf.compression": 9,
    }
    if use_tex:
        params["text.usetex"] = True
        params["text.latex.preamble"] = r"\usepackage{amsmath}\usepackage{amssymb}"
        params["font.family"] = "serif"

    mpl.rcParams.update(params)


def get_figsize(
    width: str | float = "single",
    aspect: float = 0.62,
    nrows: int = 1,
    ncols: int = 1,
) -> tuple[float, float]:
    """Compute (width, height) in inches for a grid of subplots."""

    if isinstance(width, str):
        w = {"single": SINGLE_COL_WIDTH, "double": DOUBLE_COL_WIDTH, "full": FULL_PAGE_WIDTH}[width]
    else:
        w = float(width)
    h = w * aspect * (nrows / ncols)
    return (w, h)


def add_panel_label(
    ax,
    label: str,
    title: str | None = None,
    *,
    fontsize: float | None = None,
    offset: tuple[float, float] = (-0.22, 1.04),
) -> None:
    """Add a bold panel label (``A``, ``B``, …) outside the axes frame.

    The label is anchored to the top-left, *outside* the axes box so that it
    never collides with a centred title. If a ``title`` is given it is set with
    ``loc="left"`` so that title and label share the same baseline.
    """

    fs = fontsize if fontsize is not None else mpl.rcParams["axes.titlesize"] + 2
    ax.text(
        offset[0],
        offset[1],
        label,
        transform=ax.transAxes,
        fontsize=fs,
        fontweight="bold",
        va="baseline",
        ha="left",
    )
    if title:
        ax.set_title(title, loc="left", fontsize=mpl.rcParams["axes.titlesize"])


def save_figure(
    fig,
    path,
    *,
    formats: tuple[str, ...] = ("png", "pdf"),
    dpi: int | None = None,
) -> None:
    """Save a figure in multiple formats. PDF is vector; PNG is 300 dpi by default."""

    path = Path(path)
    base = path.parent / path.stem
    for fmt in formats:
        out = base.with_suffix(f".{fmt}")
        fig.savefig(out, format=fmt, dpi=(dpi or DPI_PRINT))
    plt.close(fig)


__all__ = [
    "COLORS",
    "COLOR_CYCLE",
    "DPI_PRINT",
    "DPI_SCREEN",
    "DOUBLE_COL_WIDTH",
    "FULL_PAGE_WIDTH",
    "SINGLE_COL_WIDTH",
    "add_panel_label",
    "get_figsize",
    "save_figure",
    "setup_style",
]
