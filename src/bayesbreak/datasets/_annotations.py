"""Curated external changepoint annotations for the shipped loaders.

This module holds *citation metadata* for external annotations that would
inform BayesBreak comparator evaluation on the applied studies. Per
:mod:`bayesbreak.datasets.base` (dataset schema), we ship
citation records with empty ``interior_boundaries`` when the primary
literature reports chromosome-level or qualitative annotations rather
than probe-indexed changepoint lists — encoding fabricated boundary
indices would defeat the purpose. A curator who has hand-mapped the
primary annotations to the loader's native coordinate axis provides
``interior_boundaries`` at that time and cites the mapping in ``notes``.
"""

from __future__ import annotations

from typing import Mapping

from . import ExternalAnnotation


_TCPD_ANN_URI = "https://github.com/alan-turing-institute/TCPD/blob/master/annotations.json"
_TCPD_CITATION = (
    "van den Burg and Williams (2020). An Evaluation of Change Point "
    "Detection Algorithms. arXiv:2003.06222. Ships five independent "
    "human-annotator changepoint sets per dataset under "
    "github.com/alan-turing-institute/TCPD/annotations.json."
)


def _welllog_annotators() -> tuple[ExternalAnnotation, ...]:
    """TCPD well-log annotations on the loader's native n=4050 axis.

    The TCPD JSON series is ``well_log.txt[::6]``; a changepoint at
    JSON index ``j`` therefore corresponds to text-file index ``6 * j``
    under the standard "right-boundary of the pre-change window"
    convention. The mapping is bit-exact against the shipped
    ``well_log.txt`` (verified by matching ``txt[::6] == json`` to
    machine precision when the coordinate conversion was checked).
    """

    tcpd = {
        "6": (179, 255, 281, 311, 343, 402, 413, 422, 432, 462, 464),
        "7": (179, 255, 281, 312, 343, 402, 412, 422, 432),
        "8": (179, 255, 282, 312, 343, 402, 413, 422, 432),
        "12": (177, 467),
        "13": (4, 179, 255, 281, 311, 344, 402, 412, 422, 432, 462, 464, 521, 526, 620, 643, 661),
    }
    return tuple(
        ExternalAnnotation(
            dataset_id="welllog",
            interior_boundaries=tuple(6 * b for b in boundaries),
            source_uri=_TCPD_ANN_URI,
            citation=_TCPD_CITATION,
            notes=(
                f"TCPD human annotator {annotator}: {len(boundaries)} "
                f"changepoints on the length-675 json subset, mapped to "
                f"the length-4050 txt axis via ``txt_index = 6 * json_index``."
            ),
        )
        for annotator, boundaries in sorted(tcpd.items(), key=lambda kv: int(kv[0]))
    )


_REGISTRY: Mapping[str, tuple[ExternalAnnotation, ...]] = {
    "cgh": (
        ExternalAnnotation(
            dataset_id="cgh",
            interior_boundaries=(),
            source_uri="https://cran.r-project.org/web/packages/ecp/index.html",
            citation=(
                "Matteson and James (2014). A Nonparametric Approach for "
                "Multiple Change Point Analysis of Multivariate Data. "
                "Journal of the American Statistical Association 109(505): "
                "334-345. The ACGH dataset in the ecp package accompanies "
                "the analysis and cites Bleakley and Vert (2011, "
                "arXiv:1106.4199) for the underlying multi-subject "
                "array-CGH profiles."
            ),
            notes=(
                "Neither Matteson-James 2014 nor Bleakley-Vert 2011 "
                "publish a machine-readable shared-changepoint list on "
                "the ecp probe axis; both illustrate per-subject "
                "breakpoints qualitatively. TCPD (van den Burg-Williams "
                "2020) does not ship a CGH entry. Populating "
                "``interior_boundaries`` requires a curator to hand-map "
                "the illustrated per-subject breakpoints to the 2215-probe "
                "axis and (optionally) take a consensus."
            ),
        ),
    ),
    "welllog": _welllog_annotators(),
}


def annotations_for(dataset_id: str) -> tuple[ExternalAnnotation, ...]:
    """Return the curated citation records for ``dataset_id`` (empty tuple
    when no external annotation is currently registered)."""

    return _REGISTRY.get(dataset_id, ())
