"""Versioned dataset cards and provenance-aware loader dispatch."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace

import numpy as np

from . import DatasetBundle


@dataclass(frozen=True)
class DatasetCard:
    schema_version: str
    dataset_id: str
    source_kind: str
    source_uri: str
    source_date: str
    n_observations: int
    n_sequences: int
    stride: int
    observation_family: str
    coordinate_axis: str
    descriptor_role: str
    descriptor_hash: str | None
    data_hash: str
    preprocessing: str
    external_annotations: tuple[Mapping[str, str], ...]
    fitted_map_marker_role: str = "model-derived-map"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_DATASET_SPECS: dict[str, dict[str, str]] = {
    "welllog": {
        "observation_family": "gaussian",
        "coordinate_axis": "sample-index",
        "source_date": "source-date-not-recorded",
        "source_uri": "https://github.com/alan-turing-institute/TCPD/tree/master/datasets/well_log",
        "descriptor_role": "none",
    },
    "cgh": {
        "observation_family": "gaussian-shared-boundary",
        "coordinate_axis": "probe-index",
        "source_date": "source-date-not-recorded",
        "source_uri": "https://github.com/cran/ecp/blob/master/data/ACGH.RData",
        "descriptor_role": "gaussian-precision",
    },
    "spx": {
        "observation_family": "gaussian-log-squared-return",
        "coordinate_axis": "trading-day-index",
        "source_date": "2015-01-01/2023-12-31",
        "source_uri": "https://finance.yahoo.com/quote/%5EGSPC/history",
        "descriptor_role": "none",
    },
    "methylation": {
        "observation_family": "beta-observation",
        "coordinate_axis": "CpG-genomic-coordinate",
        "source_date": "source-date-not-recorded",
        "source_uri": "https://github.com/al2na/methylKit/blob/master/inst/extdata/test1.myCpG.txt",
        "descriptor_role": "beta-observation-precision",
    },
}


def load_with_provenance(
    dataset_id: str,
    config: Mapping[str, object] | None = None,
) -> tuple[DatasetBundle, DatasetCard]:
    """Load a supported dataset and return its synchronized provenance card."""

    from .cgh import load_cgh
    from .methylation import load_methylation
    from .spx import load_spx
    from .welllog import load_welllog

    loaders: dict[str, Callable[..., DatasetBundle]] = {
        "welllog": load_welllog,
        "cgh": load_cgh,
        "spx": load_spx,
        "methylation": load_methylation,
    }
    if dataset_id not in loaders:
        raise ValueError(f"Unknown dataset_id: {dataset_id!r}")
    options = dict(config or {})
    stride_raw = options.pop("stride", 1)
    if not isinstance(stride_raw, int) or isinstance(stride_raw, bool) or stride_raw < 1:
        raise ValueError("stride must be a positive integer")
    source_date = options.pop("source_date", None)
    if source_date is not None and not isinstance(source_date, str):
        raise TypeError("source_date must be a string when provided")
    bundle = loaders[dataset_id](**options)
    if stride_raw > 1:
        bundle = _stride_bundle(bundle, stride_raw)
    card = _card_for_bundle(bundle, stride_raw, source_date=source_date)
    return bundle, card


def _stride_bundle(bundle: DatasetBundle, stride: int) -> DatasetBundle:
    indices = np.arange(0, bundle.y.shape[0], stride, dtype=int)
    weights = None if bundle.sample_weight is None else bundle.sample_weight[indices]
    boundaries = sorted(
        {
            0,
            indices.size,
            *(
                int(np.searchsorted(indices, boundary, side="left"))
                for boundary in bundle.true_boundaries
            ),
        }
    )
    boundaries = [value for value in boundaries if 0 <= value <= indices.size]
    return replace(
        bundle,
        X=np.ascontiguousarray(bundle.X[indices]),
        y=np.ascontiguousarray(bundle.y[indices]),
        sample_weight=None if weights is None else np.ascontiguousarray(weights),
        true_boundaries=boundaries,
        metadata={**bundle.metadata, "stride": stride},
    )


def _card_for_bundle(
    bundle: DatasetBundle,
    stride: int,
    *,
    source_date: str | None,
) -> DatasetCard:
    spec = _DATASET_SPECS[bundle.name]
    descriptor = bundle.sample_weight
    source_uri = str(
        bundle.metadata.get("url") or bundle.metadata.get("csv_path") or spec["source_uri"]
    )
    if bundle.source == "simulated":
        source_uri = f"bayesbreak.datasets._simulate:{bundle.name}"
    n_sequences = 1 if bundle.y.ndim == 1 else int(bundle.y.shape[1])
    coordinate_axis = (
        "observation-index" if bundle.source == "simulated" else spec["coordinate_axis"]
    )
    descriptor_role = spec["descriptor_role"] if descriptor is not None else "none"
    return DatasetCard(
        schema_version="1.0.0",
        dataset_id=bundle.name,
        source_kind=bundle.source,
        source_uri=source_uri,
        source_date=source_date or spec["source_date"],
        n_observations=int(bundle.y.shape[0]),
        n_sequences=n_sequences,
        stride=stride,
        observation_family=spec["observation_family"],
        coordinate_axis=coordinate_axis,
        descriptor_role=descriptor_role,
        descriptor_hash=None if descriptor is None else _hash_array(descriptor),
        data_hash=_hash_arrays(bundle.X, bundle.y),
        preprocessing=f"loader-normalized; stride={stride}",
        external_annotations=(),
    )


def _hash_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _hash_array(array: np.ndarray) -> str:
    return _hash_arrays(array)
