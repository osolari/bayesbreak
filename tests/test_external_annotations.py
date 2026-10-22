"""External changepoint annotation infrastructure.

The registry ships citation records for the two applied studies whose
primary literature carries a reference boundary set (CGH — Matteson &
James 2014 / Bleakley & Vert 2011; well-log — Fearnhead 2006). The
records deliberately carry an *empty* ``interior_boundaries`` tuple
until a curator hand-maps the published breakpoints onto the loader's
coordinate axis; encoding fabricated indices would defeat the purpose.
"""

from __future__ import annotations

import numpy as np
import pytest

from bayesbreak.datasets import DatasetBundle, ExternalAnnotation
from bayesbreak.datasets._annotations import annotations_for
from bayesbreak.datasets.base import load_with_provenance


def _tiny_bundle(name: str, *, source: str) -> DatasetBundle:
    return DatasetBundle(
        X=np.arange(4, dtype=float).reshape(-1, 1),
        y=np.array([0.0, 1.0, 2.0, 3.0]),
        sample_weight=None,
        true_boundaries=[0, 4],
        name=name,
        source=source,
        description="tiny",
    )


def test_dataset_bundle_defaults_external_annotations_empty():
    b = _tiny_bundle("cgh", source="simulated")
    assert b.external_annotations == []


def test_registry_ships_cgh_and_welllog_citation_records():
    cgh_records = annotations_for("cgh")
    welllog_records = annotations_for("welllog")
    assert len(cgh_records) == 1
    # Five TCPD human annotators for well-log.
    assert len(welllog_records) == 5
    for record in (*cgh_records, *welllog_records):
        assert isinstance(record, ExternalAnnotation)
        assert record.source_uri.startswith(("http://", "https://"))
        assert record.citation and record.notes

    # CGH still carries no fabricated indices (no machine-readable source).
    assert cgh_records[0].interior_boundaries == ()


def test_welllog_tcpd_annotations_map_onto_4050_axis():
    """Every well-log annotator's boundaries lie inside the loader's
    length-4050 sample-index axis and are strictly increasing."""

    from bayesbreak.datasets._annotations import annotations_for

    records = annotations_for("welllog")
    assert len(records) == 5
    seen_counts = []
    for record in records:
        interior = record.interior_boundaries
        assert interior, "well-log annotator ships a non-empty boundary set"
        assert all(0 < b < 4050 for b in interior), "boundaries in (0, 4050)"
        assert list(interior) == sorted(interior), "strictly increasing"
        assert all(b % 6 == 0 for b in interior), (
            "TCPD-JSON stride-6 mapping means every boundary lands on a "
            "multiple of 6 in the 4050-length txt axis"
        )
        seen_counts.append(len(interior))
    # TCPD annotators produced 11, 9, 9, 2, 17 changepoints respectively.
    assert sorted(seen_counts) == [2, 9, 9, 11, 17]


def test_registry_returns_empty_tuple_for_unknown_dataset():
    assert annotations_for("does-not-exist") == ()


def test_external_annotation_frozen():
    record = ExternalAnnotation(
        dataset_id="cgh",
        interior_boundaries=(10, 20),
        source_uri="https://example.org/paper",
        citation="Example 2024",
        notes="hand-curated",
    )
    with pytest.raises(Exception):
        record.dataset_id = "welllog"  # frozen dataclass


def test_bundle_from_dict_round_trips_annotations():
    src = ExternalAnnotation(
        dataset_id="welllog",
        interior_boundaries=(100, 200, 300),
        source_uri="https://example.org/fearnhead-2006",
        citation="Fearnhead 2006",
        notes="illustrative test-only mapping",
    )
    b = DatasetBundle(
        X=np.arange(4, dtype=float).reshape(-1, 1),
        y=np.array([0.0, 1.0, 2.0, 3.0]),
        sample_weight=None,
        true_boundaries=[0, 4],
        name="welllog",
        source="downloaded",
        description="",
        external_annotations=[src],
    )
    payload = {
        "X": b.X,
        "y": b.y,
        "sample_weight": None,
        "true_boundaries": b.true_boundaries,
        "name": b.name,
        "source": b.source,
        "description": b.description,
        "metadata": b.metadata,
        "external_annotations": [
            {
                "dataset_id": src.dataset_id,
                "interior_boundaries": src.interior_boundaries,
                "source_uri": src.source_uri,
                "citation": src.citation,
                "notes": src.notes,
            }
        ],
    }
    restored = DatasetBundle.from_dict(payload)
    assert len(restored.external_annotations) == 1
    assert restored.external_annotations[0] == src


@pytest.mark.slow
def test_load_with_provenance_attaches_annotations_on_real_download():
    """When the real loader succeeds, annotations from the registry are
    attached to the returned bundle and its provenance card."""

    bundle, card = load_with_provenance("cgh")
    if bundle.source == "simulated":
        pytest.skip("CGH real download unavailable in this environment")
    assert bundle.external_annotations, (
        "downloaded CGH bundle must carry the registered citation record"
    )
    assert card.external_annotations, (
        "provenance card must mirror the bundle's annotations"
    )
    assert bundle.external_annotations[0].citation.startswith("Matteson")
