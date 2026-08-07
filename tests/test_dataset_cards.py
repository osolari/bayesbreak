from __future__ import annotations

import re
from pathlib import Path

import pytest

from bayesbreak.datasets import DatasetCard, load_with_provenance

ROOT = Path(__file__).parents[1]
DATASETS = ("welllog", "cgh", "spx", "methylation")


@pytest.mark.parametrize("dataset_id", DATASETS)
def test_simulated_bundle_and_card_are_synchronized(dataset_id: str) -> None:
    bundle, card = load_with_provenance(dataset_id, {"simulated": True, "stride": 2})
    assert isinstance(card, DatasetCard)
    assert card.schema_version == "1.0.0"
    assert card.dataset_id == bundle.name == dataset_id
    assert card.source_kind == bundle.source == "simulated"
    assert card.n_observations == bundle.y.shape[0]
    assert card.n_sequences == (1 if bundle.y.ndim == 1 else bundle.y.shape[1])
    assert card.stride == bundle.metadata["stride"] == 2
    assert re.fullmatch(r"[0-9a-f]{64}", card.data_hash)
    assert card.external_annotations == ()
    assert card.fitted_map_marker_role == "model-derived-map"


def test_precision_descriptors_are_not_called_power_weights() -> None:
    _, cgh = load_with_provenance("cgh", {"simulated": True})
    assert cgh.descriptor_role == "gaussian-precision"
    assert cgh.descriptor_hash is not None
    _, methylation = load_with_provenance("methylation", {"simulated": True})
    assert methylation.observation_family == "beta-observation"
    assert methylation.descriptor_role == "none"


def test_card_hashes_are_deterministic() -> None:
    _, first = load_with_provenance("welllog", {"simulated": True, "stride": 8})
    _, second = load_with_provenance("welllog", {"simulated": True, "stride": 8})
    assert first.data_hash == second.data_hash
    assert first.to_dict() == second.to_dict()


@pytest.mark.parametrize("dataset_id", DATASETS)
def test_static_data_card_declares_source_family_and_annotations(dataset_id: str) -> None:
    text = (ROOT / "docs" / "data_cards" / f"{dataset_id}.md").read_text(encoding="utf-8")
    assert "Source" in text
    assert "Observation family" in text
    assert "External annotations" in text
    assert "model-derived" in text or dataset_id in {"cgh", "spx", "methylation"}
