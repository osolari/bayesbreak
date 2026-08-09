from __future__ import annotations
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import bayesbreak
from bayesbreak import metrics, prediction
from bayesbreak.families import beta_obs


def test_status_is_explicitly_incomplete() -> None:
    assert bayesbreak.IMPLEMENTATION_STATUS == "interface-only"
    assert bayesbreak.SCIENTIFIC_IMPLEMENTATION_COMPLETE is False


def test_scientific_callables_fail_explicitly() -> None:
    with pytest.raises(NotImplementedError, match="CODE-BB-008"):
        metrics.match_boundaries_one_to_one([], [], 0.0)
    with pytest.raises(NotImplementedError, match="CODE-BB-007"):
        prediction.assign_to_partition([], [], [], prediction.ExtrapolationPolicy.ERROR)
    with pytest.raises(NotImplementedError, match="CODE-BB-006"):
        beta_obs.posterior_predictive_logpdf_block()
