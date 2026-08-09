from __future__ import annotations

import numpy as np
import pytest

from bayesbreak import (
    BayesBreakGaussian,
    IndependentMultivariateSegmenter,
    SharedBoundaryMultivariateSegmenter,
    SharedBoundaryReplicatesSegmenter,
    SlidingWindowSegmenter,
)
from bayesbreak.prediction import (
    ExtrapolationPolicy,
    assign_to_partition,
    posterior_predictive_logpdf,
)


@pytest.fixture
def fitted() -> BayesBreakGaussian:
    x = np.arange(6, dtype=float)
    y = np.array([0.0, 0.1, 0.2, 1.0, 1.1, 1.2])
    return BayesBreakGaussian(k_max=2).fit(x, y)


def test_default_policy_rejects_both_out_of_support_sides(fitted) -> None:
    for coordinate in (-0.1, 5.1):
        with pytest.raises(ValueError, match="must lie"):
            fitted.predict(np.array([coordinate]))
        with pytest.raises(ValueError, match="must lie"):
            fitted.transform(np.array([coordinate]))
        with pytest.raises(ValueError, match="must lie"):
            posterior_predictive_logpdf(fitted, np.array([coordinate]), np.array([0.5]))


def test_clip_policy_preserves_legacy_endpoint_assignment(fitted) -> None:
    queries = np.array([6.0, -2.0])
    endpoints = fitted.predict(np.array([5.0, 0.0]))
    clipped = fitted.predict(queries, extrapolation="clip")
    assert clipped == pytest.approx(endpoints)
    assert fitted.prediction_metadata_ == {
        "extrapolation": "clip",
        "coordinate_support": [0.0, 5.0],
    }
    assert fitted.prediction_provenance_ == fitted.prediction_metadata_


def test_directional_endpoint_policies_reject_the_opposite_side(fitted) -> None:
    fitted.predict(np.array([-1.0]), extrapolation="left_endpoint")
    with pytest.raises(ValueError, match="right-of-support"):
        fitted.predict(np.array([6.0]), extrapolation="left_endpoint")
    fitted.predict(np.array([6.0]), extrapolation="right_endpoint")
    with pytest.raises(ValueError, match="left-of-support"):
        fitted.predict(np.array([-1.0]), extrapolation="right_endpoint")


def test_exact_support_boundaries_and_unsorted_queries_are_stable(fitted) -> None:
    queries = np.array([5.0, 0.0, 3.5, 1.5])
    observed = fitted.predict(queries)
    expected = np.array([fitted.predict(np.array([value]))[0] for value in queries])
    assert observed == pytest.approx(expected)


def test_assign_to_partition_validates_and_preserves_query_order() -> None:
    fitted_coordinates = np.array([0.0, 1.0, 2.0, 3.0])
    boundaries = [0, 2, 4]
    queries = np.array([3.0, 0.0, 2.5, 1.5])
    segments = assign_to_partition(
        queries,
        fitted_coordinates,
        boundaries,
        ExtrapolationPolicy.ERROR,
    )
    assert segments.tolist() == [1, 0, 1, 0]


@pytest.mark.parametrize(
    "policy",
    ["error", "clip", "left_endpoint", "right_endpoint"],
)
def test_in_range_assignments_are_unchanged_for_every_policy(fitted, policy) -> None:
    queries = np.array([0.0, 1.5, 5.0])
    assert (
        fitted.transform(queries, extrapolation=policy).tolist()
        == fitted.transform(
            queries,
            extrapolation="error",
        ).tolist()
    )


def test_prediction_wrappers_share_the_support_policy() -> None:
    x = np.arange(6, dtype=float)
    y = np.array([0.0, 0.1, 0.2, 1.0, 1.1, 1.2])
    replicated = SharedBoundaryReplicatesSegmenter(BayesBreakGaussian(k_max=2)).fit(
        x,
        [y, y + 0.1],
    )
    multivariate = SharedBoundaryMultivariateSegmenter(BayesBreakGaussian(k_max=2), k_max=2).fit(
        x, np.column_stack([y, y + 0.1])
    )
    independent = IndependentMultivariateSegmenter(BayesBreakGaussian(k_max=2)).fit(
        x,
        np.column_stack([y, y + 0.1]),
    )
    sliding = SlidingWindowSegmenter(
        BayesBreakGaussian(k_max=2),
        window_size=4,
        overlap=1,
    ).fit(x, y)

    for estimator in (replicated, multivariate, independent, sliding):
        with pytest.raises(ValueError, match="must lie"):
            estimator.predict(np.array([-1.0]))
        output = estimator.predict(np.array([-1.0]), extrapolation="clip")
        assert output.shape[0 if output.ndim == 1 else 1] > 0
        assert estimator.prediction_metadata_["extrapolation"] == "clip"
