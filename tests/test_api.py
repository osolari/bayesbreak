"""Strict sklearn contract for the core ``BayesBreakSegmenter`` API."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone

from bayesbreak import (
    BayesBreakBernoulli,
    BayesBreakBeta,
    BayesBreakBetaObs,
    BayesBreakBinomial,
    BayesBreakGaussian,
    BayesBreakLogisticNormal,
    BayesBreakPoisson,
    make_bayesbreak,
)


@pytest.fixture
def gaussian_segmenter(gaussian_data):
    est = BayesBreakGaussian(k_max=8).fit(gaussian_data.X, gaussian_data.y)
    return est, gaussian_data


class TestFitContract:
    def test_fit_returns_self(self, gaussian_data):
        est = BayesBreakGaussian(k_max=4)
        assert est.fit(gaussian_data.X, gaussian_data.y) is est

    def test_fit_requires_y(self, gaussian_data):
        with pytest.raises(TypeError):
            BayesBreakGaussian().fit(gaussian_data.X)  # type: ignore[call-arg]

    def test_fit_rejects_1d_y_on_multivariate_wrapper(self, gaussian_data):
        from bayesbreak import SharedBoundaryMultivariateSegmenter

        est = SharedBoundaryMultivariateSegmenter(BayesBreakGaussian())
        # 1-D y gets reshaped to (n, 1) so no error; verify shape handling.
        est.fit(gaussian_data.X, gaussian_data.y)
        assert est.d_ == 1

    def test_fit_validates_k_max(self, gaussian_data):
        with pytest.raises(ValueError, match="k_max"):
            BayesBreakGaussian(k_max=0).fit(gaussian_data.X, gaussian_data.y)

    def test_fit_validates_regression_curve(self, gaussian_data):
        with pytest.raises(ValueError, match="regression_curve"):
            BayesBreakGaussian(regression_curve="bogus").fit(gaussian_data.X, gaussian_data.y)


class TestFittedAttributes:
    def test_core_attributes(self, gaussian_segmenter):
        est, data = gaussian_segmenter
        assert est.n_ == data.y.size
        assert est.x_design_.shape == (data.y.size,)
        assert est.log_block_evidence_.shape == (data.y.size + 1, data.y.size + 1)
        assert est.block_first_moment_.shape == (data.y.size + 1, data.y.size + 1)
        assert est.k_posterior_.shape == (min(8, data.y.size),)
        assert np.isfinite(est.log_evidence_)
        assert est.map_boundaries_[0] == 0
        assert est.map_boundaries_[-1] == data.y.size
        assert est.map_segment_means_.shape == (est.k_map_,)

    def test_boundary_marginals_in_unit_interval(self, gaussian_segmenter):
        est, _ = gaussian_segmenter
        assert np.all((est.boundary_marginals_ >= -1e-12) & (est.boundary_marginals_ <= 1 + 1e-12))

    def test_k_posterior_sums_to_one(self, gaussian_segmenter):
        est, _ = gaussian_segmenter
        assert float(np.sum(est.k_posterior_)) == pytest.approx(1.0, abs=1e-10)

    def test_regression_curve_optional(self, gaussian_data):
        est = BayesBreakGaussian(k_max=6, regression_curve="fixed_k").fit(
            gaussian_data.X, gaussian_data.y
        )
        assert est.bayes_curve_mean_ is not None
        assert est.bayes_curve_mean_.shape == (gaussian_data.y.size,)


class TestPredictScoreTransform:
    def test_predict_shape(self, gaussian_segmenter):
        est, data = gaussian_segmenter
        out = est.predict(data.X)
        assert out.shape == (data.y.size,)
        assert out.dtype == float

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            BayesBreakGaussian().predict(np.arange(5).reshape(-1, 1))

    def test_predict_piecewise_constant(self, gaussian_segmenter):
        est, data = gaussian_segmenter
        out = est.predict(data.X)
        # Every point should be one of the MAP segment means.
        assert np.all(np.isin(np.round(out, 6), np.round(est.map_segment_means_, 6)))

    def test_score_finite(self, gaussian_segmenter):
        est, data = gaussian_segmenter
        s = est.score(data.X, data.y)
        assert np.isfinite(s)

    def test_score_higher_on_training_than_random(self, gaussian_data):
        est = BayesBreakGaussian(k_max=8).fit(gaussian_data.X, gaussian_data.y)
        y_random = np.random.default_rng(99).standard_normal(gaussian_data.y.size) * 3.0
        assert est.score(gaussian_data.X, gaussian_data.y) > est.score(gaussian_data.X, y_random)

    def test_transform_integer_segment_indices(self, gaussian_segmenter):
        est, data = gaussian_segmenter
        seg = est.transform(data.X)
        assert seg.shape == (data.y.size,)
        assert seg.min() >= 0
        assert seg.max() <= est.k_map_ - 1

    def test_get_map_segmentation_tuple(self, gaussian_segmenter):
        est, _ = gaussian_segmenter
        k, b, mu = est.get_map_segmentation()
        assert k == est.k_map_
        assert b == est.map_boundaries_
        assert np.allclose(mu, est.map_segment_means_)


class TestSklearnPlumbing:
    def test_get_set_params(self):
        est = BayesBreakGaussian(k_max=17, estimate_hyper=False)
        assert est.get_params()["k_max"] == 17
        est.set_params(k_max=5)
        assert est.k_max == 5

    def test_clone_roundtrip(self):
        est = BayesBreakGaussian(k_max=11, regression_curve="mix_k")
        cloned = clone(est)
        assert cloned.k_max == est.k_max
        assert cloned.regression_curve == est.regression_curve
        assert cloned is not est

    def test_constructor_args_untouched(self):
        """sklearn requires constructor args be stored without coercion."""

        est = BayesBreakGaussian(k_max=11)
        # int passed in, int stored (no silent float conversion etc.).
        assert est.k_max == 11
        assert isinstance(est.k_max, int)


class TestFactory:
    @pytest.mark.parametrize(
        "name,cls",
        [
            ("gaussian", BayesBreakGaussian),
            ("normal", BayesBreakGaussian),
            ("poisson", BayesBreakPoisson),
            ("binomial", BayesBreakBinomial),
            ("beta", BayesBreakBeta),
            ("beta-obs", BayesBreakBetaObs),
            ("bernoulli", BayesBreakBernoulli),
            ("logistic-normal", BayesBreakLogisticNormal),
        ],
    )
    def test_factory_returns_class(self, name, cls):
        assert isinstance(make_bayesbreak(name), cls)

    def test_factory_passes_kwargs(self):
        est = make_bayesbreak("gaussian", k_max=25)
        assert est.k_max == 25

    def test_factory_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown family"):
            make_bayesbreak("quantum")
