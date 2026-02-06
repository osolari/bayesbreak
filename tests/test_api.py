"""Tests for the make_bayesbreak factory function and API."""

from __future__ import annotations

import numpy as np
import pytest

from bayesbreak import (
    BayesBreakBernoulli,
    BayesBreakGaussian,
    BayesBreakPoisson,
    make_bayesbreak,
    make_model,
)


class TestMakeBayesbreak:
    """Test the factory function."""

    @pytest.mark.parametrize(
        "family,expected_cls",
        [
            ("gaussian", BayesBreakGaussian),
            ("normal", BayesBreakGaussian),
            ("poisson", BayesBreakPoisson),
            ("count", BayesBreakPoisson),
            ("bernoulli", BayesBreakBernoulli),
            ("binary", BayesBreakBernoulli),
        ],
    )
    def test_factory_returns_correct_class(self, family, expected_cls):
        """Factory should return correct estimator class."""
        model = make_bayesbreak(family)
        assert isinstance(model, expected_cls)

    def test_factory_passes_kwargs(self):
        """Factory should pass kwargs to constructor."""
        model = make_bayesbreak("gaussian", k_max=25, estimate_hyper=False)
        assert model.k_max == 25
        assert model.estimate_hyper is False

    def test_factory_unknown_family_raises(self):
        """Unknown family should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown family"):
            make_bayesbreak("unknown_family")

    def test_make_model_alias(self):
        """make_model should be an alias for make_bayesbreak."""
        m1 = make_bayesbreak("gaussian", k_max=10)
        m2 = make_model("gaussian", k_max=10)
        assert type(m1) == type(m2)
        assert m1.k_max == m2.k_max


class TestEstimatorValidation:
    """Test input validation."""

    def test_k_max_must_be_positive(self):
        """k_max < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="k_max"):
            BayesBreakGaussian(k_max=0)

    def test_regression_curve_invalid_value(self):
        """Invalid regression_curve should raise ValueError."""
        with pytest.raises(ValueError, match="regression_curve"):
            BayesBreakGaussian(regression_curve="invalid")

    def test_fit_requires_data(self):
        """fit() with no data should raise ValueError."""
        model = BayesBreakGaussian()
        with pytest.raises(ValueError):
            model.fit()


class TestSklearnCompatibility:
    """Test scikit-learn API compatibility."""

    def test_fit_returns_self(self):
        """fit() should return self for chaining."""
        rng = np.random.default_rng(0)
        y = rng.standard_normal(50)
        model = BayesBreakGaussian(k_max=5)
        result = model.fit(y)
        assert result is model

    def test_fit_with_X_instead_of_y(self):
        """fit(X) should work when y is not provided."""
        rng = np.random.default_rng(1)
        X = rng.standard_normal(50)
        model = BayesBreakGaussian(k_max=5)
        model.fit(X)
        assert model.n_ == 50

    def test_predict_before_fit_raises(self):
        """predict() before fit() should raise."""
        model = BayesBreakGaussian()
        with pytest.raises(RuntimeError):
            model.predict()

    def test_get_params(self):
        """get_params should return constructor parameters."""
        model = BayesBreakGaussian(k_max=20, estimate_hyper=False)
        params = model.get_params()
        assert params["k_max"] == 20
        assert params["estimate_hyper"] is False

    def test_set_params(self):
        """set_params should update parameters."""
        model = BayesBreakGaussian(k_max=10)
        model.set_params(k_max=30)
        assert model.k_max == 30
