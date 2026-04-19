"""sklearn Pipeline / GridSearchCV compatibility tests."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from bayesbreak import BayesBreakGaussian


def test_fits_inside_pipeline(gaussian_data):
    pipe = Pipeline([("scale", StandardScaler()), ("seg", BayesBreakGaussian(k_max=6))])
    pipe.fit(gaussian_data.X, gaussian_data.y)
    pred = pipe.predict(gaussian_data.X)
    assert pred.shape == (gaussian_data.y.size,)


def test_score_in_pipeline(gaussian_data):
    pipe = Pipeline([("seg", BayesBreakGaussian(k_max=6))])
    pipe.fit(gaussian_data.X, gaussian_data.y)
    s = pipe.score(gaussian_data.X, gaussian_data.y)
    assert np.isfinite(s)


def test_grid_search_over_k_max(gaussian_data):
    cv = TimeSeriesSplit(n_splits=3)
    grid = GridSearchCV(
        BayesBreakGaussian(),
        param_grid={"k_max": [3, 6, 10]},
        cv=cv,
        refit=True,
    )
    grid.fit(gaussian_data.X, gaussian_data.y)
    assert grid.best_estimator_ is not None
    # Best k_max should be one of the grid values.
    assert grid.best_params_["k_max"] in {3, 6, 10}


def test_segmenter_transforms_as_featurizer(gaussian_data):
    """BayesBreak as a featurizer: transform returns integer segment indices."""

    seg = (
        BayesBreakGaussian(k_max=6).fit(gaussian_data.X, gaussian_data.y).transform(gaussian_data.X)
    )
    assert seg.shape == (gaussian_data.y.size,)
    assert seg.dtype.kind in "iu"
