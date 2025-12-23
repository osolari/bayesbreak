from __future__ import annotations

import numpy as np

from bayesbreak import make_model


def test_make_model_creates_estimators():
    y = np.array([0.0, 0.0, 1.0, 1.0])

    m = make_model("gaussian", k_max=3).fit(y)
    assert m.n_ == y.size

    p = make_model("poisson", k_max=3).fit(np.array([0.0, 1.0, 1.0, 2.0]))
    assert p.n_ == 4

    b = make_model("binomial", k_max=3, n_trials=2).fit(np.array([0.0, 1.0, 1.0, 2.0]))
    assert b.n_ == 4

    bb = make_model("beta", k_max=3).fit(np.array([0.2, 0.2, 0.8, 0.8]))
    assert bb.n_ == 4
