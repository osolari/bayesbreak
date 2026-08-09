from __future__ import annotations

import numpy as np
import pytest

from bayesbreak import BayesBreakGaussian, BayesBreakMixtureClassifier
from bayesbreak.mixture import _GroupState


def _small_data() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.1, 0.0, 1.1, 1.0],
        ]
    )


def _synthetic_restart(final: float, trace: list[float], *, groups: int = 1):
    pi = np.full(groups, 1.0 / groups)
    responsibilities = np.full((_small_data().shape[0], groups), 1.0 / groups)
    states = [
        _GroupState(hyper={}, template=[0, 4], k_g=1, log_score_offset=0.0) for _ in range(groups)
    ]
    table = np.zeros((5, 5))
    return final, pi, responsibilities, states, trace, [table, table]


def test_single_iteration_returns_final_trace_objective() -> None:
    fitted = BayesBreakMixtureClassifier(
        BayesBreakGaussian(k_max=2),
        n_groups=1,
        k_max=2,
        max_iter=1,
        n_restarts=1,
        random_state=0,
    ).fit(_small_data())
    assert fitted.final_objective_ == fitted.objective_trace_[-1]
    assert fitted.objective_ is fitted.objective_trace_


def test_restart_selection_uses_returned_final_objective(monkeypatch) -> None:
    outputs = iter(
        [
            _synthetic_restart(2.0, [1.0, 2.0]),
            _synthetic_restart(3.0, [1.0, 3.0]),
        ]
    )
    monkeypatch.setattr(
        BayesBreakMixtureClassifier,
        "_fit_one_restart",
        lambda self, *args, **kwargs: next(outputs),
    )
    fitted = BayesBreakMixtureClassifier(
        BayesBreakGaussian(), n_groups=1, n_restarts=2, random_state=0
    ).fit(_small_data())
    assert fitted.selected_restart_ == 1
    assert fitted.final_objective_ == 3.0
    assert fitted.objective_trace_ == [1.0, 3.0]


def test_nonmonotone_and_stale_restarts_are_excluded(monkeypatch) -> None:
    outputs = iter(
        [
            _synthetic_restart(4.0, [1.0, 0.5]),
            _synthetic_restart(5.0, [1.0, 2.0]),
            _synthetic_restart(3.0, [1.0, 3.0]),
        ]
    )
    monkeypatch.setattr(
        BayesBreakMixtureClassifier,
        "_fit_one_restart",
        lambda self, *args, **kwargs: next(outputs),
    )
    fitted = BayesBreakMixtureClassifier(
        BayesBreakGaussian(), n_groups=1, n_restarts=3, random_state=0
    ).fit(_small_data())
    assert fitted.selected_restart_ == 2
    assert [item.status for item in fitted.restart_diagnostics_] == [
        "invalid",
        "invalid",
        "valid",
    ]


def test_restart_ties_select_first_deterministically(monkeypatch) -> None:
    outputs = iter(
        [
            _synthetic_restart(2.0, [1.0, 2.0]),
            _synthetic_restart(2.0, [1.5, 2.0]),
        ]
    )
    monkeypatch.setattr(
        BayesBreakMixtureClassifier,
        "_fit_one_restart",
        lambda self, *args, **kwargs: next(outputs),
    )
    fitted = BayesBreakMixtureClassifier(
        BayesBreakGaussian(), n_groups=1, n_restarts=2, random_state=0
    ).fit(_small_data())
    assert fitted.selected_restart_ == 0


def test_failed_restart_is_recorded_and_excluded(monkeypatch) -> None:
    calls = 0

    def run_restart(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise FloatingPointError("synthetic failure")
        return _synthetic_restart(2.0, [1.0, 2.0])

    monkeypatch.setattr(BayesBreakMixtureClassifier, "_fit_one_restart", run_restart)
    fitted = BayesBreakMixtureClassifier(
        BayesBreakGaussian(), n_groups=1, n_restarts=2, random_state=0
    ).fit(_small_data())
    assert fitted.selected_restart_ == 1
    assert [item.status for item in fitted.restart_diagnostics_] == ["failed", "valid"]
    assert fitted.restart_diagnostics_[0].reason == "synthetic failure"


def test_all_invalid_restarts_raise_with_diagnostics(monkeypatch) -> None:
    monkeypatch.setattr(
        BayesBreakMixtureClassifier,
        "_fit_one_restart",
        lambda self, *args, **kwargs: _synthetic_restart(0.5, [1.0, 0.5]),
    )
    estimator = BayesBreakMixtureClassifier(
        BayesBreakGaussian(), n_groups=1, n_restarts=1, random_state=0
    )
    with pytest.raises(RuntimeError, match="No valid"):
        estimator.fit(_small_data())
    assert estimator.restart_diagnostics_[0].status == "invalid"


def test_collapsed_initial_groups_produce_finite_valid_result() -> None:
    fitted = BayesBreakMixtureClassifier(
        BayesBreakGaussian(k_max=2),
        n_groups=4,
        k_max=2,
        max_iter=3,
        n_restarts=1,
        random_state=3,
    ).fit(_small_data())
    assert np.isfinite(fitted.final_objective_)
    assert fitted.final_objective_ == fitted.objective_trace_[-1]
    assert fitted.restart_diagnostics_[0].status == "valid"


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"max_iter": 0}, "max_iter"),
        ({"n_restarts": 0}, "n_restarts"),
        ({"tol": -1.0}, "tol"),
    ],
)
def test_invalid_optimization_controls_are_rejected(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        BayesBreakMixtureClassifier(BayesBreakGaussian(), **kwargs).fit(_small_data())
