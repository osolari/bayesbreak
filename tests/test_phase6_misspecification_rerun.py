from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from scripts.phase6_misspecification_rerun import (
    cell_input_hashes,
    generate_logistic_cell,
    generate_shared_cell,
    generate_standard_cell,
    interval_summary,
    piecewise_mean,
    run_ep_bounded,
    run_standard_cell,
    summarize,
)

ROOT = Path(__file__).parents[1]


def test_piecewise_mean_returns_strict_partition() -> None:
    values, boundaries = piecewise_mean([3, 2, 4], [0.0, 1.0, -1.0])
    assert values.tolist() == [0.0, 0.0, 0.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0]
    assert boundaries == [0, 3, 5, 9]


def test_standard_generators_are_deterministic_and_match_plan_dimensions() -> None:
    expected = {
        "null-gaussian": (120, [0, 120]),
        "heavy-tail-gaussian": (120, [0, 40, 80, 120]),
        "zero-inflated-poisson": (120, [0, 40, 80, 120]),
        "dense-gaussian": (120, list(range(0, 121, 10))),
        "short-segment-gaussian": (100, [0, 48, 52, 100]),
        "prior-conflict-gaussian": (120, [0, 40, 80, 120]),
    }
    for cell_id, (n, boundaries) in expected.items():
        first = generate_standard_cell(cell_id, 123)
        second = generate_standard_cell(cell_id, 123)
        assert np.array_equal(first["values"], second["values"])
        assert first["values"].size == n
        assert first["true_boundaries"] == boundaries


def test_shared_and_logistic_generators_preserve_declared_truth() -> None:
    shared = generate_shared_cell(456)
    assert len(shared["sequences"]) == 8
    assert all(values.shape == (120,) for values in shared["sequences"])
    assert shared["common_boundaries"] == [0, 40, 80, 120]
    assert shared["subject_boundaries"][:2] == [[0, 40, 60, 80, 120]] * 2

    logistic = generate_logistic_cell(789)
    assert logistic["values"].shape == (80,)
    assert logistic["true_boundaries"] == [0, 40, 80]
    assert set(np.unique(logistic["values"])).issubset({0.0, 1.0})


def test_summary_retains_failed_and_reversed_outcomes() -> None:
    records = [
        {"status": "failed", "cell": "null-gaussian", "wall_seconds": 0.1},
        {
            "status": "executed",
            "cell": "null-gaussian",
            "wall_seconds": 0.2,
            "boundary_metrics": {"f1": 0.0},
            "k_error": 2,
            "posterior_k_entropy": 0.5,
            "posterior_mass_at_k_max": 0.2,
            "map_at_k_max": False,
            "missed_change_count": 0,
            "complete_boundary_recovery": True,
            "false_discovery_count": 2,
            "false_positive_dataset": True,
        },
    ]
    summary = summarize(records, ["null-gaussian"])["cells"]["null-gaussian"]
    assert summary["n_runs"] == 2
    assert summary["n_failed"] == 1
    assert summary["failure_rate"] == 0.5
    assert summary["false_discovery_count"]["mean"] == 2.0


def test_interval_summary_handles_single_pilot_value() -> None:
    assert interval_summary([3.0]) == {
        "mean": 3.0,
        "standard_error": 0.0,
        "ci95_lower": 3.0,
        "ci95_upper": 3.0,
        "min": 3.0,
        "max": 3.0,
    }


def test_ep_timeout_is_retained_as_scientific_outcome(monkeypatch) -> None:
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", timeout)
    record = run_ep_bounded(
        values=np.zeros(8),
        reference=None,  # type: ignore[arg-type]
        truth=[4],
        timeout_seconds=20,
    )
    assert record["status"] == "timed-out"
    assert record["timeout_seconds"] == 20
    assert record["wall_seconds"] >= 0.0


def test_logistic_summary_reports_timeout_rate() -> None:
    records = [
        {
            "status": "executed",
            "cell": "logistic-approximation-failure",
            "wall_seconds": 20.2,
            "methods": {
                "quadrature-40": {
                    "status": "executed",
                    "diagnostics": {
                        "extra": {
                            "block_error_max": 1.0,
                            "pk_tv_empirical": 0.1,
                            "conditional_partition_bounds": {"tv_upper_bound": 1.0},
                            "map_path_jaccard": 0.8,
                            "segment_error_record": {"convergence_status": "verified"},
                        }
                    },
                    "truth_metrics": {"f1": 0.9},
                },
                "laplace": {
                    "status": "executed",
                    "diagnostics": {
                        "extra": {
                            "block_error_max": 2.0,
                            "pk_tv_empirical": 0.2,
                            "conditional_partition_bounds": {"tv_upper_bound": 1.0},
                            "map_path_jaccard": 0.7,
                            "segment_error_record": {"convergence_status": "verified"},
                        }
                    },
                    "truth_metrics": {"f1": 0.8},
                },
                "ep": {"status": "timed-out", "timeout_seconds": 20},
            },
        }
    ]
    summary = summarize(records, ["logistic-approximation-failure"])["cells"][
        "logistic-approximation-failure"
    ]
    assert summary["ep_execution_rate"] == 0.0
    assert summary["ep_timeout_rate"]["mean"] == 1.0
    assert summary["ep_max_block_error"] is None


def test_prior_conflict_retains_feasible_counts() -> None:
    record = run_standard_cell("prior-conflict-gaussian", seed=311501)
    assert record["status"] == "executed"
    assert record["k_map"] in {1, 2}
    assert record["predicted_boundaries"][0] == 0
    assert record["predicted_boundaries"][-1] == 120
    assert all(
        stop - start >= 50
        for start, stop in zip(
            record["predicted_boundaries"][:-1],
            record["predicted_boundaries"][1:],
            strict=True,
        )
    )


def test_every_cell_has_complete_input_identity_hashes() -> None:
    for index, cell_id in enumerate(
        [
            "null-gaussian",
            "heavy-tail-gaussian",
            "zero-inflated-poisson",
            "dense-gaussian",
            "short-segment-gaussian",
            "prior-conflict-gaussian",
            "shared-boundary-heterogeneity",
            "logistic-approximation-failure",
        ]
    ):
        hashes = cell_input_hashes(cell_id, 261501 + 10_000 * index)
        for name in ("data_hash", "truth_hash", "effective_config_hash"):
            assert isinstance(hashes[name], str) and len(hashes[name]) == 64


def test_bounded_repilot_changes_only_ep_outcome() -> None:
    result_dir = ROOT / "results" / "phase6" / "RES-BB-SYN-006"
    original = json.loads((result_dir / "pilot.json").read_text(encoding="utf-8"))
    bounded = json.loads((result_dir / "pilot-ep-timeout.json").read_text(encoding="utf-8"))
    standard_keys = (
        "true_boundaries",
        "predicted_boundaries",
        "k_map",
        "k_error",
        "posterior_k_entropy",
        "boundary_metrics",
        "false_discovery_count",
        "missed_change_count",
        "log_evidence",
        "data_hash",
    )
    shared_keys = (
        "common_boundaries",
        "shared_boundaries",
        "shared_k_map",
        "shared_metrics",
        "subject_specific_boundary_60_selected_as_shared",
        "independent",
        "independent_mean_f1",
        "data_hash",
    )
    for before, after in zip(original["records"][:7], bounded["records"][:7], strict=True):
        assert before["cell"] == after["cell"]
        assert before["status"] == after["status"]
        if before["status"] == "failed":
            assert (before["exception_type"], before["exception_message"]) == (
                after["exception_type"],
                after["exception_message"],
            )
        elif before["cell"] == "shared-boundary-heterogeneity":
            assert all(before[key] == after[key] for key in shared_keys)
        else:
            assert all(before[key] == after[key] for key in standard_keys)

    old_logistic, new_logistic = original["records"][7], bounded["records"][7]
    assert old_logistic["data_hash"] == new_logistic["data_hash"]
    assert old_logistic["reference_boundaries"] == new_logistic["reference_boundaries"]
    for method in ("quadrature-40", "laplace"):
        assert (
            old_logistic["methods"][method]["diagnostics"]["extra"]
            == new_logistic["methods"][method]["diagnostics"]["extra"]
        )
    assert new_logistic["methods"]["ep"]["status"] == "timed-out"
