from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
PLAN_PATH = ROOT / "provenance" / "epr-bb-015-plan.json"


def _plan() -> dict:
    return json.loads(PLAN_PATH.read_text(encoding="utf-8"))


def test_misspecification_plan_has_stable_identity_and_budget() -> None:
    plan = _plan()
    assert plan["protocol_id"] == "EPR-BB-015"
    assert plan["planned_result_id"] == "RES-BB-SYN-006"
    assert plan["execution_status"] == "planned"
    assert plan["pilot_repetitions_per_cell"] == 1
    assert plan["full_repetitions_per_cell"] == 50
    assert plan["seed_base"] == 261501
    assert plan["ep_timeout_seconds"] == 20


def test_misspecification_plan_covers_every_registered_failure_regime() -> None:
    cell_ids = {cell["id"] for cell in _plan()["cells"]}
    assert cell_ids == {
        "null-gaussian",
        "heavy-tail-gaussian",
        "zero-inflated-poisson",
        "dense-gaussian",
        "short-segment-gaussian",
        "prior-conflict-gaussian",
        "shared-boundary-heterogeneity",
        "logistic-approximation-failure",
    }


def test_misspecification_plan_preserves_failed_and_reversed_outcomes() -> None:
    rules = " ".join(_plan()["abort_rules"])
    for required in ("NaN", "failed", "reversed", "nonconverged", "timed-out"):
        assert required in rules
    assert "explicitly approved" in rules


def test_execution_brief_points_to_machine_readable_plan() -> None:
    brief = (
        ROOT / "report" / "revision_artifacts" / "research" / "EPR-BB-015_EXECUTION_PLAN.md"
    ).read_text(encoding="utf-8")
    normalized = " ".join(brief.lower().split())
    assert "provenance/epr-bb-015-plan.json" in brief
    assert "full execution remains unapproved" in normalized
    assert "20-second timeout" in brief
