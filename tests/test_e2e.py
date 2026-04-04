"""End-to-end tests for the harness system.

Tests the full workflow: start → submit → evaluate → advance → complete.
Uses the subagent evaluator (no API key needed) with mocked reviewer responses.
Uses a simple test SOP without extract_requirements for core flow tests.
Structured criteria features are tested in test_p2_features.py.
"""

import json
from pathlib import Path

import pytest
import yaml

from server.evaluator import DIMENSIONS, SubagentEvaluator, create_evaluator
from server.models import SessionStage, StepStatus
from server.orchestrator import Orchestrator
from server.session_manager import SessionManager
from server.sop_registry import SOPRegistry


# Simple test SOP without extract_requirements (for core flow tests)
SIMPLE_E2E_SOP = {
    "sop_id": "e2e-test",
    "name": "E2E Test",
    "default_retry_limit": 3,
    "phases": [
        {
            "id": "phase1",
            "name": "Phase 1",
            "steps": [
                {"id": "s1", "title": "Step 1", "instruction": "Do step 1",
                 "acceptance_criteria": ["Criterion A addressed"]},
                {"id": "s2", "title": "Step 2", "instruction": "Do step 2",
                 "acceptance_criteria": ["Criterion B addressed"],
                 "depends_on": ["s1"]},
                {"id": "s3", "title": "Step 3", "instruction": "Do step 3",
                 "acceptance_criteria": ["Criterion C addressed"],
                 "depends_on": ["s2"]},
            ],
        },
    ],
}


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def harness(tmp_path):
    """Set up harness with a simple test SOP."""
    sop_dir = tmp_path / "sops"
    sop_dir.mkdir()
    (sop_dir / "e2e-test.yaml").write_text(yaml.dump(SIMPLE_E2E_SOP))
    registry = SOPRegistry(search_dirs=[sop_dir])
    manager = SessionManager(base_dir=tmp_path / "sessions")
    evaluator = SubagentEvaluator()
    return Orchestrator(registry, manager, evaluator), manager


@pytest.fixture
def harness_with_real_sops(tmp_path):
    """Set up harness with the built-in SOP templates."""
    project_sops = Path(__file__).parent.parent / "sops"
    registry = SOPRegistry(search_dirs=[project_sops])
    manager = SessionManager(base_dir=tmp_path / "sessions")
    evaluator = SubagentEvaluator()
    return Orchestrator(registry, manager, evaluator), manager


def _good_output(content="Detailed analysis with specific file paths and metrics."):
    return {
        "artifacts": [{"type": "text", "content": content}],
        "self_assessment": "All acceptance criteria addressed with project-specific details.",
    }


def _pass_eval():
    """Simulated reviewer response that passes."""
    return {
        dim: {"score": 4, "evidence": "Well done.", "gap": None}
        for dim in DIMENSIONS
    } | {"slop_flags": [], "top_3_fixes": []}


def _fail_eval(gap="Missing details"):
    """Simulated reviewer response that fails."""
    return {
        dim: {"score": 2, "evidence": "Insufficient.", "gap": gap}
        for dim in DIMENSIONS
    } | {"slop_flags": ["filler detected"], "top_3_fixes": ["Add specifics", "Remove filler"]}


# ------------------------------------------------------------------
# E2E: Complete happy path (feature-dev, all 12 steps)
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_e2e_complete_workflow(harness):
    """Walk through all 3 steps of the simple SOP with passing evaluations."""
    orch, manager = harness

    start = orch.start_session("e2e-test")
    assert start.success
    assert start.step_total == 3
    # Sequential enforcement: no steps_overview leaked
    assert "steps_overview" not in start.data
    session_id = start.session_id

    for step_num in range(3):
        resp = await orch.submit_step(session_id, _good_output(f"Step {step_num + 1} output"))
        assert resp.success
        assert resp.stage == "awaiting_evaluation"

        resp = orch.report_evaluation(session_id, _pass_eval())
        assert resp.success
        if step_num < 2:
            assert resp.stage == "awaiting_step"
            assert resp.step_index == step_num + 1
        else:
            assert resp.stage == "complete"

    status = orch.get_status(session_id)
    assert status.data["steps_completed"] == 3

    session_dir = manager._base / session_id
    assert (session_dir / "state.json").exists()
    assert (session_dir / "events.jsonl").exists()


# ------------------------------------------------------------------
# E2E: Fail → retry → pass
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_e2e_fail_retry_pass(harness):
    """First attempt fails, second attempt passes."""
    orch, _ = harness

    start = orch.start_session("e2e-test")
    session_id = start.session_id

    # Attempt 1: submit and fail
    await orch.submit_step(session_id, _good_output("Weak output"))
    resp = orch.report_evaluation(session_id, _fail_eval())
    assert "FAILED" in resp.message
    assert resp.step_index == 0
    assert "feedback" in resp.data

    # Attempt 2: submit and pass
    await orch.submit_step(session_id, _good_output("Improved output"))
    resp = orch.report_evaluation(session_id, _pass_eval())
    assert "PASSED" in resp.message
    assert resp.step_index == 1


# ------------------------------------------------------------------
# E2E: Exhaust retries → blocked → resume → pass
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_e2e_exhaust_retries_resume(harness):
    """Exhaust retries, get blocked, resume, then pass."""
    orch, _ = harness

    start = orch.start_session("e2e-test", retry_limit=2)
    session_id = start.session_id

    # Fail twice (exhausts retry_limit=2)
    for _ in range(2):
        await orch.submit_step(session_id, _good_output())
        orch.report_evaluation(session_id, _fail_eval())

    # Should be blocked now
    status = orch.get_status(session_id)
    assert status.stage == "blocked" or status.stage == "paused"

    # Resume
    resp = orch.resume_session(session_id, comment="Trying a different approach")
    assert resp.success
    assert resp.stage == "awaiting_step"

    # Now pass
    await orch.submit_step(session_id, _good_output("Better output"))
    resp = orch.report_evaluation(session_id, _pass_eval())
    assert "PASSED" in resp.message


# ------------------------------------------------------------------
# E2E: Skip steps
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_e2e_skip_and_complete(harness):
    """Skip a step, complete the rest."""
    orch, _ = harness

    start = orch.start_session("e2e-test")
    session_id = start.session_id

    # Do first step normally
    await orch.submit_step(session_id, _good_output())
    orch.report_evaluation(session_id, _pass_eval())

    # Skip step 2
    orch.skip_step(session_id, reason="Skipping for test")

    # Do last step
    await orch.submit_step(session_id, _good_output())
    resp = orch.report_evaluation(session_id, _pass_eval())

    assert resp.stage == "complete"


# ------------------------------------------------------------------
# E2E: Investigation SOP with strict profile
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_e2e_real_sops_load(harness_with_real_sops):
    """Verify all real SOP templates can start sessions."""
    orch, _ = harness_with_real_sops
    for sop_id in ["feature-dev", "investigation", "code-review"]:
        start = orch.start_session(sop_id)
        assert start.success, f"Failed to start {sop_id}: {start.message}"
        assert start.step_total >= 5
        # Sequential enforcement: no steps_overview
        assert "steps_overview" not in start.data


# ------------------------------------------------------------------
# E2E: Validation rejection doesn't consume retry
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_e2e_validation_preserves_retries(harness):
    """Validation failures should not count against retry budget."""
    orch, _ = harness

    start = orch.start_session("e2e-test", retry_limit=1)
    session_id = start.session_id

    # Submit invalid output (empty artifacts)
    bad = {"artifacts": [], "self_assessment": ""}
    resp = await orch.submit_step(session_id, bad)
    assert not resp.success
    assert "validation" in resp.message.lower()

    # Should still have full retry budget
    status = orch.get_status(session_id)
    assert status.data["current_step"]["retries_remaining"] == 1

    # Now submit valid output and pass
    await orch.submit_step(session_id, _good_output())
    resp = orch.report_evaluation(session_id, _pass_eval())
    assert "PASSED" in resp.message


# ------------------------------------------------------------------
# E2E: Feedback history accumulates
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_e2e_feedback_history(harness):
    """Feedback should accumulate across attempts."""
    orch, _ = harness

    start = orch.start_session("e2e-test")
    session_id = start.session_id

    # Attempt 1: fail
    await orch.submit_step(session_id, _good_output("First try"))
    orch.report_evaluation(session_id, _fail_eval("Missing X"))

    # Attempt 2: fail
    await orch.submit_step(session_id, _good_output("Second try"))
    orch.report_evaluation(session_id, _fail_eval("Missing Y"))

    # Check feedback history
    feedback = orch.get_feedback(session_id, step_index=0)
    assert len(feedback.data["attempts"]) == 2
    assert feedback.data["attempts"][0]["evaluation"]["verdict"] == "FAIL"
    assert feedback.data["attempts"][1]["evaluation"]["verdict"] == "FAIL"


# ------------------------------------------------------------------
# E2E: List operations
# ------------------------------------------------------------------

def test_e2e_list_sops(harness_with_real_sops):
    orch, _ = harness_with_real_sops
    sops = orch._sops.list_sops()
    sop_ids = {s["sop_id"] for s in sops}
    assert "feature-dev" in sop_ids
    assert "investigation" in sop_ids
    assert "code-review" in sop_ids


def test_e2e_list_sessions(harness):
    orch, _ = harness
    orch.start_session("e2e-test")
    orch.start_session("e2e-test")
    resp = orch.list_sessions()
    assert len(resp.data["sessions"]) == 2
