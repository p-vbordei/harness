"""Tests for the orchestrator (integration tests with mocked evaluator)."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from server.evaluator import DIMENSIONS, SubagentEvaluator, parse_evaluation_response
from server.models import DimensionScore, EvaluationResult, SessionStage, StepStatus
from server.orchestrator import Orchestrator
from server.session_manager import SessionManager
from server.sop_registry import SOPRegistry


SIMPLE_SOP = {
    "sop_id": "test-sop",
    "name": "Test SOP",
    "default_retry_limit": 2,
    "pass_threshold": 3.5,
    "phases": [
        {
            "id": "p1",
            "name": "Phase 1",
            "steps": [
                {
                    "id": "s1",
                    "title": "Step 1",
                    "instruction": "Do step 1",
                    "acceptance_criteria": ["Criterion A"],
                },
                {
                    "id": "s2",
                    "title": "Step 2",
                    "instruction": "Do step 2",
                    "acceptance_criteria": ["Criterion B"],
                    "depends_on": ["s1"],
                },
            ],
        },
    ],
}


def _pass_eval_json():
    return {
        dim: {"score": 4, "evidence": "Good", "gap": None}
        for dim in DIMENSIONS
    } | {"slop_flags": [], "top_3_fixes": []}


def _fail_eval_json():
    return {
        dim: {"score": 2, "evidence": "Bad", "gap": "Fix it"}
        for dim in DIMENSIONS
    } | {"slop_flags": ["filler"], "top_3_fixes": ["Fix X", "Fix Y"]}


VALID_OUTPUT = {
    "artifacts": [{"type": "text", "content": "My work output"}],
    "self_assessment": "I believe this meets the criteria.",
}


@pytest.fixture
def setup(tmp_path):
    """Set up orchestrator with test SOP and subagent evaluator."""
    sop_dir = tmp_path / "sops"
    sop_dir.mkdir()
    (sop_dir / "test.yaml").write_text(yaml.dump(SIMPLE_SOP))

    session_dir = tmp_path / "sessions"
    session_dir.mkdir()

    registry = SOPRegistry(search_dirs=[sop_dir])
    manager = SessionManager(base_dir=session_dir)
    evaluator = SubagentEvaluator()

    orch = Orchestrator(registry, manager, evaluator)
    return orch, manager


# ------------------------------------------------------------------
# Start session tests
# ------------------------------------------------------------------

def test_start_session(setup):
    orch, _ = setup
    resp = orch.start_session("test-sop")
    assert resp.success
    assert resp.stage == "awaiting_step"
    assert resp.step_index == 0
    assert resp.step_total == 2
    assert resp.session_id is not None
    assert resp.elicitation is not None
    assert "Step 1" in resp.elicitation["message"]


def test_start_session_unknown_sop(setup):
    orch, _ = setup
    resp = orch.start_session("nonexistent")
    assert not resp.success
    assert "not found" in resp.message


# ------------------------------------------------------------------
# Submit step tests (subagent mode)
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_submit_step_returns_evaluation_prompt(setup):
    orch, _ = setup
    start = orch.start_session("test-sop")
    session_id = start.session_id

    resp = await orch.submit_step(session_id, VALID_OUTPUT)
    assert resp.success
    assert resp.stage == "awaiting_evaluation"
    assert "evaluation_prompt" in resp.data
    assert "system_prompt" in resp.data["evaluation_prompt"]
    assert "user_prompt" in resp.data["evaluation_prompt"]


@pytest.mark.asyncio
async def test_validation_failure_doesnt_return_eval_prompt(setup):
    orch, _ = setup
    start = orch.start_session("test-sop")
    session_id = start.session_id

    bad_output = {"artifacts": [], "self_assessment": ""}
    resp = await orch.submit_step(session_id, bad_output)
    assert not resp.success
    assert "validation failed" in resp.message.lower()


# ------------------------------------------------------------------
# Report evaluation tests
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_report_evaluation_pass_advances(setup):
    orch, _ = setup
    start = orch.start_session("test-sop")
    session_id = start.session_id

    # Submit step (gets awaiting_evaluation)
    await orch.submit_step(session_id, VALID_OUTPUT)

    # Report passing evaluation
    resp = orch.report_evaluation(session_id, _pass_eval_json())
    assert resp.success
    assert "PASSED" in resp.message
    assert resp.step_index == 1  # Advanced to step 2


@pytest.mark.asyncio
async def test_report_evaluation_pass_completes_session(setup):
    orch, _ = setup
    start = orch.start_session("test-sop")
    session_id = start.session_id

    # Pass step 1
    await orch.submit_step(session_id, VALID_OUTPUT)
    orch.report_evaluation(session_id, _pass_eval_json())

    # Pass step 2
    await orch.submit_step(session_id, VALID_OUTPUT)
    resp = orch.report_evaluation(session_id, _pass_eval_json())
    assert resp.success
    assert resp.stage == "complete"


@pytest.mark.asyncio
async def test_report_evaluation_fail_allows_retry(setup):
    orch, _ = setup
    start = orch.start_session("test-sop")
    session_id = start.session_id

    await orch.submit_step(session_id, VALID_OUTPUT)
    resp = orch.report_evaluation(session_id, _fail_eval_json())
    assert resp.success  # Retry possible
    assert "FAILED" in resp.message
    assert resp.step_index == 0  # Still on step 1
    assert "feedback" in resp.data


@pytest.mark.asyncio
async def test_report_evaluation_exhausts_retries(setup):
    orch, _ = setup
    start = orch.start_session("test-sop")
    session_id = start.session_id

    # Attempt 1
    await orch.submit_step(session_id, VALID_OUTPUT)
    orch.report_evaluation(session_id, _fail_eval_json())

    # Attempt 2 (exhausts retry_limit of 2)
    await orch.submit_step(session_id, VALID_OUTPUT)
    resp = orch.report_evaluation(session_id, _fail_eval_json())
    assert not resp.success
    assert "Human review" in resp.message
    assert resp.stage == "blocked"


# ------------------------------------------------------------------
# Status and feedback tests
# ------------------------------------------------------------------

def test_get_status(setup):
    orch, _ = setup
    start = orch.start_session("test-sop")
    status = orch.get_status(start.session_id)
    assert status.success
    assert status.data["current_step"]["title"] == "Step 1"


def test_get_feedback(setup):
    orch, _ = setup
    start = orch.start_session("test-sop")
    feedback = orch.get_feedback(start.session_id)
    assert feedback.success
    assert feedback.data["attempts"] == []


def test_get_feedback_invalid_session(setup):
    orch, _ = setup
    resp = orch.get_feedback("550e8400-e29b-41d4-a716-446655440000")
    assert not resp.success
