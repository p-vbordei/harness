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


def _fail_eval_json(scores: dict | None = None):
    """Failing evaluation. `scores` overrides per-dimension scores (default 2)."""
    dim_scores = {dim: 2 for dim in DIMENSIONS}
    if scores:
        dim_scores.update(scores)
    return {
        dim: {"score": dim_scores[dim], "evidence": "Bad", "gap": "Fix it"}
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
# Stall-aware early escalation tests
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_report_evaluation_escalates_early_when_stalled(setup):
    """A flat score across attempts escalates before retries are exhausted.

    With retry_limit=3 the second failure would normally leave one retry. But
    if the score did not improve (executor is not benefiting from feedback),
    burning the last attempt is wasted work -- escalate to a human instead.
    """
    orch, _ = setup
    start = orch.start_session("test-sop", retry_limit=3)
    session_id = start.session_id

    # Attempt 1 -- fails at 2.0
    await orch.submit_step(session_id, VALID_OUTPUT)
    orch.report_evaluation(session_id, _fail_eval_json())

    # Attempt 2 -- fails at 2.0 again (no improvement); one retry still remains
    await orch.submit_step(session_id, VALID_OUTPUT)
    resp = orch.report_evaluation(session_id, _fail_eval_json())

    assert not resp.success
    assert resp.stage == "blocked"
    assert "stall" in resp.message.lower()
    assert resp.data.get("escalation") is True


@pytest.mark.asyncio
async def test_sop_stall_epsilon_overrides_orchestrator_default(tmp_path):
    """A SOP-level stall_epsilon overrides the orchestrator's default epsilon."""
    sop = {
        "sop_id": "strict-stall",
        "name": "Strict Stall SOP",
        "default_retry_limit": 3,
        "stall_epsilon": 1.0,  # demand a full point of improvement per attempt
        "phases": [{
            "id": "p1", "name": "P1", "steps": [
                {"id": "s1", "title": "Step 1", "instruction": "Do it",
                 "acceptance_criteria": ["Criterion A"]},
            ],
        }],
    }
    sop_dir = tmp_path / "sops"
    sop_dir.mkdir()
    (sop_dir / "strict.yaml").write_text(yaml.dump(sop))
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    orch = Orchestrator(
        SOPRegistry(search_dirs=[sop_dir]),
        SessionManager(base_dir=session_dir),
        SubagentEvaluator(),
    )

    start = orch.start_session("strict-stall")
    session_id = start.session_id

    # Attempt 1 -- 2.0
    await orch.submit_step(session_id, VALID_OUTPUT)
    orch.report_evaluation(session_id, _fail_eval_json())

    # Attempt 2 -- improved to ~2.45 (delta 0.45). Under the default 0.3 this
    # would keep retrying; this SOP demands >= 1.0 improvement, so it stalls.
    await orch.submit_step(session_id, VALID_OUTPUT)
    resp = orch.report_evaluation(
        session_id, _fail_eval_json({"completeness": 3, "specificity": 3})
    )

    assert not resp.success
    assert resp.stage == "blocked"
    assert "stall" in resp.message.lower()


@pytest.mark.asyncio
async def test_report_evaluation_keeps_retrying_when_improving(setup):
    """An improving score keeps its retry -- stall detection must not over-fire."""
    orch, _ = setup
    start = orch.start_session("test-sop", retry_limit=3)
    session_id = start.session_id

    # Attempt 1 -- 2.0
    await orch.submit_step(session_id, VALID_OUTPUT)
    orch.report_evaluation(session_id, _fail_eval_json())

    # Attempt 2 -- improved to ~2.45 (delta >= epsilon), still failing
    await orch.submit_step(session_id, VALID_OUTPUT)
    resp = orch.report_evaluation(
        session_id, _fail_eval_json({"completeness": 3, "specificity": 3})
    )

    assert resp.success  # retry still offered
    assert resp.stage == "awaiting_step"
    assert "retries remaining" in resp.message.lower()


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
