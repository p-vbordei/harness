"""Tests for P2 features: evaluator profiles, resume/skip, new dimensions, usage tracking."""

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from server.evaluator import (
    DIMENSIONS,
    EVALUATOR_PROFILES,
    SubagentEvaluator,
    get_profile,
    parse_evaluation_response,
)
from server.models import EvaluationResult, SessionStage, StepStatus
from server.orchestrator import Orchestrator
from server.session_manager import SessionManager
from server.sop_registry import SOPRegistry
from server.usage_tracker import UsageTracker


# ------------------------------------------------------------------
# Evaluator profile tests
# ------------------------------------------------------------------

def test_default_profile_has_6_dimensions():
    profile = get_profile("default")
    assert len(profile["dimensions"]) == 6
    assert "coherence" in profile["dimensions"]


def test_strict_profile_higher_threshold():
    profile = get_profile("strict")
    assert profile["pass_threshold"] == 4.0
    assert profile["dimensions"]["correctness"] == 0.25  # Higher weight


def test_lenient_profile_lower_threshold():
    profile = get_profile("lenient")
    assert profile["pass_threshold"] == 3.0
    assert profile["min_dimension_score"] == 2


def test_unknown_profile_falls_back_to_default():
    profile = get_profile("nonexistent")
    assert profile == EVALUATOR_PROFILES["default"]


def test_parse_with_strict_profile():
    raw = {
        dim: {"score": 4, "evidence": "Good", "gap": None}
        for dim in EVALUATOR_PROFILES["strict"]["dimensions"]
    }
    raw["slop_flags"] = []
    raw["top_3_fixes"] = []
    result = parse_evaluation_response(raw, 1, 3, profile_name="strict")
    assert result.verdict == "PASS"  # 4.0 >= 4.0


def test_parse_with_strict_profile_borderline_fail():
    raw = {
        dim: {"score": 3, "evidence": "OK", "gap": "Minor"}
        for dim in EVALUATOR_PROFILES["strict"]["dimensions"]
    }
    raw["completeness"]["score"] = 5
    raw["slop_flags"] = []
    raw["top_3_fixes"] = []
    result = parse_evaluation_response(raw, 1, 3, profile_name="strict")
    assert result.verdict == "FAIL"  # Below 4.0 threshold


# ------------------------------------------------------------------
# Coherence dimension tests
# ------------------------------------------------------------------

def test_coherence_in_default_dimensions():
    assert "coherence" in DIMENSIONS
    assert DIMENSIONS["coherence"] == 0.10


def test_parse_includes_coherence():
    raw = {
        dim: {"score": 4, "evidence": "Good", "gap": None}
        for dim in DIMENSIONS
    }
    raw["slop_flags"] = []
    raw["top_3_fixes"] = []
    result = parse_evaluation_response(raw, 1, 3)
    assert "coherence" in result.dimensions


# ------------------------------------------------------------------
# Slop penalty tests
# ------------------------------------------------------------------

def test_slop_penalty_reduces_specificity():
    raw = {
        dim: {"score": 4, "evidence": "Good", "gap": None}
        for dim in DIMENSIONS
    }
    raw["slop_flags"] = ["filler 1", "filler 2", "filler 3"]  # 3+ triggers penalty
    raw["top_3_fixes"] = []
    result = parse_evaluation_response(raw, 1, 3, profile_name="default")
    assert result.dimensions["specificity"].score == 3  # 4 - 1 = 3
    assert "Slop penalty" in result.dimensions["specificity"].evidence


def test_slop_penalty_not_applied_under_threshold():
    raw = {
        dim: {"score": 4, "evidence": "Good", "gap": None}
        for dim in DIMENSIONS
    }
    raw["slop_flags"] = ["filler 1", "filler 2"]  # Only 2, threshold is 3
    raw["top_3_fixes"] = []
    result = parse_evaluation_response(raw, 1, 3)
    assert result.dimensions["specificity"].score == 4  # No penalty


def test_slop_penalty_disabled_in_lenient_profile():
    raw = {
        dim: {"score": 4, "evidence": "Good", "gap": None}
        for dim in EVALUATOR_PROFILES["lenient"]["dimensions"]
    }
    raw["slop_flags"] = ["a", "b", "c", "d"]  # Many flags
    raw["top_3_fixes"] = []
    result = parse_evaluation_response(raw, 1, 3, profile_name="lenient")
    assert result.dimensions["specificity"].score == 4  # No penalty in lenient


# ------------------------------------------------------------------
# SOP schema: evaluator_profile + timeout
# ------------------------------------------------------------------

def test_sop_with_evaluator_profile(tmp_path):
    sop = {
        "sop_id": "test",
        "name": "Test",
        "phases": [{
            "id": "p1",
            "name": "P1",
            "steps": [{
                "id": "s1",
                "title": "S1",
                "instruction": "Do it",
                "evaluator_profile": "strict",
                "timeout": 600,
            }],
        }],
    }
    (tmp_path / "test.yaml").write_text(yaml.dump(sop))
    registry = SOPRegistry(search_dirs=[tmp_path])
    step = registry.get_step("test", "p1", "s1")
    assert step.evaluator_profile == "strict"
    assert step.timeout == 600


def test_sop_invalid_evaluator_profile(tmp_path):
    sop = {
        "sop_id": "bad",
        "name": "Bad",
        "phases": [{
            "id": "p1",
            "name": "P1",
            "steps": [{"id": "s1", "title": "S1", "instruction": "X", "evaluator_profile": "ultra"}],
        }],
    }
    (tmp_path / "bad.yaml").write_text(yaml.dump(sop))
    with pytest.raises(ValueError, match="invalid evaluator_profile"):
        SOPRegistry(search_dirs=[tmp_path])


def test_sop_invalid_timeout(tmp_path):
    sop = {
        "sop_id": "bad",
        "name": "Bad",
        "phases": [{
            "id": "p1",
            "name": "P1",
            "steps": [{"id": "s1", "title": "S1", "instruction": "X", "timeout": -1}],
        }],
    }
    (tmp_path / "bad.yaml").write_text(yaml.dump(sop))
    with pytest.raises(ValueError, match="invalid timeout"):
        SOPRegistry(search_dirs=[tmp_path])


# ------------------------------------------------------------------
# SOP templates load correctly
# ------------------------------------------------------------------

def test_investigation_template_loads():
    project_sops = Path(__file__).parent.parent / "sops"
    registry = SOPRegistry(search_dirs=[project_sops])
    sop = registry.get_sop("investigation")
    assert sop.name
    flat = registry.flatten_steps("investigation")
    assert len(flat) >= 5


def test_code_review_template_loads():
    project_sops = Path(__file__).parent.parent / "sops"
    registry = SOPRegistry(search_dirs=[project_sops])
    sop = registry.get_sop("code-review")
    assert sop.name
    flat = registry.flatten_steps("code-review")
    assert len(flat) >= 5


# ------------------------------------------------------------------
# Resume / Skip tests
# ------------------------------------------------------------------

SOP_DATA = {
    "sop_id": "test",
    "name": "Test",
    "default_retry_limit": 2,
    "phases": [{
        "id": "p1",
        "name": "P1",
        "steps": [
            {"id": "s1", "title": "Step 1", "instruction": "Do 1", "acceptance_criteria": ["C1"]},
            {"id": "s2", "title": "Step 2", "instruction": "Do 2", "depends_on": ["s1"]},
        ],
    }],
}

VALID_OUTPUT = {
    "artifacts": [{"type": "text", "content": "Work"}],
    "self_assessment": "Done.",
}

FAIL_EVAL = {
    dim: {"score": 2, "evidence": "Bad", "gap": "Fix"}
    for dim in DIMENSIONS
} | {"slop_flags": [], "top_3_fixes": ["Fix X"]}


@pytest.fixture
def orch(tmp_path):
    sop_dir = tmp_path / "sops"
    sop_dir.mkdir()
    (sop_dir / "test.yaml").write_text(yaml.dump(SOP_DATA))
    return Orchestrator(
        SOPRegistry(search_dirs=[sop_dir]),
        SessionManager(base_dir=tmp_path / "sessions"),
        SubagentEvaluator(),
    )


def test_skip_step(orch):
    start = orch.start_session("test")
    resp = orch.skip_step(start.session_id, reason="Not relevant")
    assert resp.success
    assert resp.step_index == 1  # Advanced
    assert "skipped" in resp.message.lower()


def test_skip_step_completes_session(orch):
    start = orch.start_session("test")
    orch.skip_step(start.session_id)
    resp = orch.skip_step(start.session_id)
    assert resp.stage == "complete"


@pytest.mark.asyncio
async def test_resume_blocked_session(orch):
    start = orch.start_session("test")
    sid = start.session_id

    # Exhaust retries to get blocked
    await orch.submit_step(sid, VALID_OUTPUT)
    orch.report_evaluation(sid, FAIL_EVAL)
    await orch.submit_step(sid, VALID_OUTPUT)
    blocked = orch.report_evaluation(sid, FAIL_EVAL)
    assert blocked.stage == "blocked"

    # Resume
    resp = orch.resume_session(sid, comment="Trying again")
    assert resp.success
    assert resp.stage == "awaiting_step"


def test_resume_running_session_fails(orch):
    start = orch.start_session("test")
    resp = orch.resume_session(start.session_id)
    assert not resp.success
    assert "not paused" in resp.message.lower()


def test_list_sessions(orch):
    orch.start_session("test")
    orch.start_session("test")
    resp = orch.list_sessions()
    assert resp.success
    assert len(resp.data["sessions"]) == 2


# ------------------------------------------------------------------
# Usage tracker tests
# ------------------------------------------------------------------

def test_usage_tracker(tmp_path):
    tracker = UsageTracker(tmp_path)
    tracker.start_session("s1", "test-sop")
    tracker.start_step("s1", "step-a")
    tracker.record_attempt("s1", "step-a")
    tracker.record_evaluation("s1", "step-a", "FAIL")
    tracker.record_attempt("s1", "step-a")
    tracker.record_evaluation("s1", "step-a", "PASS")

    usage = tracker.get_usage("s1")
    assert usage["total_evaluations"] == 2
    assert usage["total_passes"] == 1
    assert usage["total_fails"] == 1
    assert usage["pass_rate"] == 0.5
    assert usage["steps"]["step-a"]["attempts"] == 2


def test_usage_tracker_skip(tmp_path):
    tracker = UsageTracker(tmp_path)
    tracker.start_session("s1", "test-sop")
    tracker.record_skip("s1", "step-a")
    usage = tracker.get_usage("s1")
    assert usage["total_skips"] == 1


# ------------------------------------------------------------------
# Empty SOP validation
# ------------------------------------------------------------------

def test_empty_phases_rejected(tmp_path):
    sop = {"sop_id": "empty", "name": "Empty", "phases": []}
    (tmp_path / "empty.yaml").write_text(yaml.dump(sop))
    with pytest.raises(ValueError, match="no phases"):
        SOPRegistry(search_dirs=[tmp_path])


def test_empty_steps_rejected(tmp_path):
    sop = {
        "sop_id": "empty",
        "name": "Empty",
        "phases": [{"id": "p1", "name": "P1", "steps": []}],
    }
    (tmp_path / "empty.yaml").write_text(yaml.dump(sop))
    with pytest.raises(ValueError, match="no steps"):
        SOPRegistry(search_dirs=[tmp_path])
