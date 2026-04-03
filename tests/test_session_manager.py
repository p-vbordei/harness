"""Tests for the session manager."""

import json

import pytest

from server.models import SessionStage, StepAttempt, StepState, StepStatus
from server.session_manager import SessionManager


@pytest.fixture
def manager(tmp_path):
    return SessionManager(base_dir=tmp_path)


@pytest.fixture
def sample_steps():
    return [
        StepState(step_id="step-a", phase_id="phase1", title="Step A"),
        StepState(step_id="step-b", phase_id="phase1", title="Step B"),
    ]


def test_create_and_load_session(manager, sample_steps):
    session = manager.create_session(
        session_id="test-123",
        sop_id="test-sop",
        sop_yaml_content="sop_id: test",
        steps=sample_steps,
    )
    assert session.session_id == "test-123"
    assert session.stage == SessionStage.INITIALIZED
    assert len(session.steps) == 2

    loaded = manager.load_session("test-123")
    assert loaded.session_id == "test-123"
    assert loaded.sop_id == "test-sop"
    assert len(loaded.steps) == 2


def test_save_session_updates(manager, sample_steps):
    session = manager.create_session("s1", "sop", "yaml", sample_steps)
    session.stage = SessionStage.RUNNING
    session.steps[0].status = StepStatus.IN_PROGRESS
    manager.save_session(session)

    loaded = manager.load_session("s1")
    assert loaded.stage == SessionStage.RUNNING
    assert loaded.steps[0].status == StepStatus.IN_PROGRESS


def test_save_attempt(manager, sample_steps):
    manager.create_session("s1", "sop", "yaml", sample_steps)

    attempt = StepAttempt(
        attempt_number=1,
        submitted_at="2024-01-01T00:00:00",
        artifacts=[{"type": "text", "content": "hello"}],
        self_assessment="Looks fine.",
        evaluation={"verdict": "PASS", "weighted_score": 4.0},
    )
    manager.save_attempt("s1", sample_steps[0], attempt)

    # Verify file exists
    step_dir = manager._base / "s1" / "steps" / "phase1.step-a"
    assert (step_dir / "attempt_1.json").exists()
    data = json.loads((step_dir / "attempt_1.json").read_text())
    assert data["evaluation"]["verdict"] == "PASS"


def test_log_event(manager, sample_steps):
    manager.create_session("s1", "sop", "yaml", sample_steps)
    manager.log_event("s1", "test_event", {"key": "value"})

    events_file = manager._base / "s1" / "events.jsonl"
    lines = events_file.read_text().strip().split("\n")
    # session_created + test_event
    assert len(lines) >= 2
    last = json.loads(lines[-1])
    assert last["event_type"] == "test_event"


def test_list_sessions(manager, sample_steps):
    manager.create_session("s1", "sop1", "yaml", sample_steps)
    manager.create_session("s2", "sop2", "yaml", sample_steps)

    sessions = manager.list_sessions()
    assert len(sessions) == 2
    ids = {s["session_id"] for s in sessions}
    assert ids == {"s1", "s2"}


def test_session_exists(manager, sample_steps):
    assert not manager.session_exists("nonexistent")
    manager.create_session("s1", "sop", "yaml", sample_steps)
    assert manager.session_exists("s1")


def test_load_nonexistent_raises(manager):
    with pytest.raises(FileNotFoundError):
        manager.load_session("nonexistent")
