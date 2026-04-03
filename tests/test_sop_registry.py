"""Tests for the SOP registry."""

import tempfile
from pathlib import Path

import pytest
import yaml

from server.models import StepStatus
from server.sop_registry import SOPRegistry


MINIMAL_SOP = {
    "sop_id": "test-sop",
    "name": "Test SOP",
    "description": "A test SOP",
    "default_retry_limit": 2,
    "pass_threshold": 3.5,
    "phases": [
        {
            "id": "phase1",
            "name": "Phase One",
            "steps": [
                {
                    "id": "step-a",
                    "title": "Step A",
                    "instruction": "Do step A",
                    "acceptance_criteria": ["Criterion 1"],
                },
                {
                    "id": "step-b",
                    "title": "Step B",
                    "instruction": "Do step B",
                    "depends_on": ["step-a"],
                },
            ],
        },
        {
            "id": "phase2",
            "name": "Phase Two",
            "steps": [
                {
                    "id": "step-c",
                    "title": "Step C",
                    "instruction": "Do step C",
                },
            ],
        },
    ],
}


@pytest.fixture
def sop_dir(tmp_path):
    """Create a temp directory with a valid SOP YAML."""
    sop_file = tmp_path / "test-sop.yaml"
    sop_file.write_text(yaml.dump(MINIMAL_SOP))
    return tmp_path


@pytest.fixture
def registry(sop_dir):
    return SOPRegistry(search_dirs=[sop_dir])


def test_list_sops(registry):
    sops = registry.list_sops()
    assert len(sops) == 1
    assert sops[0]["sop_id"] == "test-sop"


def test_get_sop(registry):
    sop = registry.get_sop("test-sop")
    assert sop.name == "Test SOP"
    assert len(sop.phases) == 2


def test_get_sop_not_found(registry):
    with pytest.raises(KeyError, match="not found"):
        registry.get_sop("nonexistent")


def test_get_step(registry):
    step = registry.get_step("test-sop", "phase1", "step-a")
    assert step.title == "Step A"


def test_flatten_steps_respects_dependencies(registry):
    flat = registry.flatten_steps("test-sop")
    ids = [s.id for s in flat]
    # step-a must come before step-b (dependency)
    assert ids.index("step-a") < ids.index("step-b")
    # phase1 steps before phase2 steps
    assert ids.index("step-b") < ids.index("step-c")


def test_build_step_states(registry):
    states = registry.build_step_states("test-sop")
    assert len(states) == 3
    assert all(s.status == StepStatus.PENDING for s in states)
    assert states[0].max_attempts == 2  # from default_retry_limit


def test_malformed_sop_missing_fields(tmp_path):
    bad_file = tmp_path / "bad.yaml"
    bad_file.write_text(yaml.dump({"sop_id": "bad"}))
    with pytest.raises(ValueError, match="missing required key"):
        SOPRegistry(search_dirs=[tmp_path])


def test_cycle_detection(tmp_path):
    cyclic_sop = {
        "sop_id": "cyclic",
        "name": "Cyclic",
        "phases": [{
            "id": "p1",
            "name": "P1",
            "steps": [
                {"id": "a", "title": "A", "instruction": "Do A", "depends_on": ["b"]},
                {"id": "b", "title": "B", "instruction": "Do B", "depends_on": ["a"]},
            ],
        }],
    }
    (tmp_path / "cyclic.yaml").write_text(yaml.dump(cyclic_sop))
    registry = SOPRegistry(search_dirs=[tmp_path])
    with pytest.raises(ValueError, match="Cycle"):
        registry.flatten_steps("cyclic")


def test_feature_dev_template_loads():
    """The built-in feature-dev.yaml template should load without errors."""
    project_sops = Path(__file__).parent.parent / "sops"
    registry = SOPRegistry(search_dirs=[project_sops])
    sop = registry.get_sop("feature-dev")
    assert sop.name == "Feature Development"
    flat = registry.flatten_steps("feature-dev")
    assert len(flat) == 12
