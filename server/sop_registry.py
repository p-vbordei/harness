"""Load, validate, and query YAML SOP definitions."""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from server.models import StepState, StepStatus

# ---------------------------------------------------------------------------
# SOP dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SOPStep:
    """A single step inside a phase."""
    id: str
    title: str
    instruction: str
    acceptance_criteria: list[str] = field(default_factory=list)
    output_schema: Optional[dict[str, Any]] = None
    depends_on: list[str] = field(default_factory=list)
    on_fail: str = "retry"  # retry | skip | abort
    evaluator_profile: str = "default"  # default | strict | lenient
    timeout: Optional[int] = None  # seconds, None = no timeout


@dataclass
class SOPPhase:
    """An ordered phase containing steps."""
    id: str
    name: str
    steps: list[SOPStep] = field(default_factory=list)


@dataclass
class SOPDefinition:
    """Top-level SOP document."""
    sop_id: str
    name: str
    description: str = ""
    default_retry_limit: int = 3
    pass_threshold: float = 3.5
    phases: list[SOPPhase] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REQUIRED_STEP_FIELDS = {"id", "title", "instruction"}
_VALID_ON_FAIL = {"retry", "skip", "abort"}


def _validate_step(raw: dict, phase_id: str) -> SOPStep:
    missing = _REQUIRED_STEP_FIELDS - raw.keys()
    if missing:
        raise ValueError(
            f"Step in phase '{phase_id}' missing required fields: {missing}"
        )
    on_fail = raw.get("on_fail", "retry")
    if on_fail not in _VALID_ON_FAIL:
        raise ValueError(
            f"Step '{raw['id']}' has invalid on_fail '{on_fail}'; "
            f"expected one of {_VALID_ON_FAIL}"
        )
    evaluator_profile = raw.get("evaluator_profile", "default")
    if evaluator_profile not in ("default", "strict", "lenient"):
        raise ValueError(
            f"Step '{raw['id']}' has invalid evaluator_profile '{evaluator_profile}'; "
            f"expected one of: default, strict, lenient"
        )
    timeout = raw.get("timeout")
    if timeout is not None:
        timeout = int(timeout)
        if timeout <= 0:
            raise ValueError(f"Step '{raw['id']}' has invalid timeout: {timeout}")

    return SOPStep(
        id=raw["id"],
        title=raw["title"],
        instruction=raw["instruction"],
        acceptance_criteria=raw.get("acceptance_criteria", []),
        output_schema=raw.get("output_schema"),
        depends_on=raw.get("depends_on", []),
        on_fail=on_fail,
        evaluator_profile=evaluator_profile,
        timeout=timeout,
    )


def _validate_phase(raw: dict) -> SOPPhase:
    if "id" not in raw or "name" not in raw:
        raise ValueError(f"Phase missing 'id' or 'name': {raw}")
    steps = [_validate_step(s, raw["id"]) for s in raw.get("steps", [])]
    return SOPPhase(id=raw["id"], name=raw["name"], steps=steps)


def _validate_sop(raw: dict, source: str) -> SOPDefinition:
    for key in ("sop_id", "name", "phases"):
        if key not in raw:
            raise ValueError(f"SOP file '{source}' missing required key '{key}'")
    if not raw["phases"]:
        raise ValueError(f"SOP file '{source}' has no phases defined")
    phases = [_validate_phase(p) for p in raw["phases"]]
    total_steps = sum(len(p.steps) for p in phases)
    if total_steps == 0:
        raise ValueError(f"SOP file '{source}' has no steps defined in any phase")
    return SOPDefinition(
        sop_id=raw["sop_id"],
        name=raw["name"],
        description=raw.get("description", ""),
        default_retry_limit=raw.get("default_retry_limit", 3),
        pass_threshold=raw.get("pass_threshold", 3.5),
        phases=phases,
    )


def _topological_sort_steps(steps: list[SOPStep]) -> list[SOPStep]:
    """Topological sort of steps within a phase based on depends_on."""
    by_id: dict[str, SOPStep] = {s.id: s for s in steps}
    in_degree: dict[str, int] = {s.id: 0 for s in steps}
    dependents: dict[str, list[str]] = {s.id: [] for s in steps}

    for step in steps:
        for dep in step.depends_on:
            if dep not in by_id:
                raise ValueError(
                    f"Step '{step.id}' depends on unknown step '{dep}'"
                )
            in_degree[step.id] += 1
            dependents[dep].append(step.id)

    queue: deque[str] = deque(
        sid for sid, deg in in_degree.items() if deg == 0
    )
    ordered: list[SOPStep] = []

    while queue:
        sid = queue.popleft()
        ordered.append(by_id[sid])
        for child in dependents[sid]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    if len(ordered) != len(steps):
        raise ValueError("Cycle detected in step dependencies")

    return ordered


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DIRS = [
    _PROJECT_ROOT / "sops",
    Path.home() / ".harness" / "sops",
]


class SOPRegistry:
    """Loads, validates, and indexes SOP definitions from YAML files."""

    def __init__(self, search_dirs: Optional[list[str | Path]] = None) -> None:
        self._sops: dict[str, SOPDefinition] = {}
        dirs = [Path(d) for d in search_dirs] if search_dirs else _DEFAULT_DIRS
        self._search_dirs = dirs
        self._load_all()

    # -- loading -----------------------------------------------------------

    def _load_all(self) -> None:
        for directory in self._search_dirs:
            if not directory.is_dir():
                continue
            for path in sorted(directory.glob("*.yaml")):
                self._load_file(path)
            for path in sorted(directory.glob("*.yml")):
                self._load_file(path)

    def _load_file(self, path: Path) -> None:
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        if not isinstance(raw, dict):
            raise ValueError(f"Expected a YAML mapping in '{path}'")
        sop = _validate_sop(raw, str(path))
        self._sops[sop.sop_id] = sop

    def reload(self) -> None:
        """Re-scan directories and reload all SOPs."""
        self._sops.clear()
        self._load_all()

    # -- queries -----------------------------------------------------------

    def list_sops(self) -> list[dict[str, str]]:
        """Return a lightweight list of available SOPs."""
        return [
            {"sop_id": s.sop_id, "name": s.name, "description": s.description}
            for s in self._sops.values()
        ]

    def get_sop(self, sop_id: str) -> SOPDefinition:
        """Retrieve a full SOP definition by id."""
        try:
            return self._sops[sop_id]
        except KeyError:
            raise KeyError(f"SOP '{sop_id}' not found") from None

    def get_step(self, sop_id: str, phase_id: str, step_id: str) -> SOPStep:
        """Look up a single step by sop / phase / step ids."""
        sop = self.get_sop(sop_id)
        for phase in sop.phases:
            if phase.id == phase_id:
                for step in phase.steps:
                    if step.id == step_id:
                        return step
                raise KeyError(
                    f"Step '{step_id}' not found in phase '{phase_id}'"
                )
        raise KeyError(f"Phase '{phase_id}' not found in SOP '{sop_id}'")

    # -- flattening --------------------------------------------------------

    def flatten_steps(self, sop_id: str) -> list[SOPStep]:
        """Return all steps in execution order.

        Phases are sequential; within each phase steps are topologically
        sorted according to their ``depends_on`` declarations.
        """
        sop = self.get_sop(sop_id)
        ordered: list[SOPStep] = []
        for phase in sop.phases:
            ordered.extend(_topological_sort_steps(phase.steps))
        return ordered

    def build_step_states(self, sop_id: str) -> list[StepState]:
        """Build a list of ``StepState`` objects ready for a new session."""
        sop = self.get_sop(sop_id)
        states: list[StepState] = []
        for phase in sop.phases:
            for step in _topological_sort_steps(phase.steps):
                states.append(
                    StepState(
                        step_id=step.id,
                        phase_id=phase.id,
                        title=step.title,
                        status=StepStatus.PENDING,
                        max_attempts=sop.default_retry_limit,
                    )
                )
        return states
