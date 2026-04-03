"""Core data models for the Harness system."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class SessionStage(Enum):
    """Overall session lifecycle."""
    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class StepStatus(Enum):
    """Individual step lifecycle."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUBMITTED = "submitted"
    EVALUATING = "evaluating"
    PASSED = "passed"
    FAILED = "failed"
    BLOCKED = "blocked"  # Max retries hit
    SKIPPED = "skipped"


@dataclass
class Artifact:
    """A single artifact submitted for a step."""
    type: str  # "file_path", "code_block", "text", "json_object"
    content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class DimensionScore:
    """Score for a single evaluation dimension."""
    score: int  # 1-5
    evidence: str
    gap: Optional[str] = None


@dataclass
class EvaluationResult:
    """Result from the quality gate evaluator."""
    verdict: str  # "PASS" or "FAIL"
    weighted_score: float
    dimensions: dict[str, DimensionScore] = field(default_factory=dict)
    slop_flags: list[str] = field(default_factory=list)
    top_3_fixes: list[str] = field(default_factory=list)
    attempt: int = 1
    max_attempts: int = 3

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "weighted_score": self.weighted_score,
            "dimensions": {
                k: {"score": v.score, "evidence": v.evidence, "gap": v.gap}
                for k, v in self.dimensions.items()
            },
            "slop_flags": self.slop_flags,
            "top_3_fixes": self.top_3_fixes,
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
        }


@dataclass
class StepAttempt:
    """A single submission attempt for a step."""
    attempt_number: int
    submitted_at: str
    artifacts: list[dict]
    self_assessment: str
    evaluation: Optional[dict] = None


@dataclass
class StepState:
    """State of a single step in the workflow."""
    step_id: str
    phase_id: str
    title: str
    status: StepStatus = StepStatus.PENDING
    attempts: list[StepAttempt] = field(default_factory=list)
    current_attempt: int = 0
    max_attempts: int = 3

    @property
    def retries_remaining(self) -> int:
        return max(0, self.max_attempts - self.current_attempt)


@dataclass
class SessionState:
    """Complete state of a harness session."""
    session_id: str
    sop_id: str
    stage: SessionStage = SessionStage.INITIALIZED
    step_index: int = 0
    steps: list[StepState] = field(default_factory=list)
    context: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def step_total(self) -> int:
        return len(self.steps)

    @property
    def current_step(self) -> Optional[StepState]:
        if 0 <= self.step_index < len(self.steps):
            return self.steps[self.step_index]
        return None

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "sop_id": self.sop_id,
            "stage": self.stage.value,
            "step_index": self.step_index,
            "step_total": self.step_total,
            "context": self.context,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "steps": [
                {
                    "step_id": s.step_id,
                    "phase_id": s.phase_id,
                    "title": s.title,
                    "status": s.status.value,
                    "current_attempt": s.current_attempt,
                    "max_attempts": s.max_attempts,
                    "retries_remaining": s.retries_remaining,
                }
                for s in self.steps
            ],
        }


@dataclass
class HarnessResponse:
    """Standard response format for all harness MCP tools."""
    success: bool
    message: str
    session_id: Optional[str] = None
    stage: Optional[str] = None
    step_index: Optional[int] = None
    step_total: Optional[int] = None
    data: dict = field(default_factory=dict)
    elicitation: Optional[dict] = None

    def to_dict(self) -> dict:
        result = {
            "success": self.success,
            "message": self.message,
            "data": self.data,
        }
        if self.session_id is not None:
            result["session_id"] = self.session_id
        if self.stage is not None:
            result["stage"] = self.stage
        if self.step_index is not None:
            result["step_index"] = self.step_index
        if self.step_total is not None:
            result["step_total"] = self.step_total
        if self.elicitation is not None:
            result["elicitation"] = self.elicitation
        return result


def generate_session_id() -> str:
    """Generate a new UUID session ID."""
    return str(uuid.uuid4())
