"""Per-session usage tracking for harness workflows.

Tracks evaluation counts, token estimates, step durations, and pass/fail rates.
Persists to usage.json in the session directory.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class StepUsage:
    """Usage data for a single step."""
    step_id: str
    attempts: int = 0
    evaluations: int = 0
    passes: int = 0
    fails: int = 0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: float = 0.0


@dataclass
class SessionUsage:
    """Aggregate usage data for a session."""
    session_id: str
    sop_id: str
    total_evaluations: int = 0
    total_passes: int = 0
    total_fails: int = 0
    total_skips: int = 0
    steps: dict[str, StepUsage] = field(default_factory=dict)
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "sop_id": self.sop_id,
            "total_evaluations": self.total_evaluations,
            "total_passes": self.total_passes,
            "total_fails": self.total_fails,
            "total_skips": self.total_skips,
            "pass_rate": (
                round(self.total_passes / self.total_evaluations, 2)
                if self.total_evaluations > 0 else 0.0
            ),
            "started_at": self.started_at,
            "last_updated": self.last_updated,
            "steps": {
                k: {
                    "step_id": v.step_id,
                    "attempts": v.attempts,
                    "evaluations": v.evaluations,
                    "passes": v.passes,
                    "fails": v.fails,
                    "duration_seconds": v.duration_seconds,
                }
                for k, v in self.steps.items()
            },
        }


class UsageTracker:
    """Tracks and persists per-session usage statistics."""

    def __init__(self, sessions_base_dir: Path) -> None:
        self._base = sessions_base_dir
        self._sessions: dict[str, SessionUsage] = {}
        self._step_timers: dict[str, float] = {}  # step_key -> start time

    def start_session(self, session_id: str, sop_id: str) -> None:
        self._sessions[session_id] = SessionUsage(
            session_id=session_id, sop_id=sop_id,
        )

    def start_step(self, session_id: str, step_id: str) -> None:
        usage = self._sessions.get(session_id)
        if not usage:
            return
        if step_id not in usage.steps:
            usage.steps[step_id] = StepUsage(step_id=step_id)
        usage.steps[step_id].started_at = datetime.now().isoformat()
        self._step_timers[f"{session_id}:{step_id}"] = time.monotonic()

    def record_attempt(self, session_id: str, step_id: str) -> None:
        usage = self._sessions.get(session_id)
        if not usage:
            return
        if step_id not in usage.steps:
            usage.steps[step_id] = StepUsage(step_id=step_id)
        usage.steps[step_id].attempts += 1

    def record_evaluation(
        self, session_id: str, step_id: str, verdict: str
    ) -> None:
        usage = self._sessions.get(session_id)
        if not usage:
            return
        if step_id not in usage.steps:
            usage.steps[step_id] = StepUsage(step_id=step_id)

        step = usage.steps[step_id]
        step.evaluations += 1
        usage.total_evaluations += 1

        if verdict == "PASS":
            step.passes += 1
            usage.total_passes += 1
            step.completed_at = datetime.now().isoformat()
            timer_key = f"{session_id}:{step_id}"
            if timer_key in self._step_timers:
                step.duration_seconds = round(
                    time.monotonic() - self._step_timers.pop(timer_key), 2
                )
        else:
            step.fails += 1
            usage.total_fails += 1

        usage.last_updated = datetime.now().isoformat()
        self._save(session_id)

    def record_skip(self, session_id: str, step_id: str) -> None:
        usage = self._sessions.get(session_id)
        if not usage:
            return
        usage.total_skips += 1
        if step_id in usage.steps:
            usage.steps[step_id].completed_at = datetime.now().isoformat()
        usage.last_updated = datetime.now().isoformat()
        self._save(session_id)

    def get_usage(self, session_id: str) -> Optional[dict]:
        usage = self._sessions.get(session_id)
        return usage.to_dict() if usage else None

    def _save(self, session_id: str) -> None:
        usage = self._sessions.get(session_id)
        if not usage:
            return
        usage_path = self._base / session_id / "usage.json"
        if usage_path.parent.exists():
            try:
                usage_path.write_text(
                    json.dumps(usage.to_dict(), indent=2), encoding="utf-8"
                )
            except OSError:
                pass  # Best-effort persistence
