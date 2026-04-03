"""Filesystem-based session state persistence.

Stores all session data under ~/.harness/sessions/{session_id}/ with
JSON serialization and append-only event logging.
"""

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

from server.models import (
    SessionState,
    SessionStage,
    StepAttempt,
    StepState,
    StepStatus,
)

BASE_DIR = Path.home() / ".harness" / "sessions"
MAX_SESSIONS = 100  # Maximum concurrent sessions to prevent filesystem DOS


def _step_dir_name(phase_id: str, step_id: str) -> str:
    """Build the directory name for a step: ``{phase_id}.{step_id}``."""
    return f"{phase_id}.{step_id}"


def _session_state_from_dict(data: dict) -> SessionState:
    """Reconstruct a SessionState from its serialized dict."""
    steps: list[StepState] = []
    for s in data.get("steps", []):
        attempts = [
            StepAttempt(
                attempt_number=a["attempt_number"],
                submitted_at=a["submitted_at"],
                artifacts=a["artifacts"],
                self_assessment=a["self_assessment"],
                evaluation=a.get("evaluation"),
            )
            for a in s.get("attempts", [])
        ]
        steps.append(
            StepState(
                step_id=s["step_id"],
                phase_id=s["phase_id"],
                title=s["title"],
                status=StepStatus(s["status"]),
                attempts=attempts,
                current_attempt=s.get("current_attempt", 0),
                max_attempts=s.get("max_attempts", 3),
            )
        )

    return SessionState(
        session_id=data["session_id"],
        sop_id=data["sop_id"],
        stage=SessionStage(data["stage"]),
        step_index=data.get("step_index", 0),
        steps=steps,
        context=data.get("context", {}),
        created_at=data.get("created_at", ""),
        updated_at=data.get("updated_at", ""),
    )


def _serialize_step_state(step: StepState) -> dict:
    """Serialize a StepState including its attempts."""
    return {
        "step_id": step.step_id,
        "phase_id": step.phase_id,
        "title": step.title,
        "status": step.status.value,
        "current_attempt": step.current_attempt,
        "max_attempts": step.max_attempts,
        "attempts": [
            {
                "attempt_number": a.attempt_number,
                "submitted_at": a.submitted_at,
                "artifacts": a.artifacts,
                "self_assessment": a.self_assessment,
                "evaluation": a.evaluation,
            }
            for a in step.attempts
        ],
    }


def _full_state_dict(state: SessionState) -> dict:
    """Produce a fully serializable dict for SessionState (includes attempts)."""
    return {
        "session_id": state.session_id,
        "sop_id": state.sop_id,
        "stage": state.stage.value,
        "step_index": state.step_index,
        "context": state.context,
        "created_at": state.created_at,
        "updated_at": state.updated_at,
        "steps": [_serialize_step_state(s) for s in state.steps],
    }


class SessionManager:
    """Thread-safe, filesystem-backed session persistence."""

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self._base = base_dir or BASE_DIR
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _session_dir(self, session_id: str) -> Path:
        return self._base / session_id

    def _state_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "state.json"

    def _sop_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "sop_snapshot.yaml"

    def _events_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "events.jsonl"

    def _steps_dir(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "steps"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def session_exists(self, session_id: str) -> bool:
        """Return True if the session directory and state file exist."""
        return self._state_path(session_id).is_file()

    def create_session(
        self,
        session_id: str,
        sop_id: str,
        sop_yaml_content: str,
        steps: list[StepState],
    ) -> SessionState:
        """Create a new session on disk and return its initial state.

        Raises RuntimeError if the session quota is exceeded.
        """
        with self._lock:
            # Enforce session quota
            if self._base.is_dir():
                existing = sum(1 for c in self._base.iterdir() if c.is_dir())
                if existing >= MAX_SESSIONS:
                    raise RuntimeError(
                        f"Session quota exceeded ({MAX_SESSIONS}). "
                        "Delete old sessions before creating new ones."
                    )
            session_dir = self._session_dir(session_id)
            session_dir.mkdir(parents=True, exist_ok=True)
            self._steps_dir(session_id).mkdir(parents=True, exist_ok=True)

            # Freeze the SOP
            self._sop_path(session_id).write_text(sop_yaml_content, encoding="utf-8")

            now = datetime.now().isoformat()
            state = SessionState(
                session_id=session_id,
                sop_id=sop_id,
                stage=SessionStage.INITIALIZED,
                step_index=0,
                steps=steps,
                context={},
                created_at=now,
                updated_at=now,
            )

            self._write_state(state)
            self._append_event(session_id, "session_created", {
                "sop_id": sop_id,
                "step_count": len(steps),
            })

        return state

    def load_session(self, session_id: str) -> SessionState:
        """Load and return the SessionState from disk.

        If state.json is corrupted (JSONDecodeError), attempts to recover the
        session by replaying events from events.jsonl.

        Raises FileNotFoundError if the session does not exist and cannot be
        recovered.
        """
        with self._lock:
            path = self._state_path(session_id)
            if not path.is_file():
                raise FileNotFoundError(f"Session not found: {session_id}")
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                state = _session_state_from_dict(data)
            except json.JSONDecodeError:
                logger.warning(
                    "state.json corrupted for session %s; "
                    "attempting recovery from events",
                    session_id,
                )
                state = self._recover_session_from_events_unlocked(session_id)
                # Persist the recovered state so subsequent loads are fast
                self._write_state(state)
            return state

    def save_session(self, session_state: SessionState) -> None:
        """Persist the current session state to disk."""
        with self._lock:
            session_state.updated_at = datetime.now().isoformat()
            self._write_state(session_state)

    def save_attempt(
        self,
        session_id: str,
        step: StepState,
        attempt: StepAttempt,
    ) -> None:
        """Save a single attempt as a standalone JSON file."""
        with self._lock:
            dir_name = _step_dir_name(step.phase_id, step.step_id)
            attempt_dir = self._steps_dir(session_id) / dir_name
            attempt_dir.mkdir(parents=True, exist_ok=True)

            file_name = f"attempt_{attempt.attempt_number}.json"
            payload = {
                "attempt_number": attempt.attempt_number,
                "submitted_at": attempt.submitted_at,
                "artifacts": attempt.artifacts,
                "self_assessment": attempt.self_assessment,
                "evaluation": attempt.evaluation,
            }
            (attempt_dir / file_name).write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )

            self._append_event(session_id, "attempt_saved", {
                "phase_id": step.phase_id,
                "step_id": step.step_id,
                "attempt_number": attempt.attempt_number,
            })

    def log_event(
        self,
        session_id: str,
        event_type: str,
        data: dict[str, Any],
    ) -> None:
        """Append an event to the session's audit trail."""
        with self._lock:
            self._append_event(session_id, event_type, data)

    def log_stage_changed(
        self,
        session_id: str,
        from_stage: str,
        to_stage: str,
    ) -> None:
        """Log a session stage transition event."""
        with self._lock:
            self._append_event(session_id, "stage_changed", {
                "from": from_stage,
                "to": to_stage,
            })

    def log_step_status_changed(
        self,
        session_id: str,
        step_id: str,
        from_status: str,
        to_status: str,
    ) -> None:
        """Log a step status transition event."""
        with self._lock:
            self._append_event(session_id, "step_status_changed", {
                "step_id": step_id,
                "from": from_status,
                "to": to_status,
            })

    def recover_session_from_events(self, session_id: str) -> SessionState:
        """Reconstruct a SessionState by replaying events from events.jsonl.

        Reads events line by line and rebuilds the session state.  This is
        used as a fallback when state.json is missing or corrupted.

        Raises FileNotFoundError if the events file does not exist or
        contains no session_created event.
        """
        with self._lock:
            return self._recover_session_from_events_unlocked(session_id)

    def _recover_session_from_events_unlocked(
        self, session_id: str,
    ) -> SessionState:
        """Internal recovery logic (caller must hold the lock)."""
        events_path = self._events_path(session_id)
        if not events_path.is_file():
            raise FileNotFoundError(
                f"No events file for session: {session_id}"
            )

        events: list[dict] = []
        for line in events_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # skip malformed lines

        # --- 1. Find session_created to bootstrap state -----------------
        sop_id = ""
        created_at = ""
        step_count = 0
        for ev in events:
            if ev.get("event_type") == "session_created":
                sop_id = ev["data"].get("sop_id", "")
                step_count = ev["data"].get("step_count", 0)
                created_at = ev.get("timestamp", "")
                break
        else:
            raise FileNotFoundError(
                f"No session_created event found for session: {session_id}"
            )

        # Build skeleton steps (we won't have titles/phase_ids unless
        # attempt_saved events give us phase information).
        steps_by_id: dict[str, StepState] = {}

        # --- 2. Replay events ------------------------------------------
        stage = SessionStage.INITIALIZED
        step_index = 0
        updated_at = created_at

        for ev in events:
            etype = ev.get("event_type", "")
            data = ev.get("data", {})
            ts = ev.get("timestamp", updated_at)
            updated_at = ts

            if etype == "stage_changed":
                try:
                    stage = SessionStage(data["to"])
                except (KeyError, ValueError):
                    pass

            elif etype == "step_status_changed":
                sid = data.get("step_id", "")
                if sid and sid not in steps_by_id:
                    steps_by_id[sid] = StepState(
                        step_id=sid,
                        phase_id="",
                        title=sid,
                        status=StepStatus.PENDING,
                        attempts=[],
                        current_attempt=0,
                        max_attempts=3,
                    )
                if sid:
                    try:
                        steps_by_id[sid].status = StepStatus(data["to"])
                    except (KeyError, ValueError):
                        pass

            elif etype == "attempt_saved":
                sid = data.get("step_id", "")
                pid = data.get("phase_id", "")
                if sid and sid not in steps_by_id:
                    steps_by_id[sid] = StepState(
                        step_id=sid,
                        phase_id=pid,
                        title=sid,
                        status=StepStatus.PENDING,
                        attempts=[],
                        current_attempt=0,
                        max_attempts=3,
                    )
                if sid and pid:
                    steps_by_id[sid].phase_id = pid

            elif etype == "step_evaluated":
                sid = data.get("step_id", "")
                verdict = data.get("verdict")
                if sid and sid in steps_by_id and verdict is not None:
                    steps_by_id[sid].status = (
                        StepStatus.PASSED if verdict == "PASS" else StepStatus.FAILED
                    )

            elif etype == "session_completed":
                stage = SessionStage.COMPLETED

        # Derive step_index: first non-passed step
        steps_list = list(steps_by_id.values())
        for i, s in enumerate(steps_list):
            if s.status not in (StepStatus.PASSED, StepStatus.SKIPPED):
                step_index = i
                break
        else:
            step_index = len(steps_list)

        state = SessionState(
            session_id=session_id,
            sop_id=sop_id,
            stage=stage,
            step_index=step_index,
            steps=steps_list,
            context={},
            created_at=created_at,
            updated_at=updated_at,
        )

        logger.info(
            "Recovered session %s from events (%d events, %d steps)",
            session_id, len(events), len(steps_list),
        )
        return state

    def list_sessions(self) -> list[dict]:
        """Return summary dicts for every session on disk."""
        with self._lock:
            if not self._base.is_dir():
                return []

            sessions: list[dict] = []
            for child in sorted(self._base.iterdir()):
                state_file = child / "state.json"
                if not state_file.is_file():
                    continue
                try:
                    data = json.loads(state_file.read_text(encoding="utf-8"))
                    sessions.append({
                        "session_id": data.get("session_id", child.name),
                        "sop_id": data.get("sop_id"),
                        "stage": data.get("stage"),
                        "step_index": data.get("step_index"),
                        "step_total": len(data.get("steps", [])),
                        "created_at": data.get("created_at"),
                        "updated_at": data.get("updated_at"),
                    })
                except (json.JSONDecodeError, KeyError):
                    continue
            return sessions

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write_state(self, state: SessionState) -> None:
        """Write state.json atomically (caller must hold the lock).

        Uses temp-file-and-replace pattern to prevent corruption on crash.
        """
        path = self._state_path(state.session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        try:
            tmp_path.write_text(
                json.dumps(_full_state_dict(state), indent=2),
                encoding="utf-8",
            )
            tmp_path.replace(path)  # Atomic on POSIX
        except OSError as exc:
            # Clean up temp file on failure
            tmp_path.unlink(missing_ok=True)
            # Best-effort: log the write_error event to the audit trail
            try:
                self._append_event(state.session_id, "write_error", {
                    "error": str(exc),
                })
            except OSError:
                logger.error(
                    "Failed to log write_error event for session %s",
                    state.session_id,
                )
            raise OSError(
                f"Failed to persist session {state.session_id}: {exc}"
            ) from exc

    def _append_event(
        self,
        session_id: str,
        event_type: str,
        data: dict[str, Any],
    ) -> None:
        """Append a line to events.jsonl (caller must hold the lock)."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "session_id": session_id,
            "data": data,
        }
        path = self._events_path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
