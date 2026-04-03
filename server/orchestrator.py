"""Core workflow orchestration: submit -> validate -> evaluate -> feedback loop.

Ties together the SOP registry, session manager, validation layer, and
evaluator (subagent, Anthropic, or OpenAI-compatible).

When using the subagent evaluator (default), the flow is:
  1. harness_submit_step -> validates, returns evaluation prompt
  2. Caller dispatches a reviewer subagent with the prompt
  3. harness_report_evaluation -> parses result, advances/fails step
"""

import logging
from datetime import datetime
from typing import Any, Optional, Union

import yaml

from server.evaluator import (
    AnthropicEvaluator,
    OpenAICompatibleEvaluator,
    SubagentEvaluator,
    parse_evaluation_response,
    _error_result,
)
from server.models import (
    EvaluationResult,
    HarnessResponse,
    SessionStage,
    SessionState,
    StepAttempt,
    StepStatus,
    generate_session_id,
)
from server.session_manager import SessionManager
from server.sop_registry import SOPRegistry
from server.validation import validate_session_id, validate_sop_id, validate_submission

logger = logging.getLogger(__name__)

Evaluator = Union[SubagentEvaluator, AnthropicEvaluator, OpenAICompatibleEvaluator]


class Orchestrator:
    """Drives the harness workflow: start sessions, submit steps, iterate on feedback."""

    def __init__(
        self,
        sop_registry: SOPRegistry,
        session_manager: SessionManager,
        evaluator: Evaluator,
    ) -> None:
        self._sops = sop_registry
        self._sessions = session_manager
        self._evaluator = evaluator

    @property
    def is_subagent_mode(self) -> bool:
        return isinstance(self._evaluator, SubagentEvaluator)

    # ------------------------------------------------------------------
    # harness_start
    # ------------------------------------------------------------------

    def start_session(
        self,
        sop_id: str,
        context: dict | None = None,
        retry_limit: int | None = None,
    ) -> HarnessResponse:
        """Start a new harness session for the given SOP."""
        v = validate_sop_id(sop_id)
        if not v.is_valid:
            return HarnessResponse(success=False, message=v.message, data={"errors": v.errors})

        try:
            sop = self._sops.get_sop(sop_id)
        except KeyError:
            available = [s["sop_id"] for s in self._sops.list_sops()]
            return HarnessResponse(
                success=False,
                message=f"SOP '{sop_id}' not found.",
                data={"available_sops": available},
            )

        step_states = self._sops.build_step_states(sop_id)
        if retry_limit is not None:
            for s in step_states:
                s.max_attempts = retry_limit

        sop_yaml = yaml.dump({
            "sop_id": sop.sop_id, "name": sop.name,
            "description": sop.description,
            "default_retry_limit": sop.default_retry_limit,
            "pass_threshold": sop.pass_threshold,
            "phases": [
                {"id": p.id, "name": p.name, "steps": [
                    {"id": s.id, "title": s.title, "instruction": s.instruction,
                     "acceptance_criteria": s.acceptance_criteria, "on_fail": s.on_fail}
                    for s in p.steps
                ]}
                for p in sop.phases
            ],
        }, default_flow_style=False)

        session_id = generate_session_id()
        session = self._sessions.create_session(
            session_id=session_id, sop_id=sop_id,
            sop_yaml_content=sop_yaml, steps=step_states,
        )

        session.stage = SessionStage.RUNNING
        if session.steps:
            session.steps[0].status = StepStatus.IN_PROGRESS
        self._sessions.save_session(session)

        flat_steps = self._sops.flatten_steps(sop_id)
        steps_overview = [
            {"index": i, "id": s.id, "title": s.title}
            for i, s in enumerate(flat_steps)
        ]

        elicitation = None
        if flat_steps:
            first = flat_steps[0]
            elicitation = {
                "message": f"Step 1/{len(flat_steps)}: {first.title}",
                "instruction": first.instruction,
                "acceptance_criteria": first.acceptance_criteria,
            }

        return HarnessResponse(
            success=True,
            message=f"Session started for '{sop.name}'. Proceed with step 1.",
            session_id=session_id,
            stage="awaiting_step",
            step_index=0,
            step_total=len(step_states),
            data={"sop_name": sop.name, "steps_overview": steps_overview},
            elicitation=elicitation,
        )

    # ------------------------------------------------------------------
    # harness_submit_step
    # ------------------------------------------------------------------

    async def submit_step(
        self,
        session_id: str,
        step_output: dict,
    ) -> HarnessResponse:
        """Submit output for the current step.

        In subagent mode: validates, then returns the evaluation prompt for
        the caller to dispatch to a reviewer subagent.

        In API mode: validates, evaluates server-side, returns result directly.
        """
        # Validate session ID
        v = validate_session_id(session_id)
        if not v.is_valid:
            return HarnessResponse(success=False, message=v.message, data={"errors": v.errors})

        try:
            session = self._sessions.load_session(session_id)
        except FileNotFoundError:
            return HarnessResponse(
                success=False, message=f"Session '{session_id}' not found.",
                session_id=session_id,
            )

        if session.stage != SessionStage.RUNNING:
            return HarnessResponse(
                success=False,
                message=f"Session is '{session.stage.value}', not running.",
                session_id=session_id, stage=session.stage.value,
            )

        current = session.current_step
        if current is None:
            return HarnessResponse(
                success=False, message="No active step in session.",
                session_id=session_id,
            )

        sop = self._sops.get_sop(session.sop_id)
        flat_steps = self._sops.flatten_steps(session.sop_id)
        sop_step = flat_steps[session.step_index] if session.step_index < len(flat_steps) else None

        # Layer 1: Deterministic validation
        output_schema = sop_step.output_schema if sop_step else None
        validation = validate_submission(step_output, output_schema)
        if not validation.is_valid:
            return HarnessResponse(
                success=False,
                message=f"Submission validation failed: {validation.message}",
                session_id=session_id, stage="awaiting_step",
                step_index=session.step_index, step_total=session.step_total,
                data={"errors": validation.errors},
            )

        # Record the attempt (increment counter)
        current.current_attempt += 1
        current.status = StepStatus.EVALUATING

        attempt = StepAttempt(
            attempt_number=current.current_attempt,
            submitted_at=datetime.now().isoformat(),
            artifacts=step_output.get("artifacts", []),
            self_assessment=step_output.get("self_assessment", ""),
        )
        current.attempts.append(attempt)
        self._sessions.save_session(session)

        # Get acceptance criteria and previous attempts
        acceptance_criteria = sop_step.acceptance_criteria if sop_step else []
        previous_attempts = [
            {"attempt_number": a.attempt_number, "evaluation": a.evaluation}
            for a in current.attempts[:-1]  # all except current
            if a.evaluation is not None
        ]

        # Layer 2: Evaluate
        eval_result = await self._evaluator.evaluate(
            submission=step_output,
            acceptance_criteria=acceptance_criteria,
            previous_attempts=previous_attempts,
            step_context=sop_step.instruction if sop_step else "",
            attempt=current.current_attempt,
            max_attempts=current.max_attempts,
        )

        # Subagent mode: return prompt for external evaluation
        if isinstance(eval_result, dict) and eval_result.get("mode") == "subagent":
            # Save attempt without evaluation yet
            self._sessions.save_attempt(session_id, current, attempt)

            return HarnessResponse(
                success=True,
                message=(
                    "Submission validated. Dispatch the evaluation prompt below to "
                    "the harness-reviewer agent, then call harness_report_evaluation "
                    "with the reviewer's JSON response."
                ),
                session_id=session_id,
                stage="awaiting_evaluation",
                step_index=session.step_index,
                step_total=session.step_total,
                data={
                    "evaluation_prompt": {
                        "system_prompt": eval_result["system_prompt"],
                        "user_prompt": eval_result["user_prompt"],
                    },
                    "attempt": eval_result["attempt"],
                    "max_attempts": eval_result["max_attempts"],
                },
            )

        # Subagent auto_pass (no criteria)
        if isinstance(eval_result, dict) and eval_result.get("mode") == "auto_pass":
            eval_result = eval_result["result"]

        # API mode: evaluation is already done
        attempt.evaluation = eval_result.to_dict()
        self._sessions.save_attempt(session_id, current, attempt)
        self._sessions.save_session(session)

        self._sessions.log_event(session_id, "step_evaluated", {
            "step_id": current.step_id,
            "attempt": current.current_attempt,
            "verdict": eval_result.verdict,
            "weighted_score": eval_result.weighted_score,
        })

        if eval_result.verdict == "PASS":
            return self._advance_step(session, eval_result)
        else:
            return self._handle_failure(session, eval_result, sop_step)

    # ------------------------------------------------------------------
    # harness_report_evaluation (for subagent mode)
    # ------------------------------------------------------------------

    def report_evaluation(
        self,
        session_id: str,
        evaluation_json: dict,
    ) -> HarnessResponse:
        """Accept evaluation results from a subagent and process them.

        Called after the reviewer subagent evaluates the submission.
        The evaluation_json should match the evaluator's expected format
        (dimensions with scores, evidence, gaps, slop_flags, top_3_fixes).
        """
        v = validate_session_id(session_id)
        if not v.is_valid:
            return HarnessResponse(success=False, message=v.message)

        try:
            session = self._sessions.load_session(session_id)
        except FileNotFoundError:
            return HarnessResponse(
                success=False, message=f"Session '{session_id}' not found.",
                session_id=session_id,
            )

        current = session.current_step
        if current is None or current.status != StepStatus.EVALUATING:
            return HarnessResponse(
                success=False,
                message="No step awaiting evaluation.",
                session_id=session_id,
                stage=session.stage.value,
            )

        # Get evaluator profile from SOP step
        flat_steps = self._sops.flatten_steps(session.sop_id)
        sop_step = flat_steps[session.step_index] if session.step_index < len(flat_steps) else None
        profile_name = sop_step.evaluator_profile if sop_step else "default"

        # Parse the subagent's evaluation response
        try:
            evaluation = parse_evaluation_response(
                evaluation_json,
                attempt=current.current_attempt,
                max_attempts=current.max_attempts,
                profile_name=profile_name,
            )
        except Exception as exc:
            logger.error("Failed to parse subagent evaluation: %s", exc)
            evaluation = _error_result(
                f"Failed to parse evaluation: {exc}",
                current.current_attempt,
                current.max_attempts,
            )

        # Update the latest attempt with the evaluation
        if current.attempts:
            current.attempts[-1].evaluation = evaluation.to_dict()
            self._sessions.save_attempt(session_id, current, current.attempts[-1])

        self._sessions.save_session(session)

        self._sessions.log_event(session_id, "step_evaluated", {
            "step_id": current.step_id,
            "attempt": current.current_attempt,
            "verdict": evaluation.verdict,
            "weighted_score": evaluation.weighted_score,
            "evaluation_source": "subagent",
        })

        # Get SOP step for failure handling
        flat_steps = self._sops.flatten_steps(session.sop_id)
        sop_step = flat_steps[session.step_index] if session.step_index < len(flat_steps) else None

        if evaluation.verdict == "PASS":
            return self._advance_step(session, evaluation)
        else:
            return self._handle_failure(session, evaluation, sop_step)

    # ------------------------------------------------------------------
    # harness_get_status
    # ------------------------------------------------------------------

    def get_status(self, session_id: str) -> HarnessResponse:
        """Get the current status of a session."""
        v = validate_session_id(session_id)
        if not v.is_valid:
            return HarnessResponse(success=False, message=v.message)

        try:
            session = self._sessions.load_session(session_id)
        except FileNotFoundError:
            return HarnessResponse(
                success=False, message=f"Session '{session_id}' not found.",
                session_id=session_id,
            )

        current = session.current_step
        scores_history = []
        if current:
            for a in current.attempts:
                if a.evaluation:
                    scores_history.append({
                        "attempt": a.attempt_number,
                        "verdict": a.evaluation.get("verdict"),
                        "score": a.evaluation.get("weighted_score"),
                    })

        return HarnessResponse(
            success=True,
            message=f"Session is {session.stage.value}.",
            session_id=session_id,
            stage=session.stage.value,
            step_index=session.step_index,
            step_total=session.step_total,
            data={
                "steps_completed": sum(
                    1 for s in session.steps if s.status == StepStatus.PASSED
                ),
                "current_step": {
                    "step_id": current.step_id,
                    "title": current.title,
                    "status": current.status.value,
                    "retries_remaining": current.retries_remaining,
                } if current else None,
                "scores_history": scores_history,
            },
        )

    # ------------------------------------------------------------------
    # harness_get_feedback
    # ------------------------------------------------------------------

    def get_feedback(
        self, session_id: str, step_index: int | None = None
    ) -> HarnessResponse:
        """Get full feedback history for a step."""
        v = validate_session_id(session_id)
        if not v.is_valid:
            return HarnessResponse(success=False, message=v.message)

        try:
            session = self._sessions.load_session(session_id)
        except FileNotFoundError:
            return HarnessResponse(
                success=False, message=f"Session '{session_id}' not found.",
                session_id=session_id,
            )

        idx = step_index if step_index is not None else session.step_index
        if idx < 0 or idx >= len(session.steps):
            return HarnessResponse(
                success=False,
                message=f"Step index {idx} out of range (0-{len(session.steps) - 1}).",
                session_id=session_id,
            )

        step = session.steps[idx]
        attempts_data = [
            {
                "attempt_number": a.attempt_number,
                "submitted_at": a.submitted_at,
                "self_assessment": a.self_assessment,
                "evaluation": a.evaluation,
            }
            for a in step.attempts
        ]

        return HarnessResponse(
            success=True,
            message=f"Feedback for step '{step.title}'.",
            session_id=session_id,
            stage=session.stage.value,
            step_index=idx,
            step_total=session.step_total,
            data={
                "step_id": step.step_id,
                "title": step.title,
                "status": step.status.value,
                "attempts": attempts_data,
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # harness_resume
    # ------------------------------------------------------------------

    def resume_session(
        self, session_id: str, comment: str = ""
    ) -> HarnessResponse:
        """Resume a paused/blocked session, re-opening the current step for retry."""
        v = validate_session_id(session_id)
        if not v.is_valid:
            return HarnessResponse(success=False, message=v.message)

        try:
            session = self._sessions.load_session(session_id)
        except FileNotFoundError:
            return HarnessResponse(
                success=False, message=f"Session '{session_id}' not found.",
                session_id=session_id,
            )

        if session.stage not in (SessionStage.PAUSED, SessionStage.FAILED):
            return HarnessResponse(
                success=False,
                message=f"Session is '{session.stage.value}', not paused or failed. Cannot resume.",
                session_id=session_id, stage=session.stage.value,
            )

        current = session.current_step
        if current:
            current.status = StepStatus.IN_PROGRESS
            current.current_attempt = 0  # Reset retry counter
            current.attempts = []  # Clear previous attempts
        session.stage = SessionStage.RUNNING
        self._sessions.save_session(session)

        self._sessions.log_event(session_id, "session_resumed", {
            "comment": comment,
            "step_id": current.step_id if current else None,
        })

        flat_steps = self._sops.flatten_steps(session.sop_id)
        sop_step = flat_steps[session.step_index] if session.step_index < len(flat_steps) else None

        elicitation = None
        if sop_step:
            elicitation = {
                "message": f"Resumed: Step {session.step_index + 1}/{session.step_total}: {sop_step.title}",
                "instruction": sop_step.instruction,
                "acceptance_criteria": sop_step.acceptance_criteria,
            }

        return HarnessResponse(
            success=True,
            message=f"Session resumed. Retrying step '{current.title if current else '?'}'.",
            session_id=session_id,
            stage="awaiting_step",
            step_index=session.step_index,
            step_total=session.step_total,
            elicitation=elicitation,
        )

    # ------------------------------------------------------------------
    # harness_skip_step
    # ------------------------------------------------------------------

    def skip_step(
        self, session_id: str, reason: str = ""
    ) -> HarnessResponse:
        """Skip the current step and advance to the next one."""
        v = validate_session_id(session_id)
        if not v.is_valid:
            return HarnessResponse(success=False, message=v.message)

        try:
            session = self._sessions.load_session(session_id)
        except FileNotFoundError:
            return HarnessResponse(
                success=False, message=f"Session '{session_id}' not found.",
                session_id=session_id,
            )

        if session.stage != SessionStage.RUNNING:
            return HarnessResponse(
                success=False,
                message=f"Session is '{session.stage.value}', not running.",
                session_id=session_id, stage=session.stage.value,
            )

        current = session.current_step
        if current is None:
            return HarnessResponse(
                success=False, message="No active step to skip.",
                session_id=session_id,
            )

        current.status = StepStatus.SKIPPED
        session.step_index += 1

        self._sessions.log_event(session_id, "step_skipped", {
            "step_id": current.step_id,
            "reason": reason,
        })

        if session.step_index >= session.step_total:
            session.stage = SessionStage.COMPLETED
            self._sessions.save_session(session)
            return HarnessResponse(
                success=True,
                message="Step skipped. All steps complete.",
                session_id=session_id, stage="complete",
                step_index=session.step_index - 1,
                step_total=session.step_total,
            )

        next_step = session.steps[session.step_index]
        next_step.status = StepStatus.IN_PROGRESS
        self._sessions.save_session(session)

        flat_steps = self._sops.flatten_steps(session.sop_id)
        next_sop = flat_steps[session.step_index] if session.step_index < len(flat_steps) else None

        elicitation = None
        if next_sop:
            elicitation = {
                "message": f"Step {session.step_index + 1}/{session.step_total}: {next_sop.title}",
                "instruction": next_sop.instruction,
                "acceptance_criteria": next_sop.acceptance_criteria,
            }

        return HarnessResponse(
            success=True,
            message=f"Step '{current.title}' skipped. Advancing.",
            session_id=session_id,
            stage="awaiting_step",
            step_index=session.step_index,
            step_total=session.step_total,
            elicitation=elicitation,
        )

    # ------------------------------------------------------------------
    # harness_list_sessions
    # ------------------------------------------------------------------

    def list_sessions(self) -> HarnessResponse:
        """List all sessions with their status."""
        sessions = self._sessions.list_sessions()
        return HarnessResponse(
            success=True,
            message=f"{len(sessions)} session(s) found.",
            data={"sessions": sessions},
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _advance_step(
        self, session: SessionState, evaluation: EvaluationResult
    ) -> HarnessResponse:
        """Mark current step as passed and advance to next step or complete."""
        current = session.current_step
        current.status = StepStatus.PASSED
        session.step_index += 1

        if session.step_index >= session.step_total:
            session.stage = SessionStage.COMPLETED
            self._sessions.save_session(session)
            self._sessions.log_event(session.session_id, "session_completed", {
                "steps_total": session.step_total,
            })
            return HarnessResponse(
                success=True,
                message="All steps completed. Session finished.",
                session_id=session.session_id,
                stage="complete",
                step_index=session.step_index - 1,
                step_total=session.step_total,
                data={"evaluation": evaluation.to_dict()},
            )

        next_step = session.steps[session.step_index]
        next_step.status = StepStatus.IN_PROGRESS
        self._sessions.save_session(session)

        flat_steps = self._sops.flatten_steps(session.sop_id)
        next_sop_step = flat_steps[session.step_index] if session.step_index < len(flat_steps) else None

        elicitation = None
        if next_sop_step:
            elicitation = {
                "message": f"Step {session.step_index + 1}/{session.step_total}: {next_sop_step.title}",
                "instruction": next_sop_step.instruction,
                "acceptance_criteria": next_sop_step.acceptance_criteria,
            }

        return HarnessResponse(
            success=True,
            message=f"Step PASSED (score: {evaluation.weighted_score}). Advancing to next step.",
            session_id=session.session_id,
            stage="awaiting_step",
            step_index=session.step_index,
            step_total=session.step_total,
            data={"evaluation": evaluation.to_dict()},
            elicitation=elicitation,
        )

    def _handle_failure(
        self,
        session: SessionState,
        evaluation: EvaluationResult,
        sop_step: Any,
    ) -> HarnessResponse:
        """Handle a failed evaluation: retry, skip, abort, or escalate."""
        current = session.current_step

        if current.retries_remaining > 0:
            current.status = StepStatus.IN_PROGRESS
            self._sessions.save_session(session)

            return HarnessResponse(
                success=True,
                message=(
                    f"Step FAILED (score: {evaluation.weighted_score}). "
                    f"{current.retries_remaining} retries remaining."
                ),
                session_id=session.session_id,
                stage="awaiting_step",
                step_index=session.step_index,
                step_total=session.step_total,
                data={"feedback": evaluation.to_dict()},
                elicitation={
                    "message": f"Retry step: {current.title}. Address the feedback below.",
                    "instruction": sop_step.instruction if sop_step else "",
                    "acceptance_criteria": sop_step.acceptance_criteria if sop_step else [],
                    "feedback": evaluation.to_dict(),
                },
            )

        on_fail = sop_step.on_fail if sop_step else "retry"

        if on_fail == "skip":
            current.status = StepStatus.SKIPPED
            session.step_index += 1
            if session.step_index >= session.step_total:
                session.stage = SessionStage.COMPLETED
            else:
                session.steps[session.step_index].status = StepStatus.IN_PROGRESS
            self._sessions.save_session(session)

            return HarnessResponse(
                success=True,
                message=f"Step skipped after {current.max_attempts} failed attempts.",
                session_id=session.session_id,
                stage="awaiting_step" if session.step_index < session.step_total else "complete",
                step_index=session.step_index,
                step_total=session.step_total,
                data={"feedback": evaluation.to_dict(), "skipped": True},
            )

        if on_fail == "abort":
            session.stage = SessionStage.FAILED
            current.status = StepStatus.BLOCKED
            self._sessions.save_session(session)
            return HarnessResponse(
                success=False,
                message=f"Session aborted: step '{current.title}' failed after {current.max_attempts} attempts.",
                session_id=session.session_id,
                stage="failed",
                step_index=session.step_index,
                step_total=session.step_total,
                data={"feedback": evaluation.to_dict()},
            )

        # Default: block and escalate to human
        current.status = StepStatus.BLOCKED
        session.stage = SessionStage.PAUSED
        self._sessions.save_session(session)

        self._sessions.log_event(session.session_id, "human_escalation", {
            "step_id": current.step_id,
            "attempts_exhausted": current.max_attempts,
            "best_score": evaluation.weighted_score,
        })

        return HarnessResponse(
            success=False,
            message=(
                f"Step '{current.title}' failed after {current.max_attempts} attempts. "
                "Human review required."
            ),
            session_id=session.session_id,
            stage="blocked",
            step_index=session.step_index,
            step_total=session.step_total,
            data={
                "feedback": evaluation.to_dict(),
                "escalation": True,
                "all_attempts": [
                    {"attempt": a.attempt_number, "evaluation": a.evaluation}
                    for a in current.attempts
                ],
            },
        )
