#!/usr/bin/env python3
"""Manual interactive test for the harness system.

Run this to walk through a mini workflow and see the harness in action:
    python tests/manual_test.py

No API key needed -- uses the subagent evaluator (simulated locally).
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from server.evaluator import DIMENSIONS, SubagentEvaluator, parse_evaluation_response
from server.orchestrator import Orchestrator
from server.session_manager import SessionManager
from server.sop_registry import SOPRegistry


def print_response(label: str, resp):
    """Pretty-print a harness response."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    d = resp.to_dict()
    print(json.dumps(d, indent=2)[:2000])
    if len(json.dumps(d)) > 2000:
        print("  ... (truncated)")
    print()


async def main():
    # Setup
    project_sops = Path(__file__).parent.parent / "sops"
    tmp_dir = Path("/tmp/harness-manual-test")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    registry = SOPRegistry(search_dirs=[project_sops])
    manager = SessionManager(base_dir=tmp_dir / "sessions")
    evaluator = SubagentEvaluator()
    orch = Orchestrator(registry, manager, evaluator)

    print("\n" + "="*60)
    print("  HARNESS MANUAL TEST")
    print("="*60)

    # List available SOPs
    print("\nAvailable SOPs:")
    for sop in registry.list_sops():
        steps = registry.flatten_steps(sop["sop_id"])
        print(f"  - {sop['sop_id']}: {sop['name']} ({len(steps)} steps)")

    # Start a feature-dev session
    print("\n--- Starting feature-dev session ---")
    start = orch.start_session("feature-dev")
    print_response("harness_start", start)
    session_id = start.session_id

    # Walk through first 3 steps
    for step_num in range(3):
        step_info = start.elicitation if step_num == 0 else resp.elicitation
        if step_info:
            print(f"\n>>> Step {step_num + 1}: {step_info.get('message', '?')}")
            print(f"    Instruction: {step_info.get('instruction', '?')[:100]}...")
            print(f"    Criteria: {step_info.get('acceptance_criteria', [])}")

        # Submit
        output = {
            "artifacts": [{
                "type": "text",
                "content": f"Detailed output for step {step_num + 1} with specific "
                           f"file paths (/src/main.py, /tests/test_main.py) and "
                           f"concrete metrics (latency < 200ms, coverage > 80%)."
            }],
            "self_assessment": f"All {len(step_info.get('acceptance_criteria', []))} "
                              f"criteria addressed with project-specific details.",
        }

        submit = await orch.submit_step(session_id, output)
        print_response(f"harness_submit_step (step {step_num + 1})", submit)

        # Simulate reviewer evaluation (PASS)
        eval_json = {
            dim: {"score": 4, "evidence": "Project-specific details provided.", "gap": None}
            for dim in DIMENSIONS
        }
        eval_json["slop_flags"] = []
        eval_json["top_3_fixes"] = []

        resp = orch.report_evaluation(session_id, eval_json)
        print_response(f"harness_report_evaluation (step {step_num + 1})", resp)

    # Show status
    status = orch.get_status(session_id)
    print_response("harness_get_status", status)

    # Demo: skip a step
    print("\n--- Skipping step 4 ---")
    skip = orch.skip_step(session_id, reason="Demo skip")
    print_response("harness_skip_step", skip)

    # Demo: simulate a failure
    print("\n--- Simulating a FAIL on step 5 ---")
    output = {
        "artifacts": [{"type": "text", "content": "Vague generic output."}],
        "self_assessment": "This ensures a seamless user experience.",
    }
    submit = await orch.submit_step(session_id, output)

    fail_eval = {
        dim: {"score": 2, "evidence": "Generic boilerplate.", "gap": "No project-specific details."}
        for dim in DIMENSIONS
    }
    fail_eval["slop_flags"] = [
        "This ensures a seamless user experience",
        "comprehensive solution",
        "leverage existing patterns",
    ]
    fail_eval["top_3_fixes"] = [
        "Reference specific files and functions from this project",
        "Remove filler phrases",
        "Add concrete metrics instead of vague claims",
    ]

    resp = orch.report_evaluation(session_id, fail_eval)
    print_response("harness_report_evaluation (FAIL)", resp)

    # Show feedback
    feedback = orch.get_feedback(session_id)
    print_response("harness_get_feedback", feedback)

    # List sessions
    sessions = orch.list_sessions()
    print_response("harness_list_sessions", sessions)

    # Print session directory
    session_dir = manager._base / session_id
    print(f"\nSession files at: {session_dir}")
    for f in sorted(session_dir.rglob("*")):
        if f.is_file():
            size = f.stat().st_size
            print(f"  {f.relative_to(session_dir)} ({size} bytes)")

    print("\n" + "="*60)
    print("  MANUAL TEST COMPLETE")
    print(f"  Session ID: {session_id}")
    print(f"  Files at: {session_dir}")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
