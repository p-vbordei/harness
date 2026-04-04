#!/usr/bin/env python3
"""Manual interactive test for the harness system.

Run this to walk through a mini workflow and see the harness in action:
    python tests/manual_test.py

No API key needed -- uses the subagent evaluator (simulated locally).
Demonstrates: structured submissions, deterministic rule failures,
sequential step enforcement, and the feedback loop.
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from server.evaluator import DIMENSIONS, SubagentEvaluator, parse_evaluation_response
from server.orchestrator import Orchestrator
from server.session_manager import SessionManager
from server.sop_registry import SOPRegistry


def print_response(label: str, resp):
    d = resp.to_dict()
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    text = json.dumps(d, indent=2)
    print(text[:2500])
    if len(text) > 2500:
        print("  ... (truncated)")
    print()


async def main():
    project_sops = Path(__file__).parent.parent / "sops"
    tmp_dir = Path("/tmp/harness-manual-test")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    registry = SOPRegistry(search_dirs=[project_sops])
    manager = SessionManager(base_dir=tmp_dir / "sessions")
    evaluator = SubagentEvaluator()
    orch = Orchestrator(registry, manager, evaluator)

    print("\n" + "="*60)
    print("  HARNESS MANUAL TEST - Structured Criteria Demo")
    print("="*60)

    print("\nAvailable SOPs:")
    for sop in registry.list_sops():
        steps = registry.flatten_steps(sop["sop_id"])
        print(f"  - {sop['sop_id']}: {sop['name']} ({len(steps)} steps)")

    # --- Start session ---
    print("\n--- Starting feature-dev session ---")
    start = orch.start_session("feature-dev")
    print_response("harness_start", start)
    session_id = start.session_id
    elicitation = start.elicitation

    print(f"  Sequential enforcement: NO steps_overview leaked")
    print(f"  Required fields: {elicitation.get('required_fields', [])}")
    print(f"  Expected format shown to agent: YES")

    # --- Demo 1: Submit WITHOUT required fields (deterministic rejection) ---
    print("\n--- Demo 1: Submit plain text (missing required fields) ---")
    bad_output = {
        "artifacts": [{"type": "text", "content": "Some user stories exist."}],
        "self_assessment": "I think I covered everything.",
    }
    resp = await orch.submit_step(session_id, bad_output)
    print_response("REJECTED: Missing structured fields", resp)
    print("  ^ The rule engine caught missing user_stories/constraints/out_of_scope")
    print("  ^ No LLM call was made -- deterministic rejection (Layer 1b)")

    # --- Demo 2: Submit WITH required fields (passes Layer 1b, goes to LLM) ---
    print("\n--- Demo 2: Submit structured data (passes deterministic checks) ---")
    good_output = {
        "artifacts": [
            {
                "type": "json_object",
                "content": json.dumps({
                    "user_stories": [
                        {
                            "as_a": "warehouse manager",
                            "i_want": "scan barcodes with phone camera",
                            "so_that": "inventory updates happen in real-time",
                            "acceptance_criteria": [
                                "Barcode recognized within 2 seconds",
                                "Inventory count updates immediately after scan",
                            ],
                        }
                    ],
                    "constraints": [
                        "Must work on iOS 15+ and Android 12+",
                        "Response time under 200ms for scan-to-update",
                    ],
                    "out_of_scope": [
                        "Desktop barcode scanning",
                        "Bluetooth scanner hardware support",
                    ],
                }),
            }
        ],
        "self_assessment": "One user story with 2 acceptance criteria, 2 constraints, 2 out-of-scope items.",
    }
    resp = await orch.submit_step(session_id, good_output)
    print_response("PASSED Layer 1b: Awaiting LLM evaluation", resp)

    if resp.stage == "awaiting_evaluation":
        print("  ^ Deterministic rules PASSED. Evaluation prompt returned for reviewer agent.")

        # Simulate reviewer evaluation (PASS)
        eval_json = {
            dim: {"score": 4, "evidence": "Well-structured with specific details.", "gap": None}
            for dim in DIMENSIONS
        }
        eval_json["slop_flags"] = []
        eval_json["top_3_fixes"] = []

        resp = orch.report_evaluation(session_id, eval_json)
        print_response("PASSED: Advancing to step 2", resp)

        # Show that step 2 is now visible (but step 3+ still hidden)
        if resp.elicitation:
            print(f"  Next step revealed: {resp.elicitation['message']}")
            print(f"  Required fields: {resp.elicitation.get('required_fields', 'none')}")

    # --- Demo 3: Simulate a FAIL with slop detection ---
    print("\n--- Demo 3: Submit with AI slop (triggers slop penalty) ---")
    sloppy_output = {
        "artifacts": [
            {
                "type": "json_object",
                "content": json.dumps({
                    "files_to_modify": ["/src/scanner.py", "/src/api/barcode.py"],
                    "data_model_changes": [
                        {"table": "scans", "fields": ["barcode_value VARCHAR", "scanned_at TIMESTAMP"]}
                    ],
                    "api_changes": [
                        {"method": "POST", "path": "/api/scan", "description": "Submit barcode scan"}
                    ],
                    "testing_strategy": {
                        "unit": "Test barcode parsing",
                        "integration": "Test full scan flow",
                        "manual": "Test on real devices",
                    },
                }),
            }
        ],
        "self_assessment": "This comprehensive solution leverages existing patterns to ensure a seamless user experience.",
    }
    resp = await orch.submit_step(session_id, sloppy_output)

    if resp.stage == "awaiting_evaluation":
        fail_eval = {
            dim: {"score": 3, "evidence": "Adequate but generic.", "gap": "Needs project-specific details."}
            for dim in DIMENSIONS
        }
        fail_eval["slop_flags"] = [
            "comprehensive solution",
            "leverages existing patterns",
            "ensure a seamless user experience",
        ]
        fail_eval["top_3_fixes"] = [
            "Reference specific files from THIS project, not generic paths",
            "Remove AI filler phrases from self-assessment",
            "Add concrete metrics to testing strategy",
        ]

        resp = orch.report_evaluation(session_id, fail_eval)
        print_response("FAILED: Slop penalty applied", resp)

        if resp.data.get("feedback"):
            fb = resp.data["feedback"]
            spec_score = fb.get("dimensions", {}).get("specificity", {})
            print(f"  Specificity score: {spec_score.get('score')} (slop penalty applied)")
            print(f"  Slop flags: {fb.get('slop_flags', [])}")
            print(f"  Top fixes: {fb.get('top_3_fixes', [])}")

    # --- Status and feedback ---
    status = orch.get_status(session_id)
    print_response("harness_get_status", status)

    sessions = orch.list_sessions()
    print_response("harness_list_sessions", sessions)

    # Print session files
    session_dir = manager._base / session_id
    print(f"\nSession files at: {session_dir}")
    if session_dir.exists():
        for f in sorted(session_dir.rglob("*")):
            if f.is_file():
                print(f"  {f.relative_to(session_dir)} ({f.stat().st_size} bytes)")

    print("\n" + "="*60)
    print("  MANUAL TEST COMPLETE")
    print("  Key takeaways:")
    print("  1. Missing required fields = instant deterministic rejection (no LLM cost)")
    print("  2. Structured data + passing rules = goes to LLM evaluation")
    print("  3. AI slop detected = specificity score penalized")
    print("  4. Sequential enforcement: only current step visible")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
