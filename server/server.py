"""Harness MCP Server - quality-gated SOP execution.

Tools:
    harness_start             - Begin a session with an SOP
    harness_submit_step       - Submit output for the current step
    harness_report_evaluation - Report subagent evaluation results (subagent mode)
    harness_get_status        - Get session status
    harness_get_feedback      - Get evaluation feedback history
"""

import asyncio
import json
import logging
import os

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from server.evaluator import create_evaluator
from server.orchestrator import Orchestrator
from server.session_manager import SessionManager
from server.sop_registry import SOPRegistry

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Tool definitions
# ------------------------------------------------------------------

TOOLS = [
    Tool(
        name="harness_start",
        description=(
            "Start a new harness session for a given SOP. "
            "Returns the first step's instruction and acceptance criteria."
        ),
        inputSchema={
            "type": "object",
            "required": ["sop_id"],
            "properties": {
                "sop_id": {
                    "type": "string",
                    "description": "SOP identifier (e.g., 'feature-dev')",
                },
                "context": {
                    "type": "object",
                    "description": "Arbitrary context for the SOP",
                    "additionalProperties": True,
                },
                "retry_limit": {
                    "type": "integer",
                    "description": "Max retries per step (default: 3)",
                    "minimum": 1,
                    "maximum": 10,
                },
            },
        },
    ),
    Tool(
        name="harness_submit_step",
        description=(
            "Submit output for the current step. In subagent mode (default), "
            "returns an evaluation prompt to dispatch to the harness-reviewer agent. "
            "In API mode, evaluates server-side and returns the result directly."
        ),
        inputSchema={
            "type": "object",
            "required": ["session_id", "step_output"],
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Session UUID from harness_start",
                },
                "step_output": {
                    "type": "object",
                    "required": ["artifacts", "self_assessment"],
                    "properties": {
                        "artifacts": {
                            "type": "array",
                            "description": "List of artifacts produced",
                            "items": {
                                "type": "object",
                                "required": ["type", "content"],
                                "properties": {
                                    "type": {
                                        "type": "string",
                                        "enum": ["file_path", "code_block", "text", "json_object"],
                                    },
                                    "content": {"type": "string"},
                                    "metadata": {
                                        "type": "object",
                                        "additionalProperties": True,
                                    },
                                },
                            },
                        },
                        "self_assessment": {
                            "type": "string",
                            "description": "Agent's assessment of completeness",
                        },
                    },
                },
            },
        },
    ),
    Tool(
        name="harness_report_evaluation",
        description=(
            "Report evaluation results from the harness-reviewer subagent. "
            "Call this after dispatching the evaluation prompt from harness_submit_step "
            "to the reviewer agent and receiving its JSON response."
        ),
        inputSchema={
            "type": "object",
            "required": ["session_id", "evaluation"],
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Session UUID",
                },
                "evaluation": {
                    "type": "object",
                    "description": (
                        "The reviewer agent's JSON evaluation with dimension scores "
                        "(completeness, specificity, correctness, actionability, "
                        "format_compliance), slop_flags, and top_3_fixes."
                    ),
                },
            },
        },
    ),
    Tool(
        name="harness_get_status",
        description="Get the current status of a harness session.",
        inputSchema={
            "type": "object",
            "required": ["session_id"],
            "properties": {
                "session_id": {"type": "string", "description": "Session UUID"},
            },
        },
    ),
    Tool(
        name="harness_list_sops",
        description="List all available SOPs that can be used with harness_start.",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="harness_resume",
        description="Resume a paused or blocked session, resetting the current step for retry.",
        inputSchema={
            "type": "object",
            "required": ["session_id"],
            "properties": {
                "session_id": {"type": "string", "description": "Session UUID"},
                "comment": {
                    "type": "string",
                    "description": "Reason for resuming (logged for audit trail)",
                },
            },
        },
    ),
    Tool(
        name="harness_skip_step",
        description="Skip the current step and advance to the next one.",
        inputSchema={
            "type": "object",
            "required": ["session_id"],
            "properties": {
                "session_id": {"type": "string", "description": "Session UUID"},
                "reason": {
                    "type": "string",
                    "description": "Reason for skipping (logged for audit trail)",
                },
            },
        },
    ),
    Tool(
        name="harness_list_sessions",
        description="List all harness sessions with their current status.",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="harness_get_feedback",
        description="Get the full evaluation feedback history for a step.",
        inputSchema={
            "type": "object",
            "required": ["session_id"],
            "properties": {
                "session_id": {"type": "string", "description": "Session UUID"},
                "step_index": {
                    "type": "integer",
                    "description": "Step index (0-based). Defaults to current step.",
                },
            },
        },
    ),
]


# ------------------------------------------------------------------
# Server setup
# ------------------------------------------------------------------

def create_server() -> tuple[Server, Orchestrator]:
    """Create and configure the MCP server with all components.

    Evaluator backend is selected by environment variables:
        HARNESS_EVAL_BACKEND  - "subagent" (default), "anthropic", or "openai"
        ANTHROPIC_API_KEY     - required for "anthropic" backend
        HARNESS_EVAL_BASE_URL - required for "openai" backend (e.g., http://localhost:8000/v1)
        HARNESS_EVAL_API_KEY  - API key for "openai" backend (optional)
        HARNESS_EVAL_MODEL    - model name override (optional)
    """
    backend = os.environ.get("HARNESS_EVAL_BACKEND", "subagent")
    evaluator = create_evaluator(
        backend=backend,
        api_key=os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("HARNESS_EVAL_API_KEY"),
        base_url=os.environ.get("HARNESS_EVAL_BASE_URL"),
        model=os.environ.get("HARNESS_EVAL_MODEL"),
    )

    sop_registry = SOPRegistry()
    session_manager = SessionManager()
    orchestrator = Orchestrator(sop_registry, session_manager, evaluator)

    app = Server("harness")

    @app.list_tools()
    async def list_tools() -> list[Tool]:
        return TOOLS

    @app.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        try:
            result = await _dispatch(orchestrator, name, arguments)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as exc:
            logger.exception("Tool call failed: %s", name)
            error_response = {"success": False, "message": f"Internal error: {exc}"}
            return [TextContent(type="text", text=json.dumps(error_response, indent=2))]

    return app, orchestrator


async def _dispatch(orchestrator: Orchestrator, name: str, arguments: dict) -> dict:
    """Route tool calls to orchestrator methods."""
    if name == "harness_start":
        return orchestrator.start_session(
            sop_id=arguments["sop_id"],
            context=arguments.get("context"),
            retry_limit=arguments.get("retry_limit"),
        ).to_dict()

    elif name == "harness_submit_step":
        return (await orchestrator.submit_step(
            session_id=arguments["session_id"],
            step_output=arguments["step_output"],
        )).to_dict()

    elif name == "harness_report_evaluation":
        return orchestrator.report_evaluation(
            session_id=arguments["session_id"],
            evaluation_json=arguments["evaluation"],
        ).to_dict()

    elif name == "harness_list_sops":
        sops = orchestrator._sops.list_sops()
        return {
            "success": True,
            "message": f"{len(sops)} SOP(s) available.",
            "data": {"sops": sops},
        }

    elif name == "harness_resume":
        return orchestrator.resume_session(
            session_id=arguments["session_id"],
            comment=arguments.get("comment", ""),
        ).to_dict()

    elif name == "harness_skip_step":
        return orchestrator.skip_step(
            session_id=arguments["session_id"],
            reason=arguments.get("reason", ""),
        ).to_dict()

    elif name == "harness_list_sessions":
        return orchestrator.list_sessions().to_dict()

    elif name == "harness_get_status":
        return orchestrator.get_status(
            session_id=arguments["session_id"],
        ).to_dict()

    elif name == "harness_get_feedback":
        return orchestrator.get_feedback(
            session_id=arguments["session_id"],
            step_index=arguments.get("step_index"),
        ).to_dict()

    else:
        return {"success": False, "message": f"Unknown tool: {name}"}


async def main():
    """Run the harness MCP server over stdio."""
    logging.basicConfig(level=logging.INFO)
    app, _ = create_server()
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
