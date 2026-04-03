# Harness - Quality-Gated SOP Execution

## What This Is

An MCP server + Claude Code plugin that enforces quality gates on multi-step workflows (SOPs). Every step submission is evaluated by a separate entity (subagent with fresh context, or server-side LLM) using a 6-dimension scoring rubric with calibrated anchors and slop detection.

## Architecture

- **MCP Server** (`server/`): Python, 9 tools
- **Evaluator backends**: Subagent (default, no API key), Anthropic API, OpenAI-compatible (vLLM/Ollama)
- **6 evaluation dimensions**: completeness (25%), specificity (20%), correctness (20%), coherence (10%), actionability (15%), format compliance (10%)
- **3 evaluator profiles**: default (3.5 threshold), strict (4.0), lenient (3.0)
- **Pass condition**: Weighted average >= threshold AND no dimension below minimum
- **Slop penalty**: 3+ AI slop flags auto-deduct from specificity score
- **SOPs**: YAML files in `sops/` with phases, steps, acceptance criteria, evaluator profiles, timeouts

## MCP Tools (9)

| Tool | Purpose |
|------|---------|
| `harness_start` | Begin a session with an SOP |
| `harness_submit_step` | Submit output for current step (returns eval prompt in subagent mode) |
| `harness_report_evaluation` | Report subagent evaluation results |
| `harness_get_status` | Get session status |
| `harness_get_feedback` | Get evaluation feedback history |
| `harness_list_sops` | Discover available SOPs |
| `harness_resume` | Resume a paused/blocked session |
| `harness_skip_step` | Skip current step and advance |
| `harness_list_sessions` | List all sessions |

## Development

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v
```

## Evaluator Configuration

Default: subagent mode (no API key needed - uses Claude Code reviewer agent).

For API backends, set environment variables:
```bash
# Anthropic (Haiku)
HARNESS_EVAL_BACKEND=anthropic
ANTHROPIC_API_KEY=sk-...

# OpenAI-compatible (vLLM, Ollama)
HARNESS_EVAL_BACKEND=openai
HARNESS_EVAL_BASE_URL=http://localhost:8000/v1
HARNESS_EVAL_MODEL=my-model
```

## Key Design Decisions

1. **Evaluator is a separate entity** - Anthropic's research proves self-evaluation is biased
2. **Verdict computed server-side** - LLM provides scores, code applies pass/fail logic
3. **3 max retries** per step, then human escalation
4. **Atomic state writes** - temp file + os.replace() prevents corruption
5. **Two-layer validation** - deterministic checks first, LLM evaluation second
6. **Calibration anchors** - scored examples in evaluator prompt prevent drift
7. **Slop penalty** - AI filler phrases auto-penalize specificity score
8. **Prompt injection defense** - user content wrapped in structural delimiters
9. **Session quota** - max 100 sessions prevents filesystem DOS

## File Structure

```
server/                # MCP server (Python)
  models.py            # Data models
  sop_registry.py      # YAML SOP loader with topological sort
  session_manager.py   # Filesystem state + event sourcing recovery
  validation.py        # Layer 1 deterministic checks
  evaluator.py         # Layer 2 LLM evaluation (3 backends, 3 profiles)
  orchestrator.py      # Core workflow loop + resume/skip
  usage_tracker.py     # Per-session usage statistics
  server.py            # MCP entry point (9 tools)
sops/                  # SOP templates
  feature-dev.yaml     # Feature development (4 phases, 12 steps)
  investigation.yaml   # Security/research investigation
  code-review.yaml     # Code review workflow
  _TEMPLATE.yaml       # Template for creating new SOPs
docs/
  QUICKSTART.md        # Getting started guide
commands/              # /harness command
skills/                # Orchestration skill
agents/                # Reviewer agent
hooks/                 # Stop hook
tests/                 # 78 tests
```
