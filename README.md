# Harness - Quality-Gated SOP Execution

A **MCP server + Claude Code plugin** that enforces quality gates on any multi-step workflow. Every step submission is evaluated by a **separate, independent evaluator** -- not self-evaluation -- using a 6-dimension scoring rubric with calibration anchors and AI slop detection.

## Why This Exists

Anthropic's [own research](https://www.anthropic.com/engineering/harness-design-long-running-apps) proves that when LLMs evaluate their own work, they "confidently praise their own mediocre work." A standalone evaluator tuned for skepticism is far more tractable than making a generator self-critical.

This harness implements that insight: **the entity doing the work never judges its own output.**

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│  1. Define SOP (YAML)                                       │
│     phases → steps → acceptance_criteria                    │
│                                                             │
│  2. Start Session                                           │
│     harness_start("feature-dev") → session_id + step 1     │
│                                                             │
│  3. For each step:                                          │
│     a. Agent does the work                                  │
│     b. Agent submits via harness_submit_step                │
│     c. Separate evaluator scores on 6 dimensions            │
│     d. PASS → next step | FAIL → retry with feedback        │
│     e. 3 fails → escalate to human                          │
│                                                             │
│  4. All steps pass → session complete                       │
└─────────────────────────────────────────────────────────────┘
```

## Features

### Evaluator System
- **3 backends**: Subagent (default, no API key), Anthropic API (Haiku), OpenAI-compatible (vLLM/Ollama)
- **6 scoring dimensions**: Completeness (25%), Specificity (20%), Correctness (20%), Coherence (10%), Actionability (15%), Format Compliance (10%)
- **3 profiles**: `default` (threshold 3.5), `strict` (4.0), `lenient` (3.0)
- **AI slop detection**: 15+ flagged patterns; 3+ flags auto-penalize specificity score
- **Calibration anchors**: Scored examples prevent evaluator drift
- **Prompt injection defense**: User content wrapped in structural delimiters

### Workflow Engine
- **YAML SOP definitions** with phases, steps, dependencies (topological sort)
- **Per-step evaluator profiles** and optional timeouts
- **Session management**: Resume blocked sessions, skip steps, list all sessions
- **Event sourcing**: Automatic recovery from corrupted state files
- **Atomic writes**: Crash-safe state persistence

### MCP Tools (9)

| Tool | Purpose |
|------|---------|
| `harness_start` | Begin a session with an SOP |
| `harness_submit_step` | Submit output for current step |
| `harness_report_evaluation` | Report subagent evaluation results |
| `harness_get_status` | Get session status |
| `harness_get_feedback` | Get evaluation feedback history |
| `harness_list_sops` | Discover available SOPs |
| `harness_resume` | Resume a paused/blocked session |
| `harness_skip_step` | Skip current step |
| `harness_list_sessions` | List all sessions |

### Built-in SOP Templates

| Template | Phases | Steps | Use Case |
|----------|--------|-------|----------|
| `feature-dev` | 4 | 12 | Feature development lifecycle |
| `investigation` | 3 | 7 | Security/research investigation |
| `code-review` | 3 | 7 | Multi-perspective code review |
| `_TEMPLATE` | - | - | Commented template for new SOPs |

## Installation

### As a standalone MCP server

```bash
git clone https://github.com/p-vbordei/harness.git
cd harness
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### As a Claude Code plugin

Copy the `harness/` directory to your Claude Code plugins location, or symlink it:

```bash
ln -s /path/to/harness ~/.claude/plugins/harness
```

The `.mcp.json` at the plugin root registers the MCP server automatically.

## Configuration

### Evaluator Backends

| Backend | Env Vars | Cost | Notes |
|---------|----------|------|-------|
| `subagent` (default) | None | Free | Uses Claude Code reviewer agent |
| `anthropic` | `HARNESS_EVAL_BACKEND=anthropic`, `ANTHROPIC_API_KEY` | ~$0.001/eval | Haiku by default |
| `openai` | `HARNESS_EVAL_BACKEND=openai`, `HARNESS_EVAL_BASE_URL`, `HARNESS_EVAL_MODEL` | Varies | vLLM, Ollama, etc. |

### Evaluator Profiles

Profiles control scoring thresholds per step. Set in SOP YAML:

```yaml
steps:
  - id: security-check
    evaluator_profile: strict    # threshold 4.0, correctness weighted 25%
  - id: draft-outline
    evaluator_profile: lenient   # threshold 3.0, no slop penalty
```

## Usage

### From Claude Code CLI

```
/harness feature-dev
```

This starts a 12-step feature development workflow. For each step:
1. You receive an instruction and acceptance criteria
2. You do the work
3. You submit via `harness_submit_step`
4. The reviewer agent evaluates independently
5. You call `harness_report_evaluation` with the result
6. PASS → next step, FAIL → address feedback and retry

### Programmatic (Python)

```python
from server.orchestrator import Orchestrator
from server.evaluator import create_evaluator
from server.session_manager import SessionManager
from server.sop_registry import SOPRegistry

registry = SOPRegistry()
manager = SessionManager()
evaluator = create_evaluator()  # subagent mode

orch = Orchestrator(registry, manager, evaluator)

# Start
resp = orch.start_session("feature-dev")
session_id = resp.session_id

# Submit step output
resp = await orch.submit_step(session_id, {
    "artifacts": [{"type": "text", "content": "My requirements analysis..."}],
    "self_assessment": "All criteria addressed with specific details."
})

# In subagent mode, dispatch evaluation, then report:
resp = orch.report_evaluation(session_id, evaluation_json)
```

## Creating Custom SOPs

1. Copy `sops/_TEMPLATE.yaml` to `sops/my-workflow.yaml`
2. Edit phases, steps, and acceptance criteria
3. The harness auto-discovers YAML files in `sops/` and `~/.harness/sops/`

Key fields per step:

```yaml
- id: my-step
  title: "Human-readable title"
  instruction: "What the agent should do"
  acceptance_criteria:
    - "Specific, testable criterion 1"
    - "Specific, testable criterion 2"
  evaluator_profile: default  # or strict, lenient
  timeout: 600                # optional, seconds
  on_fail: retry              # retry, skip, or abort
  depends_on: [other-step]    # within same phase
```

## Quality Gate Scoring

Each submission is scored on 6 dimensions:

| Dimension | Weight | What It Measures |
|-----------|--------|-----------------|
| Completeness | 25% | Every acceptance criterion addressed |
| Specificity | 20% | References THIS project, not boilerplate |
| Correctness | 20% | Claims accurate, code compiles, logic holds |
| Coherence | 10% | No internal contradictions |
| Actionability | 15% | Downstream steps can act without guessing |
| Format Compliance | 10% | Matches required structure |

**Pass condition**: Weighted average >= threshold AND no single dimension below minimum.

**Slop penalty**: 3+ AI filler phrases detected → specificity score auto-reduced by 1 point.

## Session Persistence

Sessions are stored at `~/.harness/sessions/{session_id}/`:

```
state.json              # Current position (atomic writes)
sop_snapshot.yaml       # Frozen copy of SOP
steps/                  # Per-step attempt history
  phase.step/
    attempt_1.json
    attempt_2.json
events.jsonl            # Append-only audit trail
usage.json              # Usage statistics
```

If `state.json` is corrupted, the system automatically recovers by replaying `events.jsonl`.

## Testing

```bash
# Run all 78 tests
pytest tests/ -v

# Run specific test modules
pytest tests/test_evaluator.py -v      # Evaluator backends, parsing, profiles
pytest tests/test_orchestrator.py -v   # Workflow loop, submit/evaluate/advance
pytest tests/test_p2_features.py -v    # Profiles, resume/skip, slop penalty
pytest tests/test_session_manager.py -v # State persistence, atomic writes
pytest tests/test_sop_registry.py -v   # YAML loading, dependency sorting
pytest tests/test_validation.py -v     # Input validation, schema checks
```

## Architecture

The system was designed through a multi-agent research pipeline:
- 16 research agents analyzed existing harness patterns (Anthropic blog, claude-code-harness, claw-code, WaveSpeed analysis)
- 10 A-Team agents examined the design from different perspectives
- 12 improvement agents reviewed and hardened the implementation
- All validated through web research and debate rounds

Key design decisions documented in the [implementation plan](/Users/vladbordei/.claude/plans/stateful-snacking-hamming.md).

## License

Private repository.
