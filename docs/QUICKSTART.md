# Quickstart

## What is the Harness?

The Harness is a Claude Code plugin that enforces quality gates on multi-step workflows (SOPs). Every step you submit is evaluated by a separate LLM judge -- not self-evaluation -- using a 5-dimension scoring rubric, so work iterates until it genuinely passes or escalates to human review.

## Installation

1. **Copy the plugin** into your Claude Code plugins directory:

   ```bash
   cp -r /path/to/harness ~/.claude/plugins/harness
   ```

2. **Set up a virtual environment** and install dependencies:

   ```bash
   cd ~/.claude/plugins/harness
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e .
   ```

3. **Set your API key** (required for the `anthropic` evaluator backend; not needed for the default `subagent` backend):

   ```bash
   export ANTHROPIC_API_KEY="sk-ant-..."
   ```

## Running Your First Workflow

Start the built-in feature development SOP:

```
/harness feature-dev
```

You can pass optional context:

```
/harness feature-dev "Add user preference API endpoint"
```

## What Happens: The Submit-Evaluate-Iterate Loop

1. **Start** -- The harness loads the SOP and presents the first step's instruction and acceptance criteria.
2. **Work** -- You do the work as described in the instruction.
3. **Submit** -- You call `harness_submit_step` with your output and a self-assessment.
4. **Evaluate** -- A separate LLM evaluator scores your submission on 5 dimensions: completeness (30%), specificity (25%), correctness (20%), actionability (15%), and format compliance (10%).
5. **Pass or Fail** -- If the weighted average is >= 3.5 and no single dimension is below 3, you advance. Otherwise, you get the top 3 fixes and retry.
6. **Iterate** -- Address the specific feedback (not generic improvements) and resubmit. After 3 failed attempts, the step escalates to human review.
7. **Complete** -- Once all steps pass, the session is marked complete.

## Creating Your Own SOP

1. Copy the template:

   ```bash
   cp sops/_TEMPLATE.yaml sops/my-workflow.yaml
   ```

2. Edit the file. At minimum you need:
   - `sop_id` -- unique identifier (used in `/harness <sop_id>`)
   - `name` -- human-readable name
   - `phases` -- at least one phase with at least one step

3. Each step needs:
   - `id`, `title`, `instruction` -- what to do
   - `acceptance_criteria` -- list of specific, testable criteria the evaluator checks

4. Optional step fields:
   - `depends_on` -- list of step IDs (within the same phase) that must complete first
   - `on_fail` -- `retry` (default, escalates to human), `skip`, or `abort`
   - `evaluator_profile` -- `default`, `strict`, or `lenient`
   - `timeout` -- seconds before the step times out

See `sops/_TEMPLATE.yaml` for a fully commented reference with examples.

## Configuration

### Evaluator Backends

The harness supports three evaluator backends:

| Backend | Key Required | Description |
|---------|-------------|-------------|
| `subagent` (default) | No | Dispatches evaluation to a Claude Code subagent. No API key needed. The subagent runs in a fresh context window. |
| `anthropic` | Yes (`ANTHROPIC_API_KEY`) | Server-side call to Claude Haiku. Fast and cheap. |
| `openai` | Varies | Any OpenAI-compatible endpoint (vLLM, Ollama, LiteLLM). Set `base_url` and optionally `api_key`. |

The default `subagent` backend works out of the box with no configuration.

### SOP Search Directories

The harness looks for YAML SOP files in:
1. `sops/` directory in the plugin root
2. `~/.harness/sops/` for personal SOPs shared across projects

## Troubleshooting

**"SOP not found"** -- Check that your YAML file is in `sops/` and has a valid `sop_id` field. File must end in `.yaml` or `.yml`.

**"Submission validation failed"** -- Your submission did not pass Layer 1 (deterministic) checks. If the step defines an `output_schema`, your output must match it structurally. Check the `errors` field in the response.

**Step keeps failing evaluation** -- Read the `top_3_fixes` carefully. The evaluator is skeptical by design. Address each fix specifically rather than making general improvements. Check `harness_get_feedback` for the full scoring breakdown.

**"Session is paused/blocked"** -- A step exhausted its retries. Human review is needed. Look at the feedback history with `harness_get_feedback` to understand what the evaluator wanted.

**"No active step in session"** -- The session may have completed or failed. Check status with `harness_get_status`.

**Evaluator returns errors** -- For the `anthropic` backend, verify `ANTHROPIC_API_KEY` is set and valid. For the `openai` backend, verify `base_url` is reachable.
