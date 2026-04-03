---
name: harness-orchestrate
description: Drive the harness submit-evaluate-iterate loop for any SOP workflow
---

# Harness Orchestration Skill

You are executing a quality-gated SOP workflow using the harness MCP server.

## Core Loop

For each step in the workflow:

1. **Read** the step instruction and acceptance criteria from the harness response
2. **Plan** your approach - identify what artifacts you need to produce
3. **Execute** the work using your available tools (Read, Write, Edit, Bash, Grep, etc.)
4. **Self-assess** honestly against each acceptance criterion before submitting
5. **Submit** via `harness_submit_step` with:
   - `artifacts`: list of what you produced (files written, code blocks, analysis text)
   - `self_assessment`: honest evaluation of how well you met each criterion
6. **Evaluate** - after submitting, the harness returns an evaluation prompt:
   - Dispatch the `harness-reviewer` agent with the evaluation prompt from the response
   - The reviewer agent evaluates your work independently (fresh context, no access to your reasoning)
   - Call `harness_report_evaluation` with the reviewer's JSON response
7. **React to verdict**:
   - PASS: proceed to the next step
   - FAIL: read the `top_3_fixes` carefully, address each one specifically, then resubmit

## Evaluation Flow (Subagent Mode)

```
You do work
  -> harness_submit_step (validates, returns evaluation prompt)
  -> You dispatch harness-reviewer agent with the prompt
  -> Reviewer returns structured JSON scores
  -> harness_report_evaluation (processes scores, returns PASS/FAIL)
  -> If PASS: next step. If FAIL: iterate.
```

The reviewer agent is a SEPARATE entity with a fresh context. It cannot see your chain of thought. This is by design -- self-evaluation is unreliable.

## Dispatching the Reviewer

When `harness_submit_step` returns `stage: "awaiting_evaluation"`, extract the evaluation prompt from `data.evaluation_prompt` and dispatch it:

```
Use the Agent tool to launch the harness-reviewer agent with:
  - The system_prompt from data.evaluation_prompt.system_prompt
  - The user_prompt from data.evaluation_prompt.user_prompt
  - Ask it to return ONLY the JSON evaluation
```

Then take the reviewer's JSON response and call `harness_report_evaluation` with it.

## Quality Gate Dimensions

The reviewer scores on:
- **Completeness (30%)**: Did you address every acceptance criterion?
- **Specificity (25%)**: Does your output reference THIS project, not generic boilerplate?
- **Correctness (20%)**: Are claims accurate? Does code compile? Does logic hold?
- **Actionability (15%)**: Can the next step act on your output without guessing?
- **Format Compliance (10%)**: Does output match the required structure?

## Anti-Patterns to Avoid

- Submitting generic/boilerplate content (the reviewer flags "AI slop")
- Restating the instruction as a finding
- Filler phrases like "This ensures a seamless user experience"
- Submitting without checking every acceptance criterion individually
