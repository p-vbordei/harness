---
name: harness
description: Start a quality-gated SOP workflow. Usage: /harness <sop-id> [context]
arguments:
  - name: sop_id
    description: The SOP to run (e.g., 'feature-dev', 'investigation', 'code-review')
    required: true
  - name: context
    description: Optional context for the SOP (e.g., feature description, target repo)
    required: false
---

You are now operating in **harness mode** - a quality-gated workflow where every step is evaluated by an independent LLM judge.

## How It Works

1. You call `harness_start` with the SOP ID to begin a session
2. For each step, you receive an instruction and acceptance criteria
3. You do the work, then call `harness_submit_step` with your output
4. A separate Haiku LLM evaluates your work on 5 dimensions (completeness, specificity, correctness, actionability, format compliance)
5. If you PASS (score >= 3.5, no dimension below 3): advance to next step
6. If you FAIL: you receive specific feedback with top 3 fixes. Retry.
7. After 3 failed attempts: escalate to human

## Start Now

1. Call `harness_start` with sop_id: `$ARGUMENTS.sop_id`
2. Read the first step's instruction and acceptance criteria from the response
3. Do the work thoroughly - the evaluator is skeptical by design
4. Submit via `harness_submit_step` with your artifacts and a self-assessment
5. If feedback says FAIL, address the top_3_fixes specifically before resubmitting
6. Continue until all steps are complete

## Rules

- **Do not skip the quality gate.** Every step must be submitted and evaluated.
- **Address feedback specifically.** Generic improvements won't pass. Fix the exact gaps cited.
- **Self-assessment is required.** Before submitting, honestly assess your work against the acceptance criteria.
- **Check status anytime** with `harness_get_status` or review feedback with `harness_get_feedback`.
