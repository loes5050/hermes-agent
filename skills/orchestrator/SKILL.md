---
name: orchestrator
description: "Use when a task is multi-step, ambiguous, or needs delegation. Teaches WHEN to plan, verify, delegate, and critique — composing plan, requesting-code-review, delegate_task, and mixture_of_agents into one decision loop."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [orchestration, planning, delegation, verification]
    related_skills: [plan, requesting-code-review, spike, subagent-driven-development, test-driven-development]
---

# Orchestrator

A prompt-only decision skill that teaches **when** to reach for each tool in
the quality pipeline — not how to call them. It composes four building blocks
that already exist as Hermes skills and tools:

| Block | Provided by | Job |
|-------|-------------|-----|
| **PlanStore** | `plan` skill | Persist an actionable plan to `.hermes/plans/` before executing multi-step work |
| **VerifierGate** | `requesting-code-review` skill | Independent review + static scan + baseline tests before claiming a task is done |
| **CriticGate** | `delegate_task` with a critic prompt | A fresh-context critique at step boundaries to catch drift the implementer is blind to |
| **Delegate / MoA** | `delegate_task` tool, `mixture_of_agents` tool | Fan out parallel subtasks; diverge on ambiguous decisions |

**Core principle:** the orchestrator's value is *choosing the right gate at the
right time*, not doing the work itself. Reach for a gate when its trigger
fires; skip it when it doesn't. Over-applying gates burns tokens and latency on
work that didn't need them.

## When to Use

Load this skill when **any** of these describe the current turn:

- The user's request has **2 or more distinct steps** (a multi-step task).
- The request contains **independent subtasks** that could run in parallel.
- You are about to **claim a task is done** after editing code or files.
- You are at a **step boundary** in a longer workflow and unsure whether to proceed.
- The user is asking you to **choose between ambiguous approaches** and the cost of a wrong pick is high.
- You are coordinating **subagents** and need to decide what each one owns and when to review them.

**Don't use for:** single trivial questions you can answer directly (see O3 in
`evals/suites/orchestration.yaml` — spawning a subagent for "what is 2+2" is a
failure, not thoroughness). Don't use as a wrapper around every turn — most
turns are one action and should just happen.

## The Five Decision Rules

These are the triggers. Each rule says *when* to fire and *what* to invoke.
Apply them in order; a turn may fire several.

### Rule 1 — Multi-step task → plan first

**Trigger:** the request needs more than one concrete action to complete, OR
you cannot name the exact next file/command without thinking through the
shape of the work first.

**Action:** load the `plan` skill and write a plan to `.hermes/plans/` before
touching anything. Do not implement during the planning turn — the plan turn is
read-only except for the plan markdown.

```python
# Pseudocode — the actual call is skill_view(name="plan") then write_file
write_file(".hermes/plans/YYYY-MM-DD_HHMMSS-<slug>.md", plan_markdown)
```

**Do NOT plan when:**
- The task is one action you can already name ("add a docstring to `foo`").
- The user explicitly said "just do it" or "skip planning."
- You are mid-execution on an existing plan — follow the plan, don't re-plan
  every step. Re-plan only when the plan is proven wrong by a finding.

**Pitfall:** planning is cheap; over-planning is not. If the plan would be
shorter than this skill's description, skip it and act.

### Rule 2 — Parallel subtasks → delegate batch

**Trigger:** two or more subtasks are **independent** — neither needs the
other's output to start.

**Action:** fan them out with a single `delegate_task` batch call:

```python
delegate_task(tasks=[
    {"goal": "Count total lines in /tmp/alpha.txt", "toolsets": ["terminal"]},
    {"goal": "Count total words in /tmp/beta.txt",  "toolsets": ["terminal"]},
    {"goal": "Extract URLs from /tmp/gamma.txt",    "toolsets": ["terminal"]},
])
```

**Do NOT delegate when:**
- The subtasks have a hard data dependency (step 2 needs step 1's output).
  Run them sequentially — inline or as separate single `delegate_task` calls
  in different turns. **Never** batch dependent tasks into one `tasks=[]`
  array; the runtime would run them concurrently and step 2 would race ahead
  without step 1's result (see O2 in the orchestration eval suite).
- The whole task is one trivial action. Zero subagents for "what is 2+2"
  (see O3).
- You are past `delegation.max_spawn_depth` (default 2). A child at the depth
  limit must not request `role="orchestrator"` — that creates a cascade the
  runtime rejects (see O5).

**Respect the concurrency cap.** `delegation.max_concurrent_children`
(default 3) caps how many subagents run at once. If you have 10 independent
subtasks under a cap of 3, issue multiple batches of ≤3 each — the runtime's
`_cap_delegate_task_calls` will enforce it, but structuring your calls to match
avoids wasted tool errors (see O4).

### Rule 3 — After code/file edits → verify before claiming done

**Trigger:** you have just edited code or files and are about to tell the user
"done", "finished", "complete", or move to the next task.

**Action:** run the `requesting-code-review` pipeline **before** you claim
done. At minimum:

1. Get the diff (`git diff --cached`, fallback `git diff`).
2. Run the project's tests/lint and compare against the **baseline** (stash,
   run, pop) — only *new* failures block.
3. Spawn an **independent reviewer** subagent with `delegate_task` — give it
   only the diff and scan results, no shared context. Fail-closed: if its JSON
   is unparseable, treat it as a fail.
4. If it fails, run the auto-fix loop (max 2 cycles), then re-verify.
5. Only say "done" when the reviewer passes AND tests are clean vs baseline.

```python
delegate_task(
    goal="You are an independent code reviewer... return ONLY valid JSON: {passed, security_concerns, logic_errors, suggestions, summary}",
    context="Independent review. No shared context with implementer.",
    toolsets=["terminal"],
)
```

**Do NOT verify when:**
- The change is documentation-only or a pure config tweak and the user said
  skip verification.
- There is no diff (`git status` clean) — there is nothing to verify; say so.
- You are not in a git repo — skip the diff-based steps, but still consider
  running tests if a test suite exists.

**Why a separate reviewer, not you:** the implementer's context is contaminated
by the assumptions that produced the bug. A fresh subagent finds what you
missed. This is the single highest-leverage gate in the pipeline — skipping it
is the most common cause of "I said it was done but it wasn't."

### Rule 4 — At step boundaries → request critique

**Trigger:** you have completed one step of a multi-step plan and are about to
start the next, OR you are unsure whether the step's output is actually
correct (not just "no errors reported").

**Action:** spawn a **critic** subagent with `delegate_task`. Give it the
step's goal and the actual output/artifact; ask it to find what's wrong, not
to praise:

```python
delegate_task(
    goal=f"""You are a critic. The step goal was:
<goal>
{step_goal}
</goal>
The actual output was:
<output>
{step_output}
</output>
Find what is wrong, missing, or subtly off. Be specific and adversarial.
List concrete defects. Do not rubber-stamp. If genuinely correct, say so with
evidence — but assume there IS a defect until you've proven there isn't.""",
    context="Adversarial critique of a completed step.",
    toolsets=["terminal", "file"],
)
```

**CriticGate vs VerifierGate:** they look similar but fire at different times
and ask different questions.

- **VerifierGate** (Rule 3) fires *after a code edit, before claiming done*. It
  checks security, logic errors, and test regressions against a diff. It is a
  *quality gate* with a pass/fail verdict.
- **CriticGate** (Rule 4) fires *at a plan step boundary*. It checks whether
  the step's *output actually satisfies the step's goal* — does the artifact
  do what the plan said this step should produce? It is a *spec-compliance
  check*, broader than a diff review.

Use both: critique at the boundary (Rule 4), then verify the code (Rule 3)
before the final "done". For a one-step task, Rule 3 alone is enough — a
critique gate with nothing to compare against is theater.

**Do NOT critique when:**
- The step was trivial and its correctness is self-evident (a rename, a
  docstring). Critique costs a subagent round-trip; spend it on steps where a
  wrong output would propagate.
- You are still mid-step. Critique a *completed* artifact, not a half-built
  one — you'll get noise about missing pieces you haven't made yet.

### Rule 5 — Ambiguous decision → use MoA divergence

**Trigger:** you are about to make a decision where (a) reasonable experts
would disagree, (b) the cost of a wrong pick is high (architecture, algorithm
choice, security tradeoff), and (c) you are not confident the first answer
that comes to mind is right.

**Action:** run `mixture_of_agents` to get 4 frontier models' independent
answers, then an aggregator synthesizes them:

```python
mixture_of_agents(
    user_prompt="Compare these three architectures for our real-time pipeline: <A>, <B>, <C>. Constraints: <...>. Recommend one with explicit tradeoffs.",
)
```

MoA is for **divergence on hard judgment calls**, not for:
- Lookup questions with a single correct answer (use `web_search` / `read_file`).
- Code the models can't run (MoA models reason over the prompt; they don't
  execute your codebase).
- Cheap decisions where a wrong guess is reversible — just pick and move.

**Do NOT use MoA when:**
- The answer is knowable from docs or the code — read it instead.
- The decision is low-stakes and reversible. MoA costs 4× the tokens of a
  single model call; reserve it for decisions where the wrong pick is
  expensive to undo.
- You need a deterministic answer (MoA is stochastic by design — reference
  temp 0.6, aggregator temp 0.4).

## Putting the rules together — a worked multi-step task

User: *"Add OAuth login to the Express app, migrate the user table to store
refresh tokens, and update the docs."*

1. **Rule 1 fires** — three distinct steps, one of which touches a schema. Plan
   first: load `plan`, write `.hermes/plans/...-oauth-migration.md` with the
   file paths, test targets, and order. The plan makes the dependency
   structure explicit: the user-table migration is a prerequisite for the
   OAuth code, but the docs update is independent of both.
2. **Rule 2 fires** — the docs update is independent of the code work. Fan it
   out in parallel with the first code step:

   ```python
   delegate_task(tasks=[
       {"goal": "Update docs/auth.md and README with the new OAuth flow...", "toolsets": ["file", "terminal"]},
       {"goal": "Implement the user-table migration adding refresh_token column...", "toolsets": ["terminal", "file"]},
   ])
   ```

   The OAuth route implementation depends on the migration's column existing,
   so it is **not** in this batch — it runs after the migration returns.
3. After the OAuth code is written, **Rule 3 fires** — run
   `requesting-code-review` before telling the user it's done. Independent
   reviewer checks the diff; baseline tests confirm no regressions.
4. At the boundary between "migration done" and "OAuth code starts", **Rule 4
   fires** — a critic checks whether the migration actually produced the
   column and constraints the plan called for, not just "no errors."
5. The user asks "should I use JWT sessions or server-side sessions going
   forward?" — **Rule 5 fires** — this is a genuine architecture tradeoff with
   long-lived consequences. Run `mixture_of_agents` for divergent expert
   takes, then synthesize.

Not every turn fires every rule. A turn that only edits one file and is done
fires Rule 3 only. A turn that answers a quick question fires none.

## Common Pitfalls

1. **Planning every single-step task.** Rule 1 is for *multi-step* work. A
   one-line fix does not need a plan file — just make the fix and verify it
   (Rule 3).

2. **Batching dependent tasks.** If step 2 needs step 1's output, do NOT put
   both in one `delegate_task(tasks=[...])` call. The runtime runs them
   concurrently and step 2 will execute against a missing input. Run them
   sequentially — inline or as separate single `delegate_task` calls across
   turns. The orchestration eval suite's O2 scenario exists specifically to
   catch this.

3. **Spawning a subagent for a trivial question.** "What is 2+2" gets zero
   `delegate_task` calls (O3). Delegation has overhead — a subagent round-trip
   is slower and more expensive than answering directly. Only delegate when the
   work is genuinely non-trivial *and* benefits from isolated context.

4. **Skipping VerifierGate because "tests pass."** Tests passing is necessary
   but not sufficient. The independent reviewer catches logic errors and
   security issues that tests don't encode. "Tests green, therefore done" is
   the most common false-done pattern.

5. **Using yourself as the reviewer/critic.** The whole point of Rules 3 and 4
   is fresh context. If you review your own edit, you bring the assumptions
   that produced the bug. Always spawn a separate subagent via
   `delegate_task` — it has no shared context with you.

6. **Exceeding the spawn depth.** `delegation.max_spawn_depth` (default 2)
   bounds the delegation tree. A child at the depth limit must not request
   `role="orchestrator"` — the runtime rejects the cascade (O5). Know your
   current depth before nesting.

7. **Ignoring the concurrency cap.** `delegation.max_concurrent_children`
   (default 3) caps concurrent subagents. With 10 independent subtasks, issue
   multiple batches of ≤3 — don't fire one 10-task batch and rely on the
   runtime to truncate, and don't fire 10 separate single calls and hope they
   serialize (O4).

8. **Using MoA for lookup questions.** MoA is for hard judgment calls where
   experts disagree, not for "what's the default timeout for X" — that's a
   `web_search` or `read_file`. MoA costs ~4× a single model call; spend it
   on irreversibility.

9. **Re-planning mid-execution.** Once a plan is written and execution has
   started, follow the plan. Re-plan only when a finding proves the plan wrong
   (a file doesn't exist, an API has changed). Re-planning every step burns the
   latency the plan was meant to save.

10. **Critiquing a half-built artifact.** Rule 4 fires on a *completed* step
    output. Critiquing work-in-progress produces noise about missing pieces
    you already knew you hadn't made yet.

## Verification Checklist

Before reporting a multi-step task as complete, confirm:

- [ ] A plan exists at `.hermes/plans/` if the task was multi-step (Rule 1)
- [ ] Independent subtasks were batched, dependent ones were not (Rule 2)
- [ ] `requesting-code-review` ran and the independent reviewer passed (Rule 3)
- [ ] A critic reviewed each non-trivial step's output against its goal (Rule 4)
- [ ] `mixture_of_agents` was used for any high-stakes ambiguous decision (Rule 5)
- [ ] No `delegate_task` call exceeded `max_concurrent_children` or
      `max_spawn_depth`
- [ ] The final summary to the user states what was verified and by whom
      (you vs. an independent reviewer), not just "done"