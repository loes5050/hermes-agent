"""M5 StrategyCurator — learn from successful PlanStore plans.

Extends the curator with a ``strategy_pass`` that reads successful PlanStore
plans (status=``done`` with passing verification evidence on steps) and
extracts recurring step patterns. Those patterns are used to propose patches
to the orchestrator skill (``skills/orchestrator/SKILL.md``) or to create new
skills capturing the patterns.

The strategy pass runs after the regular consolidation pass when
``curator.strategy_pass`` is true (default false). It reuses the existing
curator backup (taken before any mutating pass) and respects the
``skills.write_approval`` gate — proposed writes are staged for approval
when the gate is on.

Config keys (under ``curator``):

- ``strategy_pass`` (bool, default false) — enable the strategy extraction pass
- ``strategy_min_successes`` (int, default 3) — minimum number of successful
  plans a step pattern must recur in to be considered a candidate
- ``strategy_extract_model`` (str, default '') — model override for the
  extraction fork (empty = use the curator aux model)

Design constraints (from AGENTS.md):

* **The core is a narrow waist.** StrategyCurator is a standalone module with
  no new core tools. It reuses the existing ``skill_manage`` tool surface
  (patch / create) via a forked AIAgent — the same pattern the consolidation
  pass uses.
* **Extend, don't duplicate.** The LLM fork reuses ``_resolve_review_runtime``
  from ``agent.curator`` for provider/model resolution, and the pattern
  extraction is deterministic (no LLM needed for the counting step).
* **Prompt caching is sacred.** The strategy fork runs as a separate
  ``AIAgent`` with ``skip_memory=True`` and ``skip_context_files=True`` — it
  never touches the main session's prompt cache.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

DEFAULT_STRATEGY_MIN_SUCCESSES = 3
DEFAULT_STRATEGY_EXTRACT_MODEL = ""

# ---------------------------------------------------------------------------
# Step normalization — map free-text step descriptions to canonical actions
# ---------------------------------------------------------------------------

#: Keyword → canonical-action map. First match wins. The mapping is deliberately
#: coarse — the goal is to identify structural patterns (read → edit → verify),
#: not to precisely classify every step.
_ACTION_MAP: List[Tuple[str, str]] = [
    (r"\b(read|view|inspect|examine|look|scan|search|find|list|explore|browse|open|load|fetch)\b", "read"),
    (r"\b(write|create|add|new|make|build|generate|produce|init|initialize|scaffold|seed)\b", "create"),
    (r"\b(edit|modify|update|change|patch|fix|refactor|adjust|replace|alter|rewrite|rename|move)\b", "edit"),
    (r"\b(run|execute|launch|start|invoke|call|trigger|fire)\b", "run"),
    (r"\b(test|verify|validate|assert|confirm|lint|typecheck)\b", "verify"),
    (r"\b(install|setup|configure|deploy|provision|bootstrap|register|enroll)\b", "configure"),
    (r"\b(delete|remove|drop|clean|purge|archive|teardown|deregister)\b", "delete"),
    (r"\b(analyze|review|audit|assess|evaluate|diagnose|investigate|debug|profile)\b", "analyze"),
    (r"\b(deploy|publish|release|ship|push|upload|promote|rollout)\b", "deploy"),
    (r"\b(plan|design|architect|draft|outline|sketch|specify|model)\b", "plan"),
    (r"\b(delegate|assign|dispatch|fanout|distribute|route)\b", "delegate"),
    (r"\b(document|describe|explain|comment|annotate|report)\b", "document"),
]


def _normalize_step(desc: str) -> str:
    """Map a free-text step description to a canonical action verb.

    The mapping is keyword-based: the first matching action class wins. This
    is deliberately coarse — the goal is to identify structural patterns
    (read → edit → verify), not to precisely classify every step.
    """
    desc_lower = desc.lower().strip()
    for pattern, canonical in _ACTION_MAP:
        if re.search(pattern, desc_lower):
            return canonical
    return "other"


# ---------------------------------------------------------------------------
# Plan collection — read done plans with passing evidence
# ---------------------------------------------------------------------------

def _collect_done_plans_with_evidence() -> List[Dict[str, Any]]:
    """Read all done plans whose steps have passing verification evidence.

    A plan qualifies when:
    - plan status == 'done'
    - at least one step has status == 'done' AND a non-empty evidence_event_id

    Returns a list of plan dicts (as returned by ``plan_store.get_plan_with_steps``)
    with their full step lists. Never raises — plan_store import failures or
    DB issues are caught and an empty list is returned.
    """
    try:
        from agent import plan_store
    except Exception as e:
        logger.debug("strategy_curator: plan_store import failed: %s", e)
        return []

    try:
        done_plans = plan_store.list_plans(status="done", limit=1000)
    except Exception as e:
        logger.debug("strategy_curator: list_plans failed: %s", e)
        return []

    result: List[Dict[str, Any]] = []
    for plan_row in done_plans:
        plan_id = plan_row.get("id")
        if not plan_id:
            continue
        try:
            plan = plan_store.get_plan_with_steps(plan_id)
        except Exception as e:
            logger.debug("strategy_curator: get_plan_with_steps(%s) failed: %s", plan_id, e)
            continue
        if not plan or not isinstance(plan, dict):
            continue
        steps = plan.get("steps") or []
        if not steps:
            continue
        # Qualify: at least one done step with evidence
        has_evidence = any(
            isinstance(s, dict)
            and s.get("status") == "done"
            and s.get("evidence_event_id")
            for s in steps
        )
        if not has_evidence:
            continue
        result.append(plan)
    return result


# ---------------------------------------------------------------------------
# Pattern extraction — find recurring step sequences across done plans
# ---------------------------------------------------------------------------

def _extract_patterns(
    plans: List[Dict[str, Any]],
    min_successes: int,
) -> List[Dict[str, Any]]:
    """Find recurring n-gram action patterns across done plans.

    For each plan, builds an ordered list of canonical action verbs from done
    steps, then counts all 2-grams through 5-grams across plans. A pattern
    counts once per plan even if it appears multiple times in that plan.

    Returns patterns appearing in >= min_successes plans, sorted by frequency.
    Subsumed shorter patterns (fully contained in a longer pattern with equal
    or higher count) are dropped to avoid noise.

    Each pattern dict:
    - pattern: list of action verbs (e.g. ``["read", "edit", "verify"]``)
    - length: n-gram size
    - plan_count: number of distinct plans containing this pattern
    - example_goals: up to 3 plan goals from plans containing this pattern
    - example_steps: up to 5 raw step descriptions from one matching plan
    """
    if not plans or min_successes < 1:
        return []

    # Build per-plan action sequences with metadata
    plan_data: List[Dict[str, Any]] = []
    for plan in plans:
        steps = plan.get("steps") or []
        seq: List[str] = []
        raw_steps: List[str] = []
        for step in steps:
            if not isinstance(step, dict):
                continue
            if step.get("status") != "done":
                continue
            desc = step.get("description") or ""
            if not desc.strip():
                continue
            seq.append(_normalize_step(desc))
            raw_steps.append(desc)
        if len(seq) >= 2:
            plan_data.append({
                "seq": seq,
                "goal": plan.get("goal", ""),
                "raw_steps": raw_steps,
            })

    if not plan_data:
        return []

    # Count n-grams (2-5) per plan — a pattern counts once per plan even if it
    # appears multiple times in the same plan's sequence.
    ngram_plan_indices: Dict[Tuple[str, ...], List[int]] = defaultdict(list)
    for idx, pd in enumerate(plan_data):
        seq = pd["seq"]
        seen: set = set()
        for n in range(2, min(6, len(seq) + 1)):
            for i in range(len(seq) - n + 1):
                seen.add(tuple(seq[i : i + n]))
        for ngram in seen:
            ngram_plan_indices[ngram].append(idx)

    # Filter and build results
    results: List[Dict[str, Any]] = []
    for ngram, indices in ngram_plan_indices.items():
        count = len(indices)
        if count < min_successes:
            continue
        example_goals: List[str] = []
        for idx in indices[:3]:
            goal = plan_data[idx]["goal"]
            if goal and goal not in example_goals:
                example_goals.append(goal)
        example_steps = plan_data[indices[0]]["raw_steps"][:5]
        results.append({
            "pattern": list(ngram),
            "length": len(ngram),
            "plan_count": count,
            "example_goals": example_goals,
            "example_steps": example_steps,
        })

    # Sort by plan_count descending, then length descending
    results.sort(key=lambda r: (-r["plan_count"], -r["length"]))

    # Deduplicate: drop shorter patterns fully contained in a longer pattern
    # with the same or higher plan_count. This removes noise like "read → edit"
    # when "read → edit → verify" has the same count.
    filtered: List[Dict[str, Any]] = []
    for r in results:
        r_pattern = tuple(r["pattern"])
        r_count = r["plan_count"]
        subsumed = False
        for f in filtered:
            f_pattern = tuple(f["pattern"])
            f_count = f["plan_count"]
            if f_count >= r_count and len(f_pattern) > len(r_pattern):
                # Check if r_pattern is a contiguous subsequence of f_pattern
                for i in range(len(f_pattern) - len(r_pattern) + 1):
                    if f_pattern[i : i + len(r_pattern)] == r_pattern:
                        subsumed = True
                        break
            if subsumed:
                break
        if not subsumed:
            filtered.append(r)

    return filtered[:50]  # cap at 50 patterns


# ---------------------------------------------------------------------------
# LLM prompt for the strategy extraction fork
# ---------------------------------------------------------------------------

STRATEGY_DRY_RUN_BANNER = (
    "═══════════════════════════════════════════════════════════════\n"
    "DRY-RUN — REPORT ONLY. DO NOT MUTATE THE SKILL LIBRARY.\n"
    "═══════════════════════════════════════════════════════════════\n"
    "\n"
    "This is a PREVIEW pass. Follow every instruction below EXCEPT:\n"
    "\n"
    "  • DO NOT call skill_manage with action=patch, create, delete,\n"
    "    write_file, or remove_file.\n"
    "  • DO NOT call terminal to modify any file under ~/.hermes/skills/.\n"
    "  • skills_list and skill_view are FINE — read as much as you need.\n"
    "\n"
    "Your output IS the deliverable. Produce the exact same summary and\n"
    "structured YAML block you would produce on a live run — but describe\n"
    "the patches you WOULD apply, not patches you applied.\n"
    "═══════════════════════════════════════════════════════════════"
)

STRATEGY_CURATOR_PROMPT = (
    "You are running as Hermes' background STRATEGY CURATOR (M5). This is a\n"
    "STRATEGY LEARNING pass, not a consolidation pass and not a duplicate-finder.\n\n"
    "The goal: learn from successful plans in the PlanStore ledger and encode\n"
    "recurring work patterns into the skill library so the agent gets better at\n"
    "planning over time.\n\n"
    "You have been given a set of recurring step patterns extracted from plans\n"
    "with status='done' and passing verification evidence. These patterns\n"
    "represent workflows that the agent has successfully executed multiple\n"
    "times. Your job is to capture them as durable skill improvements.\n\n"
    "Two kinds of improvement you can propose:\n\n"
    "1. PATCH THE ORCHESTRATOR SKILL — when a recurring pattern reveals a\n"
    "   workflow the orchestrator skill should teach but currently doesn't\n"
    "   (or teaches too abstractly), add a concrete subsection or decision\n"
    "   rule. Use `skill_view(name=\"orchestrator\")` to read the current\n"
    "   content, then `skill_manage(action=\"patch\", name=\"orchestrator\", ...)`\n"
    "   to add the missing guidance. The patch should be a NEW labeled\n"
    "   subsection — do not rewrite or delete existing content.\n\n"
    "2. CREATE A NEW SKILL — when a recurring pattern is domain-specific\n"
    "   (e.g. \"setup CI for a Python project\" or \"investigate a flaky test\")\n"
    "   and doesn't belong in the orchestrator, create a new class-level skill\n"
    "   with `skill_manage(action=\"create\", ...)`. The skill should capture\n"
    "   the proven step sequence as a reusable playbook, not a one-off recipe.\n\n"
    "Hard rules — do not violate:\n"
    "1. DO NOT touch bundled or hub-installed skills except `orchestrator`.\n"
    "   The orchestrator is the one skill you MAY patch — it is the strategic\n"
    "   decision skill and the natural home for cross-domain workflow patterns.\n"
    "2. DO NOT delete any skill. Archiving is the maximum destructive action.\n"
    "3. DO NOT create more than 3 new skills per pass. Quality over quantity —\n"
    "   each new skill must capture a pattern that recurs in at least\n"
    "   {min_successes} successful plans.\n"
    "4. DO NOT create narrow one-session skills. Each new skill must be a\n"
    "   class-level playbook that generalizes across sessions.\n"
    "5. Respect the write_approval gate. If `skills.write_approval` is on,\n"
    "   your skill_manage writes will be staged — that's expected. Do not\n"
    "   try to bypass the gate.\n\n"
    "How to work:\n"
    "1. Read the recurring patterns below carefully.\n"
    "2. Use `skill_view(name=\"orchestrator\")` to read the current orchestrator.\n"
    "3. For each pattern, decide: does it belong in the orchestrator (a general\n"
    "   workflow rule) or as a new skill (a domain playbook)?\n"
    "4. Apply patches via `skill_manage`. For the orchestrator, use\n"
    "   `action=patch` with a clear `## Learned Pattern: <name>` section.\n"
    "   For new skills, use `action=create`.\n"
    "5. Keep patches additive — never delete or rewrite existing content.\n\n"
    "When done, write a human summary AND a structured block:\n\n"
    "## Structured summary (required)\n"
    "```yaml\n"
    "strategy_patches:\n"
    "  - target: orchestrator\n"
    "    section: \"<section heading added>\"\n"
    "    reason: <one sentence — what pattern this captures>\n"
    "new_skills:\n"
    "  - name: <skill-name>\n"
    "    reason: <one sentence — what domain pattern this captures>\n"
    "```\n"
    "Leave a list empty if none. Do not omit the block.\n"
)


def _build_strategy_prompt(
    patterns: List[Dict[str, Any]],
    min_successes: int,
    dry_run: bool,
) -> str:
    """Assemble the full prompt for the strategy extraction fork."""
    pattern_lines: List[str] = []
    for i, p in enumerate(patterns[:20], 1):  # cap at 20 patterns in the prompt
        pattern_lines.append(
            f"  {i}. Pattern: {' → '.join(p['pattern'])}  "
            f"(recurs in {p['plan_count']} successful plans)"
        )
        if p.get("example_goals"):
            for g in p["example_goals"][:2]:
                pattern_lines.append(f"     example goal: {g[:120]}")
        if p.get("example_steps"):
            pattern_lines.append("     example steps:")
            for s in p["example_steps"][:3]:
                pattern_lines.append(f"       - {s[:120]}")
        pattern_lines.append("")

    if not pattern_lines:
        pattern_lines.append("  (no recurring patterns found)")

    patterns_block = "\n".join(pattern_lines)
    prompt_body = STRATEGY_CURATOR_PROMPT.format(min_successes=min_successes)
    prompt_body = (
        prompt_body
        + f"\n\n## Recurring patterns from {len(patterns)} pattern(s)\n\n"
        f"{patterns_block}\n"
    )

    if dry_run:
        return f"{STRATEGY_DRY_RUN_BANNER}\n\n{prompt_body}"
    return prompt_body


# ---------------------------------------------------------------------------
# LLM fork — spawn an AIAgent for the strategy extraction pass
# ---------------------------------------------------------------------------

def _resolve_strategy_model(
    cfg: Dict[str, Any],
    model_override: str,
) -> Tuple[str, str, Optional[str], Optional[str]]:
    """Resolve (provider, model, api_key, base_url) for the strategy fork.

    Precedence:
    1. Explicit model_override (non-empty) — treated as a model name; the
       provider is resolved via the canonical aux resolver. This lets users
       route the strategy fork to a different model than the consolidation fork.
    2. auxiliary.curator slot (same as the consolidation fork).
    3. Main chat model.

    Reuses ``agent.curator._resolve_review_runtime`` for cases 2 and 3 so the
    strategy fork inherits the same aux-model plumbing as the consolidation
    fork.
    """
    from agent.curator import _resolve_review_runtime

    _main = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
    _main_provider = _main.get("provider") or "auto"

    # 1. Explicit model override — use the main provider with the override model
    if model_override and model_override.strip():
        return _main_provider, model_override.strip(), None, None

    # 2/3. Reuse the curator aux resolver
    binding = _resolve_review_runtime(cfg)
    return binding.provider, binding.model, binding.explicit_api_key, binding.explicit_base_url


def _run_strategy_llm_review(
    prompt: str,
    model_override: str,
) -> Dict[str, Any]:
    """Spawn an AIAgent fork to run the strategy extraction prompt.

    Returns a dict with the same shape as ``curator._run_llm_review``:
    ``{final, summary, model, provider, tool_calls, error}``.
    """
    import contextlib

    result_meta: Dict[str, Any] = {
        "final": "",
        "summary": "",
        "model": "",
        "provider": "",
        "tool_calls": [],
        "error": None,
    }
    try:
        from run_agent import AIAgent
    except Exception as e:
        result_meta["error"] = f"AIAgent import failed: {e}"
        result_meta["summary"] = result_meta["error"]
        return result_meta

    _api_key = None
    _base_url = None
    _api_mode = None
    _resolved_provider = None
    _model_name = ""
    try:
        from hermes_cli.config import load_config
        from hermes_cli.runtime_provider import resolve_runtime_provider
        _cfg = load_config()
        _provider, _model_name, _explicit_key, _explicit_url = _resolve_strategy_model(
            _cfg, model_override,
        )
        _rp = resolve_runtime_provider(
            requested=_provider,
            target_model=_model_name,
            explicit_api_key=_explicit_key,
            explicit_base_url=_explicit_url,
        )
        _api_key = _rp.get("api_key")
        _base_url = _rp.get("base_url")
        _api_mode = _rp.get("api_mode")
        _resolved_provider = _rp.get("provider") or _provider
    except Exception as e:
        logger.debug("strategy_curator: provider resolution failed: %s", e, exc_info=True)

    result_meta["model"] = _model_name
    result_meta["provider"] = _resolved_provider or ""

    review_agent = None
    try:
        review_agent = AIAgent(
            model=_model_name,
            provider=_resolved_provider,
            api_key=_api_key,
            base_url=_base_url,
            api_mode=_api_mode,
            max_iterations=9999,
            quiet_mode=True,
            platform="strategy-curator",
            skip_context_files=True,
            skip_memory=True,
        )
        # Disable recursive nudges — the strategy curator must never spawn
        # its own review or memory nudge.
        review_agent._memory_nudge_interval = 0
        review_agent._skill_nudge_interval = 0

        # Redirect stdout/stderr to /dev/null so the fork's tool-call chatter
        # doesn't pollute the foreground terminal (same as the consolidation
        # fork in curator._run_llm_review).
        with open(os.devnull, "w", encoding="utf-8") as _devnull, \
             contextlib.redirect_stdout(_devnull), \
             contextlib.redirect_stderr(_devnull):
            conv_result = review_agent.run_conversation(user_message=prompt)

        final = ""
        if isinstance(conv_result, dict):
            final = str(conv_result.get("final_response") or "").strip()
        result_meta["final"] = final
        result_meta["summary"] = (
            (final[:240] + "…") if len(final) > 240 else (final or "no change")
        )

        # Collect tool calls for the report (same truncation as curator).
        _calls: List[Dict[str, Any]] = []
        for msg in getattr(review_agent, "_session_messages", []) or []:
            if not isinstance(msg, dict):
                continue
            tcs = msg.get("tool_calls") or []
            for tc in tcs:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                name = fn.get("name") or ""
                args_raw = fn.get("arguments") or ""
                if isinstance(args_raw, str) and len(args_raw) > 400:
                    args_raw = args_raw[:400] + "…"
                _calls.append({"name": name, "arguments": args_raw})
        result_meta["tool_calls"] = _calls
    except Exception as e:
        result_meta["error"] = f"error: {e}"
        result_meta["summary"] = result_meta["error"]
    finally:
        if review_agent is not None:
            try:
                review_agent.close()
            except Exception:
                pass
    return result_meta


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_strategy_pass(
    *,
    min_successes: int = DEFAULT_STRATEGY_MIN_SUCCESSES,
    model_override: str = DEFAULT_STRATEGY_EXTRACT_MODEL,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run the strategy extraction pass.

    Steps:
      1. Collect done plans with passing evidence from PlanStore.
      2. Extract recurring step patterns (deterministic, no LLM).
      3. If patterns are found, spawn a forked AIAgent to propose skill
         patches or new skills capturing the patterns.
      4. Return a dict with pattern stats + LLM meta.

    The pass is self-contained: it reads PlanStore, builds patterns, and
    runs the LLM fork. It does NOT write reports or update curator state —
    the caller (``curator._llm_pass``) handles that by merging the returned
    dict into its own report.

    The existing curator backup (taken before any mutating pass) covers
    strategy-pass mutations too — no separate snapshot is needed. The
    ``skills.write_approval`` gate is enforced at the ``skill_manage`` tool
    layer, not here — the forked agent's writes are automatically staged when
    the gate is on.

    Returns::

        {
            "enabled": True,
            "patterns_found": int,
            "patterns": [...],   # top patterns (capped at 10)
            "llm_meta": {...},   # same shape as curator._run_llm_review
            "dry_run": bool,
        }
    """
    # 1. Collect done plans with evidence
    plans = _collect_done_plans_with_evidence()
    if not plans:
        logger.debug("strategy_curator: no done plans with evidence found")
        return {
            "enabled": True,
            "patterns_found": 0,
            "patterns": [],
            "llm_meta": {
                "final": "",
                "summary": "skipped (no done plans with evidence)",
                "model": "",
                "provider": "",
                "tool_calls": [],
                "error": None,
            },
            "dry_run": dry_run,
        }

    # 2. Extract recurring patterns (deterministic)
    patterns = _extract_patterns(plans, min_successes)
    if not patterns:
        logger.debug(
            "strategy_curator: no recurring patterns found (min_successes=%d)",
            min_successes,
        )
        return {
            "enabled": True,
            "patterns_found": 0,
            "patterns": [],
            "llm_meta": {
                "final": "",
                "summary": f"skipped (no patterns meeting min_successes={min_successes})",
                "model": "",
                "provider": "",
                "tool_calls": [],
                "error": None,
            },
            "dry_run": dry_run,
        }

    # 3. Build prompt and run LLM fork
    logger.info(
        "strategy_curator: found %d recurring patterns, spawning LLM fork",
        len(patterns),
    )
    prompt = _build_strategy_prompt(patterns, min_successes, dry_run)
    llm_meta = _run_strategy_llm_review(prompt, model_override)

    return {
        "enabled": True,
        "patterns_found": len(patterns),
        "patterns": patterns[:10],  # cap for the report
        "llm_meta": llm_meta,
        "dry_run": dry_run,
    }