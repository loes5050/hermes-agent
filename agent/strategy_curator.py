"""Strategy curator — meta-skill growth from successful plans (M5).

Extends the curator's background maintenance with a ``strategy_pass`` that
reads successful PlanStore plans (status='done') and extracts recurring
step patterns.  Uses those patterns to propose patches to the orchestrator
skill or create new skills.

Activated by ``curator.strategy_pass: true`` in config.yaml.  The strategy
pass runs AFTER the regular skill consolidation pass, reusing the same
forked aux-model review agent, backup snapshot, and write_approval gate.

New config keys (all under ``curator`` in config.yaml):
  - strategy_pass (bool, default False)
  - strategy_min_successes (int, default 3)
  - strategy_extract_model (str, default '' — empty = inherit curator model)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# Default strategy curator config
DEFAULT_STRATEGY_CONFIG: Dict[str, Any] = {
    "strategy_pass": False,
    "strategy_min_successes": 3,
    "strategy_extract_model": "",
}


def resolve_strategy_config(curator_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Resolve strategy config with defaults."""
    cfg = dict(DEFAULT_STRATEGY_CONFIG)
    if curator_config and isinstance(curator_config, dict):
        for key in DEFAULT_STRATEGY_CONFIG:
            if key in curator_config:
                cfg[key] = curator_config[key]
    return cfg


def strategy_pass_enabled(curator_config: Optional[Dict[str, Any]] = None) -> bool:
    """Return True when the strategy extraction pass is enabled."""
    return resolve_strategy_config(curator_config).get("strategy_pass", False)


def _get_plans_db_path() -> Path:
    """Return the path to the PlanStore SQLite database."""
    return get_hermes_home() / "plans.db"


def extract_successful_plans(
    min_successes: int = 3,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Extract completed plans from the PlanStore for pattern analysis.

    Returns a list of plan dicts (goal + steps + evidence) for plans
    that reached 'done' status.  Only returns plans when the count
    meets ``min_successes`` (avoids extracting noise from too few samples).
    """
    try:
        from agent.plan_store import list_plans, get_plan_with_steps
    except ImportError:
        logger.debug("strategy_curator: plan_store not available")
        return []

    try:
        done_plans = list_plans(status="done", limit=limit)
    except Exception as e:
        logger.debug("strategy_curator: failed to list plans: %s", e)
        return []

    if len(done_plans) < min_successes:
        logger.debug(
            "strategy_curator: %d done plans < min_successes=%d — skipping",
            len(done_plans), min_successes,
        )
        return []

    enriched = []
    for plan_summary in done_plans:
        try:
            full = get_plan_with_steps(plan_summary["id"])
            if full:
                enriched.append(full)
        except Exception as e:
            logger.debug("strategy_curator: failed to read plan %s: %s",
                         plan_summary.get("id", "?"), e)

    return enriched


def _summarize_plan(plan: Dict[str, Any]) -> str:
    """Summarize a plan for the aux-model review prompt."""
    goal = plan.get("goal", "Unknown")[:200]
    steps = plan.get("steps", [])
    lines = [f"Goal: {goal}", f"Steps: {len(steps)}"]

    for step in steps:
        desc = step.get("description", "")[:120]
        status = step.get("status", "?")
        evidence = step.get("evidence_event_id", "")
        lines.append(f"  [{status}] {desc}" + (f" (evidence: {evidence})" if evidence else ""))

    return "\n".join(lines)


def build_strategy_review_prompt(plans: List[Dict[str, Any]]) -> str:
    """Build a review prompt for the aux-model strategy extraction agent.

    Presents successful plans and asks the model to identify recurring
    step patterns worth encoding as skill guidance.
    """
    plan_summaries = "\n\n---\n\n".join(
        _summarize_plan(p) for p in plans[:10]
    )

    return (
        "You are reviewing successful task plans executed by an AI agent.\n"
        "Your job is to identify recurring step patterns and extract them as "
        "strategy guidance that can improve future agent performance.\n\n"
        "## Successful Plans\n\n"
        f"{plan_summaries}\n\n"
        "## Instructions\n\n"
        "1. Identify 2-5 recurring step patterns across these plans.\n"
        "2. For each pattern, write a concrete decision rule in this format:\n"
        "   - **Trigger**: When should this rule fire?\n"
        "   - **Action**: What should the agent do?\n"
        "   - **Example**: A concrete example from the plans above.\n"
        "3. Suggest which existing skill the new rule should be added to "
        "(e.g., orchestrator, planning, requesting-code-review).\n"
        "4. Flag any anti-patterns — repeated mistakes the agent makes.\n\n"
        "Return your analysis as structured text. No preamble."
    )


def run_strategy_pass(
    curator_config: Optional[Dict[str, Any]] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the strategy extraction pass.

    Returns a dict with summary stats: plans_analyzed, patterns_found,
    skills_proposed, errors.
    """
    cfg = resolve_strategy_config(curator_config)
    if not cfg.get("strategy_pass", False):
        return {"status": "disabled", "reason": "curator.strategy_pass is false"}

    min_successes = cfg.get("strategy_min_successes", 3)

    # Phase 1: Extract successful plans
    plans = extract_successful_plans(min_successes=min_successes)
    if not plans:
        return {
            "status": "skipped",
            "reason": f"fewer than {min_successes} successful plans found",
            "plans_found": 0,
        }

    # Phase 2: Build review prompt
    prompt = build_strategy_review_prompt(plans)

    # Phase 3: Run aux-model review (reuses curator's forked agent pattern)
    # This is a placeholder — the actual aux-model call is handled by the
    # curator's existing review infrastructure (curator.py run_curator_review).
    # The strategy pass integrates by:
    #   1. Prepending this prompt to the curator's review context
    #   2. Letting the curator's forked agent process it
    #   3. The forked agent's output (proposed skill patches) goes through
    #      the existing write_approval gate

    result = {
        "status": "completed",
        "plans_analyzed": len(plans),
        "prompt_built": True,
        "prompt_length": len(prompt),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    logger.info(
        "strategy_curator: analyzed %d plans, prompt=%d chars",
        len(plans), len(prompt),
    )

    return result


def get_strategy_context(curator_config: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Return a strategy context string for the curator's review prompt.

    Called by curator.py before spawning the aux-model review agent.
    Returns the strategy analysis prompt if strategy_pass is enabled
    and there are enough successful plans; otherwise returns None.
    """
    if not strategy_pass_enabled(curator_config):
        return None

    cfg = resolve_strategy_config(curator_config)
    min_successes = cfg.get("strategy_min_successes", 3)
    plans = extract_successful_plans(min_successes=min_successes)

    if not plans:
        return None

    return build_strategy_review_prompt(plans)
