"""M3 CriticGate — automatic step-level critique via Mixture-of-Agents.

The CriticGate wraps :mod:`tools.mixture_of_agents_tool` (the existing MoA
implementation) to provide *automatic step-level critique* during a planning
session.  It fires on a cadence — every ``N`` steps (default 3) — and asks a
small panel of "devil's advocate" reference models to critique the agent's
recent work, then synthesises their feedback via an aggregator.  The resulting
critique is:

1. Persisted against the active step in the plan ledger via
   :func:`agent.plan_store.record_critique` (additive — does not change step
   status), and
2. Returned to the caller as a *synthetic user message* that is appended to
   the last user turn in the conversation — the exact same injection pattern
   ``moa_loop`` uses (a fenced block appended to the current user message's
   ``content``), so prompt-cache prefixes and role alternation are preserved.

Design constraints (from AGENTS.md):

* **Prompt caching is sacred.** The critique is injected as a synthetic
  *user* message appended to the last user turn, never as a mid-conversation
  system or assistant message.  This mirrors how ``moa_loop`` and the
  ``pre_llm_call`` hook inject context (see ``conversation_loop.py`` lines
  716–732).  No past context is mutated; the critique is ephemeral and
  API-call-time only.
* **The core is a narrow waist.** CriticGate is a standalone module with no
  new core tools.  It reuses the existing MoA tool's OpenRouter client and
  reference/aggregator infrastructure.  No ``conversation_loop.py`` edits are
  needed for the basic version — the gate is invoked either explicitly from
  the plan_store integration or via a ``pre_llm_call`` hook.

Configuration (new ``moa`` block in ``config.yaml`` / ``DEFAULT_CONFIG``)::

    moa:
      presets:
        critic:
          reference_models:
            - anthropic/claude-opus-4.6
            - google/gemini-2.5-pro
            - openai/gpt-5.4-pro
          aggregator: anthropic/claude-opus-4.6
          reference_temperature: 0.8   # higher than the default 0.6 for diversity
      auto_critique:
        enabled: false                 # opt-in; off by default
        cadence_steps: 3               # fire every N steps
        max_concurrent_critics: 3       # cap parallel reference calls
        diverge_via_delegation: false   # future: route critics through delegate_task

Usage::

    from agent.critic_gate import maybe_critique

    # Called from the plan_store integration after each step transition:
    msg = await maybe_critique(plan_store=plan, moa_config=config["moa"])
    if msg:
        messages[-1]["content"] += "\\n\\n" + msg
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Devil's-advocate framing for the critic preset
# ---------------------------------------------------------------------------

#: System-prompt prefix prepended to each reference-model call when the critic
#: preset is active.  Frames the references as adversarial reviewers whose job
#: is to find flaws, not to solve the problem — this is what distinguishes the
#: ``critic`` preset from the default MoA preset (which asks references to
#: *answer* the query).
CRITIC_REFERENCE_SYSTEM_PROMPT = (
    "You are a rigorous devil's-advocate reviewer.  Do NOT solve the task.  "
    "Your job is to critique the agent's recent work on the step below.  "
    "Identify: (1) logical errors or incorrect assumptions, (2) missing "
    "edge cases or unhandled failure modes, (3) premature convergence on a "
    "single approach, (4) verification gaps (claims not backed by evidence), "
    "and (5) any sign the step is drifting from the plan's goal.  Be "
    "concrete and cite the specific action or output you are challenging.  "
    "If the work is genuinely sound, say so briefly — do not invent "
    "criticism.  Keep your critique under 300 words.\n\n"
)

#: Aggregator system prompt for the critic preset.  Synthesises the
#: reference critiques into a single, actionable critique blob.
CRITIC_AGGREGATOR_SYSTEM_PROMPT = (
    "You have been provided with a set of critiques from several reviewer "
    "models evaluating an agent's recent step.  Your task is to synthesise "
    "these critiques into a single, concise, actionable critique.  "
    "De-duplicate overlapping points, rank by severity (blocker > major > "
    "minor > nit), and strip any redundant hedging.  If the reviewers "
    "disagree, surface the disagreement rather than averaging it away.  "
    "Output a JSON object with keys: \"summary\" (str), \"issues\" (list of "
    "{\"severity\": str, \"point\": str}), \"verdict\": one of "
    "\"proceed\" | \"revise\" | \"block\".  If there are no real issues, "
    "return an empty issues list and verdict \"proceed\".\n\n"
    "Critiques from reviewers:"
)

#: Fenced-block header used when injecting the critique as a synthetic user
#: message.  Mirrors the ``build_memory_context_block`` fence style.
CRITIQUE_BLOCK_HEADER = "🧪 Auto-critique (M3 CriticGate)"


# ---------------------------------------------------------------------------
# Default config constants (also exposed for external callers that build
# config dicts programmatically)
# ---------------------------------------------------------------------------

DEFAULT_CRITIC_PRESET: Dict[str, Any] = {
    "reference_models": [],       # empty → fall back to MoA tool defaults
    "aggregator": "",             # empty → fall back to MoA tool default
    "reference_temperature": 0.8,  # higher than MoA default (0.6) for diversity
}

DEFAULT_AUTO_CRITIQUE_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "cadence_steps": 3,
    "max_concurrent_critics": 3,
    "diverge_via_delegation": False,
}


# ---------------------------------------------------------------------------
# Preset resolution
# ---------------------------------------------------------------------------

def resolve_critic_preset(moa_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Resolve the ``critic`` preset from the ``moa`` config block.

    Falls back to the module-level defaults in :mod:`tools.mixture_of_agents_tool`
    when the config is absent or incomplete, so the gate works out-of-the-box
    without explicit config.
    """
    if not moa_config or not isinstance(moa_config, dict):
        moa_config = {}
    presets = moa_config.get("presets") or {}
    critic = presets.get("critic") or {}

    preset: Dict[str, Any] = dict(DEFAULT_CRITIC_PRESET)
    for key in DEFAULT_CRITIC_PRESET:
        if key in critic:
            preset[key] = critic[key]

    # Resolve empty reference_models / aggregator to MoA tool defaults.
    if not preset["reference_models"]:
        try:
            from tools.mixture_of_agents_tool import (
                REFERENCE_MODELS as _DEFAULT_REFS,
                AGGREGATOR_MODEL as _DEFAULT_AGG,
            )
            preset["reference_models"] = list(_DEFAULT_REFS)
            if not preset["aggregator"]:
                preset["aggregator"] = _DEFAULT_AGG
        except Exception:
            # If the MoA tool can't be imported (missing dependency), leave
            # empty — the gate will log and skip.
            pass

    preset["reference_temperature"] = float(preset["reference_temperature"])
    return preset


def resolve_auto_critique_config(
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve the ``auto_critique`` sub-block with defaults.

    Accepts either the full config dict (looks for ``config["moa"]["auto_critique"]``)
    or just the ``moa`` block directly (looks for ``moa_config["auto_critique"]``).
    """
    cfg = dict(DEFAULT_AUTO_CRITIQUE_CONFIG)
    if not config:
        return cfg
    if not isinstance(config, dict):
        return cfg

    # Accept both full-config and moa-block-only call styles.
    moa_cfg = config.get("moa", config) if "moa" in config else config
    if not isinstance(moa_cfg, dict):
        return cfg
    auto_cfg = moa_cfg.get("auto_critique", {})
    if not isinstance(auto_cfg, dict):
        return cfg
    for key in DEFAULT_AUTO_CRITIQUE_CONFIG:
        if key in auto_cfg:
            cfg[key] = auto_cfg[key]

    cfg["cadence_steps"] = int(cfg["cadence_steps"])
    cfg["max_concurrent_critics"] = int(cfg["max_concurrent_critics"])
    cfg["enabled"] = bool(cfg["enabled"])
    cfg["diverge_via_delegation"] = bool(cfg["diverge_via_delegation"])
    return cfg


def auto_critique_enabled(config: Optional[Dict[str, Any]] = None) -> bool:
    """Quick check: is auto-critique enabled?"""
    return resolve_auto_critique_config(config).get("enabled", False)


# ---------------------------------------------------------------------------
# Core: aggregate_moa_context (the reusable MoA wrapper)
# ---------------------------------------------------------------------------

async def aggregate_moa_context(
    prompt: str,
    *,
    preset: Optional[Dict[str, Any]] = None,
    max_concurrent: int = 3,
    reference_preamble: str = "",
    aggregator_system_prompt: str = "",
    max_retries: int = 3,
) -> str:
    """Run a 2-layer MoA pass and return the aggregated text.

    This is the reusable wrapper the task specification refers to as
    ``moa_loop.aggregate_moa_context``.  It wraps the existing
    :mod:`tools.mixture_of_agents_tool` primitives
    (``_run_reference_model_safe`` + ``_run_aggregator_model``) so we reuse
    the proven OpenRouter client, retry logic, and content extraction.

    Args:
        prompt: The user query / context to critique.
        preset: Resolved preset dict with ``reference_models``, ``aggregator``,
            ``reference_temperature``.  When ``None``, resolves the critic
            preset from default config.
        max_concurrent: Cap on parallel reference-model calls.  Defaults to 3.
        reference_preamble: Optional system-prompt prefix prepended to each
            reference call (used by the critic preset to frame references as
            devil's advocates).
        aggregator_system_prompt: Base system prompt for the aggregator.  When
            empty, falls back to the MoA default synthesis prompt.
        max_retries: Per-reference-model retry count.

    Returns:
        The final aggregated text.  On total failure, returns an empty string
        (the gate treats empty as "skip injection").
    """
    if preset is None:
        preset = resolve_critic_preset()

    from tools.mixture_of_agents_tool import (
        _run_reference_model_safe,
        _run_aggregator_model,
        _construct_aggregator_prompt,
        AGGREGATOR_SYSTEM_PROMPT as _DEFAULT_AGG_PROMPT,
        AGGREGATOR_TEMPERATURE as _DEFAULT_AGG_TEMP,
    )

    ref_models: List[str] = preset["reference_models"]
    ref_temp: float = float(preset["reference_temperature"])
    agg_model: str = preset["aggregator"]

    if not ref_models:
        logger.warning("CriticGate: no reference models available; skipping MoA")
        return ""

    # Cap concurrency to avoid hammering OpenRouter when the preset has many
    # reference models.  A semaphore is lighter than chunking and lets faster
    # models start the aggregator sooner.
    semaphore = asyncio.Semaphore(max(1, max_concurrent))

    async def _run_one(model: str) -> Tuple[str, str, bool]:
        async with semaphore:
            # Prepend the devil's-advocate preamble to the user prompt for
            # critic presets.  For the default (non-critic) use case the
            # preamble is empty and the prompt passes through unchanged.
            user_prompt = (
                reference_preamble + prompt if reference_preamble else prompt
            )
            return await _run_reference_model_safe(
                model,
                user_prompt,
                temperature=ref_temp,
                max_retries=max_retries,
            )

    logger.debug(
        "CriticGate: running MoA with %d reference models (concurrency=%d)",
        len(ref_models),
        max_concurrent,
    )

    results = await asyncio.gather(*[_run_one(m) for m in ref_models])

    successful: List[str] = []
    failed: List[str] = []
    for model_name, content, success in results:
        if success and content:
            successful.append(content)
        else:
            failed.append(model_name)

    if failed:
        logger.warning(
            "CriticGate: %d/%d reference models failed: %s",
            len(failed),
            len(ref_models),
            ", ".join(failed),
        )

    if not successful:
        logger.error("CriticGate: all reference models failed; skipping critique")
        return ""

    # Layer 2 — aggregate.  Use the caller-provided aggregator prompt or the
    # MoA default.  The aggregator temperature is intentionally NOT exposed in
    # the critic preset — the critic benefits from focused (low-temp)
    # synthesis, and the MoA default (0.4) already encodes that.
    base_agg_prompt = aggregator_system_prompt or _DEFAULT_AGG_PROMPT
    aggregator_system = _construct_aggregator_prompt(base_agg_prompt, successful)

    try:
        final = await _run_aggregator_model(
            aggregator_system,
            prompt,
            temperature=_DEFAULT_AGG_TEMP,
        )
        return final or ""
    except Exception as exc:
        logger.error("CriticGate: aggregator failed: %s", exc, exc_info=True)
        return ""


# ---------------------------------------------------------------------------
# Critique injection (synthetic user message — moa_loop pattern)
# ---------------------------------------------------------------------------

def _build_critique_block(critique_text: str) -> str:
    """Wrap critique text in a fenced block for injection into a user message.

    Mirrors the ``build_memory_context_block`` fence style used by the memory
    manager and ``pre_llm_call`` hooks in ``conversation_loop.py``.
    """
    return (
        f"--- {CRITIQUE_BLOCK_HEADER} ---\n"
        f"{critique_text}\n"
        f"--- end critique ---"
    )


def _inject_critique_as_user_message(
    messages: List[Dict[str, Any]],
    critique_text: str,
) -> bool:
    """Append a critique block to the last user message in ``messages``.

    This is the moa_loop injection pattern: the critique is appended to the
    *content* of the last user turn, not inserted as a new mid-conversation
    message.  This preserves:

    * **Role alternation** — no new same-role message is created.
    * **Prompt-cache prefix** — the message list length and roles are
      unchanged; only the trailing user message's content grows.

    The mutation is applied to the *caller's* list in place.  Callers that
    need to preserve the original should pass a copy.

    Returns ``True`` if injected, ``False`` if there was no user message to
    inject into (the caller should handle this by skipping the critique).
    """
    if not critique_text or not messages:
        return False
    # Find the last user message (walking backwards).
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") == "user":
            block = _build_critique_block(critique_text)
            base = msg.get("content", "")
            if isinstance(base, str):
                msg["content"] = base + "\n\n" + block if base else block
            else:
                # Non-string content (list of parts) — append a text part.
                if isinstance(base, list):
                    base.append({"type": "text", "text": block})
                else:
                    # Unknown shape; fall back to string conversion.
                    msg["content"] = str(base) + "\n\n" + block
            return True
    return False


# ---------------------------------------------------------------------------
# Step-context extraction from the plan store
# ---------------------------------------------------------------------------

def _extract_step_context(
    plan_store: Any,
    *,
    plan_id: Optional[str] = None,
    step_id: Optional[str] = None,
    db_path: Optional[Any] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Build a textual snapshot of the active step for the critic prompt.

    Reads the active plan + steps from the plan ledger and returns a compact
    string describing the goal, the active step, and its attempts/critique
    history.  Returns ``(context_string, resolved_step_id)`` or
    ``(None, None)`` if there is no active plan or step to critique.

    ``plan_store`` is the :mod:`agent.plan_store` module (passed in explicitly
    to avoid a hard import dependency at module load time — the gate must be
    importable even when the plan DB does not exist).

    ``db_path`` is forwarded to every plan_store call so tests and custom
    layouts can pin the ledger location.  When ``None`` the plan_store's
    default resolution (``HERMES_HOME/plans.db``) is used.
    """
    try:
        if plan_id:
            plan = plan_store.get_plan_with_steps(plan_id, db_path=db_path)
        else:
            # Find the most recent active plan.
            plans = plan_store.list_plans(status="active", limit=1, db_path=db_path)
            if not plans:
                return None, None
            plan = plan_store.get_plan_with_steps(plans[0]["id"], db_path=db_path)
        if not plan:
            return None, None

        steps = plan.get("steps") or []
        if not steps:
            return None, None

        # Pick the target step: explicit step_id, else the first non-done step.
        target = None
        if step_id:
            for s in steps:
                if s.get("id") == step_id:
                    target = s
                    break
        if target is None:
            for s in steps:
                if s.get("status") not in ("done", "superseded"):
                    target = s
                    break
        if target is None:
            return None, None

        lines = [
            f"Plan goal: {plan.get('goal', '(unknown)')}",
            f"Active step (idx={target.get('idx')}): {target.get('description', '')}",
            f"Step status: {target.get('status', 'pending')}",
            f"Attempts: {target.get('attempts', 0)}",
        ]
        prior_critique = target.get("critique")
        if prior_critique:
            lines.append(
                "Prior critique: "
                + json.dumps(prior_critique, ensure_ascii=False)[:500]
            )
        return "\n".join(lines), target.get("id")
    except Exception as exc:
        logger.warning("CriticGate: failed to extract step context: %s", exc)
        return None, None


# ---------------------------------------------------------------------------
# Public entry point: maybe_critique
# ---------------------------------------------------------------------------

async def maybe_critique(
    plan_store: Any,
    moa_config: Optional[Dict[str, Any]] = None,
    *,
    step_count: int = 0,
    messages: Optional[List[Dict[str, Any]]] = None,
    plan_id: Optional[str] = None,
    step_id: Optional[str] = None,
    db_path: Optional[Any] = None,
) -> Optional[str]:
    """Run an automatic critique if the cadence fires, else return ``None``.

    This is the main entry point for the M3 CriticGate.  It is designed to be
    called on every step transition (e.g. from the plan_store integration or a
    ``pre_llm_call`` hook) with the current ``step_count``.  The critique only
    fires when:

    1. ``auto_critique.enabled`` is ``True`` (default ``False`` — opt-in), AND
    2. ``step_count`` is a positive multiple of ``cadence_steps`` (default 3).

    When the gate fires it:

    1. Extracts the active step context from the plan ledger.
    2. Runs a 2-layer MoA pass with the ``critic`` preset (references framed as
       devil's advocates, aggregator synthesises an actionable critique).
    3. Persists the critique against the step via
       :func:`agent.plan_store.record_critique`.
    4. If ``messages`` is provided, injects the critique as a synthetic user
       message (appended to the last user turn — the moa_loop pattern).

    Args:
        plan_store: The :mod:`agent.plan_store` module (or a duck-typed shim
            with the same API).  Pass ``None`` to skip plan-ledger integration
            (the critique still runs against ``messages`` content if
            provided, but is not persisted).
        moa_config: The ``moa`` config block dict.  When ``None``, defaults
            are used (auto_critique disabled, critic preset from MoA defaults).
        step_count: The current step index (0-based or 1-based — only the
            modulo matters).  The gate fires when
            ``step_count > 0 and step_count % cadence_steps == 0``.
        messages: Optional conversation message list.  When provided, the
            critique is injected into the last user message in place.
        plan_id: Optional explicit plan id to critique.  When ``None``, the
            most recent ``active`` plan is used.
        step_id: Optional explicit step id to critique.  When ``None``, the
            first non-done step in the plan is used.
        db_path: Optional plan DB path (for tests / custom layouts).

    Returns:
        The critique text (for the caller to use), or ``None`` if the gate did
        not fire.  When ``messages`` is provided, the injection is already done
        and the returned text is the same critique block content.
    """
    ac = resolve_auto_critique_config(moa_config)

    # --- Gate check -------------------------------------------------------
    if not ac["enabled"]:
        return None
    cadence = ac["cadence_steps"]
    if cadence < 1:
        cadence = 3  # sane default
    if step_count <= 0 or step_count % cadence != 0:
        return None

    logger.info(
        "CriticGate: firing auto-critique at step %d (cadence=%d)",
        step_count,
        cadence,
    )

    # --- Build the critique prompt ----------------------------------------
    # The critic needs something to critique.  Prefer the structured step
    # context from the plan ledger; fall back to the recent conversation
    # messages if no plan is available.
    target_step_id: Optional[str] = step_id
    step_context: Optional[str] = None
    if plan_store is not None:
        step_context, target_step_id = _extract_step_context(
            plan_store, plan_id=plan_id, step_id=step_id, db_path=db_path
        )

    if step_context:
        critique_prompt = (
            "Critique the agent's recent work on this planning step:\n\n"
            + step_context
        )
    elif messages:
        # Fall back: summarise the last few messages as the critique target.
        recent = messages[-6:]
        parts = []
        for m in recent:
            role = m.get("role", "?")
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            parts.append(f"[{role}] {str(content)[:400]}")
        critique_prompt = (
            "Critique the agent's recent conversation actions:\n\n"
            + "\n".join(parts)
        )
    else:
        logger.debug("CriticGate: no step context and no messages; skipping")
        return None

    # --- Run the MoA critique --------------------------------------------
    preset = resolve_critic_preset(moa_config)
    if not preset["reference_models"]:
        logger.warning("CriticGate: no reference models configured; skipping")
        return None

    t0 = time.monotonic()
    critique_text = await aggregate_moa_context(
        critique_prompt,
        preset=preset,
        max_concurrent=ac["max_concurrent_critics"],
        reference_preamble=CRITIC_REFERENCE_SYSTEM_PROMPT,
        aggregator_system_prompt=CRITIC_AGGREGATOR_SYSTEM_PROMPT,
    )
    elapsed = time.monotonic() - t0

    if not critique_text:
        logger.warning("CriticGate: MoA returned empty critique; skipping")
        return None

    logger.info("CriticGate: critique complete (%.1fs, %d chars)", elapsed, len(critique_text))

    # --- Persist to plan ledger (additive — does not change step status) --
    if plan_store is not None and target_step_id:
        try:
            critique_blob: Dict[str, Any] = {
                "text": critique_text,
                "step_count": step_count,
                "preset": "critic",
                "models": {
                    "reference_models": preset["reference_models"],
                    "aggregator": preset["aggregator"],
                },
                "elapsed_s": round(elapsed, 2),
            }
            plan_store.record_critique(target_step_id, critique_blob, db_path=db_path)
            logger.debug("CriticGate: critique persisted to step %s", target_step_id)
        except Exception as exc:
            logger.warning("CriticGate: failed to persist critique: %s", exc)

    # --- Inject as synthetic user message (moa_loop pattern) --------------
    if messages:
        injected = _inject_critique_as_user_message(messages, critique_text)
        if not injected:
            logger.debug("CriticGate: no user message to inject into; returning text")

    return critique_text


# ---------------------------------------------------------------------------
# Synchronous convenience (for hook callers that are not async)
# ---------------------------------------------------------------------------

def maybe_critique_sync(
    plan_store: Any,
    moa_config: Optional[Dict[str, Any]] = None,
    *,
    step_count: int = 0,
    messages: Optional[List[Dict[str, Any]]] = None,
    plan_id: Optional[str] = None,
    step_id: Optional[str] = None,
    db_path: Optional[Any] = None,
) -> Optional[str]:
    """Synchronous wrapper around :func:`maybe_critique`.

    For callers that are not already in an async context (e.g. a synchronous
    ``pre_llm_call`` hook).  Runs the async gate in a fresh event loop.
    """
    try:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                maybe_critique(
                    plan_store,
                    moa_config,
                    step_count=step_count,
                    messages=messages,
                    plan_id=plan_id,
                    step_id=step_id,
                    db_path=db_path,
                )
            )
        finally:
            loop.close()
    except Exception as exc:
        logger.error("CriticGate sync wrapper failed: %s", exc, exc_info=True)
        return None


__all__ = [
    "aggregate_moa_context",
    "maybe_critique",
    "maybe_critique_sync",
    "resolve_critic_preset",
    "resolve_auto_critique_config",
    "auto_critique_enabled",
    "CRITIC_REFERENCE_SYSTEM_PROMPT",
    "CRITIC_AGGREGATOR_SYSTEM_PROMPT",
    "DEFAULT_CRITIC_PRESET",
    "DEFAULT_AUTO_CRITIQUE_CONFIG",
]