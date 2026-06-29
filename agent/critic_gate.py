"""Critic gate — automatic step-level critique via MoA divergence (M3).

Wraps ``moa_loop.aggregate_moa_context()`` to provide advisory critique at
plan-step boundaries.  Unlike MoA (which is per-turn, opt-in via ``/moa``),
the critic gate fires automatically every ``planning.critique_cadence``
steps (default 3), using a dedicated ``critic`` MoA preset.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_CRITIC_PRESET: Dict[str, Any] = {
    "reference_models": [],
    "aggregator": {},
    "reference_temperature": 0.8,
    "aggregator_temperature": 0.4,
    "enabled": True,
}

DEFAULT_AUTO_CRITIQUE_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "cadence_steps": 3,
    "max_concurrent_critics": 3,
    "diverge_via_delegation": False,
}

CRITIC_REFERENCE_SYSTEM_PROMPT = (
    "You are a critic advisor. You are NOT the acting agent and you do NOT "
    "execute anything. Your job is to find what the agent may have missed, "
    "gotten wrong, or is about to do suboptimally. Be a constructive devil's "
    "advocate: challenge assumptions, surface alternatives, identify risks."
)


def resolve_critic_preset(moa_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    preset = dict(DEFAULT_CRITIC_PRESET)
    if moa_config and "presets" in moa_config:
        critic_cfg = moa_config["presets"].get("critic", {})
        if isinstance(critic_cfg, dict):
            for key in DEFAULT_CRITIC_PRESET:
                if key in critic_cfg:
                    preset[key] = critic_cfg[key]
    return preset


def resolve_auto_critique_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = dict(DEFAULT_AUTO_CRITIQUE_CONFIG)
    if config and "moa" in config:
        moa_cfg = config.get("moa", {})
        if isinstance(moa_cfg, dict):
            auto_cfg = moa_cfg.get("auto_critique", {})
            if isinstance(auto_cfg, dict):
                for key in DEFAULT_AUTO_CRITIQUE_CONFIG:
                    if key in auto_cfg:
                        cfg[key] = auto_cfg[key]
    return cfg


def auto_critique_enabled(config: Optional[Dict[str, Any]] = None) -> bool:
    return resolve_auto_critique_config(config).get("enabled", False)


def maybe_critique(
    plan_context: Dict[str, Any],
    moa_config: Optional[Dict[str, Any]] = None,
    auto_cfg: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    if auto_cfg is None:
        auto_cfg = resolve_auto_critique_config()
    if not auto_cfg.get("enabled", False):
        return None
    cadence = auto_cfg.get("cadence_steps", 3)
    current_step = plan_context.get("current_step_idx", 0)
    if current_step <= 0 or current_step % cadence != 0:
        return None
    critic_preset = resolve_critic_preset(moa_config)
    if not critic_preset.get("enabled", True):
        return None
    if not critic_preset.get("reference_models"):
        logger.debug("critic_gate: no reference models — skipping")
        return None
    try:
        from agent.moa_loop import aggregate_moa_context
        plan_goal = plan_context.get("goal", "Unknown")
        current_step_desc = plan_context.get("current_step_desc", "")
        steps_done = plan_context.get("steps_done", [])
        user_prompt = (
            f"Plan: {plan_goal}\n"
            f"Steps done: {len(steps_done)}\n"
            f"Current step ({current_step}): {current_step_desc}\n\n"
            f"Critique the agent's progress. What has it missed? "
            f"What risks exist? What should it do differently?"
        )
        critique = aggregate_moa_context(
            user_prompt=user_prompt,
            api_messages=[{"role": "user", "content": user_prompt}],
            reference_models=critic_preset.get("reference_models", []),
            aggregator=critic_preset.get("aggregator", {}),
            temperature=float(critic_preset.get("reference_temperature", 0.8)),
            aggregator_temperature=float(critic_preset.get("aggregator_temperature", 0.4)),
        )
        if critique:
            return f"[CRITIQUE — step {current_step}]\n\n{critique}"
    except ImportError:
        logger.debug("critic_gate: moa_loop not available")
    except Exception:
        logger.debug("critic_gate: critique failed", exc_info=True)
    return None
