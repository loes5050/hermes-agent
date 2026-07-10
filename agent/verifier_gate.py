"""Verifier gate — bounded re-entry when edits lack passing evidence.

This module upgrades the policy-only ``verification_stop`` nudge into a
*verifier gate*: when ``agent.verify_on_stop`` is ``"gate"``, the gate can
re-enter the conversation loop after a text_response if any edited path lacks
fresh passing verification evidence. The gate is bounded by
``verify_gate.max_attempts`` (default 3); after the limit is exhausted the
gate degrades to the existing advisory nudge from ``verification_stop``.

Design constraints (from AGENTS.md):

- **Prompt caching is sacred.** The gate never mutates the system prompt or
  past context. It appends a synthetic *user* continuation (mirroring the
  existing verification_stop seam) so role alternation is preserved and the
  cached prefix stays byte-stable.
- **Narrow core.** The gate is a pure helper invoked from the one verified
  seam in ``conversation_loop.py`` — no new model tools, no toolset changes.
- **Extend, don't duplicate.** Reuses ``verification_evidence.verification_status``
  and ``coding_context.project_facts_for`` rather than re-deriving state.

Config keys (all in ``config.yaml`` under ``agent.verify_gate`` — no env vars):

    verify_gate:
      max_attempts: 3          # bounded re-entry attempts before degrade-to-nudge
      require_canonical: true   # only gate when canonical verify commands exist
      allow_ad_hoc: true        # also accept ad-hoc temp-script evidence as passing
      messaging_surface: false  # gate even on messaging surfaces (default off)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Reuse the non-code path filter from verification_stop so we gate on the same
# set of verifiable paths the nudge uses — no duplication, single source of truth.
try:
    from agent.verification_stop import _filter_verifiable_paths, _session_is_messaging_surface
except Exception:  # pragma: no cover - import guard
    _filter_verifiable_paths = lambda paths: list(paths)  # noqa: E731
    _session_is_messaging_surface = lambda: False  # noqa: E731

# ──────────────────────────────────────────────────────────────────────────
# Default config (mirrors config.yaml agent.verify_gate)
# ──────────────────────────────────────────────────────────────────────────

DEFAULT_VERIFY_GATE_CONFIG: Dict[str, Any] = {
    "max_attempts": 3,
    "require_canonical": True,
    "allow_ad_hoc": True,
    "messaging_surface": False,
}

# Backward-compatible alias
_DEFAULT_GATE_CONFIG = DEFAULT_VERIFY_GATE_CONFIG

_PASSING_STATUSES = frozenset({"passed"})


# ──────────────────────────────────────────────────────────────────────────
# Config helpers
# ──────────────────────────────────────────────────────────────────────────

def _load_gate_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return the effective ``agent.verify_gate`` config, merged with defaults."""
    if config is None:
        try:
            from hermes_cli.config import load_config

            config = load_config()
        except Exception:
            config = {}
    agent_cfg = (config or {}).get("agent") if isinstance(config, dict) else None
    gate_cfg = agent_cfg.get("verify_gate") if isinstance(agent_cfg, dict) else None
    if not isinstance(gate_cfg, dict):
        gate_cfg = {}
    merged = dict(_DEFAULT_GATE_CONFIG)
    merged.update(gate_cfg)
    return merged


def resolve_gate_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Public alias for ``_load_gate_config`` — used by conversation_loop callers."""
    return _load_gate_config(config)


def _resolve_verify_on_stop(config: Optional[Dict[str, Any]] = None) -> str:
    """Return the raw ``agent.verify_on_stop`` value as a lowercased string.

    Returns one of: ``"true"``, ``"false"``, ``"auto"``, ``"gate"`` — or
    ``""`` when unset/unknown. The ``"gate"`` sentinel is the new mode this
    module implements.
    """
    env = os.environ.get("HERMES_VERIFY_ON_STOP")
    if env is not None:
        return env.strip().lower()
    if config is None:
        try:
            from hermes_cli.config import load_config

            config = load_config()
        except Exception:
            config = {}
    agent_cfg = (config or {}).get("agent") if isinstance(config, dict) else None
    cfg_val = agent_cfg.get("verify_on_stop") if isinstance(agent_cfg, dict) else None
    if isinstance(cfg_val, bool):
        return "true" if cfg_val else "false"
    if isinstance(cfg_val, str):
        return cfg_val.strip().lower()
    return ""


def gate_mode_active(config: Optional[Dict[str, Any]] = None) -> bool:
    """Return True when ``verify_on_stop`` is ``"gate"`` and the gate should fire.

    The gate respects the same messaging-surface default as verification_stop:
    if ``verify_gate.messaging_surface`` is false (the default) and the current
    session is a human messaging surface (Telegram, Discord, etc.), the gate
    is suppressed — the verification narrative would be chat noise.
    """
    if _resolve_verify_on_stop(config) != "gate":
        return False
    gate_cfg = _load_gate_config(config)
    if not gate_cfg.get("messaging_surface", False):
        if _session_is_messaging_surface():
            return False
    return True


def verify_on_stop_gate_enabled(config: Optional[Dict[str, Any]] = None) -> bool:
    """Backward-compatible alias for ``gate_mode_active``."""
    return gate_mode_active(config)


# ──────────────────────────────────────────────────────────────────────────
# Evidence checking
# ──────────────────────────────────────────────────────────────────────────

def _workspace_has_passing_evidence(
    *,
    session_id: Optional[str],
    changed_paths: List[str],
    gate_cfg: Dict[str, Any],
) -> Tuple[bool, List[str], Optional[Dict[str, Any]]]:
    """Return ``(all_passing, unverified_paths, first_failing_status)``.

    Checks each edited workspace root via ``verification_evidence.verification_status``.
    A path is "passing" when the latest evidence for its workspace root has
    status ``"passed"`` and the evidence is not stale (edited after the
    evidence was recorded). When ``require_canonical`` is false, missing
    verification commands do not block the gate (the agent is expected to
    create ad-hoc evidence). When ``allow_ad_hoc`` is true, ad-hoc temp-script
    evidence counts as passing.
    """
    try:
        from agent.coding_context import project_facts_for
        from agent.verification_evidence import verification_status
    except Exception:
        # If we can't import the evidence ledger, don't block — degrade to nudge.
        return True, [], None

    require_canonical = bool(gate_cfg.get("require_canonical", True))
    allow_ad_hoc = bool(gate_cfg.get("allow_ad_hoc", True))

    # Group changed paths by workspace root so we issue one status check per root.
    root_to_paths: Dict[str, List[str]] = {}
    for raw_path in changed_paths:
        if not raw_path:
            continue
        try:
            p = Path(str(raw_path)).expanduser()
            candidate = p if p.is_dir() else p.parent
            root_str = str(candidate.resolve())
        except Exception:
            continue
        root_to_paths.setdefault(root_str, []).append(str(raw_path))

    if not root_to_paths:
        return True, [], None

    unverified: List[str] = []
    first_failing_status: Optional[Dict[str, Any]] = None

    for root_str, paths in root_to_paths.items():
        facts = project_facts_for(root_str)
        if not facts:
            # Not a code workspace — nothing to verify for this path group.
            continue

        verify_commands = list(facts.get("verifyCommands") or [])
        if require_canonical and not verify_commands:
            # No canonical verify commands and require_canonical is true —
            # we can't gate meaningfully, so treat this root as passing.
            continue

        status = verification_status(session_id=session_id, cwd=root_str)
        state = str(status.get("status") or "unverified")

        if state in _PASSING_STATUSES:
            # Confirm the evidence is not stale (edited after evidence was recorded).
            evidence = status.get("evidence")
            if isinstance(evidence, dict):
                kind = str(evidence.get("kind") or "")
                if kind == "ad_hoc" and not allow_ad_hoc:
                    # Ad-hoc evidence doesn't count when allow_ad_hoc is false.
                    unverified.extend(paths)
                    if first_failing_status is None:
                        first_failing_status = status
                    continue
            continue

        # status is "stale", "failed", "unverified", or anything else — gate.
        unverified.extend(paths)
        if first_failing_status is None:
            first_failing_status = status

    return (len(unverified) == 0), unverified, first_failing_status


# ──────────────────────────────────────────────────────────────────────────
# Gate decision
# ──────────────────────────────────────────────────────────────────────────

def gate_continuation(
    *,
    session_id: Optional[str],
    changed_paths: Iterable[str],
    attempts: int = 0,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Return a synthetic user continuation when the gate should re-enter.

    Returns ``None`` when:

    - gate mode is not active (``verify_on_stop != "gate"``),
    - all edited paths have fresh passing evidence,
    - the attempt count has reached ``max_attempts`` (degrade to nudge —
      the caller falls back to ``build_verify_on_stop_nudge`` as before),
    - or no changed paths are verifiable code paths.

    When the gate fires, returns a continuation string suitable as a
    synthetic *user* message that preserves role alternation and does not
    mutate the system prompt or past context.
    """
    if not gate_mode_active(config):
        return None

    gate_cfg = _load_gate_config(config)
    max_attempts = int(gate_cfg.get("max_attempts", 3) or 3)

    if attempts >= max_attempts:
        # Exhausted gate budget — degrade to the existing nudge path.
        # The caller (conversation_loop) already calls build_verify_on_stop_nudge
        # when this returns None, so we just signal "no gate action."
        logger.debug(
            "verifier_gate: attempts exhausted (%d/%d) — degrading to nudge",
            attempts, max_attempts,
        )
        return None

    # Filter to verifiable code paths (drop docs/prose/markdown/skills).
    paths = sorted({str(p) for p in _filter_verifiable_paths(changed_paths)})
    if not paths:
        return None

    all_passing, unverified_paths, first_status = _workspace_has_passing_evidence(
        session_id=session_id,
        changed_paths=paths,
        gate_cfg=gate_cfg,
    )
    if all_passing:
        logger.debug("verifier_gate: all paths verified — gate passes")
        return None

    return _build_gate_continuation(
        unverified_paths=unverified_paths,
        status=first_status,
        attempts=attempts,
        max_attempts=max_attempts,
    )


def build_verify_gate_continuation(
    *,
    session_id: Optional[str],
    changed_paths: Iterable[str],
    cwd: Optional[str] = None,
    attempts: int = 0,
    max_attempts: int = 3,
    allow_ad_hoc: bool = True,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Backward-compatible wrapper around ``gate_continuation``.

    Accepts the explicit-parameter style used by the alternative
    conversation_loop patch and delegates to the canonical implementation.
    ``cwd`` and ``max_attempts``/``allow_ad_hoc`` overrides are folded into
    a synthetic config dict so the single code path is used.
    """
    # Build an override config so explicit params win over config.yaml.
    override_cfg = config if isinstance(config, dict) else {}
    if override_cfg:
        agent = dict(override_cfg.get("agent") or {})
        gate = dict(agent.get("verify_gate") or {})
    else:
        gate = {}
    gate["max_attempts"] = max_attempts
    gate["allow_ad_hoc"] = allow_ad_hoc
    agent_cfg = {"verify_gate": gate, "verify_on_stop": "gate"}
    merged = {"agent": agent_cfg}
    return gate_continuation(
        session_id=session_id,
        changed_paths=changed_paths,
        attempts=attempts,
        config=merged,
    )


def _build_gate_continuation(
    *,
    unverified_paths: List[str],
    status: Optional[Dict[str, Any]],
    attempts: int,
    max_attempts: int,
) -> str:
    """Build the synthetic user continuation for a gate re-entry."""
    remaining = max_attempts - attempts

    # Status detail (reuse verification_stop's formatter when available).
    try:
        from agent.verification_stop import _status_detail

        status_detail = _status_detail(status) if status else "unverified"
    except Exception:
        state = str((status or {}).get("status") or "unverified")
        status_detail = state

    # Path summary (bounded — mirror verification_stop's cap).
    max_paths = 8
    shown = unverified_paths[:max_paths]
    path_lines = [f"- `{p}`" for p in shown]
    leftover = len(unverified_paths) - len(shown)
    if leftover > 0:
        path_lines.append(f"- ... and {leftover} more")
    path_block = "\n".join(path_lines) if path_lines else "(none)"

    return (
        "[System: Verification gate — the following edited paths lack fresh "
        f"passing verification evidence (attempt {attempts + 1}/{max_attempts}, "
        f"{remaining} gate attempt(s) remaining before degrade-to-nudge).\n\n"
        f"Verification status: {status_detail}\n\n"
        f"Unverified paths:\n{path_block}\n\n"
        "Run the relevant verification command(s) now, read any failure output, "
        "repair the code, and only finish when verification passes — or explain "
        "the concrete blocker if verification is genuinely impossible.]"
    )


__all__ = [
    "gate_mode_active",
    "gate_continuation",
    "build_verify_gate_continuation",
    "verify_on_stop_gate_enabled",
    "resolve_gate_config",
    "DEFAULT_VERIFY_GATE_CONFIG",
]