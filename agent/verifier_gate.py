"""Turn-end verifier gate — evidence-bound completion gating.

Extends verification_stop (policy-only nudge) with a gate mode that re-enters
the conversation loop when code edits lack passing verification evidence.

Activated by ``agent.verify_on_stop: "gate"`` in config.yaml.  When enabled,
the turn-finalizer checks ``verification_evidence.verification_status()``
and, if any edited path lacks passing evidence and attempts are below
``verify_gate.max_attempts``, injects a continuation (synthetic user message)
into the message stream — exactly as ``verification_stop`` does today.

Bounded by ``verify_gate.max_attempts`` (default 3).  After exhausting
attempts, degrades to the existing advisory nudge so the user always gets
a response.  Never gates on messaging surfaces (Telegram, Discord, etc.).

New config keys (all under ``agent.verify_gate``, in config.yaml):
  - max_attempts (int, default 3)
  - require_canonical (bool, default True)
  - allow_ad_hoc (bool, default True)
  - messaging_surface (bool, default False)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from agent.verification_evidence import verification_status, VerificationEvidence

# project_facts_for may not exist in older codebase versions
try:
    from agent.coding_context import project_facts_for
except ImportError:
    project_facts_for = None

logger = logging.getLogger(__name__)

# Default gate configuration
DEFAULT_VERIFY_GATE_CONFIG: Dict[str, Any] = {
    "max_attempts": 3,
    "require_canonical": True,
    "allow_ad_hoc": True,
    "messaging_surface": False,
}


def resolve_gate_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Resolve verify_gate config with defaults for any missing keys."""
    gate_cfg = dict(DEFAULT_VERIFY_GATE_CONFIG)
    if config and isinstance(config.get("verify_gate"), dict):
        user_cfg = config["verify_gate"]
        for key in DEFAULT_VERIFY_GATE_CONFIG:
            if key in user_cfg:
                gate_cfg[key] = user_cfg[key]
    return gate_cfg


def verify_on_stop_gate_enabled(config: Optional[Dict[str, Any]] = None) -> bool:
    """Return True when verify_on_stop is set to 'gate' mode."""
    if not config:
        return False
    agent_cfg = config.get("agent", {})
    if isinstance(agent_cfg, dict):
        return agent_cfg.get("verify_on_stop") == "gate"
    return False


def build_verify_gate_continuation(
    session_id: str,
    changed_paths: set,
    cwd: str,
    attempts: int,
    max_attempts: int,
    allow_ad_hoc: bool = True,
) -> Optional[str]:
    """Build a gate continuation message if verification is needed.

    Checks ``verification_evidence.verification_status()`` for the session.
    If any edited path lacks ``passed`` evidence and attempts remain,
    returns a synthetic user message that re-enters the conversation loop.

    Returns None when:
      - No changed paths to verify.
      - All paths have passing evidence.
      - Attempts exhausted (caller should degrade to nudge).
      - Session is a messaging surface (never gates).
    """
    if not changed_paths:
        return None

    if attempts >= max_attempts:
        logger.debug(
            "verifier_gate: attempts exhausted (%d/%d) — degrading to nudge",
            attempts, max_attempts,
        )
        return None

    # Query the evidence ledger
    status = verification_status(session_id, cwd)
    evidence = status.get("evidence", {}) if isinstance(status, dict) else {}

    # Find paths without passing evidence
    unverified_paths: List[str] = []
    for path in sorted(changed_paths):
        path_evidence = evidence.get(path, {})
        if path_evidence.get("status") != "passed":
            unverified_paths.append(path)

    if not unverified_paths:
        logger.debug("verifier_gate: all paths verified — gate passes")
        return None

    # Build the continuation message
    path_list = "\n".join(f"  - {p}" for p in unverified_paths[:8])
    remaining = max_attempts - attempts

    # Discover canonical verify commands for the workspace
    verify_commands = _discover_verify_commands(cwd) if cwd else []

    cmd_hint = ""
    if verify_commands:
        cmd_hint = "\n\nDetected verify commands:\n" + "\n".join(
            f"  {cmd}" for cmd in verify_commands[:3]
        )

    return (
        f"[VERIFICATION GATE — attempt {attempts + 1}/{max_attempts}]\n\n"
        f"The following files were edited but lack passing verification evidence:\n\n"
        f"{path_list}\n\n"
        f"Before claiming this task is complete, run verification on these files.\n"
        f"{cmd_hint}\n\n"
        f"If you cannot verify these changes, explain why. "
        f"You have {remaining} verification attempt(s) remaining."
    )


def _discover_verify_commands(cwd: str) -> List[str]:
    """Discover canonical verification commands for a workspace."""
    if project_facts_for is None:
        return []
    try:
        facts = project_facts_for(cwd)
        if isinstance(facts, dict):
            return facts.get("verifyCommands", [])
    except Exception:
        pass
    return []
