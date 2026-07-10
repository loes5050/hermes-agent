"""Unit tests for the verifier gate wiring.

Tests that:
- gate_continuation fires when verify_on_stop == "gate" and paths lack evidence,
- gate_continuation does NOT fire when mode is off / false / auto / true,
- gate_mode_active resolves correctly for each mode,
- the gate degrades (returns None) after max_attempts,
- the config default never ships with "gate" as the default.
"""

import json
from pathlib import Path
from unittest import mock

import pytest

from agent.verifier_gate import (
    DEFAULT_VERIFY_GATE_CONFIG,
    gate_continuation,
    gate_mode_active,
)


# ──────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clear_gate_env(monkeypatch):
    """Clear HERMES_VERIFY_ON_STOP so tests control mode via config dict only."""
    for var in (
        "HERMES_VERIFY_ON_STOP",
        "HERMES_SESSION_PLATFORM",
        "HERMES_PLATFORM",
    ):
        monkeypatch.delenv(var, raising=False)
    # Force messaging-surface detection off so gate_mode_active isn't suppressed.
    monkeypatch.setattr(
        "agent.verifier_gate._session_is_messaging_surface", lambda: False
    )
    yield


def _gate_config(mode, gate_overrides=None):
    """Build a config dict with the given verify_on_stop mode and gate overrides."""
    gate = dict(DEFAULT_VERIFY_GATE_CONFIG)
    if gate_overrides:
        gate.update(gate_overrides)
    return {"agent": {"verify_on_stop": mode, "verify_gate": gate}}


def _node_project(root: Path) -> None:
    (root / "package.json").write_text(
        json.dumps({"scripts": {"test": "vitest", "lint": "eslint ."}}),
        encoding="utf-8",
    )


# ──────────────────────────────────────────────────────────────────────────
# gate_mode_active — mode resolution
# ──────────────────────────────────────────────────────────────────────────

class TestGateModeActive:
    def test_gate_mode_active_when_gate(self):
        assert gate_mode_active(_gate_config("gate")) is True

    def test_gate_mode_inactive_when_off(self):
        assert gate_mode_active(_gate_config(False)) is False

    def test_gate_mode_inactive_when_auto(self):
        assert gate_mode_active(_gate_config("auto")) is False

    def test_gate_mode_inactive_when_true(self):
        # "true" enables the nudge but NOT the gate — gate is "gate" only.
        assert gate_mode_active(_gate_config(True)) is False

    def test_gate_mode_inactive_when_empty(self):
        assert gate_mode_active(_gate_config("")) is False

    def test_gate_mode_inactive_when_unset(self):
        # No agent section at all.
        assert gate_mode_active({}) is False

    def test_gate_mode_suppressed_on_messaging_surface(self, monkeypatch):
        monkeypatch.setattr(
            "agent.verifier_gate._session_is_messaging_surface", lambda: True
        )
        # Default messaging_surface=False suppresses the gate on chat surfaces.
        assert gate_mode_active(_gate_config("gate")) is False

    def test_gate_mode_allowed_on_messaging_surface_when_overridden(self, monkeypatch):
        monkeypatch.setattr(
            "agent.verifier_gate._session_is_messaging_surface", lambda: True
        )
        cfg = _gate_config("gate", gate_overrides={"messaging_surface": True})
        assert gate_mode_active(cfg) is True


# ──────────────────────────────────────────────────────────────────────────
# gate_continuation — fires only when mode is "gate"
# ──────────────────────────────────────────────────────────────────────────

class TestGateContinuationMode:
    """The core wiring contract: gate fires when mode==gate, not when off."""

    def test_returns_none_when_mode_off(self):
        result = gate_continuation(
            session_id="s1",
            changed_paths=["src/app.py"],
            attempts=0,
            config=_gate_config(False),
        )
        assert result is None

    def test_returns_none_when_mode_auto(self):
        result = gate_continuation(
            session_id="s1",
            changed_paths=["src/app.py"],
            attempts=0,
            config=_gate_config("auto"),
        )
        assert result is None

    def test_returns_none_when_mode_true(self):
        result = gate_continuation(
            session_id="s1",
            changed_paths=["src/app.py"],
            attempts=0,
            config=_gate_config(True),
        )
        assert result is None

    def test_returns_none_when_mode_unset(self):
        result = gate_continuation(
            session_id="s1",
            changed_paths=["src/app.py"],
            attempts=0,
            config={},
        )
        assert result is None

    def test_returns_continuation_when_gate_mode_and_unverified(self):
        """When mode==gate and paths lack evidence, gate fires."""
        with mock.patch(
            "agent.verifier_gate._workspace_has_passing_evidence",
            return_value=(False, ["src/app.py"], {"status": "unverified"}),
        ):
            result = gate_continuation(
                session_id="s1",
                changed_paths=["src/app.py"],
                attempts=0,
                config=_gate_config("gate"),
            )
        assert result is not None
        assert "Verification gate" in result
        assert "src/app.py" in result

    def test_returns_none_when_gate_mode_but_all_verified(self):
        """When mode==gate but all paths have passing evidence, gate passes."""
        with mock.patch(
            "agent.verifier_gate._workspace_has_passing_evidence",
            return_value=(True, [], None),
        ):
            result = gate_continuation(
                session_id="s1",
                changed_paths=["src/app.py"],
                attempts=0,
                config=_gate_config("gate"),
            )
        assert result is None

    def test_returns_none_when_attempts_exhausted(self):
        """After max_attempts, the gate degrades (returns None for nudge fallback)."""
        with mock.patch(
            "agent.verifier_gate._workspace_has_passing_evidence",
            return_value=(False, ["src/app.py"], {"status": "failed"}),
        ):
            result = gate_continuation(
                session_id="s1",
                changed_paths=["src/app.py"],
                attempts=3,  # at default max_attempts=3
                config=_gate_config("gate"),
            )
        assert result is None

    def test_returns_none_when_attempts_below_max_still_fires(self):
        """Attempts below max_attempts still fires the gate."""
        with mock.patch(
            "agent.verifier_gate._workspace_has_passing_evidence",
            return_value=(False, ["src/app.py"], {"status": "failed"}),
        ):
            result = gate_continuation(
                session_id="s1",
                changed_paths=["src/app.py"],
                attempts=2,  # one remaining before max=3
                config=_gate_config("gate"),
            )
        assert result is not None
        assert "1 gate attempt" in result  # remaining = 3 - 2 = 1

    def test_returns_none_when_no_verifiable_paths(self):
        """Doc/markdown-only edits never gate (no verifiable code paths)."""
        result = gate_continuation(
            session_id="s1",
            changed_paths=["README.md", "docs/guide.rst"],
            attempts=0,
            config=_gate_config("gate"),
        )
        assert result is None

    def test_returns_none_when_empty_changed_paths(self):
        result = gate_continuation(
            session_id="s1",
            changed_paths=[],
            attempts=0,
            config=_gate_config("gate"),
        )
        assert result is None


# ──────────────────────────────────────────────────────────────────────────
# Config default safety — verify_on_stop never defaults to "gate"
# ──────────────────────────────────────────────────────────────────────────

class TestConfigDefaultSafe:
    def test_default_verify_gate_config_does_not_enable_gate(self):
        """The shipped DEFAULT_VERIFY_GATE_CONFIG is inert without mode=gate."""
        cfg = {"agent": {"verify_on_stop": False, "verify_gate": DEFAULT_VERIFY_GATE_CONFIG}}
        assert gate_mode_active(cfg) is False

    def test_default_config_verify_on_stop_is_not_gate(self):
        """The DEFAULT_CONFIG verify_on_stop value must be 'auto' or false, never 'gate'."""
        try:
            from hermes_cli.config import DEFAULT_CONFIG
        except Exception:
            pytest.skip("hermes_cli.config not importable in this environment")
        agent = DEFAULT_CONFIG.get("agent", {})
        vos = agent.get("verify_on_stop")
        assert vos != "gate", (
            "verify_on_stop must never default to 'gate' — it is opt-in only"
        )