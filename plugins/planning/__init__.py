"""Planning plugin — registers the ``hermes plan`` CLI subcommand.

M1 scope: operator-facing CLI only. No model tools are added (the planning
*skill* drives decomposition; this plugin exposes the ledger CLI). The
underlying storage lives in :mod:`agent.plan_store`.
"""

from __future__ import annotations

import logging

from plugins.planning.cli import plan_command, register_cli

logger = logging.getLogger(__name__)


def register(ctx) -> None:
    """Register the ``hermes plan`` CLI subcommand via the plugin manager."""
    ctx.register_cli_command(
        name="plan",
        help="Plan ledger — create plans, complete steps, inspect status",
        setup_fn=register_cli,
        handler_fn=plan_command,
        description=(
            "Operator CLI for the plan ledger (M1 PlanStore). Create plans "
            "with steps, mark a step done (cross-checks verification evidence), "
            "and list/inspect plan status. Storage: <HERMES_HOME>/plans.db."
        ),
    )