"""CLI commands for the ``hermes plan`` subcommand.

The argparse tree is built by :func:`register_cli` and dispatched by
:func:`plan_command`. Subcommands:

    hermes plan create   --goal ... [--steps s1 -s s2 ...] [--session-id ...]
    hermes plan add-step  <plan-id> <description>
    hermes plan step done <step-id> --evidence <event-id> [--force]
    hermes plan status    [plan-id] [--session-id ...]
    hermes plan list      [--session-id ...] [--status ...] [--limit N]
    hermes plan delete    <plan-id>
    hermes plan prune     [--days N]

The ``step done`` path cross-checks
``agent.verification_evidence.verification_status()`` and only flips a step to
``done`` when the evidence status is ``passed`` (or when ``--force`` is given as
the operator escape hatch). See :func:`agent.plan_store.complete_step`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

from agent import plan_store

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# argparse setup
# ---------------------------------------------------------------------------

def register_cli(subparser: argparse.ArgumentParser) -> None:
    """Build the ``hermes plan <action>`` subparser tree."""
    subs = subparser.add_subparsers(dest="plan_action")

    # ── create ──────────────────────────────────────────────────────────────
    create_p = subs.add_parser(
        "create",
        help="Create a plan with a goal and optional ordered steps",
    )
    create_p.add_argument("--goal", required=True, help="Plan goal (required)")
    create_p.add_argument(
        "--steps", "-s", action="append", default=[],
        help="Step description (repeat -s for each ordered step)",
    )
    create_p.add_argument(
        "--session-id", default=None,
        help="Associate the plan with a session id",
    )
    create_p.add_argument(
        "--parent-plan-id", default=None,
        help="Parent plan id (for sub-plans)",
    )
    create_p.add_argument(
        "--status", default="pending",
        choices=list(plan_store.VALID_STATUSES),
        help="Initial plan status (default: pending)",
    )

    # ── add-step ────────────────────────────────────────────────────────────
    addstep_p = subs.add_parser(
        "add-step",
        help="Append a step to an existing plan",
    )
    addstep_p.add_argument("plan_id", help="Plan id to append to")
    addstep_p.add_argument("description", help="Step description")

    # ── step done ───────────────────────────────────────────────────────────
    stepdone_p = subs.add_parser(
        "step-done",
        aliases=["step-done", "done"],
        help="Mark a step done (cross-checks verification evidence)",
    )
    stepdone_p.add_argument("step_id", help="Step id to complete")
    stepdone_p.add_argument(
        "--evidence", default=None,
        help="Verification evidence event id to record on the step",
    )
    stepdone_p.add_argument(
        "--force", action="store_true",
        help="Complete without evidence verification (operator escape hatch)",
    )
    stepdone_p.add_argument(
        "--critique", default=None,
        help="Optional JSON-encoded critique blob to store with the step",
    )
    stepdone_p.add_argument(
        "--cwd", default=None,
        help="Working directory for the verification cross-check (default: process cwd)",
    )
    stepdone_p.add_argument(
        "--session-id", default=None,
        help="Session id for the verification cross-check (default: step's plan session)",
    )

    # ── step status (low-level status setter, no evidence gate) ─────────────
    stepstatus_p = subs.add_parser(
        "step-status",
        help="Set a step's status directly (no evidence check)",
    )
    stepstatus_p.add_argument("step_id")
    stepstatus_p.add_argument(
        "status", choices=list(plan_store.VALID_STATUSES),
        help="New status",
    )
    stepstatus_p.add_argument(
        "--evidence", default=None,
        help="Optional evidence event id to record",
    )

    # ── status ──────────────────────────────────────────────────────────────
    status_p = subs.add_parser(
        "status",
        help="Show a plan with its ordered steps",
    )
    status_p.add_argument("plan_id", help="Plan id (omit with --session-id to show active)")
    status_p.add_argument(
        "--session-id", default=None,
        help="Show the most recent active plan for this session",
    )

    # ── list ────────────────────────────────────────────────────────────────
    list_p = subs.add_parser("list", aliases=["ls"], help="List plans")
    list_p.add_argument("--session-id", default=None)
    list_p.add_argument(
        "--status", default=None, choices=list(plan_store.VALID_STATUSES),
    )
    list_p.add_argument("--limit", type=int, default=100)

    # ── delete ──────────────────────────────────────────────────────────────
    delete_p = subs.add_parser("delete", help="Delete a plan and its steps")
    delete_p.add_argument("plan_id")

    # ── prune ───────────────────────────────────────────────────────────────
    prune_p = subs.add_parser(
        "prune",
        help="Delete done/failed/superseded plans older than --days",
    )
    prune_p.add_argument(
        "--days", type=int, default=30,
        help="Retention window in days (default: 30)",
    )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def _print_json(obj: Any) -> None:
    json.dump(obj, sys.stdout, indent=2, ensure_ascii=False, default=str)
    sys.stdout.write("\n")


def _print_plan(plan: Optional[Dict[str, Any]]) -> None:
    if plan is None:
        print("not found")
        return
    _print_json(plan)


def plan_command(args: argparse.Namespace) -> int:
    """Dispatch a ``hermes plan <action>`` invocation. Returns an exit code."""
    action = getattr(args, "plan_action", None)
    if action is None:
        print("usage: hermes plan <create|add-step|step-done|step-status|status|list|delete|prune>",
              file=sys.stderr)
        return 2

    try:
        return _dispatch(action, args)
    except Exception as exc:  # surface a clean CLI error, not a traceback
        logger.debug("hermes plan %s failed: %s", action, exc, exc_info=True)
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _dispatch(action: str, args: argparse.Namespace) -> int:
    if action == "create":
        plan = plan_store.create_plan(
            args.goal,
            session_id=args.session_id,
            steps=args.steps or None,
            parent_plan_id=args.parent_plan_id,
            status=args.status,
        )
        _print_plan(plan)
        return 0

    if action == "add-step":
        step = plan_store.add_step(args.plan_id, args.description)
        _print_json(step)
        return 0

    if action in ("step-done", "step_done", "done"):
        critique: Optional[Dict[str, Any]] = None
        if args.critique:
            try:
                critique = json.loads(args.critique)
            except json.JSONDecodeError as exc:
                print(f"error: --critique is not valid JSON: {exc}", file=sys.stderr)
                return 2
        cwd = args.cwd if args.cwd is not None else os.getcwd()
        ok, reason = plan_store.complete_step(
            args.step_id,
            evidence_event_id=args.evidence,
            force=args.force,
            critique=critique,
            cwd=cwd,
            session_id=args.session_id,
        )
        if not ok:
            print(f"not done: {reason}", file=sys.stderr)
            return 1
        print(f"done: {reason}")
        return 0

    if action == "step-status":
        updated = plan_store.update_step_status(
            args.step_id,
            args.status,
            evidence_event_id=args.evidence,
        )
        if not updated:
            print("not found", file=sys.stderr)
            return 1
        print(f"ok: {args.step_id} -> {args.status}")
        return 0

    if action == "status":
        plan_id: Optional[str] = args.plan_id
        if not plan_id and args.session_id:
            plans = plan_store.list_plans(
                session_id=args.session_id, status="active", limit=1,
            )
            if not plans:
                print("no active plan for session", file=sys.stderr)
                return 1
            plan_id = plans[0]["id"]
        if not plan_id:
            print("error: plan_id required (or --session-id with an active plan)",
                  file=sys.stderr)
            return 2
        plan = plan_store.get_plan_with_steps(plan_id)
        _print_plan(plan)
        return 0 if plan else 1

    if action in ("list", "ls"):
        plans = plan_store.list_plans(
            session_id=args.session_id,
            status=args.status,
            limit=args.limit,
        )
        _print_json(plans)
        return 0

    if action == "delete":
        deleted = plan_store.delete_plan(args.plan_id)
        print("deleted" if deleted else "not found")
        return 0 if deleted else 1

    if action == "prune":
        n = plan_store.prune_older_than(args.days)
        print(f"pruned {n} plan(s)")
        return 0

    print(f"error: unknown plan action {action!r}", file=sys.stderr)
    return 2