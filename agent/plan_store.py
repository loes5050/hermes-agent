"""SQLite-backed plan ledger for Hermes Agent (M1 PlanStore).

A *plan* is a goal plus an ordered list of *steps*. The agent (driven by the
``planning`` skill) decomposes a complex request into a plan, works the steps
one at a time, and records verification evidence against each step before it
is marked ``done``. Plans are persisted so a long-lived conversation can
resume mid-plan after a restart, and so ``hermes plan status`` / ``hermes plan
list`` can report progress out-of-band.

Storage mirrors the proven patterns in ``hermes_state.py`` /
``hermes_cli/kanban_db.py``:

* Single SQLite file at ``<HERMES_HOME>/plans.db``.
* ``journal_mode=WAL`` via the shared ``hermes_state.apply_wal_with_fallback``
  helper, with DELETE fallback on WAL-incompatible filesystems (NFS/SMB/FUSE).
* A module-level ``threading.Lock`` serializes schema init across same-process
  threads (gateway / dashboard start several); SQLite's WAL lock handles
  cross-process serialization.
* Schema is versioned via a ``schema_version`` pragma row; an idempotent
  additive migration runs on every fresh connection so new columns can be
  introduced without a destructive dump/reload.

The module is **side-effect-free at import time** — no DB file is created until
:func:`connect` is first called. It has no dependency on the CLI, the agent
runtime, or the model; the CLI shim in ``plugins/planning/cli.py`` is the only
caller that touches argparse.

Schema (M1):

    plans(id TEXT PRIMARY KEY, session_id TEXT, goal TEXT,
          created_at TEXT, status TEXT, parent_plan_id TEXT)
    steps(id TEXT PRIMARY KEY, plan_id TEXT, idx INTEGER,
          description TEXT, status TEXT, evidence_event_id TEXT,
          attempts INTEGER, critique_json TEXT)

``status`` is one of: ``pending``, ``active``, ``done``, ``blocked``,
``failed``, ``superseded``.

Step completion is gated on verification evidence: the ``step done`` CLI
subcommand cross-checks ``verification_evidence.verification_status()`` (when
that module is present) and only flips a step to ``done`` when the evidence
status is ``passed``. When the verification module is unavailable the call
fails closed — the step is NOT silently marked done — and the caller is told
to pass ``--force`` to override (operator escape hatch, never the default).
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: All legal plan/step statuses. Order is significant only for display sorting.
VALID_STATUSES: Tuple[str, ...] = (
    "pending",
    "active",
    "done",
    "blocked",
    "failed",
    "superseded",
)
_VALID_STATUS_SET = frozenset(VALID_STATUSES)

#: Schema version recorded in ``schema_meta``. Bump when the additive
#: migration in :data:`_MIGRATIONS` grows a new step.
SCHEMA_VERSION = 1

#: Default DB filename under HERMES_HOME.
DEFAULT_DB_FILENAME = "plans.db"


# ---------------------------------------------------------------------------
# Schema SQL
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS plans (
    id             TEXT PRIMARY KEY,
    session_id     TEXT,
    goal           TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'pending',
    parent_plan_id TEXT
);

CREATE TABLE IF NOT EXISTS steps (
    id               TEXT PRIMARY KEY,
    plan_id          TEXT NOT NULL,
    idx              INTEGER NOT NULL,
    description     TEXT NOT NULL,
    status           TEXT NOT NULL DEFAULT 'pending',
    evidence_event_id TEXT,
    attempts         INTEGER NOT NULL DEFAULT 0,
    critique_json    TEXT,
    FOREIGN KEY (plan_id) REFERENCES plans(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_steps_plan_id ON steps(plan_id);
CREATE INDEX IF NOT EXISTS idx_steps_plan_idx ON steps(plan_id, idx);
CREATE INDEX IF NOT EXISTS idx_plans_session ON plans(session_id);
CREATE INDEX IF NOT EXISTS idx_plans_status ON plans(status);
CREATE INDEX IF NOT EXISTS idx_plans_parent ON plans(parent_plan_id);

CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def plan_db_path() -> Path:
    """Resolve the plan ledger DB path.

    Honors ``HERMES_HOME`` (via the shared ``hermes_constants.get_hermes_home``
    resolver, which is profile-aware) and places the file at
    ``<HERMES_HOME>/plans.db``. An explicit ``HERMES_PLANS_DB`` env var, when
    set, pins the file path directly — useful for tests and unusual layouts.
    """
    override = os.environ.get("HERMES_PLANS_DB", "").strip()
    if override:
        return Path(override)
    try:
        from hermes_constants import get_hermes_home  # local import; avoid cycles
        home = get_hermes_home()
    except Exception:
        home = Path(os.path.expanduser("~/.hermes"))
    return home / DEFAULT_DB_FILENAME


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------

#: Serializes schema init across same-process threads (gateway/dashboard).
_INIT_LOCK = threading.Lock()
#: Paths already initialized in this process — cheap fast-path for reconnects.
_INITIALIZED_PATHS: set[str] = set()


def _sqlite_connect(path: Path) -> sqlite3.Connection:
    """Open a SQLite connection with a generous busy timeout.

    The busy timeout lets SQLite serialize concurrent writers via its WAL lock
    rather than surfacing ``database is locked`` to callers. 5s is the same
    headroom kanban_db uses and is plenty for the low-write plan ledger.
    """
    conn = sqlite3.connect(str(path), timeout=5.0, isolation_level=None)
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def _apply_pragmas(conn: sqlite3.Connection, *, db_label: str) -> None:
    """Apply durable+concurrent pragmas, with WAL fallback on incompatible FS."""
    from hermes_state import apply_wal_with_fallback
    apply_wal_with_fallback(conn, db_label=db_label)
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("PRAGMA foreign_keys=ON")


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create tables if absent and stamp the schema version (idempotent)."""
    conn.executescript(_SCHEMA_SQL)
    row = conn.execute(
        "SELECT value FROM schema_meta WHERE key='schema_version'"
    ).fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO schema_meta(key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
    else:
        # Forward-only version stamp. M1 has no destructive migrations; future
        # additive ALTER TABLEs would land in a _migrate() helper keyed off
        # the stored version.
        try:
            stored = int(row[0])
        except (TypeError, ValueError):
            stored = 0
        if stored < SCHEMA_VERSION:
            conn.execute(
                "UPDATE schema_meta SET value=? WHERE key='schema_version'",
                (str(SCHEMA_VERSION),),
            )


def connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Open (and initialize if needed) the plan ledger DB.

    WAL mode is enabled on every connection; it's a no-op after the first
    connection to the same path. The first connection auto-runs schema setup so
    callers never need a separate init step. Subsequent connections skip the
    schema pass via :data:`_INITIALIZED_PATHS`.

    Args:
        db_path: explicit file path (tests, overrides). When ``None`` the path
            is resolved via :func:`plan_db_path`.
    """
    path = db_path if db_path is not None else plan_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved = str(path.resolve())
    conn = _sqlite_connect(path)
    try:
        conn.row_factory = sqlite3.Row
        with _INIT_LOCK:
            _apply_pragmas(conn, db_label=f"plans.db ({path.name})")
            if resolved not in _INITIALIZED_PATHS:
                _ensure_schema(conn)
                _INITIALIZED_PATHS.add(resolved)
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        raise
    return conn


@contextmanager
def connect_closing(db_path: Optional[Path] = None):
    """Open a plan DB connection and guarantee it is closed on exit.

    Prefer this over ``with connect() as conn:`` — sqlite3's built-in context
    manager only commits/rollbacks; it does NOT close the file descriptor.
    In long-lived processes (gateway, dashboard) unclosed connections
    accumulate as open FDs and eventually hit ``[Errno 24] Too many open
    files``.
    """
    conn = connect(db_path=db_path)
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    """UTC timestamp in ISO-8601 (``Z`` suffix), stable across platforms."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _new_id(prefix: str) -> str:
    """Short, URL-safe, collision-resistant id with a readable prefix."""
    return f"{prefix}_{secrets.token_hex(6)}"


def _validate_status(status: str, *, field: str = "status") -> None:
    if status not in _VALID_STATUS_SET:
        raise ValueError(
            f"invalid {field} {status!r}; expected one of {sorted(_VALID_STATUS_SET)}"
        )


def _row_to_plan(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    return d


def _row_to_step(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    # Lazy-parse critique_json for callers; keep raw too.
    raw = d.get("critique_json")
    if isinstance(raw, str) and raw:
        try:
            d["critique"] = json.loads(raw)
        except json.JSONDecodeError:
            d["critique"] = None
    else:
        d["critique"] = None
    return d


# ---------------------------------------------------------------------------
# Verification evidence cross-check
# ---------------------------------------------------------------------------

def _evidence_passed(
    *,
    session_id: Optional[str],
    cwd: Optional[str],
) -> Tuple[bool, str]:
    """Return ``(passed, reason)`` by cross-checking ``verification_evidence``.

    Calls ``agent.verification_evidence.verification_status(session_id=...,
    cwd=...)`` when that module is importable. That function returns a *dict*
    with a ``"status"`` key (one of ``passed``, ``failed``, ``stale``,
    ``unverified``, ``not_applicable``). This helper returns ``(True, "passed")``
    only when that status is exactly ``"passed"``. Any other status, missing
    result, or import failure returns ``(False, <reason>)`` — the call fails
    CLOSED so a broken/missing verification path can never silently mark a step
    done. The caller must pass ``--force`` to override (operator escape hatch).
    """
    try:
        from agent.verification_evidence import verification_status  # type: ignore
    except Exception as exc:  # pragma: no cover — module optional in M1
        return False, f"verification_evidence unavailable: {exc}"
    try:
        result = verification_status(session_id=session_id, cwd=cwd)
    except Exception as exc:
        return False, f"verification_status() raised: {exc}"
    if not isinstance(result, dict):
        return False, f"verification_status() returned {type(result).__name__}, expected dict"
    status = result.get("status")
    if status == "passed":
        return True, "passed"
    return False, f"evidence status is {status!r} (expected 'passed')"


# ---------------------------------------------------------------------------
# Plan CRUD
# ---------------------------------------------------------------------------

def create_plan(
    goal: str,
    *,
    session_id: Optional[str] = None,
    steps: Optional[Sequence[str]] = None,
    parent_plan_id: Optional[str] = None,
    status: str = "pending",
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Create a plan with an optional ordered list of step descriptions.

    Returns the created plan row as a dict (including ``id``). Steps are
    inserted with sequential ``idx`` starting at 0.
    """
    if not goal or not goal.strip():
        raise ValueError("goal must be a non-empty string")
    _validate_status(status, field="plan status")
    plan_id = _new_id("plan")
    now = _now_iso()
    with connect_closing(db_path) as conn:
        conn.execute(
            "INSERT INTO plans(id, session_id, goal, created_at, status, parent_plan_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (plan_id, session_id, goal.strip(), now, status, parent_plan_id),
        )
        step_rows: List[Tuple[str, str, int, str, str]] = []
        for idx, desc in enumerate(steps or ()):
            if not desc or not desc.strip():
                raise ValueError(f"step {idx} has empty description")
            step_rows.append(
                (_new_id("step"), plan_id, idx, desc.strip(), "pending")
            )
        if step_rows:
            conn.executemany(
                "INSERT INTO steps(id, plan_id, idx, description, status) "
                "VALUES (?, ?, ?, ?, ?)",
                step_rows,
            )
    return get_plan(plan_id, db_path=db_path)  # type: ignore[return-value]


def get_plan(plan_id: str, *, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Return a plan row by id, or ``None`` if not found."""
    with connect_closing(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM plans WHERE id=?", (plan_id,)
        ).fetchone()
        if row is None:
            return None
        return _row_to_plan(row)


def get_plan_with_steps(
    plan_id: str, *, db_path: Optional[Path] = None
) -> Optional[Dict[str, Any]]:
    """Return a plan dict augmented with its ordered ``steps`` list."""
    with connect_closing(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM plans WHERE id=?", (plan_id,)
        ).fetchone()
        if row is None:
            return None
        plan = _row_to_plan(row)
        step_rows = conn.execute(
            "SELECT * FROM steps WHERE plan_id=? ORDER BY idx ASC", (plan_id,)
        ).fetchall()
        plan["steps"] = [_row_to_step(r) for r in step_rows]
        return plan


def list_plans(
    *,
    session_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """List plans, optionally filtered by session and/or status.

    Results are newest-first (``created_at`` descending) and capped at
    ``limit`` (default 100, max 1000).
    """
    if status is not None:
        _validate_status(status)
    limit = max(1, min(int(limit), 1000))
    clauses: List[str] = []
    params: List[Any] = []
    if session_id is not None:
        clauses.append("session_id=?")
        params.append(session_id)
    if status is not None:
        clauses.append("status=?")
        params.append(status)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = f"SELECT * FROM plans{where} ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    with connect_closing(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_row_to_plan(r) for r in rows]


def update_plan_status(
    plan_id: str,
    status: str,
    *,
    db_path: Optional[Path] = None,
) -> bool:
    """Set a plan's status. Returns ``True`` if a row was updated."""
    _validate_status(status, field="plan status")
    with connect_closing(db_path) as conn:
        cur = conn.execute(
            "UPDATE plans SET status=? WHERE id=?", (status, plan_id)
        )
        return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Step CRUD
# ---------------------------------------------------------------------------

def get_step(step_id: str, *, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Return a step row by id, or ``None`` if not found."""
    with connect_closing(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM steps WHERE id=?", (step_id,)
        ).fetchone()
        if row is None:
            return None
        return _row_to_step(row)


def list_steps(plan_id: str, *, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Return the ordered steps for a plan."""
    with connect_closing(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM steps WHERE plan_id=? ORDER BY idx ASC", (plan_id,)
        ).fetchall()
    return [_row_to_step(r) for r in rows]


def add_step(
    plan_id: str,
    description: str,
    *,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Append a step to a plan (idx = max existing idx + 1)."""
    if not description or not description.strip():
        raise ValueError("description must be non-empty")
    with connect_closing(db_path) as conn:
        # Verify plan exists.
        exists = conn.execute(
            "SELECT 1 FROM plans WHERE id=?", (plan_id,)
        ).fetchone()
        if not exists:
            raise KeyError(f"plan {plan_id!r} not found")
        row = conn.execute(
            "SELECT COALESCE(MAX(idx), -1) FROM steps WHERE plan_id=?", (plan_id,)
        ).fetchone()
        next_idx = (row[0] + 1) if row is not None else 0
        step_id = _new_id("step")
        conn.execute(
            "INSERT INTO steps(id, plan_id, idx, description, status) "
            "VALUES (?, ?, ?, ?, 'pending')",
            (step_id, plan_id, next_idx, description.strip()),
        )
    result = get_step(step_id, db_path=db_path)
    assert result is not None  # just inserted
    return result


def update_step_status(
    step_id: str,
    status: str,
    *,
    evidence_event_id: Optional[str] = None,
    critique: Optional[Dict[str, Any]] = None,
    db_path: Optional[Path] = None,
) -> bool:
    """Set a step's status, optionally recording evidence id / critique.

    This is the low-level setter — it does NOT gate ``done`` on evidence. The
    CLI ``step done`` path uses :func:`complete_step` for the evidence
    cross-check; direct programmatic callers may use this to set intermediate
    statuses (``active``, ``blocked``, ``failed``).

    Returns ``True`` if a row was updated.
    """
    _validate_status(status, field="step status")
    sets: List[str] = ["status=?"]
    params: List[Any] = [status]
    if evidence_event_id is not None:
        sets.append("evidence_event_id=?")
        params.append(evidence_event_id)
    if critique is not None:
        sets.append("critique_json=?")
        params.append(json.dumps(critique, ensure_ascii=False))
    if status in ("active", "failed"):
        sets.append("attempts=attempts+1")
    params.append(step_id)
    sql = f"UPDATE steps SET {', '.join(sets)} WHERE id=?"
    with connect_closing(db_path) as conn:
        cur = conn.execute(sql, params)
        return cur.rowcount > 0


def complete_step(
    step_id: str,
    *,
    evidence_event_id: Optional[str] = None,
    force: bool = False,
    critique: Optional[Dict[str, Any]] = None,
    cwd: Optional[str] = None,
    session_id: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> Tuple[bool, str]:
    """Mark a step ``done``, gated on verification evidence.

    The gate cross-checks ``agent.verification_evidence.verification_status(
    session_id=..., cwd=...)`` and only flips the step to ``done`` when the
    returned status is ``"passed"``. The session id is resolved from the
    step's parent plan (falling back to the ``session_id`` argument), and
    ``cwd`` defaults to the process's current directory. The
    ``evidence_event_id`` (if provided) is recorded on the step as the
    evidence reference regardless of the gate outcome.

    When ``agent.verification_evidence`` is not importable, the call fails
    CLOSED (step stays in its current status) unless ``force=True`` — the
    operator escape hatch, never the default. Any non-``passed`` evidence
    status also fails closed; ``force`` overrides the gate.

    Returns ``(updated: bool, reason: str)``.
    """
    if not force:
        # Resolve the plan's session_id for the verification cross-check.
        resolved_sid = session_id
        if resolved_sid is None:
            with connect_closing(db_path) as conn:
                row = conn.execute(
                    "SELECT p.session_id FROM steps s JOIN plans p ON p.id = s.plan_id "
                    "WHERE s.id=?",
                    (step_id,),
                ).fetchone()
                if row is not None:
                    resolved_sid = row["session_id"] if row[0] else None
        passed, reason = _evidence_passed(session_id=resolved_sid, cwd=cwd)
        if not passed:
            return False, f"evidence check failed: {reason}"
    updated = update_step_status(
        step_id,
        "done",
        evidence_event_id=evidence_event_id,
        critique=critique,
        db_path=db_path,
    )
    if not updated:
        return False, f"step {step_id!r} not found"
    return True, "done"


def record_critique(
    step_id: str,
    critique: Dict[str, Any],
    *,
    db_path: Optional[Path] = None,
) -> bool:
    """Store a critique blob against a step (additive — does not change status)."""
    if not isinstance(critique, dict):
        raise TypeError("critique must be a dict")
    with connect_closing(db_path) as conn:
        cur = conn.execute(
            "UPDATE steps SET critique_json=? WHERE id=?",
            (json.dumps(critique, ensure_ascii=False), step_id),
        )
        return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------

def delete_plan(plan_id: str, *, db_path: Optional[Path] = None) -> bool:
    """Delete a plan and its steps (cascades via FK). Returns ``True`` if deleted."""
    with connect_closing(db_path) as conn:
        cur = conn.execute("DELETE FROM plans WHERE id=?", (plan_id,))
        return cur.rowcount > 0


def prune_older_than(days: int, *, db_path: Optional[Path] = None) -> int:
    """Delete plans (and their steps) older than ``days`` days.

    Only ``done``/``failed``/``superseded`` plans are eligible — an in-flight
    plan is never pruned regardless of age. Returns the number of plans
    deleted. Intended for the ``planning.ledger_retention_days`` janitor.
    """
    if days < 0:
        raise ValueError("days must be non-negative")
    cutoff = (time.time() - days * 86400.0)
    cutoff_dt = datetime.fromtimestamp(cutoff, tz=timezone.utc)
    cutoff_iso = cutoff_dt.isoformat(timespec="seconds").replace("+00:00", "Z")
    eligible = ("done", "failed", "superseded")
    with connect_closing(db_path) as conn:
        cur = conn.execute(
            "DELETE FROM plans WHERE created_at < ? AND status IN (?, ?, ?)",
            (cutoff_iso, *eligible),
        )
        return cur.rowcount


__all__ = [
    "VALID_STATUSES",
    "SCHEMA_VERSION",
    "DEFAULT_DB_FILENAME",
    "plan_db_path",
    "connect",
    "connect_closing",
    "create_plan",
    "get_plan",
    "get_plan_with_steps",
    "list_plans",
    "update_plan_status",
    "get_step",
    "list_steps",
    "add_step",
    "update_step_status",
    "complete_step",
    "record_critique",
    "delete_plan",
    "prune_older_than",
]