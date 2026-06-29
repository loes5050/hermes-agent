#!/usr/bin/env python3
"""CI eval runner — orchestrates tiered eval suites for GitHub Actions.

Invoked by CI to run one or more eval suites via ``evals/runners/run_suite.py``
and aggregate the results into a single JSON report with baseline comparison
and gate enforcement.

Tiers
-----

* **Tier 1 — deterministic, no API keys.** Runs the
  ``orchestration``, ``cost_cache``, ``subagent_verify`` and
  ``memory_recall`` suites with ``--deterministic-only`` so they exercise
  structural invariants without any live model calls. Safe on every PR.

* **Tier 2 — live model.** Runs the ``code_task`` and ``research_citation``
  suites against a real provider. Requires ``OPENROUTER_API_KEY`` (or the
  appropriate provider secret). Typically run on merge to ``main`` or on a
  labeled PR, not on every push.

* **Tier 3 — comprehensive.** Runs every suite available in
  ``evals/suites/`` (Tier 1 deterministic + Tier 2 live). Used for nightly /
  release validation.

Gate semantics
--------------

* **Hard gates** — ``cache_break_events == 0``, ``verify_rate >= 0.90``,
  ``recall_at_3 >= 0.85``, ``plan_score >= 0.80``. A hard-gate failure exits
  non-zero so branch protection blocks the PR.
* **Soft gates** — pass-rate regressions against the stored baseline. These
  print warnings but do not fail the build.

Exit codes
----------

* ``0`` — all suites ran and every hard gate passed (soft-gate warnings are
  printed but do not change the exit code).
* ``1`` — at least one hard gate failed, a suite errored, or a required
  secret was missing.
* ``2`` — invalid arguments (argparse handles this itself).

Usage
-----

    # Tier 1 on every PR (no API keys needed)
    python scripts/ci/run_evals.py --tier 1

    # Tier 2 on merge-to-main (needs OPENROUTER_API_KEY)
    python scripts/ci/run_evals.py --tier 2

    # Tier 3 nightly / release
    python scripts/ci/run_evals.py --tier 3

    # Single suite override
    python scripts/ci/run_evals.py --tier 1 --suite orchestration
    python scripts/ci/run_evals.py --tier 2 --suite code_task

    # Custom provider/model for Tier 2+
    python scripts/ci/run_evals.py --tier 2 --provider openrouter \
        --model anthropic/claude-haiku-4.5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_WORKTREE = Path(__file__).resolve().parent.parent.parent
_EVALS_DIR = _WORKTREE / "evals"
_SUITES_DIR = _EVALS_DIR / "suites"
_BASELINES_DIR = _EVALS_DIR / "baselines"
_REPORTS_DIR = _EVALS_DIR / "reports"
_RUNNER = _EVALS_DIR / "runners" / "run_suite.py"
_LATEST_REPORT = _REPORTS_DIR / "latest.json"

# ---------------------------------------------------------------------------
# Tier → suite mapping
# ---------------------------------------------------------------------------
# Order matters for human-readable output; keep deterministic suites first.
_TIER1_SUITES: List[str] = [
    "orchestration",
    "cost_cache",
    "subagent_verify",
    "memory_recall",
]
_TIER2_SUITES: List[str] = [
    "code_task",
    "research_citation",
]

# Hard-gate thresholds. Each gate is keyed by a metric name that the runner
# emits either at the suite level or inside per-scenario ``details``.
# A gate is "hard" — failing it exits non-zero.
HARD_GATES: Dict[str, Tuple[str, float, str]] = {
    # metric_name: (operator, threshold, human description)
    "cache_break_events": ("==", 0.0, "prompt-cache break events must be zero"),
    "verify_rate": (">=", 0.90, "subagent verification rate"),
    "recall_at_3": (">=", 0.85, "memory recall@3"),
    "plan_score": (">=", 0.80, "orchestration plan quality"),
}

# Soft gates: pass-rate regression tolerance vs baseline (fractional drop).
_SOFT_REGRESSION_TOLERANCE = 0.05  # 5 percentage points

log = logging.getLogger("run_evals")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
    )


def _suite_yaml_path(suite: str) -> Path:
    return _SUITES_DIR / f"{suite}.yaml"


def _baseline_path(suite: str) -> Path:
    return _BASELINES_DIR / f"{suite}.json"


def _suites_for_tier(tier: int, override: Optional[List[str]]) -> List[str]:
    """Resolve the list of suites to run for a given tier / override."""
    if override:
        return override
    if tier == 1:
        return list(_TIER1_SUITES)
    if tier == 2:
        return list(_TIER2_SUITES)
    if tier == 3:
        # comprehensive: deterministic suites first, then live suites
        return list(_TIER1_SUITES) + list(_TIER2_SUITES)
    raise ValueError(f"Unknown tier: {tier}")


def _is_deterministic_suite(suite: str) -> bool:
    """Tier 1 suites are run deterministic-only; Tier 2 suites are live."""
    return suite in _TIER1_SUITES


# ---------------------------------------------------------------------------
# Runner invocation
# ---------------------------------------------------------------------------
def run_single_suite(
    suite: str,
    tier: int,
    provider: str,
    model: str,
    output_path: Path,
) -> Dict[str, Any]:
    """Invoke ``evals/runners/run_suite.py`` for one suite and return its report.

    The runner writes its own per-suite JSON to ``output_path``; we read it
    back so the aggregator can merge everything into ``latest.json``.
    """
    suite_yaml = _suite_yaml_path(suite)
    if not suite_yaml.exists():
        log.error("Suite YAML not found: %s (skipping)", suite_yaml)
        return {
            "suite": suite,
            "error": f"suite YAML not found: {suite_yaml}",
            "total": 0,
            "passed": 0,
            "failed": 0,
            "errored": 0,
            "pass_rate": 0.0,
            "scenarios": [],
        }

    cmd: List[str] = [
        sys.executable,
        str(_RUNNER),
        "--suite", suite,
        "--provider", provider,
        "--model", model,
        "--output", str(output_path),
        "--quiet",
    ]
    if _is_deterministic_suite(suite):
        cmd.append("--deterministic-only")

    baseline = _baseline_path(suite)
    if baseline.exists():
        cmd.extend(["--baseline", str(baseline)])

    log.info("Running suite '%s' (tier %d): %s", suite, tier, " ".join(cmd))
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(_WORKTREE),
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min hard cap per suite
        )
    except subprocess.TimeoutExpired:
        log.error("Suite '%s' timed out after 1800s", suite)
        return {
            "suite": suite,
            "error": "timeout after 1800s",
            "total": 0,
            "passed": 0,
            "failed": 0,
            "errored": 0,
            "pass_rate": 0.0,
            "scenarios": [],
            "duration_s": 1800,
        }
    except FileNotFoundError:
        log.error("Runner not found at %s — is the repo layout correct?", _RUNNER)
        return {
            "suite": suite,
            "error": f"runner not found: {_RUNNER}",
            "total": 0,
            "passed": 0,
            "failed": 0,
            "errored": 0,
            "pass_rate": 0.0,
            "scenarios": [],
        }

    elapsed = round(time.time() - t0, 2)
    if proc.returncode != 0:
        log.warning(
            "Suite '%s' runner exited %d — stderr: %s",
            suite,
            proc.returncode,
            (proc.stderr or "").strip()[:500],
        )

    # The runner writes the report JSON to output_path regardless of exit
    # code (it exits non-zero on low pass-rate or baseline regression). Read
    # it back so we can aggregate.
    report: Dict[str, Any]
    if output_path.exists():
        try:
            report = json.loads(output_path.read_text(encoding="utf-8"))
        except Exception as e:
            log.error("Could not parse report for '%s': %s", suite, e)
            report = {
                "suite": suite,
                "error": f"unparseable report: {e}",
                "total": 0,
                "passed": 0,
                "failed": 0,
                "errored": 0,
                "pass_rate": 0.0,
                "scenarios": [],
            }
    else:
        log.error("Runner produced no report file at %s", output_path)
        report = {
            "suite": suite,
            "error": "no report file produced",
            "total": 0,
            "passed": 0,
            "failed": 0,
            "errored": 0,
            "pass_rate": 0.0,
            "scenarios": [],
        }

    report.setdefault("duration_s", elapsed)
    report.setdefault("runner_exit_code", proc.returncode)
    if proc.stderr:
        report.setdefault("runner_stderr", proc.stderr.strip()[-2000:])
    return report


# ---------------------------------------------------------------------------
# Gate evaluation
# ---------------------------------------------------------------------------
def _extract_metric(report: Dict[str, Any], metric: str) -> Optional[float]:
    """Pull a gate metric out of a suite report.

    Metrics can live at suite level (e.g. ``verify_rate``) or inside the
    ``details`` of individual scenarios (e.g. ``cache_break_events``,
    ``plan_score``, ``recall_at_3``). We check suite-level first, then
    aggregate from scenarios.
    """
    # 1. Suite-level field
    if metric in report and isinstance(report[metric], (int, float)):
        return float(report[metric])

    # 2. Scenario-level details — take the max (worst case) for ==/>= gates
    values: List[float] = []
    for s in report.get("scenarios", []):
        details = s.get("details", {}) or {}
        if isinstance(details, dict) and metric in details:
            try:
                values.append(float(details[metric]))
            except (TypeError, ValueError):
                pass
    if values:
        # For "==0" gates (cache_break_events) the worst case is the max;
        # for ">=" gates the worst case is the min. We return max here and
        # let the operator decide; callers that need min can recompute.
        return max(values)

    return None


def _gate_pass(metric: str, value: float) -> bool:
    op, threshold, _desc = HARD_GATES[metric]
    if op == "==":
        return value == threshold
    if op == ">=":
        return value >= threshold
    return False  # unknown operator → fail safe


def evaluate_hard_gates(reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Evaluate hard gates across all suite reports.

    Returns a list of gate-result dicts. Any ``passed == False`` entry means
    the CI run must exit non-zero.
    """
    results: List[Dict[str, Any]] = []
    # Map metrics to the suite(s) that should produce them.
    metric_to_suites: Dict[str, List[str]] = {
        "cache_break_events": ["cost_cache"],
        "verify_rate": ["subagent_verify"],
        "recall_at_3": ["memory_recall"],
        "plan_score": ["orchestration"],
    }

    for metric, (op, threshold, desc) in HARD_GATES.items():
        relevant = [r for r in reports if r.get("suite") in metric_to_suites.get(metric, [])]
        if not relevant:
            # No suite produced this metric — treat as not-evaluated (skip).
            results.append({
                "metric": metric,
                "status": "not_evaluated",
                "description": desc,
                "value": None,
                "threshold": threshold,
                "operator": op,
                "passed": None,
            })
            continue

        # Worst-case value across relevant suites
        vals = []
        for r in relevant:
            v = _extract_metric(r, metric)
            if v is not None:
                vals.append(v)

        if not vals:
            results.append({
                "metric": metric,
                "status": "missing",
                "description": desc,
                "value": None,
                "threshold": threshold,
                "operator": op,
                "passed": False,  # missing a hard-gate metric is a failure
                "reason": f"suite(s) {metric_to_suites[metric]} ran but did not emit '{metric}'",
            })
            continue

        worst = max(vals) if op == "==" else min(vals)
        passed = _gate_pass(metric, worst)
        results.append({
            "metric": metric,
            "status": "pass" if passed else "fail",
            "description": desc,
            "value": worst,
            "threshold": threshold,
            "operator": op,
            "passed": passed,
        })
    return results


def compare_baselines(reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Soft-gate: compare each suite's pass_rate to its baseline file.

    Returns a list of soft-gate result dicts. These never cause a non-zero
    exit; they only produce warnings.
    """
    soft: List[Dict[str, Any]] = []
    for r in reports:
        suite = r.get("suite", "unknown")
        baseline_path = _baseline_path(suite)
        if not baseline_path.exists():
            soft.append({
                "suite": suite,
                "status": "no_baseline",
                "message": f"no baseline at {baseline_path}",
            })
            continue
        try:
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        except Exception as e:
            soft.append({
                "suite": suite,
                "status": "baseline_unreadable",
                "message": str(e),
            })
            continue

        old_rate = float(baseline.get("pass_rate", 0))
        new_rate = float(r.get("pass_rate", 0))
        delta = new_rate - old_rate
        status = "regression" if delta < -_SOFT_REGRESSION_TOLERANCE else (
            "improvement" if delta > _SOFT_REGRESSION_TOLERANCE else "stable"
        )
        soft.append({
            "suite": suite,
            "status": status,
            "baseline_pass_rate": round(old_rate, 4),
            "current_pass_rate": round(new_rate, 4),
            "delta": round(delta, 4),
        })
    return soft


# ---------------------------------------------------------------------------
# Aggregation + output
# ---------------------------------------------------------------------------
def build_aggregate_report(
    tier: int,
    suites: List[str],
    suite_reports: List[Dict[str, Any]],
    hard_gates: List[Dict[str, Any]],
    soft_gates: List[Dict[str, Any]],
    provider: str,
    model: str,
) -> Dict[str, Any]:
    total = sum(r.get("total", 0) for r in suite_reports)
    passed = sum(r.get("passed", 0) for r in suite_reports)
    failed = sum(r.get("failed", 0) for r in suite_reports)
    errored = sum(r.get("errored", 0) for r in suite_reports)
    overall_rate = (passed / total) if total else 0.0

    hard_failures = [g for g in hard_gates if g.get("passed") is False]

    return {
        "schema": "hermes-eval-report/v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tier": tier,
        "suites_requested": suites,
        "provider": provider,
        "model": model,
        "summary": {
            "total_scenarios": total,
            "passed": passed,
            "failed": failed,
            "errored": errored,
            "overall_pass_rate": round(overall_rate, 4),
        },
        "hard_gates": hard_gates,
        "hard_gates_passed": len(hard_failures) == 0,
        "soft_gates": soft_gates,
        "suites": suite_reports,
    }


def write_report(report: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Aggregate report written to %s", path)


# ---------------------------------------------------------------------------
# Human-readable summary
# ---------------------------------------------------------------------------
def print_summary(report: Dict[str, Any]) -> None:
    out: List[str] = []
    bar = "=" * 64
    out.append("")
    out.append(bar)
    out.append("Hermes Eval CI Runner — Summary")
    out.append(bar)
    out.append(f"  Tier:     {report['tier']}")
    out.append(f"  Provider: {report.get('provider', 'n/a')}")
    out.append(f"  Model:    {report.get('model', 'n/a')}")
    out.append(f"  Time:     {report['timestamp']}")
    out.append("")

    s = report["summary"]
    out.append(f"  Scenarios:  {s['total_scenarios']}")
    out.append(f"  Passed:     {s['passed']}")
    out.append(f"  Failed:     {s['failed']}")
    out.append(f"  Errored:    {s['errored']}")
    out.append(f"  Pass rate:  {s['overall_pass_rate']:.1%}")
    out.append("")

    # Per-suite breakdown
    out.append("  Suites:")
    for sr in report.get("suites", []):
        suite = sr.get("suite", "?")
        if sr.get("error") and not sr.get("scenarios"):
            out.append(f"    ✗ {suite}: ERROR — {sr['error']}")
            continue
        rate = sr.get("pass_rate", 0.0)
        status = "✅" if rate >= 0.5 else "❌"
        out.append(
            f"    {status} {suite}: {sr.get('passed', 0)}/{sr.get('total', 0)} "
            f"({rate:.0%})  errors={sr.get('errored', 0)}  "
            f"{sr.get('duration_s', 0):.1f}s"
        )
    out.append("")

    # Hard gates
    out.append("  Hard gates:")
    for g in report.get("hard_gates", []):
        metric = g["metric"]
        if g["passed"] is True:
            icon = "✅"
            detail = f"value={g['value']}"
        elif g["passed"] is False:
            icon = "❌"
            detail = f"value={g['value']} {g['operator']} {g['threshold']} FAILED"
            if g.get("reason"):
                detail += f" ({g['reason']})"
        else:
            icon = "⚪"
            detail = g.get("status", "not evaluated")
        out.append(f"    {icon} {metric}: {detail}")
    out.append("")

    # Soft gates
    soft = report.get("soft_gates", [])
    if soft:
        out.append("  Soft gates (warnings only):")
        for sg in soft:
            suite = sg.get("suite", "?")
            status = sg.get("status", "?")
            if status == "regression":
                out.append(
                    f"    ⚠️  {suite}: REGRESSION "
                    f"Δ={sg.get('delta', 0):+.2%} "
                    f"({sg.get('baseline_pass_rate', 0):.1%} → {sg.get('current_pass_rate', 0):.1%})"
                )
            elif status == "improvement":
                out.append(
                    f"    📈 {suite}: improvement "
                    f"Δ={sg.get('delta', 0):+.2%}"
                )
            elif status == "stable":
                out.append(f"    ✓  {suite}: stable (Δ={sg.get('delta', 0):+.2%})")
            else:
                out.append(f"    •  {suite}: {status} — {sg.get('message', '')}")
        out.append("")

    out.append(bar)
    if report["hard_gates_passed"]:
        out.append("  RESULT: PASS — all hard gates satisfied")
    else:
        out.append("  RESULT: FAIL — one or more hard gates failed")
    out.append(bar)
    out.append("")

    text = "\n".join(out)
    print(text)
    # Also emit a concise GitHub Actions annotation for hard-gate failures
    if not report["hard_gates_passed"]:
        for g in report.get("hard_gates", []):
            if g.get("passed") is False:
                print(
                    f"::error::Hard gate failed: {g['metric']} "
                    f"({g['operator']}{g['threshold']}) got {g['value']}",
                    file=sys.stderr,
                )


# ---------------------------------------------------------------------------
# Secret check
# ---------------------------------------------------------------------------
def _check_secrets(tier: int) -> Optional[str]:
    """Return an error message if a required secret is missing, else None."""
    if tier == 1:
        return None  # deterministic, no secrets
    # Tier 2 and 3 need a provider key. We check the common ones.
    provider_keys = ["OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"]
    if not any(os.environ.get(k) for k in provider_keys):
        return (
            f"Tier {tier} requires a live model API key "
            f"(one of: {', '.join(provider_keys)}). Set OPENROUTER_API_KEY in CI secrets."
        )
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        prog="run_evals.py",
        description="CI eval runner — orchestrate tiered Hermes eval suites.",
    )
    parser.add_argument(
        "--tier",
        type=int,
        choices=[1, 2, 3],
        required=True,
        help="Eval tier: 1=deterministic (no API keys), 2=live model, 3=comprehensive.",
    )
    parser.add_argument(
        "--suite",
        action="append",
        dest="suites",
        metavar="SUITE",
        help=(
            "Run a specific suite instead of the full tier. Can be passed "
            "multiple times (e.g. --suite orchestration --suite cost_cache)."
        ),
    )
    parser.add_argument(
        "--provider",
        default="openrouter",
        help="LLM provider for Tier 2+ suites (default: openrouter).",
    )
    parser.add_argument(
        "--model",
        default="anthropic/claude-haiku-4.5",
        help="Model name for Tier 2+ suites.",
    )
    parser.add_argument(
        "--output",
        default=str(_LATEST_REPORT),
        help=f"Aggregate JSON report path (default: {_LATEST_REPORT}).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    _setup_logging(verbose=args.verbose)

    # 1. Resolve suites
    try:
        suites = _suites_for_tier(args.tier, args.suites)
    except ValueError as e:
        log.error("%s", e)
        return 2

    if not suites:
        log.error("No suites resolved for tier %d", args.tier)
        return 1
    log.info("Tier %d → suites: %s", args.tier, ", ".join(suites))

    # 2. Secret check (Tier 2/3)
    secret_err = _check_secrets(args.tier)
    if secret_err:
        log.error("%s", secret_err)
        print(f"::error::{secret_err}", file=sys.stderr)
        return 1

    # 3. Run each suite
    suite_reports: List[Dict[str, Any]] = []
    for suite in suites:
        # Per-suite report path; the aggregate report overwrites latest.json
        per_suite_output = _REPORTS_DIR / f"{suite}.json"
        report = run_single_suite(
            suite=suite,
            tier=args.tier,
            provider=args.provider,
            model=args.model,
            output_path=per_suite_output,
        )
        suite_reports.append(report)

    # 4. Evaluate gates
    hard_gates = evaluate_hard_gates(suite_reports)
    soft_gates = compare_baselines(suite_reports)

    # 5. Build + write aggregate report
    aggregate = build_aggregate_report(
        tier=args.tier,
        suites=suites,
        suite_reports=suite_reports,
        hard_gates=hard_gates,
        soft_gates=soft_gates,
        provider=args.provider,
        model=args.model,
    )
    write_report(aggregate, Path(args.output))

    # 6. Human-readable summary
    print_summary(aggregate)

    # 7. Exit code: non-zero if any hard gate failed or any suite errored
    hard_fail = not aggregate["hard_gates_passed"]
    any_error = any(r.get("error") and not r.get("scenarios") for r in suite_reports)
    if hard_fail:
        log.error("Exiting non-zero: hard gate failure")
        return 1
    if any_error:
        log.error("Exiting non-zero: one or more suites errored")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())