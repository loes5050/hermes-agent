#!/usr/bin/env python3
"""CI entry point for Hermes Agent eval suites.

Invoked by GitHub Actions (.github/workflows/evals.yml) and locally.
Aggregates suite reports, compares against baselines, and enforces gates.

Usage:
    python scripts/ci/run_evals.py --tier 1 [--output evals/reports/tier1.json]
    python scripts/ci/run_evals.py --tier 2 [--output evals/reports/tier2.json]
    python scripts/ci/run_evals.py --tier 3 [--output evals/reports/nightly.json]
    python scripts/ci/run_evals.py --suite orchestration,cost_cache [--output ...]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

_HERE = Path(__file__).resolve().parent
_WORKTREE = _HERE.parent.parent
_EVALS_DIR = _WORKTREE / "evals"
_RUNNER = _EVALS_DIR / "runners" / "run_suite.py"
_REPORTS_DIR = _EVALS_DIR / "reports"
_BASELINES_DIR = _EVALS_DIR / "baselines"

# ---------------------------------------------------------------------------
# Gate thresholds (hard = exit non-zero on fail, soft = warn only)
# ---------------------------------------------------------------------------
HARD_GATES: Dict[str, Dict[str, float]] = {
    "orchestration": {"pass_rate": 0.50, "plan_score": 0.80},
    "cost_cache": {"pass_rate": 0.80, "cache_break_events": 0},
    "subagent_verify": {"pass_rate": 0.60, "verify_rate": 0.90},
    "memory_recall": {"pass_rate": 0.60, "recall_at_3": 0.85},
    "windows_reliability": {"pass_rate": 0.80},
}

SOFT_GATES: Dict[str, Dict[str, float]] = {
    "code_task": {"pass_rate": 0.70},
    "research_citation": {"pass_rate": 0.60, "unjustified_cite_rate": 0.05},
}

TIER1_SUITES = ["orchestration", "cost_cache", "subagent_verify", "memory_recall"]
TIER2_SUITES = ["code_task", "research_citation"]
TIER3_SUITES = TIER1_SUITES + TIER2_SUITES


def run_suite(suite_name: str, deterministic: bool = False, provider: str = "openrouter", model: str = "anthropic/claude-haiku-4.5") -> dict:
    """Run a single suite via the runner and return its report."""
    output_path = _REPORTS_DIR / f"{suite_name}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(_RUNNER),
        "--suite", suite_name,
        "--output", str(output_path),
        "--quiet",
    ]
    if deterministic:
        cmd.append("--deterministic-only")
    else:
        cmd.extend(["--provider", provider, "--model", model])

    # Check for baseline
    baseline_path = _BASELINES_DIR / f"{suite_name}_baseline.json"
    if baseline_path.exists():
        cmd.extend(["--baseline", str(baseline_path)])

    print(f"  Running: {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        print(f"  WARNING: Runner exited with code {result.returncode}", file=sys.stderr)
        if result.stderr:
            print(f"  stderr: {result.stderr[:500]}", file=sys.stderr)

    # Load the report
    if output_path.exists():
        try:
            return json.loads(output_path.read_text(encoding="utf-8"))
        except Exception as e:
            return {"suite": suite_name, "error": f"Failed to parse report: {e}", "total": 0, "passed": 0, "failed": 0}
    return {"suite": suite_name, "error": "No report generated", "total": 0, "passed": 0, "failed": 0}


def check_gates(reports: List[dict], tier: int) -> tuple[bool, List[str]]:
    """Check all reports against gate thresholds. Returns (all_passed, violations)."""
    violations: List[str] = []
    all_passed = True

    gates_to_check = {}
    if tier == 1:
        gates_to_check = HARD_GATES
    elif tier == 2:
        gates_to_check = {**HARD_GATES, **SOFT_GATES}
    else:
        gates_to_check = {**HARD_GATES, **SOFT_GATES}

    for report in reports:
        suite_name = report.get("suite", "")
        gates = gates_to_check.get(suite_name, {})

        for metric, threshold in gates.items():
            if metric == "pass_rate":
                actual = report.get("pass_rate", 0)
                if actual < threshold:
                    msg = f"HARD GATE FAIL: {suite_name}.{metric}={actual:.2%} < {threshold:.2%}"
                    violations.append(msg)
                    if suite_name in HARD_GATES:
                        all_passed = False
            elif metric == "cache_break_events":
                # Check from individual scenarios
                for s in report.get("scenarios", []):
                    breaks = s.get("details", {}).get("cache_breaks", 0)
                    if breaks > threshold:
                        msg = f"HARD GATE FAIL: {suite_name}.{s['id']}.cache_breaks={breaks} > {threshold}"
                        violations.append(msg)
                        all_passed = False

    return all_passed, violations


def print_report(reports: List[dict], violations: List[str]) -> None:
    """Print a human-readable aggregated report."""
    print(f"\n{'='*70}")
    print(f"  Hermes Agent Eval Report — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*70}")

    total_passed = sum(r.get("passed", 0) for r in reports)
    total_scenarios = sum(r.get("total", 0) for r in reports)
    total_failed = sum(r.get("failed", 0) for r in reports)
    total_errored = sum(r.get("errored", 0) for r in reports)

    print(f"  Suites: {len(reports)}  |  Scenarios: {total_scenarios}  |  "
          f"Passed: {total_passed}  |  Failed: {total_failed}  |  Errors: {total_errored}")

    if total_scenarios > 0:
        rate = total_passed / total_scenarios
        print(f"  Overall Pass Rate: {rate:.1%}")
    print(f"{'='*70}")

    for report in reports:
        suite = report.get("suite", "?")
        pr = report.get("pass_rate", 0)
        status = "✅" if pr >= 0.8 else ("⚠️" if pr >= 0.5 else "❌")
        print(f"  {status} {suite}: {report.get('passed',0)}/{report.get('total',0)} ({pr:.1%})")
        for s in report.get("scenarios", []):
            s_status = "✅" if s["pass"] else "❌"
            print(f"      {s_status} {s['id']}: score={s.get('score',0):.2f}")

    if violations:
        print(f"\n{'='*70}")
        print(f"  GATE VIOLATIONS ({len(violations)})")
        print(f"{'='*70}")
        for v in violations:
            print(f"  ❌ {v}")
    else:
        print(f"\n  ✅ All gates passed.")

    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Hermes Agent Eval CI Runner")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3], help="CI tier (1=fast/deterministic, 2=live, 3=nightly)")
    parser.add_argument("--suite", help="Comma-separated suite names to run (overrides --tier)")
    parser.add_argument("--output", help="Output JSON path for aggregated report")
    parser.add_argument("--provider", default="openrouter", help="LLM provider for live suites")
    parser.add_argument("--model", default="anthropic/claude-haiku-4.5", help="Model for live suites")
    parser.add_argument("--no-gates", action="store_true", help="Skip gate enforcement (always exit 0)")
    args = parser.parse_args()

    # Determine suites to run
    if args.suite:
        suites = [s.strip() for s in args.suite.split(",")]
    elif args.tier == 1:
        suites = TIER1_SUITES
    elif args.tier == 2:
        suites = TIER2_SUITES
    elif args.tier == 3:
        suites = TIER3_SUITES
    else:
        print("ERROR: Must specify --tier or --suite", file=sys.stderr)
        sys.exit(1)

    # Determine if deterministic
    deterministic = args.tier == 1 if args.tier else False

    print(f"=== Hermes Eval CI — Tier {args.tier or 'custom'} ({len(suites)} suites) ===", file=sys.stderr)

    reports = []
    for suite_name in suites:
        print(f"\n--- Suite: {suite_name} ---", file=sys.stderr)
        report = run_suite(suite_name, deterministic=deterministic, provider=args.provider, model=args.model)
        reports.append(report)

    # Check gates
    all_passed, violations = check_gates(reports, args.tier or 1)
    print_report(reports, violations)

    # Write aggregated report
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        aggregated = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tier": args.tier,
            "suites_run": suites,
            "total_scenarios": sum(r.get("total", 0) for r in reports),
            "total_passed": sum(r.get("passed", 0) for r in reports),
            "total_failed": sum(r.get("failed", 0) for r in reports),
            "violations": violations,
            "all_gates_passed": all_passed,
            "reports": reports,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(aggregated, f, indent=2, ensure_ascii=False)
        print(f"\nAggregated report: {output_path}", file=sys.stderr)

    if not args.no_gates and not all_passed:
        print("\n❌ GATE FAILURES DETECTED — exiting non-zero", file=sys.stderr)
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
