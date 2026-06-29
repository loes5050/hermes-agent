"""Code task rubric for Hermes Agent eval suite.

Grades end-to-end coding scenarios: bug fixes, feature implementation, refactoring.
For live-model runs only (Tier 2).
"""

import re


def grade(scenario: dict, result: dict) -> dict:
    """Score a code task scenario.

    Checks: no tool errors, test output contains PASS/success indicators,
    and the final response indicates completion.
    """
    sid = scenario.get("id", "?")
    final = result.get("final_response", "")
    messages = result.get("messages", [])
    error = result.get("error")
    api_calls = result.get("api_calls", 0)

    if error:
        return {
            "pass": False,
            "score": 0.0,
            "details": {"error": error, "reason": "scenario errored"},
        }

    # Check terminal output for test pass indicators
    test_passed = False
    terminal_outputs = []
    for msg in messages:
        if msg.get("role") == "tool" and msg.get("name") == "terminal":
            content = str(msg.get("content", ""))
            terminal_outputs.append(content)
            if _has_test_pass(content):
                test_passed = True
                break

    # Check for tool errors
    has_error = False
    error_details = []
    for msg in messages:
        if msg.get("role") == "tool":
            content = str(msg.get("content", ""))
            if _is_real_error(content):
                has_error = True
                error_details.append(content[:200])

    conditions = scenario.get("pass_conditions", [])
    checks_passed = 0
    details = {
        "test_passed": test_passed,
        "has_tool_error": has_error,
        "api_calls": api_calls,
    }

    for cond in conditions:
        ctype = cond.get("type", "")
        if ctype == "no_tool_error":
            if not has_error:
                checks_passed += 1
        elif ctype == "response_contains":
            val = cond.get("value", "")
            found = val.lower() in final.lower() or any(val.lower() in t.lower() for t in terminal_outputs)
            details[f"contains_{val[:30]}"] = found
            if found:
                checks_passed += 1
        else:
            checks_passed += 1

    total = len(conditions) if conditions else 1
    if total == 0:
        total = 1

    # Bonus: test actually passed
    if test_passed and not has_error:
        checks_passed = max(checks_passed, total)

    score = min(checks_passed / total, 1.0)
    return {
        "pass": score >= 0.6 and not has_error,
        "score": round(score, 3),
        "details": details,
    }


def _has_test_pass(output: str) -> bool:
    """Check if terminal output indicates tests passed."""
    lower = output.lower()
    # Python unittest/pytest patterns
    if re.search(r"\bOK\b", output) and "FAIL" not in output:
        return True
    if "passed" in lower and "failed" not in lower:
        return True
    if re.search(r"\bPASS\b", output):
        return True
    if "all tests" in lower and "pass" in lower:
        return True
    # Generic success patterns
    if lower.strip().endswith("pass") or lower.strip().endswith("pass."):
        return True
    return False


def _is_real_error(output: str) -> bool:
    """Check if tool output contains a real error (not benign mentions)."""
    lower = output.lower()
    if "traceback (most recent call last)" in lower:
        return True
    if "error:" in lower and "no error" not in lower and "0 error" not in lower:
        return True
    if "assertionerror" in lower or "assertion error" in lower:
        return True
    if "syntaxerror" in lower:
        return True
    if "importerror" in lower or "modulenotfounderror" in lower:
        return True
    return False
