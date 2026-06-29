"""Orchestration rubric for Hermes Agent eval suite.

Grades agent decomposition quality, delegation choices, and concurrency discipline.
"""


def grade(scenario: dict, result: dict) -> dict:
    """Score an orchestration scenario."""
    sid = scenario.get("id", "?")
    final = result.get("final_response", "")
    messages = result.get("messages", [])
    error = result.get("error")
    api_calls = result.get("api_calls", 0)

    if error:
        return {"pass": False, "score": 0.0, "details": {"error": error}}

    # Count delegate_task calls
    delegate_calls = 0
    tool_names = []
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                name = tc.get("function", {}).get("name", "")
                tool_names.append(name)
                if name == "delegate_task":
                    delegate_calls += 1

    # Check for tool errors
    has_error = False
    for msg in messages:
        if msg.get("role") == "tool":
            content = str(msg.get("content", ""))
            lower = content.lower()
            if "traceback" in lower:
                has_error = True
                break

    # Score against pass conditions
    conditions = scenario.get("pass_conditions", [])
    checks_passed = 0
    details = {
        "delegate_calls": delegate_calls,
        "tools_used": list(set(tool_names)),
        "api_calls": api_calls,
        "has_error": has_error,
    }

    for cond in conditions:
        ctype = cond.get("type", "")
        if ctype == "delegate_call_count":
            min_val = cond.get("min", 0)
            max_val = cond.get("max", 999)
            if min_val <= delegate_calls <= max_val:
                checks_passed += 1
            details["delegate_range"] = f"{min_val}-{max_val}"
        elif ctype == "no_tool_error":
            if not has_error:
                checks_passed += 1
        elif ctype == "response_contains":
            val = cond.get("value", "")
            found = val.lower() in final.lower()
            details[f"contains_{val[:30]}"] = found
            if found:
                checks_passed += 1
        else:
            checks_passed += 1

    total = len(conditions) if conditions else 1
    score = checks_passed / total

    # O3_no_spawn_trivial: penalize if delegate_task used for trivial
    if sid == "O3_no_spawn_trivial" and delegate_calls > 0:
        score = max(0.0, score - 0.5)

    return {
        "pass": score >= 0.5 and not has_error,
        "score": round(score, 3),
        "details": details,
    }
