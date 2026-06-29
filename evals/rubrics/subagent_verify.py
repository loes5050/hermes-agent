"""Subagent verification rubric for Hermes Agent eval suite.

Checks that the parent verifies subagent outputs instead of blindly trusting them.
"""


def grade(scenario: dict, result: dict) -> dict:
    """Score a subagent verification scenario."""
    sid = scenario.get("id", "?")
    final = result.get("final_response", "")
    messages = result.get("messages", [])
    error = result.get("error")
    api_calls = result.get("api_calls", 0)

    if error:
        return {"pass": False, "score": 0.0, "details": {"error": error}}

    # Count delegate_task calls and verification steps
    delegate_indices = []
    verify_tools_after = []
    tool_sequence = []

    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                name = tc.get("function", {}).get("name", "")
                tool_sequence.append((i, name))
                if name == "delegate_task":
                    delegate_indices.append(i)

    # Check if there's a verification tool call after delegate_task
    verify_tools = {"read_file", "search_files", "terminal"}
    for di in delegate_indices:
        for msg_i, tname in tool_sequence:
            if msg_i > di and tname in verify_tools:
                verify_tools_after.append(tname)
                break

    verify_rate = len(verify_tools_after) / len(delegate_indices) if delegate_indices else 1.0

    # Check for tool errors
    has_error = False
    for msg in messages:
        if msg.get("role") == "tool":
            content = str(msg.get("content", ""))
            lower = content.lower()
            if "traceback" in lower:
                has_error = True
                break

    conditions = scenario.get("pass_conditions", [])
    checks_passed = 0
    details = {
        "delegate_calls": len(delegate_indices),
        "verify_tools_after": verify_tools_after,
        "verify_rate": round(verify_rate, 3),
        "api_calls": api_calls,
        "has_error": has_error,
    }

    for cond in conditions:
        ctype = cond.get("type", "")
        if ctype == "no_tool_error":
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

    # Bonus for verification behavior
    if verify_rate > 0:
        score = min(1.0, score + 0.2)

    return {
        "pass": score >= 0.5 and not has_error,
        "score": round(score, 3),
        "details": details,
    }
