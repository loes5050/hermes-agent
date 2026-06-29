"""Research citation rubric for Hermes Agent eval suite.

Grades web-grounded research scenarios: source citation, disagreement surfacing,
uncertainty flagging, and multi-hop chaining.
"""

import re


def grade(scenario: dict, result: dict) -> dict:
    """Score a research citation scenario.

    Checks: presence of URLs, source citations, honesty about uncertainty,
    and multi-hop completeness.
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

    # Count citations (URLs in response)
    urls = re.findall(r"https?://[^\s\)\]>]+", final)
    citation_count = len(set(urls))  # Unique URLs

    # Check for fabricated-looking citations
    has_credible_urls = any(
        domain in url.lower()
        for url in urls
        for domain in [
            "python.org", "wikipedia.org", "github.com", "arxiv.org",
            "docs.", ".edu", ".gov", "medium.com", "dev.to",
            "stackoverflow.com", "npmjs.com", "pypi.org",
            "nobelprize.org", "britannica.com",
        ]
    )

    # Check for tool errors
    has_error = False
    for msg in messages:
        if msg.get("role") == "tool":
            content = str(msg.get("content", ""))
            lower = content.lower()
            if "error" in lower and "no error" not in lower:
                has_error = True
                break

    # Check web_search usage
    search_count = 0
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                name = tc.get("function", {}).get("name", "")
                if name in ("web_search", "web_extract"):
                    search_count += 1

    conditions = scenario.get("pass_conditions", [])
    checks_passed = 0
    details = {
        "citation_count": citation_count,
        "urls": urls[:5],
        "has_credible_urls": has_credible_urls,
        "search_count": search_count,
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
            found = val.lower() in final.lower()
            details[f"contains_{val[:30]}"] = found
            if found:
                checks_passed += 1
        else:
            checks_passed += 1

    # Bonus: citations present
    if citation_count > 0 and not has_error:
        checks_passed += 1

    total = len(conditions) + 1 if conditions else 2  # +1 for bonus
    score = min(checks_passed / total, 1.0)

    # Special: R3_no_source should NOT fabricate citations when answer unknown
    if sid == "R3_no_source":
        if citation_count > 0 and not has_credible_urls:
            score = max(0.0, score - 0.5)

    return {
        "pass": score >= 0.5 and not has_error,
        "score": round(score, 3),
        "details": details,
    }
