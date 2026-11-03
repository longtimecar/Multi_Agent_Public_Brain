"""Token estimation for injection budgeting.

Deliberately conservative and dependency-free. A memory block must fit a hard
cap, so over-estimating is the safe direction. CJK codepoints are counted as
roughly one token each, which is much closer to the truth than len()/4 and is
why naive character heuristics blow the budget on Chinese text.
"""

from __future__ import annotations

import unicodedata


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    cjk = 0
    other = 0
    for ch in text:
        if unicodedata.east_asian_width(ch) in ("W", "F"):
            cjk += 1
        else:
            other += 1
    return cjk + (other + 3) // 4


def fits(text: str, budget: int) -> bool:
    return estimate_tokens(text) <= budget


def truncate_to_budget(text: str, budget: int) -> str:
    """Trim at a line boundary rather than mid-token.

    Returns empty string when not even one line fits, so callers can drop the
    whole item instead of injecting a fragment.
    """
    if fits(text, budget):
        return text
    lines = text.splitlines()
    kept: list[str] = []
    used = 0
    for line in lines:
        cost = estimate_tokens(line) + 1
        if used + cost > budget:
            break
        kept.append(line)
        used += cost
    return "\n".join(kept)
