"""Text cleaning shared by capture and evidence validation."""
from __future__ import annotations

import re

_INJECTED_BLOCK = re.compile(
    r"<(memory|personal_memory|system-reminder|local-command-stdout|command-name|command-message|command-args|bash-input|bash-stdout|bash-stderr|task-notification|message_quote|user_info|project_context)\b[^>]*>.*?(?:</\1\s*>|\Z)",
    re.IGNORECASE | re.DOTALL,
)


_HEDGE = re.compile(
    r"也许|可能|大概|或许|说不定|不确定|应该是|估计|似乎|好像|万一|假如|如果|不知道会不会"
    r"|\b(?:maybe|perhaps|might|possibly|probably|not sure|i think|i guess|i suppose|if)\b",
    re.IGNORECASE,
)
_CLAUSE_BREAK = "，,。.；;！!？?\n"


def is_hedged(text: str) -> bool:
    """Whether the user themselves wrapped the statement in doubt."""
    return bool(_HEDGE.search(text or ""))


def widen_to_clause(episode_text: str, quote: str) -> str:
    """Return the clause around `quote` when it carries a hedge the quote dropped.

    The extractor may quote only a sub-span such as "参加一个比武大赛" and lose
    the "也许" before it. Confidence is judged on the quote, so a dropped hedge
    would otherwise look like a plain statement.
    """
    text = clean_turn(episode_text)
    start = text.find(quote)
    if start < 0 or is_hedged(quote):
        return quote
    end = start + len(quote)
    left = max((text.rfind(ch, 0, start) for ch in _CLAUSE_BREAK), default=-1) + 1
    rights = [i for i in (text.find(ch, end) for ch in _CLAUSE_BREAK) if i >= 0]
    right = min(rights) if rights else len(text)
    clause = text[left:right].strip()
    return clause if clause and is_hedged(clause) and len(clause) <= 400 else quote


def clean_turn(text: str) -> str:
    """Remove injected and command material before using text as evidence."""
    if not isinstance(text, str):
        return ""
    text = _INJECTED_BLOCK.sub("", text).strip()
    if re.match(r"^/[a-z][\w-]*(?:\s|$)", text, re.IGNORECASE):
        return ""
    return text
