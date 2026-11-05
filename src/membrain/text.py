"""Text cleaning shared by capture and evidence validation."""
from __future__ import annotations

import re
from datetime import date, timedelta

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


# --- Time-aware recall ---------------------------------------------------
# "我昨晚说的" only resolves against a date window. These are the coarse phrases
# people actually type, matched deterministically: a recall-time LLM call would
# put a model in the hot path, and an LLM guess about today's date is worse than
# reading the clock. Order matters — the specific phrase must win over the general
# one, so "昨晚" is tested before "昨天".

_TIME_WINDOW: list[tuple[str, int, int]] = [
    # (pattern, days_back_start, days_back_end); window = [today-start, today-end]
    (r"昨晚|昨天晚上|昨夜|yesterday evening|last night", 1, 1),
    (r"前天晚上|前晚|the night before last", 2, 2),
    (r"昨天|昨日|yesterday", 1, 1),
    (r"前天|day before yesterday", 2, 2),
    (r"大前天", 3, 3),
    (r"今天|今日|刚刚|刚才|today|just now", 0, 0),
    (r"上周|上一周|上个星期|last week", 7, 13),
    (r"本周|这周|这个星期|this week", 0, 6),
    (r"上个月|上月|last month", 30, 60),
    (r"这个月|本月|this month", 0, 29),
    (r"最近几天|近几天|这几天|in the last few days|past few days", 0, 7),
    (r"最近|recently|lately", 0, 14),
]

_N_DAYS_AGO = re.compile(r"(\d+)\s*(?:天|日)前|(\d+)\s*days?\s*ago", re.IGNORECASE)


def parse_time_window(query: str, today: date | None = None) -> tuple[date, date] | None:
    """Map a natural time phrase to an inclusive window of operator-local dates.

    Always returns (start, end) with start <= end, so callers can compare
    directly instead of normalising at every use site. Returns None when the
    query carries no time expression, so the caller keeps unfiltered recall
    instead of silently narrowing it.
    """
    low = (query or "").casefold()
    if not low.strip():
        return None
    today = today or date.today()
    for pattern, back_start, back_end in _TIME_WINDOW:
        if re.search(pattern, low, re.IGNORECASE):
            near, far = sorted((back_start, back_end))
            return (today - timedelta(days=far), today - timedelta(days=near))
    m = _N_DAYS_AGO.search(low)
    if m:
        n = int(m.group(1) or m.group(2))
        if 0 <= n <= 3650:
            return (today - timedelta(days=n), today - timedelta(days=n))
    return None
