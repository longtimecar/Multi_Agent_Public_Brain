"""Search modes aligned with OpenViking find/search thought (gap 4).

QUICK: single query, no intent analysis, no rerank (find-like).
THINKING: intent analysis + optional L0 rerank with 2x candidate pool (search-like).
CONTEXT: multi-type quota assembly + token trim (search mode=context thought).

Intent/rerank remain fail-open and still default OFF unless mode=thinking
or the caller explicitly sets use_intent/use_rerank. mode=quick forces both off
even if settings defaults are on.
"""

from __future__ import annotations

from typing import Any

MODE_QUICK = "quick"
MODE_THINKING = "thinking"
MODE_CONTEXT = "context"
VALID_MODES = frozenset({MODE_QUICK, MODE_THINKING, MODE_CONTEXT})


def normalize_mode(mode: str | None) -> str | None:
    if mode is None:
        return None
    raw = str(mode).strip().lower()
    if not raw:
        return None
    if raw in VALID_MODES:
        return raw
    raise ValueError(f"unknown search mode: {mode!r} (expected quick|thinking|context)")


def resolve_intent_rerank(
    mode: str | None,
    *,
    use_intent: bool | None,
    use_rerank: bool | None,
    intent_default: bool = False,
    rerank_default: bool = False,
) -> tuple[bool, bool, list[str]]:
    """Resolve effective intent/rerank flags for a recall call.

    Explicit True/False from the caller wins over mode/defaults.
    mode=quick forces False unless caller explicitly set True (we still honor
    explicit True for tests/debug, but notes record the conflict).
    mode=thinking turns both on when caller left them as None.
    """
    notes: list[str] = []
    normalized = normalize_mode(mode)
    if normalized:
        notes.append(f"search_mode:{normalized}")

    intent: bool
    rerank: bool

    if use_intent is None:
        if normalized == MODE_QUICK:
            intent = False
        elif normalized == MODE_THINKING:
            intent = True
        elif normalized == MODE_CONTEXT:
            # Context assembly uses find-like recall per OV; intent stays off unless default/explicit.
            intent = bool(intent_default)
        else:
            intent = bool(intent_default)
    else:
        intent = bool(use_intent)

    if use_rerank is None:
        if normalized == MODE_QUICK:
            rerank = False
        elif normalized == MODE_THINKING:
            rerank = True
        elif normalized == MODE_CONTEXT:
            rerank = bool(rerank_default)
        else:
            rerank = bool(rerank_default)
    else:
        rerank = bool(use_rerank)

    if normalized == MODE_QUICK and (intent or rerank) and (use_intent is True or use_rerank is True):
        notes.append("quick_mode:explicit_flags_override")

    return intent, rerank, notes


def mode_label(mode: str | None, fallback: str) -> str:
    normalized = normalize_mode(mode)
    if normalized:
        return f"{fallback}+{normalized}" if fallback and not fallback.endswith(normalized) else normalized
    return fallback
