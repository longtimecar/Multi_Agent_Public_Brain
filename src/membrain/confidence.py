"""Confidence is computed here, in code, and never accepted from a model.

Models asked for a confidence number cluster around 0.8-0.9 whether or not
they are right, so a self-reported score is worse than no score. Everything
below derives from evidence structure, which the model cannot inflate without
producing more verbatim quotes from more independent episodes.
"""

from __future__ import annotations

from datetime import date

from .models import Fact, FactType
from .text import is_hedged

# How much a single well-supported claim of each type is worth before
# corroboration. Constraints are high because they are usually explicit and
# unambiguous; preferences start lower because they are the easiest thing to
# over-read from one remark.
BASE: dict[FactType, float] = {
    FactType.CONSTRAINT: 0.90,
    FactType.FACT: 0.78,
    FactType.PROJECT: 0.72,
    FactType.GOAL: 0.70,
    FactType.PREFERENCE: 0.62,
    FactType.ENTITY: 0.74,
    FactType.EVENT: 0.70,
    FactType.CASE: 0.68,
    FactType.TOOL: 0.72,
    FactType.BELIEF: 0.60,
}

SINGLE_EPISODE_CAP = 0.40
MAX_CONFIDENCE = 0.95
PROMOTE_AT = 0.60

# A plain first-person statement such as "I'm allergic to peanuts" is not a
# guess, so one occurrence is enough to state it. A statement that carries a
# hedge ("maybe", "也许") stays under the single-episode cap until it recurs.
EXPLICIT_CEILING = 0.70


def compute_confidence(fact: Fact, today: date | None = None) -> float:
    """Return a confidence in [0, 1] derived only from evidence structure."""
    today = today or date.today()
    episodes = fact.evidence
    n = len(episodes)
    if n == 0:
        return 0.0

    base = BASE.get(fact.type, 0.65)

    # Diminishing returns on repeated evidence: 1 -> 0.00, 2 -> 0.55, 3 -> 0.80, 4 -> 0.91
    support = 1.0 - 0.45 ** (n - 1)

    # Evidence gathered in one sitting is one observation, not several. This is
    # the main defence against a single long conversation manufacturing certainty.
    dates = sorted({e.date for e in episodes})
    spread_days = (dates[-1] - dates[0]).days
    independence = min(1.0, spread_days / 14.0)
    if spread_days >= 14:
        spread_bonus = 0.08
    elif spread_days >= 7:
        spread_bonus = 0.04
    else:
        spread_bonus = 0.0

    stated_ratio = sum(1 for e in episodes if e.origin != "inferred") / n
    # Hedge words are read in the owner's own wording. An imported claim is a distilled rule, and a rule
    # about uncertainty ("when unsure, search first") is not itself doubtful. Importing already skips
    # tentative notes.
    explicit = any((e.origin == "stated" and not is_hedged(e.quote)) or e.origin == "imported" for e in episodes)

    # Corroboration only counts when it is independent; a burst of same-day
    # evidence therefore cannot lift a claim above its single-observation value.
    corroboration = support * (0.35 + 0.65 * independence)

    conf = base * (0.45 + 0.55 * corroboration) * (0.65 + 0.35 * stated_ratio)
    conf += spread_bonus

    if n == 1:
        conf = min(conf, SINGLE_EPISODE_CAP)
    if explicit:
        # Extra evidence must never lower what one plain statement already earned.
        conf = max(conf, min(base, EXPLICIT_CEILING))

    if fact.status.value == "contested":
        conf = min(conf, 0.45)

    return round(max(0.0, min(MAX_CONFIDENCE, conf)), 3)


def is_promoted(fact: Fact, today: date | None = None) -> bool:
    """Whether a claim may be injected as fact rather than as a hedge."""
    return compute_confidence(fact, today) >= PROMOTE_AT


def render_hedged(fact: Fact) -> str:
    """Low-confidence claims are phrased as questions, not assertions.

    An unchallenged hedge produces no new evidence, which is exactly right:
    silence never counts as confirmation.
    """
    if is_promoted(fact):
        return fact.claim
    return f"Unconfirmed note based on previous user-provided evidence: {fact.claim} (not independently corroborated)."
