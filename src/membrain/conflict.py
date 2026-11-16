"""Contradiction detection and deterministic 4-tier auto-resolution.

Detection finds pairs among live active facts sharing a subject that look like
contradictions. `constraint` facts are NEVER auto-resolved and always stay
`contested` for the owner. Other pairs follow a strict, reversible 4-tier rule:
  1. Owner's unhedged local `stated` statement beats external `imported`/`inferred` summary
  2. Newer explicit change statement (unhedged, with change cue, date >= opponent)
  3. More independent evidence: (firm_days, total_days, distinct_episodes)
  4. Exact tie -> stay `CONTESTED` for human / health-scan review

Every auto-resolution writes reversible events to `ledger.jsonl`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .models import Evidence, Fact, FactType, Status
from .text import is_hedged

NEGATION_CUES = (
    "not", "no", "never", "dont", "doesnt", "didnt", "cant", "cannot",
    "avoid", "hate", "dislike", "without", "instead", "rather",
)
# Function words carry no topical signal and inflate/deflate overlap.
STOPWORDS = frozenset(
    """
    a an the is are was were be been being am do does did doing have has had
    having i me my we our you your he she it they them this that these those
    of in on at to for with from by as and or but if then than so such very
    too also just really quite pretty much more most less least
    """.split()
)
STOPWORDS |= frozenset(NEGATION_CUES)

# Antonym pairs that flip the meaning of an otherwise-similar claim.
ANTONYMS = (
    ("love", "hate"), ("like", "dislike"), ("prefer", "avoid"),
    ("always", "never"), ("verbose", "terse"), ("concise", "verbose"),
    ("short", "long"), ("fast", "slow"), ("simple", "complex"),
    ("beginner", "expert"), ("enabled", "disabled"), ("use", "avoid"),
    ("want", "refuse"),
)

_WORD_RE = re.compile(r"[a-z0-9]+|[一-鿿]+", re.UNICODE)
_CJK_NEG = ("不", "别", "勿", "没", "无")
_EXPLICIT_CHANGE_RE = re.compile(
    r"改用|改成|换成|不再|改为|从此|现在改|现在用|已经换|重新换|"
    r"\b(?:switched|switching|changed|now i|no longer|instead of|moved to)\b",
    re.IGNORECASE,
)


@dataclass
class Conflict:
    a: Fact
    b: Fact
    reason: str


@dataclass(frozen=True)
class ConflictDecision:
    keep_id: str
    drop_id: str
    tier: int
    reason: str
    conflict_reason: str = ""


def _stem(word: str) -> str:
    """Crude but adequate suffix folding: eats->eat, using->use, prefers->prefer."""
    for suffix in ("ing", "ies", "es", "ed", "s"):
        if len(word) > 3 and word.endswith(suffix):
            stem = word[: -len(suffix)]
            if suffix == "ies":
                return stem + "y"
            return stem
    return word


def _content(text: str) -> set[str]:
    return {
        _stem(w) for w in _WORD_RE.findall(text.lower()) if w not in STOPWORDS and len(w) > 1
    }


def _has_negation(text: str) -> bool:
    low = f" {text.lower()} "
    if any(f" {cue}" in low or f"{cue} " in low for cue in NEGATION_CUES):
        return True
    return any(ch in text for ch in _CJK_NEG)


def _antonym_pair(a: str, b: str) -> tuple[str, str] | None:
    ca, cb = _content(a), _content(b)
    for x, y in ANTONYMS:
        sx, sy = _stem(x), _stem(y)
        if (sx in ca and sy in cb) or (sy in ca and sx in cb):
            return (x, y)
    return None


def _containment(a: str, b: str) -> float:
    ca, cb = _content(a), _content(b)
    if not ca or not cb:
        return 0.0
    return len(ca & cb) / min(len(ca), len(cb))


def _firm_evidence(fact: Fact) -> list[Evidence]:
    return [e for e in fact.evidence if e.origin in ("stated", "imported") and not is_hedged(e.quote)]


def _has_firm_stated(fact: Fact) -> bool:
    return any(e.origin == "stated" and not is_hedged(e.quote) for e in fact.evidence)


def _all_external_or_inferred(fact: Fact) -> bool:
    return bool(fact.evidence) and all(e.origin in ("imported", "inferred") for e in fact.evidence)


def _has_explicit_change(fact: Fact) -> bool:
    firm = _firm_evidence(fact)
    if not firm:
        return False
    if _EXPLICIT_CHANGE_RE.search(fact.claim):
        return True
    return any(_EXPLICIT_CHANGE_RE.search(e.quote) for e in firm)


def _latest_date(fact: Fact):
    return max((e.date for e in fact.evidence), default=fact.valid_from)


def _evidence_rank(fact: Fact) -> tuple[int, int, int]:
    firm_days = len({e.date for e in _firm_evidence(fact)})
    total_days = len({e.date for e in fact.evidence})
    distinct_episodes = len({e.episode for e in fact.evidence})
    return (firm_days, total_days, distinct_episodes)


def decide_conflict(a: Fact, b: Fact, conflict_reason: str = "") -> ConflictDecision | None:
    """Resolve a contradiction deterministically when safe, else return None (CONTESTED)."""
    if FactType.CONSTRAINT in {a.type, b.type}:
        return None

    # Tier 1: owner's direct firm local statement beats external imported / inferred summary.
    a_firm_stated = _has_firm_stated(a)
    b_firm_stated = _has_firm_stated(b)
    if a_firm_stated and _all_external_or_inferred(b):
        return ConflictDecision(
            keep_id=a.id,
            drop_id=b.id,
            tier=1,
            reason="本地明确口述优先于外部导入或推断摘要",
            conflict_reason=conflict_reason,
        )
    if b_firm_stated and _all_external_or_inferred(a):
        return ConflictDecision(
            keep_id=b.id,
            drop_id=a.id,
            tier=1,
            reason="本地明确口述优先于外部导入或推断摘要",
            conflict_reason=conflict_reason,
        )

    # Tier 2: explicit change cue ("改用/不再/switched...") with non-older evidence wins.
    change_a = _has_explicit_change(a)
    change_b = _has_explicit_change(b)
    date_a = _latest_date(a)
    date_b = _latest_date(b)
    if change_a and not change_b and date_a >= date_b:
        return ConflictDecision(
            keep_id=a.id,
            drop_id=b.id,
            tier=2,
            reason="含明确变更表述且不早于对方陈述",
            conflict_reason=conflict_reason,
        )
    if change_b and not change_a and date_b >= date_a:
        return ConflictDecision(
            keep_id=b.id,
            drop_id=a.id,
            tier=2,
            reason="含明确变更表述且不早于对方陈述",
            conflict_reason=conflict_reason,
        )
    if change_a and change_b and date_a != date_b:
        keep, drop = (a, b) if date_a > date_b else (b, a)
        return ConflictDecision(
            keep_id=keep.id,
            drop_id=drop.id,
            tier=2,
            reason="双方均为明确变更，保留日期更新的一条",
            conflict_reason=conflict_reason,
        )

    # Tier 3: more independent evidence wins: (firm_days, total_days, distinct_episodes).
    rank_a = _evidence_rank(a)
    rank_b = _evidence_rank(b)
    if rank_a != rank_b:
        keep, drop, kr, dr = (a, b, rank_a, rank_b) if rank_a > rank_b else (b, a, rank_b, rank_a)
        return ConflictDecision(
            keep_id=keep.id,
            drop_id=drop.id,
            tier=3,
            reason=f"更多独立证据支持（{kr[0]}天/{kr[2]}条 vs {dr[0]}天/{dr[2]}条）",
            conflict_reason=conflict_reason,
        )

    # Tier 4: exact tie stays contested for owner / health scan.
    return None


def detect_conflicts(
    facts: list[Fact],
    content_floor: float = 0.5,
    near_identical: float = 0.8,
) -> list[Conflict]:
    """Find candidate contradictions among live facts sharing a subject."""
    live = [f for f in facts if f.status == Status.ACTIVE and f.valid_to is None]
    by_subject: dict[str, list[Fact]] = {}
    for f in live:
        by_subject.setdefault(f.subject, []).append(f)

    out: list[Conflict] = []
    for group in by_subject.values():
        for i, a in enumerate(group):
            for b in group[i + 1 :]:
                if a.claim.strip().lower() == b.claim.strip().lower():
                    continue  # identical claim, not a conflict

                pair = _antonym_pair(a.claim, b.claim)
                if pair:
                    out.append(Conflict(a, b, f"antonym cue on same subject ({pair[0]}/{pair[1]})"))
                    continue

                overlap = _containment(a.claim, b.claim)
                neg_a, neg_b = _has_negation(a.claim), _has_negation(b.claim)

                if overlap >= content_floor and neg_a != neg_b:
                    out.append(Conflict(a, b, "same subject, one claim negated"))
                    continue

                # Same subject, high lexical containment, neither negated, but the
                # claims differ — typically a swapped value ("uses Rust"/"uses Python").
                if overlap >= near_identical and not (neg_a or neg_b):
                    out.append(Conflict(a, b, "same subject, conflicting value"))
    return out
