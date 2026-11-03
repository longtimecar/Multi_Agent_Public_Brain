"""Contradiction detection.

Detection is deliberately mechanical and never resolves anything. It answers
only "these two claims look like they are about the same thing and do not say
the same thing" — a similarity question, safe for code. Deciding WHICH is true
is a fact about the user that exists nowhere in the logs, so it stays with the
human.

Signals, cheapest first:
  1. antonym cue on the same subject   ("likes X" / "dislikes X")
  2. one claim negated, the other not, same subject, overlapping content
  3. same subject, near-identical phrasing, different value ("uses Rust"/"uses Python")

Measurement notes, learned the hard way: plain Jaccard with raw tokens misses
real conflicts. `does not eat spicy food` vs `eats spicy food` scores 0.33
because "does" and "not" are noise and "eat"/"eats" do not match. So tokens are
stemmed and stop-worded first, and overlap is measured by containment
(|A∩B| / min|A|,|B|) rather than by union.

Anything flagged gets `status: contested`, which both hedges it at retrieval and
surfaces it in `list_contested`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .models import Fact, Status

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
    ("always", "never"), ("verbose", "terse"), ("short", "long"),
    ("fast", "slow"), ("simple", "complex"), ("beginner", "expert"),
    ("enabled", "disabled"), ("use", "avoid"), ("want", "refuse"),
)

_WORD_RE = re.compile(r"[a-z0-9]+|[一-鿿]+", re.UNICODE)
_CJK_NEG = ("不", "别", "勿", "没", "无")


@dataclass
class Conflict:
    a: Fact
    b: Fact
    reason: str


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
