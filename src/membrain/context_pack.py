"""Context-mode type quotas + token trim (OpenViking gap 6 thought, not code).

Assembles a balanced memory pack by FactType buckets, then fills leftover
slots from spare hits. Always respects a hard token budget; never truncates
mid-item.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .budget import estimate_tokens
from .models import Fact, FactType

# Soft first-pass ceilings (chat-oriented). Unused slots backfill from spare.
DEFAULT_TYPE_QUOTAS: dict[str, int] = {
    FactType.PREFERENCE.value: 2,
    FactType.FACT.value: 3,
    FactType.ENTITY.value: 2,
    FactType.EVENT.value: 2,
    FactType.GOAL.value: 1,
    FactType.PROJECT.value: 1,
    FactType.CONSTRAINT.value: 2,
    FactType.CASE.value: 1,
    FactType.TOOL.value: 1,
    FactType.BELIEF.value: 0,  # sensitive beliefs never get a default quota bucket
}

PURPOSE_QUOTAS: dict[str, dict[str, int]] = {
    "chat": dict(DEFAULT_TYPE_QUOTAS),
    "coding": {
        FactType.PREFERENCE.value: 1,
        FactType.FACT.value: 2,
        FactType.ENTITY.value: 2,
        FactType.EVENT.value: 1,
        FactType.GOAL.value: 1,
        FactType.PROJECT.value: 2,
        FactType.CONSTRAINT.value: 2,
        FactType.CASE.value: 1,
        FactType.TOOL.value: 2,
        FactType.BELIEF.value: 0,
    },
}


@dataclass(frozen=True)
class PackedHit:
    fact: Fact
    score: float
    bucket: str


def normalize_quotas(
    quotas: Mapping[str, int] | None = None,
    purpose: str | None = None,
) -> dict[str, int]:
    base = dict(PURPOSE_QUOTAS.get((purpose or "chat").strip().lower(), DEFAULT_TYPE_QUOTAS))
    if quotas:
        for key, value in quotas.items():
            k = str(key).strip().lower()
            try:
                n = int(value)
            except (TypeError, ValueError):
                continue
            if k:
                base[k] = max(0, n)
    return base


def pack_by_type_quotas(
    ranked: Iterable[tuple[Fact, float]],
    *,
    quotas: Mapping[str, int] | None = None,
    purpose: str | None = None,
    top_k: int = 8,
    budget: int = 1500,
    include_sensitive: bool = False,
) -> tuple[list[Fact], str, int, list[str]]:
    """Return chosen facts, rendered block, tokens, notes."""
    qmap = normalize_quotas(quotas, purpose)
    notes = [f"context_quotas:{ {k: v for k, v in qmap.items() if v > 0} }"]
    buckets: dict[str, list[PackedHit]] = {k: [] for k in qmap}
    spare_key = "_spare"
    buckets[spare_key] = []

    for fact, score in ranked:
        if fact.sensitive and not include_sensitive:
            continue
        key = fact.type.value if fact.type.value in qmap else spare_key
        if key not in buckets:
            buckets[key] = []
        buckets[key].append(PackedHit(fact=fact, score=float(score), bucket=key))

    first: list[PackedHit] = []
    spare: list[PackedHit] = []
    for key, quota in qmap.items():
        items = sorted(buckets.get(key, []), key=lambda h: h.score, reverse=True)
        if quota <= 0:
            # Explicit zero = excluded from default pack (e.g. belief).
            continue
        first.extend(items[:quota])
        # Overflow stays out: typed quotas are hard ceilings in membrain.
    spare.extend(buckets.get(spare_key, []))
    spare.sort(key=lambda h: h.score, reverse=True)

    ordered = first + spare
    header = "<memory mode=\"context\">"
    footer = "</memory>"
    used = estimate_tokens(header + footer)
    chosen: list[Fact] = []
    lines: list[str] = []
    seen: set[str] = set()
    type_counts: dict[str, int] = {}

    for hit in ordered:
        if hit.fact.id in seen:
            continue
        if len(chosen) >= top_k:
            break
        tname = hit.fact.type.value
        if tname in qmap:
            if type_counts.get(tname, 0) >= qmap[tname]:
                continue
        tags = f"({hit.fact.type.value}, {hit.fact.subject}, conf {hit.fact.confidence:.2f}"
        if hit.fact.uri:
            tags += f", {hit.fact.uri}"
        if hit.fact.sensitive:
            tags += ", sensitive"
        tags += ")"
        line = f"- {hit.fact.claim} {tags}"
        cost = estimate_tokens(line) + 1
        if used + cost > budget:
            continue
        lines.append(line)
        chosen.append(hit.fact)
        seen.add(hit.fact.id)
        used += cost
        type_counts[tname] = type_counts.get(tname, 0) + 1

    text = ""
    if lines:
        text = header + "\n" + "\n".join(lines) + "\n" + footer
    notes.append(f"context_packed={len(chosen)}")
    return chosen, text, estimate_tokens(text), notes
