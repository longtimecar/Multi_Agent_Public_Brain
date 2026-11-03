"""L0/L1 layers for facts (OpenViking-port Scheme A).

L0 abstract: one-line summary of a single fact (stored on the fact).
L1 overview: per type or type/subject rollup, derived on demand from live facts.
SessionStart injects profile + L0/L1; full claims are on-demand via search/read.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .models import Fact, FactType
from .uri import URI_PREFIX, build_uri

# Topics that mark a fact sensitive when present in claim/subject (religion/politics).
SENSITIVE_MARKERS = (
    "religion", "religious", "faith", "god", "church", "temple", "buddh", "christian",
    "islam", "muslim", "hindu", "daoist", "taoist", "atheis",
    "政治", "宗教", "信仰", "佛教", "基督教", "伊斯兰", "清真", "教会", "寺庙",
    "共产党", "民主", "共和", "选举", "党派", "左派", "右派",
    "politics", "political", "democrat", "republican", "liberal", "conservative",
    "election", "partisan", "ideolog",
)


def looks_sensitive(text: str, fact_type: str | FactType | None = None) -> bool:
    """Heuristic: belief type is always sensitive; religion/politics markers too."""
    if fact_type is not None:
        value = fact_type.value if isinstance(fact_type, FactType) else str(fact_type)
        if value == FactType.BELIEF.value:
            return True
    low = (text or "").casefold()
    return any(marker.casefold() in low for marker in SENSITIVE_MARKERS)


def make_abstract(claim: str, limit: int = 120) -> str:
    """L0: compress a claim to one short line."""
    flat = " ".join((claim or "").split())
    if len(flat) <= limit:
        return flat
    cut = flat[: limit - 1].rsplit(" ", 1)[0]
    return (cut or flat[: limit - 1]).rstrip(",.;:") + "…"


def ensure_fact_layers(fact: Fact) -> Fact:
    """Fill uri / abstract / sensitive defaults without mutating claim text."""
    updates: dict = {}
    if not fact.uri:
        updates["uri"] = build_uri(fact.type.value, fact.subject, fact.peer)
    if not fact.abstract:
        updates["abstract"] = make_abstract(fact.claim)
    if fact.sensitive is False and looks_sensitive(f"{fact.subject} {fact.claim}", fact.type):
        updates["sensitive"] = True
    if not updates:
        return fact
    return fact.model_copy(update=updates)


def group_key(fact: Fact) -> tuple[str, str]:
    return (fact.type.value, fact.subject)


def build_l1_overviews(facts: Iterable[Fact], *, include_sensitive: bool = False) -> list[dict]:
    """Build L1 overview rows grouped by type+subject.

    Sensitive facts are omitted unless include_sensitive=True (topic-relevant recall).
    """
    buckets: dict[tuple[str, str], list[Fact]] = defaultdict(list)
    for fact in facts:
        if not fact.is_live:
            continue
        if fact.sensitive and not include_sensitive:
            continue
        buckets[group_key(fact)].append(fact)

    rows: list[dict] = []
    for (ftype, subject), items in sorted(buckets.items()):
        items = sorted(items, key=lambda f: (f.valid_from, f.created_at, f.id), reverse=True)
        abstracts = [f.abstract or make_abstract(f.claim) for f in items[:8]]
        overview = "; ".join(abstracts)
        if len(items) > 8:
            overview += f"; …(+{len(items) - 8} more)"
        rows.append({
            "uri": build_uri(ftype, subject),
            "type": ftype,
            "subject": subject,
            "level": 1,
            "count": len(items),
            "overview": overview,
            "ids": [f.id for f in items],
            "sensitive": any(f.sensitive for f in items),
        })
    return rows


def render_l0_lines(facts: Iterable[Fact], *, include_sensitive: bool = False) -> list[str]:
    """One L0 line per fact for injection."""
    lines: list[str] = []
    for fact in facts:
        if not fact.is_live:
            continue
        if fact.sensitive and not include_sensitive:
            continue
        abstract = fact.abstract or make_abstract(fact.claim)
        lines.append(f"- [L0 {fact.type.value}/{fact.subject}] {abstract}")
    return lines


def render_l1_block(overviews: list[dict], budget_lines: int = 40) -> str:
    if not overviews:
        return ""
    lines = ["<memory_overview level=\"1\">"]
    for row in overviews[:budget_lines]:
        lines.append(f"- [{row['type']}/{row['subject']}] ({row['count']}) {row['overview']}")
    lines.append("</memory_overview>")
    return "\n".join(lines)


def scope_prefix_for_type(fact_type: str | None) -> str | None:
    if not fact_type:
        return None
    return f"{URI_PREFIX}{fact_type.strip().lower()}/"
