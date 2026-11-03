"""IntentAnalyzer → TypedQuery (OpenViking gap 2).

Default off (use_intent=False). LLM failure / timeout fail-open to a single
TypedQuery with the original query. Chit-chat may yield 0 queries (no recall).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence


class JSONClient(Protocol):
    def complete_json(self, system: str, user: str) -> dict[str, Any]: ...


@dataclass(frozen=True)
class TypedQuery:
    query: str
    context_type: str = "memory"  # memory | resource | skill (resource/skill may no-op)
    intent: str = ""
    priority: int = 3

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "context_type": self.context_type,
            "intent": self.intent,
            "priority": self.priority,
        }


@dataclass
class IntentPlan:
    queries: list[TypedQuery] = field(default_factory=list)
    reasoning: str = ""
    fallback: bool = False
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "queries": [q.as_dict() for q in self.queries],
            "reasoning": self.reasoning,
            "fallback": self.fallback,
            "notes": list(self.notes),
        }


_CHAT_RE = re.compile(
    r"^(?:hi|hello|hey|thanks|thank you|ok|okay|嗯+|好的?|谢谢|你好|嗨)[\s!.。！？]*$",
    re.IGNORECASE,
)


def looks_like_chitchat(query: str) -> bool:
    text = " ".join((query or "").strip().split())
    if not text:
        return True
    if len(text) <= 8 and _CHAT_RE.match(text):
        return True
    return False


def _clamp_priority(value: Any) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return 3
    return max(1, min(5, n))


def _normalize_context_type(value: Any) -> str:
    raw = str(value or "memory").strip().lower()
    if raw in {"memory", "resource", "skill"}:
        return raw
    return "memory"


class IntentAnalyzer:
    """Optional LLM planner. Always fail-open."""

    MAX_QUERIES = 5
    MAX_RECENT = 5

    def __init__(self, llm: JSONClient | None = None) -> None:
        self.llm = llm

    def analyze(
        self,
        query: str,
        *,
        recent_messages: Sequence[dict[str, str]] | None = None,
        session_overview: str = "",
    ) -> IntentPlan:
        q = " ".join((query or "").split())
        if not q:
            return IntentPlan(queries=[], reasoning="empty query", notes=["empty"])

        if self.llm is None:
            if looks_like_chitchat(q):
                return IntentPlan(queries=[], reasoning="chitchat heuristic", notes=["no_llm_chitchat"])
            return IntentPlan(
                queries=[TypedQuery(query=q, context_type="memory", intent="fallback", priority=3)],
                fallback=True,
                notes=["no_llm"],
            )

        recent = list(recent_messages or [])[-self.MAX_RECENT :]
        payload = {
            "current_query": q,
            "session_overview": (session_overview or "")[:4000],
            "recent_messages": recent,
        }
        system = (
            "You plan retrieval queries for a personal memory system. "
            "Return JSON only: {\"reasoning\": str, \"queries\": ["
            "{\"query\": str, \"context_type\": \"memory\"|\"resource\"|\"skill\", "
            "\"intent\": str, \"priority\": 1-5}]}. "
            "Use 0 queries for pure chit-chat/greetings with no memory need. "
            "For preference/profile questions emit at least one memory query. "
            "Prefer memory context_type. Keep queries short. Max 5 queries."
        )
        try:
            raw = self.llm.complete_json(system, json.dumps(payload, ensure_ascii=False))
        except Exception as exc:  # fail-open
            return IntentPlan(
                queries=[TypedQuery(query=q, context_type="memory", intent="fallback", priority=3)],
                reasoning="",
                fallback=True,
                notes=[f"llm_failed:{type(exc).__name__}"],
            )

        queries_raw = raw.get("queries") if isinstance(raw, dict) else None
        if queries_raw is None and isinstance(raw, list):
            queries_raw = raw
        if not isinstance(queries_raw, list):
            return IntentPlan(
                queries=[TypedQuery(query=q, context_type="memory", intent="fallback", priority=3)],
                fallback=True,
                notes=["bad_queries_shape"],
            )

        out: list[TypedQuery] = []
        for item in queries_raw[: self.MAX_QUERIES]:
            if not isinstance(item, dict):
                continue
            text = " ".join(str(item.get("query") or "").split())
            if not text:
                continue
            out.append(
                TypedQuery(
                    query=text,
                    context_type=_normalize_context_type(item.get("context_type")),
                    intent=str(item.get("intent") or ""),
                    priority=_clamp_priority(item.get("priority")),
                )
            )

        reasoning = ""
        if isinstance(raw, dict) and isinstance(raw.get("reasoning"), str):
            reasoning = raw["reasoning"]
        return IntentPlan(queries=out, reasoning=reasoning, fallback=False, notes=["ok"])
