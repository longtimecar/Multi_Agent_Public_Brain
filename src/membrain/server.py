"""MCP server. This is the portable floor: any MCP client gets the same memory.

Deliberate omissions:
  * There is no tool that writes `constraint` facts. Hard constraints come from
    the human, through the CLI, so a poisoned tool result can never become a
    standing instruction that re-enters every future session.
  * There is no tool that resolves a conflict. Conflicts surface; the human
    decides. Only `correct_memory` (an explicit user correction) changes
    lifecycle state, and it can retract but never silently pick a winner.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Literal

from mcp.server.mcpserver import MCPServer

from .archive import Archive
from .config import Settings, load_settings
from .embeddings import Embedder
from .index import Index
from .models import AGENT_WRITABLE_TYPES, Episode, Evidence, FactType, Status
from .retrieval import Retriever
from .store import Store, StoreError
from .text import clean_turn

INSTRUCTIONS = """\
Long-term memory about this user, shared across every agent they use.

Use `search_memory` before asking the user to repeat context (optional scope=
URI prefix, level=0/1/2, type=). Use `write_memory` when the user states a
durable fact, preference, goal, project, entity, event, case, tool or belief -
quote them verbatim in `evidence`. Beliefs and religion/politics are stored as
sensitive and recalled only when topic-relevant. Do not write speculation, and
do not write anything they did not actually say.

Past session records, lessons and reports live in the archive: use `search_archive`,
then `read_archive`. The archive is never injected automatically.

If this agent has no capture hooks, call `record_user_message` once per user turn
with the user's exact words before replying. It returns the episode id that
`write_memory` needs as evidence. Hook-based agents already record turns and do
not need it. Recorded messages are extracted into memory in the background, so
`write_memory` is only needed for something the user wants saved right now.

If a retrieved memory is wrong, call `correct_memory` with its id. Do not attempt
to reconcile contradictions yourself; `list_contested` shows what needs a human
decision.
"""

TypeLiteral = Literal[
    "fact", "preference", "goal", "project",
    "entity", "event", "case", "tool", "belief",
]


class MemoryService:
    """Wires the store, index and retriever together for one process."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        self.store = Store(self.settings.home)
        self.index = Index(self.settings.db_path)
        self.embedder = Embedder(self.settings)
        self.retriever = Retriever(self.store, self.index, self.embedder, self.settings)
        self.archive = Archive(self.store, self.index.conn, self.embedder, self.settings)

    def ensure_index(self) -> None:
        if self.index.stats()["facts"] == 0 and any(self.store.facts_dir.glob("f_*.md")):
            self.retriever.reindex()
        try:
            self.archive.ensure_current()
        except Exception:  # a damaged archive index must never stop the memory server starting
            pass

    def reindex_all(self) -> dict[str, Any]:
        """Rebuild every derived table: facts, then the archive."""
        facts = self.retriever.reindex()
        return {**facts, "archive": self.archive.reindex()}


def build_server(service: MemoryService | None = None) -> MCPServer:
    svc = service or MemoryService()
    svc.ensure_index()

    server = MCPServer(
        name="membrain",
        title="Personal Memory",
        description="User-owned long-term memory shared across AI agents",
        instructions=INSTRUCTIONS,
        version="0.1.0",
        log_level="WARNING",
    )

    @server.tool(
        description=(
            "Search the user's long-term memory. Call this before asking the user to "
            "re-explain context. Returns facts relevant to the query, already trimmed "
            "to an injection budget."
        )
    )
    def search_memory(
        query: str,
        top_k: int = 8,
        scope: str | None = None,
        level: int = 2,
        type: str | None = None,
        include_sensitive: bool = False,
        use_intent: bool = False,
        use_rerank: bool = False,
        session: str | None = None,
        mode: str | None = None,
        peer: str | None = None,
        peer_scope: str = "all",
        purpose: str | None = None,
    ) -> dict[str, Any]:
        """Search memory with optional URI scope / level / type filters.

        scope: URI prefix e.g. mb://user/memories/preference or shorthand preference/
        level: 0=L0 abstract, 1=L1 overview, 2=full claim (default)
        type: fact type filter
        include_sensitive: if false (default), religion/politics/belief only when topic-relevant
        use_intent / use_rerank: optional OV-gap features (default false; fail-open)
        mode: quick|thinking|context (OV find/search/context thought); default None keeps flag defaults
        peer / peer_scope: optional; default peer_scope=all keeps cross-project recall
        purpose: chat|coding preset for context mode quotas
        Default recall is cross-project (peer is ignored unless peer_scope=actor).
        """
        llm = None
        want_intent = bool(use_intent) or (str(mode or "").lower() == "thinking")
        if want_intent:
            from .llm import LLMClient
            llm = LLMClient(svc.settings)
        intent_flag: bool | None = True if use_intent else None
        rerank_flag: bool | None = True if use_rerank else None
        if str(mode or "").lower() == "quick":
            if not use_intent:
                intent_flag = False
            if not use_rerank:
                rerank_flag = False
        result = svc.retriever.recall(
            query,
            top_k=max(1, min(top_k, 20)),
            scope=scope,
            level=level,
            type=type,
            include_sensitive=include_sensitive,
            use_intent=intent_flag,
            use_rerank=rerank_flag,
            session=session,
            llm=llm,
            mode=mode,
            peer=peer,
            peer_scope=peer_scope or "all",
            purpose=purpose,
        )
        return {
            "block": result.text,
            "tokens": result.tokens,
            "mode": result.mode,
            "level": result.level,
            "notes": result.notes,
            "entries": [
                {
                    "id": f.id,
                    "type": f.type.value,
                    "subject": f.subject,
                    "claim": f.claim,
                    "abstract": f.abstract,
                    "uri": f.uri,
                    "sensitive": f.sensitive,
                    "peer": f.peer,
                    "confidence": f.confidence,
                    "status": f.status.value,
                    "source": f.source.model_dump(mode="json"),
                    "evidence": [e.model_dump(mode="json") for e in f.evidence],
                }
                for f in result.hits
            ],
        }


    @server.tool(
        description=(
            "Quick memory find (OV find-like): single query, no intent analysis, no rerank. "
            "Prefer this for simple lookups; use search_memory(mode=\"thinking\") for complex tasks."
        )
    )
    def find_memory(
        query: str,
        top_k: int = 8,
        scope: str | None = None,
        level: int = 2,
        type: str | None = None,
        include_sensitive: bool = False,
        peer: str | None = None,
        peer_scope: str = "all",
    ) -> dict[str, Any]:
        return search_memory(
            query=query,
            top_k=top_k,
            scope=scope,
            level=level,
            type=type,
            include_sensitive=include_sensitive,
            use_intent=False,
            use_rerank=False,
            mode="quick",
            peer=peer,
            peer_scope=peer_scope,
        )

    @server.tool(
        description=(
            "Read memory_diff.json for a completed session archive (audit of adds/updates/"
            "ledger retract-supersede deletes). Read-only."
        )
    )
    def read_memory_diff(archive: str) -> dict[str, Any]:
        from . import session_archive
        path = session_archive.resolve_archive_path(svc.store, archive)
        return session_archive.read_memory_diff(path)

    @server.tool(
        description=(
            "Search the user's archive of long notes (past session records, lessons learned, reports, study "
            "material, imported from other memory systems). Use it when `search_memory` has no answer and the "
            "user asks about something done or written earlier. Returns titles and one-line abstracts; call "
            "`read_archive` with an id for the full text."
        )
    )
    def search_archive(query: str, source: str | None = None, category: str | None = None,
                       top_k: int = 8) -> dict[str, Any]:
        results, notes = svc.archive.search(query, source, category, top_k=max(1, min(top_k, 20)))
        return {"results": results, "notes": notes}

    @server.tool(
        description=(
            "Browse the archive by layer. With no arguments: counts per source and category. With `source` "
            "and/or `category`: the entries there, newest first, with abstracts."
        )
    )
    def browse_archive(source: str | None = None, category: str | None = None) -> dict[str, Any]:
        return svc.archive.browse(source, category)

    @server.tool(description="Read the full text of one archive entry by id, in slices of up to 4000 characters.")
    def read_archive(entry_id: str, offset: int = 0, limit: int = 4000) -> dict[str, Any]:
        try:
            return svc.archive.read(entry_id, offset, limit)
        except StoreError as exc:
            return {"ok": False, "error": str(exc)}

    @server.tool(
        description=(
            "For agents without capture hooks: record the user's latest message verbatim, "
            "once per user turn, before replying. Pass only text the user typed, never your own "
            "words, tool output or retrieved memory. Returns the episode id to cite as evidence."
        )
    )
    def record_user_message(text: str, agent: str = "unknown", session: str | None = None) -> dict[str, Any]:
        cleaned = clean_turn(text)
        if not cleaned:
            return {"ok": False, "error": "nothing to record after removing injected or command text"}
        session = session or f"relay-{agent}"
        now = datetime.now(timezone.utc)
        # A hook-based agent already recorded this turn; reuse it so one message stays one episode.
        for ep in svc.store.iter_episodes():
            if (ep.role == "user" and clean_turn(ep.text) == cleaned
                    and (ep.session == session or (now - ep.ts).total_seconds() < 600)):
                return {"ok": True, "episode": ep.id, "duplicate": True}
        uid = hashlib.sha256(f"relay|{session}|{now.isoformat()}|{cleaned}".encode("utf-8")).hexdigest()
        episode = Episode(id=f"ep_{uid[:16]}", ts=now, session=session, agent=agent, role="user",
                          text=cleaned[:20000], relayed=True)
        svc.store.append_episodes([episode])
        try:
            from .hooks import _start_extractor
            _start_extractor(svc)
        except Exception:  # extraction can be retried later; recording already succeeded
            pass
        return {"ok": True, "episode": episode.id, "duplicate": False}

    @server.tool(description="Find captured user turns containing an exact quote, for use as memory evidence.")
    def find_user_evidence(quote: str, session: str | None = None) -> dict[str, Any]:
        if not quote.strip():
            return {"matches": []}
        matches = []
        for episode in svc.store.iter_episodes():
            if (episode.role == "user" and (session is None or episode.session == session)
                    and quote.strip() in clean_turn(episode.text)):
                matches.append({"episode": episode.id, "session": episode.session,
                                "date": episode.ts.date().isoformat()})
        return {"matches": matches[-20:]}

    @server.tool(
        description=(
            "Store a durable fact about the user. Only call this for things the user "
            "actually stated. `evidence` must contain a verbatim quote from the "
            "conversation and the episode id it came from."
        )
    )
    def write_memory(
        type: TypeLiteral,
        subject: str,
        claim: str,
        evidence: list[dict[str, str]],
        agent: str = "unknown",
        session: str | None = None,
        peer: str | None = None,
        sensitive: bool | None = None,
    ) -> dict[str, Any]:
        try:
            try:
                fact_type = FactType(type)
            except ValueError as exc:
                raise StoreError(f"unknown fact type: {type}") from exc
            if fact_type not in AGENT_WRITABLE_TYPES:
                raise StoreError("agent-written constraints are not allowed")
            captured = {ep.id: ep for ep in svc.store.iter_episodes()}
            if not evidence:
                raise StoreError("at least one captured user quote is required")
            for item in evidence:
                episode = captured.get(item.get("episode", ""))
                quote = item.get("quote", "")
                if episode is None or episode.role != "user" or not isinstance(quote, str) or not quote.strip() or quote.strip() not in clean_turn(episode.text):
                    raise StoreError("evidence quote must match a captured user turn")
            parsed = [
                Evidence(
                    episode=e["episode"],
                    quote=e["quote"],
                    date=captured[e["episode"]].ts.date(),
                    origin="stated",
                )
                for e in evidence
            ]
            fact = svc.store.draft_fact(
                type=fact_type,
                subject=subject,
                claim=claim,
                evidence=parsed,
                source_agent=agent,
                source_session=session,
                peer=peer,
                sensitive=sensitive,
            )
            stored = svc.store.write_fact(fact)
            svc.retriever.sync_fact(stored)
            flagged = svc.retriever.scan_conflicts()
        except (KeyError, ValueError, StoreError) as exc:
            return {"ok": False, "error": str(exc)}
        return {
            "ok": True,
            "id": stored.id,
            "uri": stored.uri,
            "sensitive": stored.sensitive,
            "abstract": stored.abstract,
            "confidence": stored.confidence,
            "conflicts": len(flagged),
            "note": "single-observation claims are injected as questions, not as facts",
        }

    @server.tool(
        description=(
            "Correct or retract a memory the user said is wrong. This is the only "
            "supported way to change memory lifecycle from a conversation."
        )
    )
    def correct_memory(
        memory_id: str,
        reason: str,
        replacement_claim: str | None = None,
        agent: str = "unknown",
        session: str | None = None,
        episode_id: str | None = None,
    ) -> dict[str, Any]:
        try:
            old = svc.store.get_fact(memory_id)
            if old.type == FactType.CONSTRAINT:
                raise StoreError("only the user may change a hard constraint")
            correction_episode = next((ep for ep in svc.store.iter_episodes() if ep.id == episode_id), None)
            if (correction_episode is None or correction_episode.role != "user" or not reason.strip()
                    or reason.strip() not in clean_turn(correction_episode.text)):
                raise StoreError("correction requires a reason quoted from a captured user episode")
        except StoreError as exc:
            return {"ok": False, "error": str(exc)}

        new_id = None
        if replacement_claim:
            evidence = [
                Evidence(
                    episode=correction_episode.id,
                    quote=reason,
                    date=correction_episode.ts.date(),
                    origin="stated",
                )
            ]
            new = svc.store.draft_fact(
                type=old.type,
                subject=old.subject,
                claim=replacement_claim,
                evidence=evidence,
                source_agent=agent,
                source_session=session,
                supersedes=old.id,
            )
            stored = svc.store.write_fact(new)
            svc.store.supersede(old.id, stored.id)
            new_id = stored.id
            svc.retriever.sync_fact(stored)
        else:
            svc.store.retract(memory_id, reason)

        svc.retriever.sync_fact(svc.store.get_fact(memory_id))
        return {"ok": True, "retracted": memory_id, "replacement": new_id}

    @server.tool(description="List memories that have unresolved contradictions. These need a human decision.")
    def list_contested() -> dict[str, Any]:
        items = svc.retriever.contested()
        return {
            "count": len(items),
            "contested": [
                {
                    "id": f.id,
                    "subject": f.subject,
                    "claim": f.claim,
                    "confidence": f.confidence,
                    "valid_from": f.valid_from.isoformat(),
                }
                for f in items
            ],
        }

    @server.tool(description="Return the user's profile and hard constraints (always-injected core).")
    def get_profile() -> dict[str, Any]:
        path = svc.settings.profile_path
        if not path.exists():
            return {"profile": "", "exists": False}
        return {"profile": path.read_text(encoding="utf-8"), "exists": True}

    @server.tool(description="Index statistics and backend status.")
    def memory_stats() -> dict[str, Any]:
        live = svc.store.live_facts()
        by_type: dict[str, int] = {}
        sensitive_n = 0
        for f in live:
            by_type[f.type.value] = by_type.get(f.type.value, 0) + 1
            if f.sensitive:
                sensitive_n += 1
        return {
            "home": str(svc.settings.home),
            "facts_total": len(svc.store.all_facts()),
            "facts_live": len(live),
            "contested": len([f for f in live if f.status == Status.CONTESTED]),
            "sensitive_live": sensitive_n,
            "by_type": by_type,
            "index": svc.index.stats(),
            "episodes": sum(1 for _ in svc.store.iter_episodes()),
            "scheme_a": {
                "uri_prefix": "mb://user/memories/<type>/<subject>",
                "auto_commit_tokens": svc.settings.auto_commit_token_threshold,
                "auto_commit_idle_minutes": svc.settings.auto_commit_idle_minutes,
            },
            **svc.settings.describe(),
        }

    return server


def main() -> None:
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
