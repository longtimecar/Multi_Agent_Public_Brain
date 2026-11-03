"""Hybrid retrieval with a hard injection budget.

Fusion is reciprocal-rank, which needs no score normalisation and degrades
gracefully when one arm has no results (no embeddings configured, or a query
with no lexical overlap at all).

Budgeting is the part that matters most in practice: over-retrieval is the
dominant failure mode of memory injection. Each item is only included if it
fits whole; nothing is ever truncated mid-item.

Scheme A: optional scope (URI prefix), level (0=abstract / 1=overview / 2=claim),
type filter, and sensitive gating (religion/politics/belief excluded from
default inject unless topic-relevant).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
import re

from .budget import estimate_tokens
from .config import Settings
from .confidence import is_promoted, render_hedged
from .embeddings import Embedder, EmbeddingError
from .index import Index, Hit
from .models import Fact, FactType, Status
from .overview import build_l1_overviews, looks_sensitive, make_abstract, render_l1_block
from .peer import peer_allows
from .search_mode import MODE_CONTEXT, normalize_mode, resolve_intent_rerank
from .store import Store
from .uri import uri_matches_scope

RRF_K = 60
CANDIDATES = 50
DEFAULT_TOP_K = 8


def _overview_scope(query: str) -> str | None:
    """Recognize explicit requests to summarize saved memory or preferences.

    These are category queries, not semantic similarity queries: asking for a
    list of preferences need not resemble the content of any one preference.
    Narrow topic questions continue through the ordinary hybrid retrieval path.
    """
    text = " ".join(query.casefold().split())
    compact = re.sub(r"\s+", "", text)
    preference_terms = ("我的喜好", "我的偏好", "我的爱好", "我的兴趣", "我平时爱干嘛",
                        "我平常爱干嘛", "我平时喜欢做什么", "我的兴趣爱好")
    if any(term in compact for term in preference_terms) or re.search(
        r"\b(?:my (?:preferences|interests|hobbies)|what do i (?:like|enjoy))\b", text
    ):
        return "preferences"
    if re.search(
        r"(?:查询|查看|列出|总结|概括|展示)(?:一下|下)?(?:我|我的)(?:全部|所有|个人)?记忆(?:$|[，。！？?!、；;])"
        r"|(?:我|我的)(?:全部|所有|个人)?记忆(?:有哪些|是什么|都有什么)", compact
    ):
        return "all"
    if re.search(r"\b(?:what do you (?:remember|know) about me|(?:list|summari[sz]e|show) my (?:memories|memory))\b", text):
        return "all"
    return None


def _topic_wants_sensitive(query: str) -> bool:
    """Only surface sensitive beliefs when the user is clearly on that topic."""
    return looks_sensitive(query)


@dataclass
class RecallResult:
    hits: list[Fact]
    text: str
    tokens: int
    mode: str
    notes: list[str] = field(default_factory=list)
    level: int = 2


class Retriever:
    def __init__(self, store: Store, index: Index, embedder: Embedder, settings: Settings) -> None:
        self.store = store
        self.index = index
        self.embedder = embedder
        self.settings = settings

    # ------------------------------------------------------------ ingestion

    def reindex(self) -> dict[str, int]:
        """Rebuild derived state from the canonical Markdown. Idempotent."""
        self.index.reset()
        facts = self.store.all_facts()
        to_embed: list[tuple[Fact, str]] = []
        for fact in facts:
            path = self.store.fact_path(fact.id)
            body = path.read_text(encoding="utf-8") if path.exists() else ""
            self.index.upsert(fact, body)
            if fact.is_live:
                to_embed.append((fact, f"{fact.subject}\n{fact.claim}"))
        embedded = 0
        if self.embedder.enabled and to_embed:
            try:
                matrix = self.embedder.embed([t for _, t in to_embed])
                if matrix is not None:
                    self.index.claim_model(self.embedder.model)  # reset() cleared any old claim
                    for (fact, _), vec in zip(to_embed, matrix):
                        self.index.put_vector(fact.id, vec)
                    embedded = len(to_embed)
            except EmbeddingError:
                pass  # lexical-only index is still a working index
        self.index.commit()
        return {"facts": len(facts), "embedded": embedded}

    def sync_fact(self, fact: Fact) -> None:
        """Index a single fact after a write, keeping the index warm."""
        path = self.store.fact_path(fact.id)
        body = path.read_text(encoding="utf-8") if path.exists() else ""
        self.index.upsert(fact, body)
        # Vectors from a different model live in a different space; skip them
        # here and let `reindex` re-embed everything with the current model.
        if fact.is_live and self.embedder.enabled and self.index.claim_model(self.embedder.model):
            try:
                vec = self.embedder.embed([f"{fact.subject}\n{fact.claim}"])
                if vec is not None:
                    self.index.put_vector(fact.id, vec[0])
            except EmbeddingError:
                pass
        self.index.commit()

    # -------------------------------------------------------------- filters

    def _filter_facts(
        self,
        facts: list[Fact],
        *,
        scope: str | None = None,
        type: str | FactType | None = None,
        include_sensitive: bool = False,
        allow_sensitive_topic: bool = False,
        peer: str | None = None,
        peer_scope: str = "all",
    ) -> list[Fact]:
        type_value = None
        if type is not None:
            type_value = type.value if isinstance(type, FactType) else str(type).strip().lower()
        out: list[Fact] = []
        for fact in facts:
            if type_value and fact.type.value != type_value:
                continue
            if scope and not uri_matches_scope(fact.uri or "", scope):
                continue
            if fact.sensitive and not (include_sensitive or allow_sensitive_topic):
                continue
            if not peer_allows(fact.uri, fact.peer, peer=peer, peer_scope=peer_scope):
                continue
            out.append(fact)
        return out

    # -------------------------------------------------------------- retrieval

    def _candidates(self, query: str, live: dict[str, Fact]) -> tuple[dict[str, Hit], str, list[str]]:
        notes: list[str] = []
        got: dict[str, Hit] = {}

        lex = self.index.lexical(query, limit=CANDIDATES)
        for fid, score in lex.items():
            if fid in live:
                got[fid] = Hit(fact=live[fid], lexical=score)

        mode = "lexical"
        if self.embedder.enabled:
            ids, mat = self.index.vector_matrix()
            if mat is not None and self.index.model_matches(self.embedder.model):
                try:
                    qv = self.embedder.embed_queries([query])
                except EmbeddingError:
                    qv = None
                    notes.append("embedding request failed; lexical only")
                if qv is not None:
                    qv = qv[0]
                    sims = mat @ qv
                    # Guard lexical matches too when they have a vector: FTS
                    # can otherwise inject unrelated facts on common words or
                    # metadata despite a low semantic score. Facts with no
                    # vector retain lexical fallback.
                    scores = {fid: float(sims[i]) for i, fid in enumerate(ids)}
                    for fid in list(got):
                        if fid in scores and not scores[fid] >= self.settings.semantic_floor:
                            del got[fid]
                    keep = min(CANDIDATES, len(ids))
                    order = sims.argsort()[::-1][:keep]
                    for i in order:
                        fid = ids[int(i)]
                        if fid in live and float(sims[int(i)]) >= self.settings.semantic_floor:
                            hit = got.setdefault(fid, Hit(fact=live[fid]))
                            hit.semantic = float(sims[int(i)])
                    mode = "hybrid"
            elif mat is not None:
                notes.append("index built with a different embedding model; lexical only")
            else:
                notes.append("no vectors indexed yet; lexical only")
        else:
            notes.append("no embedding endpoint configured; lexical only")
        return got, mode, notes

    def _rank(self, hits: dict[str, Hit]) -> list[Hit]:
        lex_order = sorted(hits.values(), key=lambda h: h.lexical, reverse=True)
        sem_order = sorted(hits.values(), key=lambda h: h.semantic, reverse=True)
        rrf: dict[str, float] = {}
        for rank, hit in enumerate(lex_order):
            if hit.lexical > 0:
                rrf[hit.fact.id] = rrf.get(hit.fact.id, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, hit in enumerate(sem_order):
            if hit.semantic > 0:
                rrf[hit.fact.id] = rrf.get(hit.fact.id, 0.0) + 1.0 / (RRF_K + rank + 1)
        # Relevance decides the order; confidence and recency only break ties.
        def key(hit: Hit) -> tuple[float, float, float]:
            age_days = max(0, (date.today() - hit.fact.valid_from).days)
            return (round(rrf.get(hit.fact.id, 0.0), 9), hit.fact.confidence, -age_days)

        return sorted(hits.values(), key=key, reverse=True)

    def recall(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        budget: int | None = None,
        exclude_ids: set[str] | None = None,
        *,
        scope: str | None = None,
        level: int = 2,
        type: str | FactType | None = None,
        include_sensitive: bool = False,
        use_intent: bool | None = None,
        use_rerank: bool | None = None,
        session: str | None = None,
        llm: object | None = None,
        mode: str | None = None,
        peer: str | None = None,
        peer_scope: str = "all",
        purpose: str | None = None,
        type_quotas: dict[str, int] | None = None,
    ) -> RecallResult:
        budget = budget if budget is not None else self.settings.turn_token_budget
        level = 0 if level <= 0 else (1 if level == 1 else 2)
        search_mode = normalize_mode(mode)
        use_intent, use_rerank, mode_notes = resolve_intent_rerank(
            search_mode,
            use_intent=use_intent,
            use_rerank=use_rerank,
            intent_default=bool(getattr(self.settings, "use_intent_default", False)),
            rerank_default=bool(getattr(self.settings, "use_rerank_default", False)),
        )
        allow_sensitive_topic = include_sensitive or _topic_wants_sensitive(query)
        notes: list[str] = list(mode_notes)

        # Gap 2: optional intent → 0..N TypedQuery (fail-open). Default off.
        planned_queries: list[str] = [query]
        if use_intent:
            try:
                from .intent import IntentAnalyzer
                recent = []
                overview = ""
                if session:
                    eps = [ep for ep in self.store.iter_episodes() if ep.session == session]
                    recent = [{"role": ep.role, "text": ep.text[:500]} for ep in eps[-5:]]
                    try:
                        from .session_archive import history_dir
                        hist = history_dir(self.store, session)
                        dones = sorted(hist.glob("archive_*"))
                        for arch in reversed(dones):
                            ov = arch / ".overview.md"
                            if ov.exists():
                                overview = ov.read_text(encoding="utf-8")[:2000]
                                break
                    except Exception:
                        overview = ""
                plan = IntentAnalyzer(llm).analyze(query, recent_messages=recent, session_overview=overview)
                notes.extend([f"intent:{n}" for n in plan.notes])
                memory_qs = [tq for tq in plan.queries if tq.context_type == "memory"]
                if not plan.queries:
                    notes.append("intent:zero_queries_chitchat")
                    return RecallResult(hits=[], text="", tokens=0, mode="intent-skip", notes=notes, level=level)
                if memory_qs:
                    memory_qs = sorted(memory_qs, key=lambda t: t.priority)[: int(getattr(self.settings, "intent_max_queries", 5) or 5)]
                    planned_queries = [tq.query for tq in memory_qs]
                    notes.append(f"intent:planned={len(planned_queries)}")
                else:
                    notes.append("intent:no_memory_queries")
                    planned_queries = [query]
            except Exception as exc:  # fail-open
                notes.append(f"intent_failed:{exc.__class__.__name__}")
                planned_queries = [query]

        live_list = self._filter_facts(
            self.store.live_facts(),
            scope=scope,
            type=type,
            include_sensitive=include_sensitive,
            allow_sensitive_topic=allow_sensitive_topic,
            peer=peer,
            peer_scope=peer_scope,
        )
        if peer and (peer_scope or "all").lower() in {"actor", "peer", "strict"}:
            notes.append(f"peer_scope:{peer_scope}:{peer}")
        elif peer:
            notes.append("peer_noted_default_cross_project")
        if allow_sensitive_topic and not include_sensitive:
            notes.append("sensitive facts eligible: topic-relevant")
        elif not include_sensitive:
            notes.append("sensitive facts excluded from default recall")

        # Gap 6: context mode — type quotas + token trim (fail-open to normal path on error).
        if search_mode == MODE_CONTEXT and level == 2 and not _overview_scope(query):
            try:
                from .context_pack import pack_by_type_quotas
                live = {f.id: f for f in live_list}
                merged: dict[str, Hit] = {}
                mode_name = "lexical"
                for pq in planned_queries:
                    hits, mode_name, cand_notes = self._candidates(pq, live)
                    notes = notes + cand_notes
                    for fid, hit in hits.items():
                        prev = merged.get(fid)
                        if prev is None:
                            merged[fid] = hit
                        else:
                            prev.lexical = max(prev.lexical, hit.lexical)
                            prev.semantic = max(prev.semantic, hit.semantic)
                ranked = self._rank(merged)
                if exclude_ids:
                    ranked = [h for h in ranked if h.fact.id not in exclude_ids]
                scored = []
                for i, h in enumerate(ranked):
                    score = float(h.semantic or 0.0) * 10.0 + float(h.lexical or 0.0) + (len(ranked) - i) * 0.001
                    scored.append((h.fact, score))
                chosen, text, tokens, pack_notes = pack_by_type_quotas(
                    scored,
                    quotas=type_quotas,
                    purpose=purpose,
                    top_k=top_k,
                    budget=budget,
                    include_sensitive=allow_sensitive_topic,
                )
                notes.extend(pack_notes)
                return RecallResult(
                    hits=chosen,
                    text=text,
                    tokens=tokens,
                    mode="context",
                    notes=notes,
                    level=level,
                )
            except Exception as exc:
                notes.append(f"context_pack_failed:{exc.__class__.__name__}")

        overview_kind = _overview_scope(query)
        if level == 1 or (overview_kind and level != 0):
            facts = live_list
            if exclude_ids:
                facts = [f for f in facts if f.id not in exclude_ids]
            if overview_kind == "preferences":
                facts = [f for f in facts if f.type == FactType.PREFERENCE]
            # Newest first keeps overview hits stable vs FS listing order (lifecycle/budget tests).
            facts = sorted(facts, key=lambda fact: (fact.valid_from, fact.created_at, fact.id), reverse=True)
            overviews = build_l1_overviews(facts, include_sensitive=allow_sensitive_topic)
            block = render_l1_block(overviews)
            # Also attach ids for injection dedupe; preserve newest-first hit order
            ids = [fid for row in overviews for fid in row["ids"]]
            id_set = set(ids)
            hit_facts = [f for f in facts if f.id in id_set]
            text = block
            if text and estimate_tokens(text) > budget:
                # trim overview rows
                trimmed = render_l1_block(overviews[: max(1, budget // 40)])
                text = trimmed if estimate_tokens(trimmed) <= budget else ""
            return RecallResult(
                hits=hit_facts[:top_k],
                text=text,
                tokens=estimate_tokens(text),
                mode="overview-l1" if level == 1 else "overview",
                notes=notes + ([f"explicit {overview_kind} overview; L1"] if overview_kind else ["L1 overview"]),
                level=1,
            )

        if overview_kind:
            facts = live_list
            if overview_kind == "preferences":
                facts = [f for f in facts if f.type == FactType.PREFERENCE]
            facts.sort(key=lambda fact: (fact.valid_from, fact.created_at, fact.id), reverse=True)
            ranked = [Hit(fact=fact) for fact in facts]
            mode = "overview"
            notes = notes + [f"explicit {overview_kind} overview; newest first"]
        else:
            live = {f.id: f for f in live_list}
            merged: dict[str, Hit] = {}
            mode = "lexical"
            for pq in planned_queries:
                hits, mode, cand_notes = self._candidates(pq, live)
                notes = notes + cand_notes
                for fid, hit in hits.items():
                    prev = merged.get(fid)
                    if prev is None:
                        merged[fid] = hit
                    else:
                        prev.lexical = max(prev.lexical, hit.lexical)
                        prev.semantic = max(prev.semantic, hit.semantic)
            ranked = self._rank(merged)
            # Gap 3: optional L0 rerank (fail-open). Fetch 2× then cut.
            if use_rerank and ranked:
                try:
                    from .overview import make_abstract
                    from .rerank import rerank_scores
                    factor = max(1, int(getattr(self.settings, "rerank_candidate_factor", 2) or 2))
                    pool = ranked[: max(top_k * factor, top_k)]
                    docs = [(h.fact.abstract or make_abstract(h.fact.claim)) for h in pool]
                    # fallback: position-derived score (higher better)
                    fallback = [float(len(pool) - i) for i in range(len(pool))]
                    use_llm_flag = bool(getattr(self.settings, "_rerank_use_llm", False))
                    rr = rerank_scores(
                        query,
                        docs,
                        fallback,
                        llm=llm,
                        use_llm=use_llm_flag,
                        provider=getattr(self.settings, "rerank_provider", None),
                        settings=self.settings,
                    )
                    notes.extend([f"rerank:{n}" for n in rr.notes])
                    order = sorted(range(len(pool)), key=lambda i: rr.scores[i], reverse=True)
                    ranked = [pool[i] for i in order] + ranked[len(pool):]
                except Exception as exc:
                    notes.append(f"rerank_failed:{exc.__class__.__name__}")

        header = "<memory>"
        footer = "</memory>"
        used = estimate_tokens(header + footer)
        chosen: list[Fact] = []
        lines: list[str] = []

        for hit in ranked:
            if exclude_ids and hit.fact.id in exclude_ids:
                continue
            if len(chosen) >= top_k:
                break
            fact = hit.fact
            if level == 0:
                claimed = fact.abstract or make_abstract(fact.claim)
            else:
                claimed = render_hedged(fact)
            tags = f"({fact.type.value}, {fact.subject}, conf {fact.confidence:.2f}"
            if fact.uri:
                tags += f", {fact.uri}"
            if fact.sensitive:
                tags += ", sensitive"
            if fact.status == Status.CONTESTED:
                tags += ", CONTESTED - conflicting entries exist"
            tags += ")"
            line = f"- {claimed} {tags}"
            cost = estimate_tokens(line) + 1
            if used + cost > budget:
                continue  # skip, never truncate mid-item
            lines.append(line)
            chosen.append(fact)
            used += cost

        text = ""
        if lines:
            text = header + "\n" + "\n".join(lines) + "\n" + footer
        out_mode = mode
        if search_mode == "quick" and mode not in {"intent-skip", "overview", "overview-l1"}:
            out_mode = f"quick:{mode}"
        elif search_mode == "thinking" and mode not in {"intent-skip", "overview", "overview-l1"}:
            out_mode = f"thinking:{mode}"
        return RecallResult(
            hits=chosen,
            text=text,
            tokens=estimate_tokens(text),
            mode=out_mode,
            notes=notes,
            level=level,
        )

    def contested(self) -> list[Fact]:
        return [f for f in self.store.all_facts() if f.status == Status.CONTESTED]

    def scan_conflicts(self) -> list:
        """Detect contradictions across the live store and flag them.

        Called after every write, so contradictions become visible on the turn
        they appear instead of accumulating silently. Flagging is all it does;
        resolution stays human.
        """
        from .conflict import detect_conflicts

        found = detect_conflicts(self.store.live_facts())
        if not found:
            return []
        self.store.contest_pairs([(c.a.id, c.b.id, c.reason) for c in found])
        for fact in self.store.all_facts():
            if fact.id in {c.a.id for c in found} | {c.b.id for c in found}:
                self.sync_fact(fact)
        return found
