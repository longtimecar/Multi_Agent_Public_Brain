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

import json
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
import re

from .budget import estimate_tokens
from .agents import is_disabled
from .config import Settings
from .confidence import is_promoted, render_hedged
from .embeddings import Embedder, EmbeddingError
from .index import Index, Hit, _match_terms
from .models import Episode, Fact, FactType, Status
from .overview import build_l1_overviews, looks_sensitive, make_abstract, render_l1_block
from .peer import peer_allows
from .search_mode import MODE_CONTEXT, normalize_mode, resolve_intent_rerank
from .store import Store
from .text import clean_turn, parse_time_window
from .uri import uri_matches_scope

RRF_K = 60
CANDIDATES = 50
DEFAULT_TOP_K = 8
# Below this many curated hits, recall also reads the raw transcript. Two rather
# than one: a single fact is often the wrong one, and the point of the fallback
# is to catch the case where extraction missed the turn entirely.
EPISODE_FALLBACK_BELOW = 2
# Texts per embedding request. A whole-store batch exceeds the read timeout on
# real endpoints; this size was verified against the owner's own 221 live facts.
EMBED_BATCH = 16
# CJK substring matches are not bm25 scores, so they get their own scale. Only
# the order of lexical scores feeds RRF, not their magnitude, so this weight
# decides where CJK matches sit among the FTS matches — above them, because
# when the question is Chinese the FTS match is the meaningless one.
CJK_LEXICAL_WEIGHT = 1e-3
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
# Above the CJK weight: a question that names a type and a day has an exact
# answer, and it should outrank anything found by fuzzy text overlap.
TYPE_WINDOW_WEIGHT = 1e-2
_TASK_QUESTION = re.compile(
    r"任务|待办|要做的事|该做什么|下一步做什么|还有什么没|没做完|"
    r"todo|to-do|my tasks?|what .{0,16}task|open (?:tasks|items)",
    re.IGNORECASE,
)


def _asked_type(query: str) -> FactType | None:
    """The memory type a question is asking ABOUT, as opposed to its topic."""
    if _TASK_QUESTION.search(query or ""):
        return FactType.TASK
    return None


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
    # "我的工作偏好是什么" is the same category ask as "我的偏好" with one noun
    # in between. Listing the exact phrases missed every qualified variant, so a
    # direct question about preferences fell through to similarity search and
    # came back empty: no single preference resembles "我的工作偏好".
    if re.search(
        r"(?:我的?[\u4e00-\u9fa5]{0,6}(?:偏好|喜好|爱好|习惯|风格|口味|忌讳))"
        r"|(?:我).{0,8}(?:喜欢什么|偏好什么|有哪些偏好|有哪些习惯)",
        compact,
    ) and re.search(r"(?:是什么|有哪些|都是什么|有什么|总结|概括|回顾|说说|列一下|列出|吗|？|\?)", compact):
        return "preferences"
    if re.search(r"\b(?:my|what are my)\b[^?]{0,30}\b(?:preferences|habits|interests|hobbies)\b", text):
        return "preferences"
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
    # Raw captured turns that matched when curated facts came up thin. Labelled
    # so the caller can say where the answer came from instead of presenting a
    # transcript fragment as a settled memory.
    episode_hits: list[dict] = field(default_factory=list)
    time_window: tuple[date, date] | None = None


class Retriever:
    def __init__(self, store: Store, index: Index, embedder: Embedder, settings: Settings) -> None:
        self.store = store
        self.index = index
        self.embedder = embedder
        self.settings = settings
        # Agents the owner disconnected from. Their memories stay on disk but
        # leave every recall path until the agent is connected again.
        self.disabled_agents: set[str] = set()
        # Recall tracker: which facts were surfaced when. Drives the decay
        # policy — details fade from injection after a month without use.
        self._tracker: dict[str, dict] = {}
        self._tracker_dirty = False
        self._tracker_last_flush = 0.0
        self._tracker_lock = threading.Lock()
        self._load_tracker()

    # ------------------------------------------------------- recall tracking

    @property
    def tracker_path(self):
        return self.settings.home / ".state" / "recall-tracker.json"

    def _load_tracker(self) -> None:
        try:
            data = json.loads(self.tracker_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self._tracker = {k: v for k, v in data.items() if isinstance(v, dict)}
        except (OSError, ValueError):
            self._tracker = {}

    def flush_tracker(self, force: bool = False) -> None:
        """Persist recall counters. Throttled: at most once a minute unless forced."""
        with self._tracker_lock:
            if not self._tracker_dirty:
                return
            now = time.monotonic()
            if not force and now - self._tracker_last_flush < 60:
                return
            path = self.tracker_path
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                path.write_text(json.dumps(self._tracker, ensure_ascii=False), encoding="utf-8")
                self._tracker_dirty = False
                self._tracker_last_flush = now
            except OSError:
                pass  # losing a flush only delays the decay clock, never corrupts data

    def _track_recalled(self, fact_ids: list[str]) -> None:
        now_iso = datetime.now(timezone.utc).date().isoformat()
        with self._tracker_lock:
            for fact_id in fact_ids:
                entry = self._tracker.setdefault(fact_id, {"count": 0})
                entry["last_recalled"] = now_iso
                entry["count"] = int(entry.get("count", 0)) + 1
            self._tracker_dirty = True
        self.flush_tracker()

    def recall_days_since_use(self, fact: Fact) -> int:
        """Days since the fact entered the store or was last surfaced, whichever is newer."""
        today = date.today()
        created = fact.created_at.date() if fact.created_at else fact.valid_from
        entry = self._tracker.get(fact.id) or {}
        last = entry.get("last_recalled")
        since_use = (today - date.fromisoformat(last)).days if last else None
        age = max(0, (today - created).days)
        return min(age, since_use) if since_use is not None else age

    def fact_is_stale(self, fact: Fact, window: int = 30) -> bool:
        """True when neither fresh nor surfaced inside the decay window."""
        return self.recall_days_since_use(fact) > window

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
            # One request for the whole store times out on a real endpoint: 221
            # facts exceeded the read timeout while batches of 16 were fine, and
            # the EmbeddingError was swallowed — leaving the index silently
            # lexical-only with no signal anywhere. Chunk it and keep whatever
            # succeeds, so a slow endpoint degrades instead of emptying.
            for start in range(0, len(to_embed), EMBED_BATCH):
                chunk = to_embed[start:start + EMBED_BATCH]
                try:
                    matrix = self.embedder.embed([t for _, t in chunk])
                except EmbeddingError:
                    continue  # lexical-only index is still a working index
                if matrix is None:
                    continue
                self.index.claim_model(self.embedder.model)  # reset() cleared any old claim
                for (fact, _), vec in zip(chunk, matrix):
                    self.index.put_vector(fact.id, vec)
                embedded += len(chunk)
        episodes = self.sync_episodes(self.store.iter_episodes())
        self.index.commit()
        return {"facts": len(facts), "embedded": embedded, "episodes": episodes}

    def sync_episodes(self, episodes: Iterable[Episode]) -> int:
        """Index raw turns so recall can fall back to what was actually said."""
        count = 0
        for ep in episodes:
            self.index.upsert_episode(ep)
            count += 1
        if count:
            self.index.commit()
        return count

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
            if is_disabled(fact.source.agent, self.disabled_agents):
                continue
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

    def _candidates(self, query: str, live: dict[str, Fact],
                    time_window: tuple[date, date] | None = None) -> tuple[dict[str, Hit], str, list[str]]:
        notes: list[str] = []
        got: dict[str, Hit] = {}

        lex = self.index.lexical(query, limit=CANDIDATES)
        for fid, score in lex.items():
            if fid in live:
                got[fid] = Hit(fact=live[fid], lexical=score)

        for fid, score in self._cjk_lexical(query, live).items():
            hit = got.get(fid)
            if hit is None:
                got[fid] = Hit(fact=live[fid], lexical=score)
            else:
                hit.lexical = max(hit.lexical, score)

        typed = self._typed_window_hits(query, live, time_window)
        for fid, score in typed.items():
            hit = got.get(fid)
            if hit is None:
                got[fid] = Hit(fact=live[fid], typed=score)
            else:
                hit.typed = max(hit.typed, score)

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
                        if fid in typed:
                            # A type+day match is a structured answer, not a
                            # similarity claim. Vetoing it on cosine made the
                            # result depend on which language the extractor
                            # happened to phrase the claim in: a Chinese question
                            # about an English claim scored under the floor and
                            # the correct task disappeared.
                            continue
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

    def _rank(self, hits: dict[str, Hit], time_window: tuple[date, date] | None = None) -> list[Hit]:
        lex_order = sorted(hits.values(), key=lambda h: h.lexical, reverse=True)
        sem_order = sorted(hits.values(), key=lambda h: h.semantic, reverse=True)
        rrf: dict[str, float] = {}
        for rank, hit in enumerate(lex_order):
            if hit.lexical > 0:
                rrf[hit.fact.id] = rrf.get(hit.fact.id, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, hit in enumerate(sem_order):
            if hit.semantic > 0:
                rrf[hit.fact.id] = rrf.get(hit.fact.id, 0.0) + 1.0 / (RRF_K + rank + 1)

        def in_window(hit: Hit) -> float:
            """1.0 when some evidence date falls inside the asked-about window.

            A demotion, never a hard filter: a fact stated earlier can still be
            the right answer to "昨晚说的", and dropping it outright would repeat
            the mistake of auto-invalidating on weak signals.
            """
            if time_window is None:
                return 1.0
            lo, hi = time_window
            dates = [e.date for e in hit.fact.evidence] or [hit.fact.valid_from]
            return 1.0 if any(lo <= d <= hi for d in dates) else 0.0

        # A question that names a type and a day has an exact answer: those
        # hits come first, then relevance, then time match, confidence, recency.
        def key(hit: Hit) -> tuple[float, float, float, float, float]:
            age_days = max(0, (date.today() - hit.fact.valid_from).days)
            return (hit.typed, round(rrf.get(hit.fact.id, 0.0), 9), in_window(hit),
                    hit.fact.confidence, -age_days)

        return sorted(hits.values(), key=key, reverse=True)

    def _cjk_lexical(self, query: str, live: dict[str, Fact]) -> dict[str, float]:
        """Substring scoring for CJK queries, which FTS5 cannot match.

        unicode61 treats a whole CJK run as a single token, so a Chinese
        question never matches a Chinese quote token-wise and recall came back
        empty even when the memory was there. Scanning the curated facts is
        bounded by the fact count, not by the transcript, so it stays cheap.
        """
        terms = [t for t in _match_terms(query) if _CJK_RE.search(t)]
        if not terms:
            return {}
        out: dict[str, float] = {}
        for fid, fact in live.items():
            haystack = " ".join([fact.claim, fact.subject,
                                 *(e.quote for e in fact.evidence)]).casefold()
            hits = sum(1 for t in terms if t in haystack)
            if hits:
                out[fid] = CJK_LEXICAL_WEIGHT * hits / len(terms)
        return out

    def _typed_window_hits(self, query: str, live: dict[str, Fact],
                           time_window: tuple[date, date] | None) -> dict[str, float]:
        """Facts matching the question's TYPE and day, regardless of wording.

        "我昨晚说的任务是什么" names a type and a date, not a topic. No amount of
        text similarity connects 任务 to a claim that says 时间戳修复, so leaving
        this to embeddings made the answer depend on which words the extractor
        happened to choose. It is a structured lookup.
        """
        wanted = _asked_type(query)
        if wanted is None:
            return {}
        out: dict[str, float] = {}
        for fid, fact in live.items():
            if fact.type != wanted:
                continue
            if wanted == FactType.TASK and fact.closed_at is not None:
                continue  # finished work is not an answer to "what is my task"
            if time_window is not None:
                lo, hi = time_window
                dates = [e.date for e in fact.evidence] or [fact.valid_from]
                if not any(lo <= d <= hi for d in dates):
                    continue
            out[fid] = TYPE_WINDOW_WEIGHT
        return out

    def _needs_episode_fallback(self, chosen: list[Fact],
                                time_window: tuple[date, date] | None) -> bool:
        """True when curated facts cannot answer, so raw turns are worth reading."""
        if not chosen:
            return True
        if time_window is None:
            return len(chosen) < EPISODE_FALLBACK_BELOW
        lo, hi = time_window
        return not any(
            lo <= d <= hi
            for fact in chosen
            for d in ([e.date for e in fact.evidence] or [fact.valid_from])
        )

    def _episode_fallback(self, query: str, time_window: tuple[date, date] | None,
                          limit: int = 5) -> list[dict]:
        """Raw captured turns matching the query, newest first.

        Curated facts are a lossy summary of what was said. When they come up
        thin the transcript still answers "what did I say last night", so recall
        reads it instead of returning nothing.
        """
        start, end = time_window if time_window else (None, None)
        try:
            rows = self.index.search_episodes(query, start=start, end=end, limit=limit * 3)
        except Exception:
            return []
        out: list[dict] = []
        for row in rows:
            text = clean_turn(row.get("text") or "")
            if not text:
                continue
            if self._episode_agent_hidden(row.get("agent") or ""):
                continue
            out.append({**row, "text": text[:300]})
            if len(out) >= limit:
                break
        return out

    def _episode_agent_hidden(self, agent: str) -> bool:
        """Turns from a disconnected agent stay hidden, same as its facts."""
        return is_disabled(agent, self.disabled_agents)

    def open_tasks(self, limit: int = 20) -> list[Fact]:
        """Tasks the owner has not said are finished, most recently stated first."""
        tasks = [f for f in self.store.live_facts() if f.is_open_task]
        tasks.sort(key=lambda f: f.valid_from, reverse=True)
        return tasks[:limit]

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
        # "我昨晚说的" is a date question, not a similarity question. Parsing it
        # deterministically keeps the recall hot path free of model calls.
        time_window = parse_time_window(query)
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
                    hits, mode_name, cand_notes = self._candidates(pq, live, time_window)
                    notes = notes + cand_notes
                    for fid, hit in hits.items():
                        prev = merged.get(fid)
                        if prev is None:
                            merged[fid] = hit
                        else:
                            prev.lexical = max(prev.lexical, hit.lexical)
                            prev.semantic = max(prev.semantic, hit.semantic)
                ranked = self._rank(merged, time_window)
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
                hits, mode, cand_notes = self._candidates(pq, live, time_window)
                notes = notes + cand_notes
                for fid, hit in hits.items():
                    prev = merged.get(fid)
                    if prev is None:
                        merged[fid] = hit
                    else:
                        prev.lexical = max(prev.lexical, hit.lexical)
                        prev.semantic = max(prev.semantic, hit.semantic)
            ranked = self._rank(merged, time_window)
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
            stale = self.fact_is_stale(fact)
            if level == 0 or stale:
                # Decay: facts nobody surfaced for a month collapse to their
                # one-line gist. The claim and type stay; the supporting
                # quote stops being injected (and re-reads stay possible via
                # detail views). This is the "gradual forgetting" policy.
                claimed = fact.abstract or make_abstract(fact.claim)
            else:
                claimed = render_hedged(fact)
            tags = f"({fact.type.value}, {fact.subject}, conf {fact.confidence:.2f}"
            if fact.uri:
                tags += f", {fact.uri}"
            if fact.sensitive:
                tags += ", sensitive"
            if stale and level > 0:
                tags += ", aged - long-unused summary only"
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
        self._track_recalled([fact.id for fact in chosen])

        text = ""
        if lines:
            text = header + "\n" + "\n".join(lines) + "\n" + footer
        out_mode = mode
        if search_mode == "quick" and mode not in {"intent-skip", "overview", "overview-l1"}:
            out_mode = f"quick:{mode}"
        elif search_mode == "thinking" and mode not in {"intent-skip", "overview", "overview-l1"}:
            out_mode = f"thinking:{mode}"
        # When the curated layer has nothing usable, read what was actually
        # said. "Usable" means: some hit, and — if the question named a time —
        # a hit whose evidence falls in that window. A correct fact from the
        # asked-about day is a complete answer; dumping loosely-matching raw
        # turns next to it only invites the reader to answer from the wrong one.
        episode_hits: list[dict] = []
        if self._needs_episode_fallback(chosen, time_window):
            episode_hits = self._episode_fallback(query, time_window)
            if episode_hits:
                notes.append("来自会话记录（记忆条目不足，已回看原始对话）")
        return RecallResult(
            hits=chosen,
            text=text,
            tokens=estimate_tokens(text),
            mode=out_mode,
            notes=notes,
            level=level,
            episode_hits=episode_hits,
            time_window=time_window,
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
