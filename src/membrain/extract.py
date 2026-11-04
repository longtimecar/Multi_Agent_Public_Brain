"""Turn captured user turns into validated memory operations."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from .embeddings import EmbeddingError
from .llm import LLMError
from .locking import file_lock
from .injection import load as load_injections
from .models import Episode, Evidence, Fact, FactType
from .retrieval import Retriever
from .store import Store, StoreError, _atomic_write
from .text import clean_turn, widen_to_clause

STATE_FILE = ".state/extraction.json"
MAX_BATCH = 20


class JSONClient(Protocol):
    def complete_json(self, system: str, user: str) -> dict[str, Any]: ...


@dataclass
class ExtractionReport:
    sessions: int = 0
    episodes_seen: int = 0
    user_turns: int = 0
    operations: int = 0
    facts_added: int = 0
    corroborated: int = 0
    superseded: int = 0
    retracted: int = 0
    skipped: int = 0
    errors: list[str] | None = None

    def as_dict(self) -> dict[str, Any]:
        result = dict(self.__dict__)
        result["errors"] = self.errors or []
        return result


def _state_path(store: Store) -> Path:
    path = store.home / STATE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _load_state(store: Store) -> dict[str, Any]:
    path = _state_path(store)
    if not path.exists():
        return {"processed": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        processed = data["processed"]
        if not isinstance(processed, dict) or not all(isinstance(k, str) and isinstance(v, list)
                                                      and all(isinstance(x, str) for x in v)
                                                      for k, v in processed.items()):
            raise ValueError("invalid processed sessions")
        return {"processed": processed}
    except (OSError, ValueError, KeyError, AttributeError, TypeError) as exc:
        raise StoreError("extraction state is unreadable; refusing to replay episodes") from exc


def _save_state(store: Store, state: dict[str, Any]) -> None:
    _atomic_write(_state_path(store), json.dumps(state, ensure_ascii=False, indent=2))


def _normal(text: str) -> str:
    return " ".join(text.casefold().split())


def _explicit_change(text: str) -> bool:
    low = text.casefold()
    return any(token in low for token in (
        "switched", "switching", "changed", "change", "now i", "instead of",
        "改成", "换成", "不再", "改为", "从此",
    )) or _explicit_correction(text)


def _explicit_correction(text: str) -> bool:
    low = text.casefold()
    return any(token in low for token in (
        "wrong", "incorrect", "not true", "that's false", "that is false", "mistake",
        "错误", "错了", "不对", "不正确", "不是这样", "记错", "撤回",
    ))


_TASK_SPECS = re.compile(
    r"帮我(写|做|拟|起草)|写一(封|份|个|条)|这(封|份|篇)(信|邮件|文档|报告|文案)|"
    r"交付可直接|this (email|message|document|reply|post)\b|for this (task|assignment|deliverable)\b",
    re.IGNORECASE,
)
_DURABLE_MARK = re.compile(
    r"以后|每次|总是|一直|不要|别再|从今|从此|一律|习惯|prefer|always|from now on|going forward|"
    r"every time|in general|generally|by default",
    re.IGNORECASE,
)


def _one_off_task_spec(text: str) -> bool:
    """True when the turn is a deliverable spec for THIS task with no durable wording."""
    return bool(_TASK_SPECS.search(text)) and not _DURABLE_MARK.search(text)


def _quote_episode(episodes: dict[str, Episode], episode_id: str, quote: str) -> Episode:
    episode = episodes.get(episode_id)
    if episode is None or episode.role != "user":
        raise StoreError("evidence episode must be a captured user turn")
    text = clean_turn(episode.text)
    if not text or not quote.strip() or quote.strip() not in text:
        raise StoreError("evidence quote must be a verbatim substring of the user turn")
    return episode


def _evidence(episode: Episode, quote: str) -> Evidence:
    return Evidence(episode=episode.id, quote=widen_to_clause(episode.text, quote),
                    date=episode.ts.date(), origin="stated")


class Extractor:
    def __init__(self, store: Store, retriever: Retriever, llm: JSONClient) -> None:
        self.store = store
        self.retriever = retriever
        self.llm = llm

    def _context(self, users: list[Episode], all_episodes: list[Episode]) -> str:
        first = min((all_episodes.index(ep) for ep in users), default=0)
        previous_assistant = next((ep for ep in reversed(all_episodes[:first])
                                   if ep.session == users[0].session and ep.role == "assistant"), None)
        context = []
        if previous_assistant:
            context.append({"role": "assistant", "evidence": False, "text": clean_turn(previous_assistant.text)})
        context.extend({"role": "user", "evidence": True, "episode": ep.id, "text": clean_turn(ep.text)} for ep in users)

        candidates: dict[str, Fact] = {}
        for episode in users:
            result = self.retriever.recall(episode.text, top_k=8, budget=1200)
            for fact in result.hits:
                candidates[fact.id] = fact
        injected_ids = load_injections(self.store, users[0].session)
        for fact in self.store.live_facts():
            if fact.id in injected_ids:
                candidates[fact.id] = fact
        facts = [{"id": f.id, "type": f.type.value, "subject": f.subject, "claim": f.claim,
                  "status": f.status.value, "confidence": f.confidence} for f in candidates.values()]
        return json.dumps({"turns": context, "similar_live_facts": facts}, ensure_ascii=False)

    def _prompt(self, payload: str) -> tuple[str, str]:
        system = (
            "You extract durable facts about one user from captured conversation. "
            "Only user turns marked evidence=true can support a write. Assistant text is context, never evidence. "
            "What counts as a memory: stable identity and background; durable preferences the user wants applied "
            "across future situations (keep the claim SHORT and GENERAL, e.g. 'Gives feedback emails a clear "
            "three-part structure' — never a step-by-step template); ongoing goals and projects; entities "
            "(people/orgs/places/objects) the user will keep caring about; dated events; reusable lessons (case); "
            "tools and workflows the user actually uses; beliefs and values. "
            "What does NOT count: one-off task instructions and deliverable specs (checklists for THIS email, doc, "
            "script or reply), templates, step lists, temporary context, passing opinions, small talk, hypotheticals, "
            "facts about other people, tool output, and suggestions the user did not adopt. "
            "The test for every add: would recalling this a month from now in an UNRELATED task be helpful? "
            "If not, do not store it. When in doubt, return fewer operations. "
            "Ignore tasks, small talk, passing emotions, hypotheticals, facts about other people, tool output, "
            "and suggestions the user did not adopt. Claims must be concise English. Quotes must be exact substrings. "
            "Use only these operations: add, corroborate, supersede, retract, none. "
            "Return only {\"operations\":[...]}. For add/supersede use type, subject, claim, episode, quote, durable; "
            "supersede also requires fact_id of the prior fact. "
            "For corroborate use fact_id, episode, quote. For retract use fact_id, episode, quote. "
            "Use supersede only for an explicit change and retract only for an explicit correction. "
            "Every operation MUST have an 'op' field, separate from the fact 'type' field. "
            "On every add set \"durable\": true only if the one-month test passes; if it does not, drop the operation "
            "entirely instead of writing durable:false. "
            "Example add: {\"operations\":[{\"op\":\"add\",\"type\":\"preference\","
            "\"subject\":\"dietary\",\"claim\":\"Does not eat spicy food\","
            "\"episode\":\"e1\",\"quote\":\"I do not eat spicy food\",\"durable\":true}]}. "
            "Example reject: user turn '写一封请同事反馈初稿的邮件，交付可直接复制的邮件：主题行，第一段说明事项在走到哪一步……' "
            "describes THIS task's deliverable, not a durable preference — return {\"operations\":[]}. "
            "Allowed fact types are exactly: fact, preference, goal, project, entity, event, case, tool, belief. "
            "Never output constraint. Use preference for habits/needs; fact for biography (job, school, location); "
            "entity for people/orgs/places the user cares about; event for dated occurrences; case for reusable "
            "lessons/patterns; tool for tools/workflows the user uses; belief for values including religion or "
            "politics (these are stored and marked sensitive automatically — do not omit them). "
            "Use an empty operations array if nothing qualifies."
        )
        return system, payload

    def _duplicate(self, subject: str, claim: str) -> Fact | None:
        live = self.store.live_facts()
        canonical = _normal(claim)
        for fact in live:
            if fact.type != FactType.CONSTRAINT and fact.subject == subject and _normal(fact.claim) == canonical:
                return fact
        live = [fact for fact in live if fact.type != FactType.CONSTRAINT]
        if not live:
            return None
        by_id = {fact.id: fact for fact in live}
        candidates = live
        if self.retriever.embedder.enabled:
            try:
                vectors = self.retriever.embedder.embed(
                    [f"{subject}\n{claim}"] + [f"{fact.subject}\n{fact.claim}" for fact in live])
            except EmbeddingError as exc:
                raise LLMError("cannot check duplicates: embedding request failed") from exc
            if vectors is None:
                raise LLMError("cannot check duplicates: embeddings unavailable")
            scores = vectors[1:] @ vectors[0]
            candidates = [live[int(i)] for i in scores.argsort()[::-1]
                          if float(scores[int(i)]) >= self.retriever.settings.duplicate_floor]
        if not candidates:
            return None
        result = self.llm.complete_json(
            "Compare personal facts. Return JSON {\"duplicate_id\": id or null}. "
            "Select an id only if the claims assert the same fact. Contradictions, changes, "
            "different values and merely related facts are not duplicates.",
            json.dumps({"claim": claim, "subject": subject,
                        "candidates": [{"id": f.id, "claim": f.claim, "subject": f.subject} for f in candidates[:10]]},
                       ensure_ascii=False),
        )
        if "duplicate_id" not in result or (result["duplicate_id"] is not None and not isinstance(result["duplicate_id"], str)):
            raise LLMError("duplicate decision must contain duplicate_id as a string or null")
        selected = result["duplicate_id"]
        if selected is not None and selected not in {f.id for f in candidates[:10]}:
            raise LLMError("duplicate decision referenced an unknown candidate")
        if selected:
            return by_id[selected]
        return None

    def _apply(self, op: dict[str, Any], episodes: dict[str, Episode], report: ExtractionReport) -> None:
        if not isinstance(op.get("op"), str):
            raise StoreError("extraction operation is missing its op field")
        name = op["op"].lower()
        if name == "none":
            report.skipped += 1
            return
        episode_id = str(op.get("episode", ""))
        quote = str(op.get("quote", "")).strip()
        episode = _quote_episode(episodes, episode_id, quote)
        evidence = _evidence(episode, quote)

        # Gap 8: type merge policy soft gate (ledger still owns truth; no hard delete).
        try:
            from .merge_policy import advise_skip
            type_hint = op.get("type")
            if not type_hint and op.get("fact_id"):
                try:
                    type_hint = self.store.get_fact(str(op.get("fact_id"))).type.value
                except Exception:
                    type_hint = None
            advice = advise_skip(name, type_hint)
            if advice and name in {"corroborate", "supersede"}:
                report.skipped += 1
                return
        except Exception:
            pass

        if name == "add":
            if op.get("durable") is False or _one_off_task_spec(episode.text):
                report.skipped += 1
                return
            try:
                type_ = FactType(str(op.get("type", "fact")))
            except ValueError as exc:
                raise StoreError("unknown fact type") from exc
            if type_ == FactType.CONSTRAINT:
                type_ = FactType.PREFERENCE
            subject = str(op.get("subject", "")).strip()
            claim = str(op.get("claim", "")).strip()
            if not subject or not claim:
                raise StoreError("add requires subject and claim")
            duplicate = self._duplicate(subject.casefold(), claim)
            if duplicate:
                if self.store.corroborate(duplicate.id, evidence):
                    report.corroborated += 1
                else:
                    report.skipped += 1
                return
            from .overview import looks_sensitive
            sensitive = type_ == FactType.BELIEF or looks_sensitive(f"{subject} {claim}", type_)
            fact = self.store.draft_fact(type=type_, subject=subject, claim=claim,
                                         evidence=[evidence], source_agent=episode.agent,
                                         source_session=episode.session, sensitive=sensitive,
                                         peer=getattr(episode, "project", None))
            stored = self.store.write_fact(fact)
            self.retriever.sync_fact(stored)
            report.facts_added += 1
        elif name == "corroborate":
            fact = self.store.get_fact(str(op.get("fact_id", "")))
            if not fact.is_live:
                raise StoreError("cannot corroborate a non-live fact")
            if fact.type == FactType.CONSTRAINT:
                raise StoreError("only the user may change a hard constraint")
            if self.store.corroborate(fact.id, evidence):
                report.corroborated += 1
        elif name == "supersede":
            if not _explicit_change(episode.text):
                raise StoreError("supersede requires an explicit change in the user turn")
            old = self.store.get_fact(str(op.get("fact_id", "")))
            if not old.is_live:
                completed = [f for f in self.store.all_facts() if f.supersedes == old.id
                             and any(e.episode == episode.id and e.quote == quote for e in f.evidence)
                             and _normal(f.claim) == _normal(str(op.get("claim", "")))]
                if completed:
                    self.retriever.sync_fact(completed[0])
                    self.retriever.sync_fact(old)
                    report.skipped += 1
                    return
                raise StoreError("cannot supersede a non-live fact")
            if old.type == FactType.CONSTRAINT:
                raise StoreError("only the user may change a hard constraint")
            type_ = FactType(str(op.get("type", old.type.value)))
            if type_ == FactType.CONSTRAINT:
                type_ = FactType.PREFERENCE
            claim = str(op.get("claim", "")).strip()
            if not claim:
                raise StoreError("supersede requires replacement claim")
            existing = next((f for f in self.store.all_facts() if f.supersedes == old.id
                             and any(e.episode == episode.id and e.quote == quote for e in f.evidence)
                             and _normal(f.claim) == _normal(claim)), None)
            if existing:
                stored = existing
            else:
                new = self.store.draft_fact(type=type_, subject=str(op.get("subject", old.subject)),
                                            claim=claim, evidence=[evidence], source_agent=episode.agent,
                                            source_session=episode.session, supersedes=old.id)
                stored = self.store.write_fact(new)
            self.store.supersede(old.id, stored.id)
            self.retriever.sync_fact(stored)
            self.retriever.sync_fact(self.store.get_fact(old.id))
            report.superseded += 1
        elif name == "retract":
            if not _explicit_correction(episode.text):
                raise StoreError("retract requires an explicit correction in the user turn")
            fact = self.store.get_fact(str(op.get("fact_id", "")))
            if not fact.is_live:
                if fact.retracted_reason == quote:
                    self.retriever.sync_fact(fact)
                    report.skipped += 1
                    return
                raise StoreError("cannot retract a non-live fact")
            if fact.type == FactType.CONSTRAINT:
                raise StoreError("only the user may change a hard constraint")
            self.store.retract(fact.id, quote)
            self.retriever.sync_fact(self.store.get_fact(fact.id))
            report.retracted += 1
        else:
            raise StoreError(f"unsupported extraction operation: {name}")
        report.operations += 1

    def extract_session(self, session: str, episodes: list[Episode], state: dict[str, Any], report: ExtractionReport) -> None:
        processed = set(state["processed"].get(session, []))
        pending = [ep for ep in episodes if ep.session == session and ep.id not in processed]
        if not pending:
            return
        users = [ep for ep in pending if ep.role == "user" and clean_turn(ep.text)]
        if not users:
            state["processed"][session] = sorted(processed | {ep.id for ep in pending})
            _save_state(self.store, state)
            return
        for start in range(0, len(users), MAX_BATCH):
            batch = users[start:start + MAX_BATCH]
            payload = self._context(batch, episodes)
            system, user = self._prompt(payload)
            response = self.llm.complete_json(system, user)
            operations = response.get("operations")
            if not isinstance(operations, list):
                raise LLMError("extraction response must contain operations")
            episode_map = {ep.id: ep for ep in batch}
            for raw in operations:
                if not isinstance(raw, dict):
                    raise StoreError("each extraction operation must be an object")
                self._apply(raw, episode_map, report)
                self.retriever.scan_conflicts()
            processed.update(ep.id for ep in batch)
            state["processed"][session] = sorted(processed)
            _save_state(self.store, state)
            report.user_turns += len(batch)
        report.episodes_seen += len(pending)
        report.sessions += 1

    def run(self, session: str | None = None) -> ExtractionReport:
        report = ExtractionReport(errors=[])
        episodes = list(self.store.iter_episodes())
        sessions = sorted({ep.session for ep in episodes if session is None or ep.session == session})
        with file_lock(self.store.home / ".state" / "extraction.lock"):
            state = _load_state(self.store)
            for current in sessions:
                try:
                    self.extract_session(current, episodes, state, report)
                except (LLMError, StoreError, ValueError) as exc:
                    report.errors.append(f"{current}: {exc}")
        return report
