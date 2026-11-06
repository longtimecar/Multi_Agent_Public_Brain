"""Append-only storage.

Two kinds of file are canonical:
  facts/<id>.md       one claim per file, YAML front-matter + body
  episodes/YYYY-MM-DD.jsonl

A third is the event ledger (`ledger.jsonl`). Lifecycle changes are APPENDED as
events rather than written in place, so the fact files stay immutable and the
current state is always a fold over the ledger. This is what makes memory
replayable: improve the extractor, re-fold, keep the history.
"""

from __future__ import annotations

import json
import os
import random
import re
import string
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

import yaml
from pydantic import ValidationError

from .confidence import compute_confidence
from .locking import file_lock
from .models import ID_RE, Episode, Evidence, Fact, FactType, Status

FM_RE = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n?(.*)\Z", re.DOTALL)
_SLUG_RE = re.compile(r"[^a-z0-9]+")


class StoreError(Exception):
    """Raised for invalid writes. The MCP layer turns this into a tool error."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _suffix(n: int = 6) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))


def new_fact_id(d: date | None = None) -> str:
    return f"f_{(d or date.today()).strftime('%Y%m%d')}_{_suffix()}"


def new_episode_id(ts: datetime, n: int) -> str:
    return f"ep_{ts.strftime('%Y%m%d_%H%M%S')}_{n}"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


class Store:
    def __init__(self, home: Path) -> None:
        self.home = Path(home)
        self.facts_dir = self.home / "facts"
        self.episodes_dir = self.home / "episodes"
        self.ledger_path = self.home / "ledger.jsonl"
        self.ledger_lock_path = self.home / ".state" / "ledger.lock"
        self.facts_dir.mkdir(parents=True, exist_ok=True)
        self.episodes_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- facts

    def fact_path(self, fact_id: str) -> Path:
        if not isinstance(fact_id, str) or not ID_RE.fullmatch(fact_id):
            raise StoreError("invalid fact id")
        return self.facts_dir / f"{fact_id}.md"

    @staticmethod
    def render(fact: Fact) -> str:
        meta = fact.model_dump(mode="json", exclude={"claim"})
        body = f"# {fact.claim}\n"
        return (
            "---\n"
            + yaml.safe_dump(meta, sort_keys=False, allow_unicode=True, width=100)
            + "---\n"
            + body
        )

    @staticmethod
    def parse(text: str) -> Fact:
        m = FM_RE.match(text)
        if not m:
            raise StoreError("fact file missing YAML front-matter")
        meta = yaml.safe_load(m.group(1)) or {}
        body = m.group(2).strip()
        heading = body.splitlines()[0].lstrip("# ").strip() if body else ""
        if "claim" not in meta and heading:
            meta["claim"] = heading
        try:
            return Fact(**meta)
        except ValidationError as exc:  # surface as a clean store error
            raise StoreError(f"invalid fact: {exc.errors()[0]['msg']}") from exc

    def write_fact(self, fact: Fact) -> Fact:
        """Validate, score, then persist. Confidence is always recomputed here."""
        fact = fact.model_copy(update={"confidence": compute_confidence(fact)})
        path = self.fact_path(fact.id)
        if path.exists():
            raise StoreError(f"fact {fact.id} already exists; memory is append-only")
        _atomic_write(path, self.render(fact))
        self._append_ledger({"op": "create", "id": fact.id, "ts": _now().isoformat(), "confidence": fact.confidence})
        return fact

    def get_fact(self, fact_id: str) -> Fact:
        path = self.fact_path(fact_id)
        if not path.exists():
            raise StoreError(f"no such fact: {fact_id}")
        return self.apply_ledger(self.parse(path.read_text(encoding="utf-8")))

    def all_facts(self) -> list[Fact]:
        state = self._ledger_state()
        out: list[Fact] = []
        for path in sorted(self.facts_dir.glob("f_*.md")):
            try:
                fact = self.parse(path.read_text(encoding="utf-8"))
            except (StoreError, ValidationError):
                continue  # a malformed file must never break the whole store
            out.append(self._apply_events(fact, state.get(fact.id, [])))
        return out

    def live_facts(self) -> list[Fact]:
        return [f for f in self.all_facts() if f.is_live]

    # -------------------------------------------------------------- ledger

    def _append_ledger(self, event: dict) -> None:
        with file_lock(self.ledger_lock_path):
            self._append_ledger_locked(event)

    def _append_ledger_locked(self, event: dict) -> None:
        with open(self.ledger_path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _ledger_state(self) -> dict[str, list[dict]]:
        if not self.ledger_path.exists():
            return {}
        state: dict[str, list[dict]] = {}
        with open(self.ledger_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("op") == "create":
                    state.setdefault(event["id"], [])
                elif "id" in event:
                    state.setdefault(event["id"], []).append(event)
        return state

    @staticmethod
    def _apply_events(fact: Fact, events: Iterable[dict]) -> Fact:
        status = fact.status
        valid_to = fact.valid_to
        reason = fact.retracted_reason
        closed_at = fact.closed_at
        evidence = list(fact.evidence)
        seen = {item.episode for item in evidence}
        for ev in events:
            op = ev.get("op")
            if op == "corroborate":
                try:
                    item = Evidence.model_validate(ev.get("evidence"))
                except ValidationError:
                    continue
                if item.episode not in seen:
                    evidence.append(item)
                    seen.add(item.episode)
            elif op == "retract":
                status = Status.RETRACTED
                reason = ev.get("reason")
            elif op == "contest":
                if status not in (Status.RETRACTED, Status.SUPERSEDED):
                    status = Status.CONTESTED
            elif op == "resolve":
                if status not in (Status.RETRACTED, Status.SUPERSEDED):
                    status = Status.ACTIVE if ev.get("keep") else Status.RETRACTED
            elif op == "supersede":
                if status != Status.RETRACTED:
                    status = Status.SUPERSEDED
                raw = ev.get("valid_to")
                if raw:
                    valid_to = date.fromisoformat(raw)
            elif op == "close":
                raw = ev.get("closed_at")
                try:
                    closed_at = date.fromisoformat(raw) if raw else date.today()
                except ValueError:
                    continue
            elif op == "reopen":
                closed_at = None
        folded = fact.model_copy(update={"status": status, "valid_to": valid_to, "retracted_reason": reason,
                                         "evidence": evidence, "closed_at": closed_at})
        return folded.model_copy(update={"confidence": compute_confidence(folded)})

    def apply_ledger(self, fact: Fact) -> Fact:
        return self._apply_events(fact, self._ledger_state().get(fact.id, []))

    # ----------------------------------------------------------- lifecycle

    def corroborate(self, fact_id: str, evidence: Evidence) -> bool:
        """Add one distinct episode to a live fact without editing its file."""
        with file_lock(self.ledger_lock_path):
            fact = self.get_fact(fact_id)
            if not fact.is_live:
                raise StoreError("cannot corroborate a non-live fact")
            if any(item.episode == evidence.episode for item in fact.evidence):
                return False
            self._append_ledger_locked({"op": "corroborate", "id": fact_id,
                                        "evidence": evidence.model_dump(mode="json"), "ts": _now().isoformat()})
            return True

    def supersede(self, old_id: str, new_id: str, when: date | None = None) -> None:
        when = when or date.today()
        if not self.fact_path(old_id).exists() or not self.fact_path(new_id).exists():
            raise StoreError("supersede requires both facts to exist")
        self._append_ledger({"op": "supersede", "id": old_id, "by": new_id, "valid_to": when.isoformat(), "ts": _now().isoformat()})
        self._append_ledger({"op": "link", "id": new_id, "supersedes": old_id, "ts": _now().isoformat()})

    def retract(self, fact_id: str, reason: str) -> None:
        if not self.fact_path(fact_id).exists():
            raise StoreError(f"no such fact: {fact_id}")
        self._append_ledger({"op": "retract", "id": fact_id, "reason": reason, "ts": _now().isoformat()})

    def close_task(self, fact_id: str, when: date | None = None) -> None:
        """Mark a task finished. Append-only: the fact file is never rewritten.

        Closing is an owner decision, never a calendar rule — an unfinished task
        must not retire just because nobody mentioned it for a while.
        """
        fact = self.get_fact(fact_id)
        if fact.type != FactType.TASK:
            raise StoreError("only task memories can be closed")
        if not fact.is_live:
            raise StoreError("cannot close a non-live fact")
        self._append_ledger({"op": "close", "id": fact_id,
                             "closed_at": (when or date.today()).isoformat(), "ts": _now().isoformat()})

    def reopen_task(self, fact_id: str) -> None:
        """Undo a close, for when the work turns out not to be finished."""
        fact = self.get_fact(fact_id)
        if fact.type != FactType.TASK:
            raise StoreError("only task memories can be reopened")
        self._append_ledger({"op": "reopen", "id": fact_id, "ts": _now().isoformat()})

    def contest(self, fact_ids: list[str], note: str) -> None:
        for fid in fact_ids:
            self._append_ledger({"op": "contest", "id": fid, "note": note, "ts": _now().isoformat()})

    def contest_pairs(self, pairs: list[tuple[str, str, str]]) -> int:
        """Flag (a, b, reason) pairs as contested. Idempotent.

        Detection lives in conflict.py; this only records the outcome. Nothing
        here picks a winner — that is the human's call, via `resolve`.
        """
        existing = {e.get("note") for fid in self._ledger_state().values() for e in fid if e.get("op") == "contest"}
        marked = 0
        for a, b, reason in pairs:
            note = f"{min(a, b)}~{max(a, b)}: {reason}"
            if note in existing:
                continue
            for fid in (a, b):
                self._append_ledger({"op": "contest", "id": fid, "note": note, "ts": _now().isoformat()})
            marked += 1
        return marked

    def resolve(self, keep: str, drop: str) -> None:
        """Human-decided resolution. The only path from contested to settled."""
        self._append_ledger({"op": "resolve", "id": keep, "keep": True, "dropped": drop, "ts": _now().isoformat()})
        self._append_ledger({"op": "retract", "id": drop, "reason": f"resolved in favour of {keep}", "ts": _now().isoformat()})

    # ------------------------------------------------------------ episodes

    def append_event(self, event: dict) -> None:
        """Append a lifecycle event for something other than a fact (archive, import)."""
        self._append_ledger({**event, "ts": _now().isoformat()})

    def ledger_events(self, op: str) -> list[dict]:
        """All ledger events of one kind, in order."""
        return [e for events in self._ledger_state().values() for e in events if e.get("op") == op]

    def append_episodes(self, episodes: list[Episode]) -> list[Episode]:
        """Append captured turns. Never interprets, never blocks on a model."""
        if not episodes:
            return []
        by_day: dict[str, list[Episode]] = {}
        for ep in episodes:
            # Shard by the operator's calendar day, not UTC. A turn spoken at
            # 07:00 in UTC+8 is 23:00 UTC the previous day; filing it under the
            # UTC day makes every "昨晚" window look in the wrong file.
            by_day.setdefault(ep.local_date.isoformat(), []).append(ep)
        written: list[Episode] = []
        for day, batch in by_day.items():
            path = self.episodes_dir / f"{day}.jsonl"
            existing = 0
            if path.exists():
                with open(path, encoding="utf-8") as fh:
                    existing = sum(1 for line in fh if line.strip())
            with open(path, "a", encoding="utf-8", newline="\n") as fh:
                for i, ep in enumerate(batch):
                    stamped = ep.model_copy(update={"id": ep.id or new_episode_id(ep.ts, existing + i)})
                    fh.write(stamped.model_dump_json() + "\n")
                    written.append(stamped)
        return written

    def iter_episodes(self, since: date | None = None) -> Iterator[Episode]:
        for path in sorted(self.episodes_dir.glob("*.jsonl")):
            if since and path.stem < since.isoformat():
                continue
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield Episode(**json.loads(line))
                    except (json.JSONDecodeError, ValidationError):
                        continue

    # ------------------------------------------------------------ drafting

    def draft_fact(
        self,
        *,
        type: FactType,
        subject: str,
        claim: str,
        evidence: list[Evidence],
        source_agent: str = "unknown",
        source_session: str | None = None,
        supersedes: str | None = None,
        peer: str | None = None,
        sensitive: bool | None = None,
        uri: str | None = None,
        abstract: str | None = None,
        created_at: "datetime | None" = None,
    ) -> Fact:
        """Build a Fact with a fresh id; confidence is filled in by write_fact.

        Scheme A fields (uri/abstract/peer/sensitive) are optional; Fact validators
        fill uri/abstract and may auto-mark sensitive for belief/religion/politics.
        """
        from .overview import looks_sensitive, make_abstract
        from .uri import build_uri

        slug = _SLUG_RE.sub("-", subject.lower()).strip("-") or "general"
        if sensitive is None:
            sensitive = type == FactType.BELIEF or looks_sensitive(f"{slug} {claim}", type)
        fact = Fact(
            id=new_fact_id(),
            type=type,
            subject=slug,
            claim=claim.strip(),
            evidence=evidence,
            confidence=0.0,
            # A merge continues an existing memory; stamping "now" would make an
            # old statement look like it was first recorded today.
            **({"created_at": created_at} if created_at is not None else {}),
            valid_from=max(e.date for e in evidence),
            supersedes=supersedes,
            source={"agent": source_agent, "session": source_session},
            peer=peer,
            sensitive=bool(sensitive),
            uri=uri or build_uri(type.value, slug, peer),
            abstract=abstract or make_abstract(claim.strip()),
        )
        return fact

    def migrate_scheme_a(self) -> dict[str, int]:
        """Backfill uri/abstract/sensitive onto existing fact files (additive rewrite).

        Checks raw front-matter keys on disk (validators fill in-memory defaults),
        then rewrites files that lack Scheme A fields. Ledger is not touched.
        """
        from .overview import looks_sensitive, make_abstract
        from .uri import build_uri

        updated = 0
        scanned = 0
        for path in sorted(self.facts_dir.glob("f_*.md")):
            scanned += 1
            try:
                text = path.read_text(encoding="utf-8")
                m = FM_RE.match(text)
                if not m:
                    continue
                meta = yaml.safe_load(m.group(1)) or {}
                fact = self.parse(text)
            except (StoreError, ValidationError, yaml.YAMLError):
                continue
            disk_missing = (
                "uri" not in meta
                or "abstract" not in meta
                or ("sensitive" not in meta and (
                    fact.type == FactType.BELIEF
                    or looks_sensitive(f"{fact.subject} {fact.claim}", fact.type)
                ))
            )
            if not disk_missing:
                continue
            updates: dict = {
                "uri": meta.get("uri") or fact.uri or build_uri(fact.type.value, fact.subject, fact.peer),
                "abstract": meta.get("abstract") or fact.abstract or make_abstract(fact.claim),
                "sensitive": bool(meta.get("sensitive", fact.sensitive)),
            }
            if not updates["sensitive"] and (
                fact.type == FactType.BELIEF
                or looks_sensitive(f"{fact.subject} {fact.claim}", fact.type)
            ):
                updates["sensitive"] = True
            migrated = fact.model_copy(update=updates)
            _atomic_write(path, self.render(migrated))
            updated += 1
        return {"scanned": scanned, "updated": updated}

