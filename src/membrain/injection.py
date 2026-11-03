"""Per-session injection history and token accounting."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .models import Fact
from .store import Store, _atomic_write


def _path(store: Store, session: str) -> Path:
    digest = hashlib.sha256(session.encode("utf-8")).hexdigest()
    return store.home / ".state" / "injections" / f"{digest}.json"


def fingerprint(fact: Fact) -> str:
    body = json.dumps(fact.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def load(store: Store, session: str) -> dict[str, str]:
    path = _path(store, session)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in data.items()):
        raise ValueError("invalid injection history")
    return data


def save(store: Store, session: str, state: dict[str, str], event: str, tokens: int, ids: list[str]) -> None:
    _atomic_write(_path(store, session), json.dumps(state, ensure_ascii=False, sort_keys=True))
    log = store.home / ".state" / "injection-log.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "session": session,
                                 "event": event, "tokens": tokens, "fact_ids": ids}, ensure_ascii=False) + "\n")
