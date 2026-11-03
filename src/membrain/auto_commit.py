"""Auto-commit policy inside membrain (Scheme A / P3).

Triggers extraction when either:
  * estimated session tokens since last commit >= threshold (default 8000), or
  * idle minutes since last user episode >= idle_minutes (default 30).

LLM quota use is intentional and allowed. This module never touches protected
agent hook configs — it only records state and starts the existing extractor.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .budget import estimate_tokens
from .locking import file_lock
from .models import Episode
from .store import Store, _atomic_write

STATE_FILE = ".state/auto_commit.json"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _state_path(store: Store) -> Path:
    path = store.home / STATE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_state(store: Store) -> dict[str, Any]:
    path = _state_path(store)
    if not path.exists():
        return {"sessions": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"sessions": {}}
    if not isinstance(data, dict):
        return {"sessions": {}}
    sessions = data.get("sessions")
    if not isinstance(sessions, dict):
        data["sessions"] = {}
    return data


def save_state(store: Store, state: dict[str, Any]) -> None:
    _atomic_write(_state_path(store), json.dumps(state, ensure_ascii=False, indent=2))


def _session_entry(state: dict[str, Any], session: str) -> dict[str, Any]:
    sessions = state.setdefault("sessions", {})
    entry = sessions.get(session)
    if not isinstance(entry, dict):
        entry = {"tokens": 0, "last_activity": None, "last_commit": None, "pending": False}
        sessions[session] = entry
    return entry


def note_activity(store: Store, session: str, text: str, *, threshold: int, idle_minutes: int) -> dict[str, Any]:
    """Accumulate tokens for a session and mark pending when threshold/idle hit.

    Call on each captured user turn. Does not start the extractor itself.
    """
    with file_lock(store.home / ".state" / "auto_commit.lock"):
        state = load_state(store)
        entry = _session_entry(state, session)
        now = _now()
        cost = estimate_tokens(text or "")
        entry["tokens"] = int(entry.get("tokens") or 0) + cost
        entry["last_activity"] = now.isoformat()
        reason = None
        if entry["tokens"] >= threshold:
            entry["pending"] = True
            reason = "token_threshold"
        save_state(store, state)
        return {"session": session, "tokens": entry["tokens"], "pending": bool(entry.get("pending")), "reason": reason}


def check_idle(store: Store, *, threshold: int, idle_minutes: int) -> list[str]:
    """Mark sessions pending when idle long enough with uncommitted tokens."""
    pending_sessions: list[str] = []
    with file_lock(store.home / ".state" / "auto_commit.lock"):
        state = load_state(store)
        now = _now()
        for session, entry in list(state.get("sessions", {}).items()):
            if not isinstance(entry, dict):
                continue
            tokens = int(entry.get("tokens") or 0)
            if tokens <= 0:
                continue
            last_raw = entry.get("last_activity")
            if not last_raw:
                continue
            try:
                last = datetime.fromisoformat(str(last_raw).replace("Z", "+00:00"))
                if last.tzinfo is None:
                    last = last.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            idle = (now - last).total_seconds() / 60.0
            if tokens >= threshold or idle >= idle_minutes:
                entry["pending"] = True
                pending_sessions.append(session)
        save_state(store, state)
    return pending_sessions


def take_pending(store: Store) -> list[str]:
    """Return and clear sessions flagged for commit."""
    with file_lock(store.home / ".state" / "auto_commit.lock"):
        state = load_state(store)
        ready: list[str] = []
        for session, entry in list(state.get("sessions", {}).items()):
            if isinstance(entry, dict) and entry.get("pending"):
                ready.append(session)
                entry["pending"] = False
                entry["tokens"] = 0
                entry["last_commit"] = _now().isoformat()
        save_state(store, state)
        return ready


def should_commit_session(store: Store, session: str, *, threshold: int, idle_minutes: int) -> tuple[bool, str | None]:
    """Inspect one session without clearing pending."""
    state = load_state(store)
    entry = state.get("sessions", {}).get(session) or {}
    tokens = int(entry.get("tokens") or 0)
    if tokens >= threshold:
        return True, "token_threshold"
    last_raw = entry.get("last_activity")
    if last_raw and tokens > 0:
        try:
            last = datetime.fromisoformat(str(last_raw).replace("Z", "+00:00"))
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            if (_now() - last).total_seconds() / 60.0 >= idle_minutes:
                return True, "idle"
        except ValueError:
            pass
    if entry.get("pending"):
        return True, "pending"
    return False, None


def estimate_episode_tokens(episodes: list[Episode]) -> int:
    return sum(estimate_tokens(ep.text) for ep in episodes)

def commit_pending_with_archive(store: Store, *, reason: str = "auto_commit") -> dict[str, Any]:
    """Take pending sessions, run Phase1 archive cut, return task ids.

    Phase2 (summary/extract/diff) is started by the caller (hooks/CLI).
    """
    from . import session_archive

    sessions = take_pending(store)
    phase1 = session_archive.commit_sessions(store, sessions, reason=reason) if sessions else []
    return {"sessions": sessions, "phase1": phase1}
