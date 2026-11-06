"""Auto-commit policy inside membrain (Scheme A / P3).

Triggers extraction when either:
  * estimated session tokens since last commit >= threshold (default 8000), or
  * idle minutes since last user episode >= idle_minutes (default 30).

LLM quota use is intentional and allowed. This module never touches protected
agent hook configs — it only records state and starts the existing extractor.
"""

from __future__ import annotations

import hashlib
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
    ready: list[str] = []
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
                ready.append(session)
        save_state(store, state)
    return ready


def _session_fingerprint(store: Store, session: str) -> str:
    """Content fingerprint of a session's captured turns.

    Episode ids are minted at capture time, so the same batch always hashes to
    the same value and a batch that gained turns hashes differently. This is
    what makes a repeated commit of an already-committed batch a no-op.
    """
    ids = sorted(ep.id for ep in store.iter_episodes() if ep.session == session)
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def pending_sessions(store: Store) -> list[str]:
    """Return sessions flagged for commit. Read-only: nothing is cleared.

    Clearing before a successful write is how sessions used to disappear
    forever, so callers now claim nothing until the archive write succeeded.
    """
    with file_lock(store.home / ".state" / "auto_commit.lock"):
        return _pending_locked(load_state(store))


def _pending_locked(state: dict[str, Any]) -> list[str]:
    return [
        session
        for session, entry in state.get("sessions", {}).items()
        if isinstance(entry, dict) and entry.get("pending")
    ]


def take_pending(store: Store) -> list[str]:
    """Compatibility alias for :func:`pending_sessions` (no longer clears).

    It used to reset ``pending``/``tokens`` and stamp ``last_commit`` *before*
    anything was written, so an exception in the archive step dropped the
    session for good. Only :func:`mark_committed` clears a session, and only
    after its archive write returned.
    """
    return pending_sessions(store)


def _record_commit_failure(store: Store, session: str, message: str) -> None:
    """Keep the failure visible and the session queued for the next run."""
    with file_lock(store.home / ".state" / "auto_commit.lock"):
        state = load_state(store)
        entry = _session_entry(state, session)
        entry["pending"] = True
        entry["last_error"] = message
        entry["last_error_at"] = _now().isoformat()
        save_state(store, state)


def mark_committed(store: Store, items: list[dict[str, Any]]) -> None:
    """Drop sessions from the queue after their archive write succeeded.

    ``items`` carries the token snapshot and fingerprint taken before the
    write, so tokens accumulated while the write was in flight are kept.
    """
    if not items:
        return
    with file_lock(store.home / ".state" / "auto_commit.lock"):
        state = load_state(store)
        now = _now().isoformat()
        for item in items:
            entry = _session_entry(state, str(item["session"]))
            consumed = int(item.get("tokens") or 0)
            entry["tokens"] = max(0, int(entry.get("tokens") or 0) - consumed)
            entry["pending"] = False
            entry["last_commit"] = now
            entry["committed_fingerprint"] = item["fingerprint"]
            entry.pop("last_error", None)
            entry.pop("last_error_at", None)
        save_state(store, state)


def _pending_snapshot(store: Store) -> dict[str, dict[str, Any]]:
    """Queue contents plus per-session tokens/fingerprint at this instant."""
    with file_lock(store.home / ".state" / "auto_commit.lock"):
        state = load_state(store)
        sessions = _pending_locked(state)
        entries = {s: dict(state.get("sessions", {}).get(s) or {}) for s in sessions}
    return {
        session: {
            "tokens": int(entries[session].get("tokens") or 0),
            "committed_fingerprint": entries[session].get("committed_fingerprint"),
            "fingerprint": _session_fingerprint(store, session),
        }
        for session in sessions
    }


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
    """Commit pending sessions in write-then-clear order.

    Phase 1 (the archive write) runs first. A session leaves the queue only
    after its write returned; on any exception it stays pending with a readable
    ``last_error`` so the next run retries it. One failing session never stops
    the rest of the batch. A batch whose fingerprint matches the last committed
    one is reported as ``already_committed`` instead of being written twice.

    Phase2 (summary/extract/diff) is started by the caller (hooks/CLI).
    """
    from . import session_archive

    snapshot = _pending_snapshot(store)
    sessions = list(snapshot)
    phase1: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    committed: list[dict[str, Any]] = []

    for session, snap in snapshot.items():
        fingerprint = str(snap["fingerprint"])
        if snap.get("committed_fingerprint") == fingerprint:
            phase1.append({
                "status": "already_committed",
                "task_id": None,
                "archived": False,
                "session": session,
                "reason": "duplicate_batch",
            })
            committed.append({"session": session, "tokens": snap["tokens"], "fingerprint": fingerprint})
            continue
        try:
            result = session_archive.phase1_commit(store, session, reason=reason)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            errors.append({"session": session, "error": message})
            try:
                _record_commit_failure(store, session, message)
            except Exception as state_exc:
                errors[-1]["state_error"] = f"{type(state_exc).__name__}: {state_exc}"
            continue
        phase1.append(result)
        committed.append({"session": session, "tokens": snap["tokens"], "fingerprint": fingerprint})

    mark_committed(store, committed)
    return {
        "sessions": sessions,
        "phase1": phase1,
        "errors": errors,
        "committed": [item["session"] for item in committed],
    }
