"""Two-phase session archive + memory_diff (OpenViking gap 1).

Phase 1 (sync, request path): allocate archive_NNN, write messages.jsonl + meta,
return task_id. Phase 2 (async/background): write .abstract.md / .overview.md,
run existing extract, write memory_diff.json (even when empty), mark .done.

Deletes in the diff map to ledger retract/supersede — never hard-delete fact files.
Sensitive facts are not dumped into SessionStart by this module.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from .locking import file_lock
from .models import Episode
from .overview import make_abstract
from .store import Store, _atomic_write

SESSIONS_ROOT = "sessions"
ARCHIVE_RE = re.compile(r"^archive_(\d{3,})$")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_session(session: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", (session or "unknown").strip()) or "unknown"
    return cleaned[:120]


def sessions_home(store: Store) -> Path:
    path = store.home / SESSIONS_ROOT
    path.mkdir(parents=True, exist_ok=True)
    return path


def session_dir(store: Store, session: str) -> Path:
    path = sessions_home(store) / _safe_session(session)
    path.mkdir(parents=True, exist_ok=True)
    return path


def history_dir(store: Store, session: str) -> Path:
    path = session_dir(store, session) / "history"
    path.mkdir(parents=True, exist_ok=True)
    return path


def archive_uri(session: str, archive_name: str) -> str:
    return f"mb://user/sessions/{_safe_session(session)}/history/{archive_name}"


def empty_memory_diff(archive_uri_str: str, *, extracted_at: str | None = None) -> dict[str, Any]:
    return {
        "archive_uri": archive_uri_str,
        "extracted_at": extracted_at or _now().isoformat(),
        "operations": {"adds": [], "updates": [], "deletes": []},
        "skipped_operations": [],
        "summary": {
            "total_adds": 0,
            "total_updates": 0,
            "total_deletes": 0,
            "total_skipped": 0,
            "facts_added": 0,
            "corroborated": 0,
            "superseded": 0,
            "retracted": 0,
            "skipped": 0,
            "errors": [],
        },
    }


def _next_archive_name(history: Path) -> str:
    nums = []
    for child in history.iterdir() if history.exists() else []:
        if child.is_dir():
            m = ARCHIVE_RE.match(child.name)
            if m:
                nums.append(int(m.group(1)))
    n = (max(nums) + 1) if nums else 1
    return f"archive_{n:03d}"


def _already_archived_episode_ids(store: Store, session: str) -> set[str]:
    seen: set[str] = set()
    hist = history_dir(store, session)
    for child in sorted(hist.glob("archive_*")):
        meta_path = child / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for eid in meta.get("episode_ids") or []:
            if isinstance(eid, str):
                seen.add(eid)
    return seen


def _episodes_for_session(store: Store, session: str) -> list[Episode]:
    return [ep for ep in store.iter_episodes() if ep.session == session]


def phase1_commit(
    store: Store,
    session: str,
    *,
    reason: str = "auto_commit",
    agent: str | None = None,
) -> dict[str, Any]:
    """Sync cut: write messages.jsonl + meta + task_id. No LLM."""
    safe = _safe_session(session)
    with file_lock(store.home / ".state" / "session_archive.lock"):
        hist = history_dir(store, session)
        already = _already_archived_episode_ids(store, session)
        episodes = [ep for ep in _episodes_for_session(store, session) if ep.id not in already]
        if not episodes:
            return {
                "status": "skipped",
                "task_id": None,
                "archived": False,
                "session": safe,
                "reason": "no_new_messages",
                "archive_path": None,
                "archive_uri": None,
            }
        name = _next_archive_name(hist)
        archive_path = hist / name
        archive_path.mkdir(parents=True, exist_ok=False)
        task_id = str(uuid.uuid4())
        uri = archive_uri(session, name)
        messages_path = archive_path / "messages.jsonl"
        with open(messages_path, "w", encoding="utf-8", newline="\n") as fh:
            for ep in episodes:
                row = {
                    "id": ep.id,
                    "ts": ep.ts.isoformat() if hasattr(ep.ts, "isoformat") else str(ep.ts),
                    "session": ep.session,
                    "agent": ep.agent,
                    "role": ep.role,
                    "text": ep.text,
                    "project": ep.project,
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        meta = {
            "task_id": task_id,
            "session": safe,
            "original_session": session,
            "archive_name": name,
            "archive_uri": uri,
            "reason": reason,
            "agent": agent,
            "phase1_at": _now().isoformat(),
            "phase1_status": "ready",
            "episode_ids": [ep.id for ep in episodes],
            "message_count": len(episodes),
            "phase2_status": "pending",
        }
        _atomic_write(archive_path / "meta.json", json.dumps(meta, ensure_ascii=False, indent=2))
        # queue marker for background phase2
        queue_path = store.home / ".state" / "session_archive_queue.jsonl"
        queue_path.parent.mkdir(parents=True, exist_ok=True)
        with open(queue_path, "a", encoding="utf-8", newline="\n") as qh:
            qh.write(json.dumps({
                "task_id": task_id,
                "session": session,
                "archive_path": str(archive_path),
                "archive_uri": uri,
                "enqueued_at": _now().isoformat(),
            }, ensure_ascii=False) + "\n")
        return {
            "status": "accepted",
            "task_id": task_id,
            "archived": True,
            "session": safe,
            "reason": reason,
            "archive_path": str(archive_path),
            "archive_uri": uri,
            "message_count": len(episodes),
        }


def _heuristic_summary(messages: list[dict[str, Any]]) -> tuple[str, str]:
    """Fail-open abstract/overview without LLM."""
    users = [m for m in messages if m.get("role") == "user"]
    texts = [(" ".join(str(m.get("text") or "").split())) for m in users if str(m.get("text") or "").strip()]
    if not texts:
        texts = [(" ".join(str(m.get("text") or "").split())) for m in messages if str(m.get("text") or "").strip()]
    if not texts:
        abstract = "Empty session archive"
        overview = "# Session overview\n\nNo user turns captured.\n"
        return abstract, overview
    first = make_abstract(texts[0], limit=160)
    abstract = f"Session ({len(messages)} msgs / {len(users)} user): {first}"
    bullets = "\n".join(f"- {make_abstract(t, limit=120)}" for t in texts[:12])
    overview = (
        "# Session overview\n\n"
        f"**One-line**: {abstract}\n\n"
        "## User turns (sample)\n"
        f"{bullets}\n"
    )
    return abstract, overview


def load_archive_messages(archive_path: Path) -> list[dict[str, Any]]:
    path = Path(archive_path) / "messages.jsonl"
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def build_memory_diff(
    archive_uri_str: str,
    *,
    report: Any | None = None,
    ledger_events: Iterable[dict[str, Any]] | None = None,
    facts_by_id: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Build OV-shaped memory_diff from ExtractionReport + optional ledger slice.

    deletes[] means ledger retract/supersede, never filesystem hard-delete.
    """
    diff = empty_memory_diff(archive_uri_str)
    adds: list[dict[str, Any]] = []
    updates: list[dict[str, Any]] = []
    deletes: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    events = list(ledger_events or [])
    for ev in events:
        op = str(ev.get("op") or "")
        fid = str(ev.get("id") or "")
        fact = None
        if facts_by_id and fid:
            try:
                fact = facts_by_id(fid)
            except Exception:
                fact = None
        uri = getattr(fact, "uri", None) if fact is not None else None
        claim = getattr(fact, "claim", None) if fact is not None else None
        ftype = getattr(getattr(fact, "type", None), "value", None) if fact is not None else None
        if op == "create":
            adds.append({
                "uri": uri or f"mb://user/memories/fact/{fid}",
                "memory_type": ftype or "fact",
                "fact_id": fid,
                "after": claim or "",
            })
        elif op == "corroborate":
            updates.append({
                "uri": uri or f"mb://user/memories/fact/{fid}",
                "memory_type": ftype or "fact",
                "fact_id": fid,
                "before": claim or "",
                "after": claim or "",
                "note": "corroborated",
            })
        elif op == "supersede":
            deletes.append({
                "uri": uri or f"mb://user/memories/fact/{fid}",
                "memory_type": ftype or "fact",
                "fact_id": fid,
                "deleted_content": claim or "",
                "ledger_op": "supersede",
                "by": ev.get("by"),
            })
        elif op == "retract":
            deletes.append({
                "uri": uri or f"mb://user/memories/fact/{fid}",
                "memory_type": ftype or "fact",
                "fact_id": fid,
                "deleted_content": claim or "",
                "ledger_op": "retract",
                "reason": ev.get("reason"),
            })

    if report is not None:
        # Report counters always land in summary; skipped ops when no ledger detail.
        summary = diff["summary"]
        summary["facts_added"] = int(getattr(report, "facts_added", 0) or 0)
        summary["corroborated"] = int(getattr(report, "corroborated", 0) or 0)
        summary["superseded"] = int(getattr(report, "superseded", 0) or 0)
        summary["retracted"] = int(getattr(report, "retracted", 0) or 0)
        summary["skipped"] = int(getattr(report, "skipped", 0) or 0)
        summary["errors"] = list(getattr(report, "errors", None) or [])
        for err in summary["errors"]:
            skipped.append({
                "reason_code": "extract_error",
                "reason": str(err),
            })
        # If extract reported skips but no ledger events, keep an audit row.
        if summary["skipped"] and not events:
            skipped.append({
                "reason_code": "no_qualifying_ops",
                "reason": f"extractor skipped {summary['skipped']} candidate(s)",
            })

    diff["operations"] = {"adds": adds, "updates": updates, "deletes": deletes}
    diff["skipped_operations"] = skipped
    diff["summary"]["total_adds"] = len(adds)
    diff["summary"]["total_updates"] = len(updates)
    diff["summary"]["total_deletes"] = len(deletes)
    diff["summary"]["total_skipped"] = len(skipped)
    return diff


def _ledger_since(store: Store, since_iso: str) -> list[dict[str, Any]]:
    path = store.ledger_path
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        since = datetime.fromisoformat(since_iso.replace("Z", "+00:00"))
        if since.tzinfo is None:
            since = since.replace(tzinfo=timezone.utc)
    except ValueError:
        since = None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if not isinstance(ev, dict):
                continue
            ts_raw = ev.get("ts")
            if since is not None and isinstance(ts_raw, str):
                try:
                    ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    if ts < since:
                        continue
                except ValueError:
                    pass
            if ev.get("op") in {"create", "corroborate", "supersede", "retract"}:
                out.append(ev)
    return out


def phase2_process(
    store: Store,
    archive_path: str | Path,
    *,
    run_extract: bool = True,
    extractor_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Async/background work: summary sidecars, extract, memory_diff, .done."""
    archive_path = Path(archive_path)
    meta_path = archive_path / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"archive meta missing: {archive_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    uri = str(meta.get("archive_uri") or archive_uri(str(meta.get("session")), archive_path.name))
    session = str(meta.get("original_session") or meta.get("session") or "")
    messages = load_archive_messages(archive_path)
    abstract, overview = _heuristic_summary(messages)
    _atomic_write(archive_path / ".abstract.md", abstract.rstrip() + "\n")
    _atomic_write(archive_path / ".overview.md", overview.rstrip() + "\n")

    phase2_started = _now().isoformat()
    report = None
    extract_errors: list[str] = []
    if run_extract and session:
        try:
            if extractor_factory is not None:
                extractor = extractor_factory()
            else:
                extractor = None
            if extractor is not None:
                report = extractor.run(session=session)
        except Exception as exc:  # fail-open: still write empty/partial diff
            extract_errors.append(f"extract: {exc}")

    events = _ledger_since(store, phase2_started)

    def _get_fact(fid: str) -> Any:
        try:
            return store.get_fact(fid)
        except Exception:
            return None

    diff = build_memory_diff(
        uri,
        report=report,
        ledger_events=events,
        facts_by_id=_get_fact,
    )
    if extract_errors:
        diff["skipped_operations"].extend(
            {"reason_code": "extract_error", "reason": e} for e in extract_errors
        )
        diff["summary"]["total_skipped"] = len(diff["skipped_operations"])
        diff["summary"]["errors"] = list(diff["summary"].get("errors") or []) + extract_errors

    _atomic_write(archive_path / "memory_diff.json", json.dumps(diff, ensure_ascii=False, indent=2))
    done = {
        "task_id": meta.get("task_id"),
        "completed_at": _now().isoformat(),
        "archive_uri": uri,
        "phase2_status": "completed",
        "message_count": len(messages),
        "diff_summary": {
            "total_adds": diff["summary"]["total_adds"],
            "total_updates": diff["summary"]["total_updates"],
            "total_deletes": diff["summary"]["total_deletes"],
            "total_skipped": diff["summary"]["total_skipped"],
        },
    }
    _atomic_write(archive_path / ".done", json.dumps(done, ensure_ascii=False, indent=2) + "\n")
    meta["phase2_status"] = "completed"
    meta["phase2_at"] = done["completed_at"]
    _atomic_write(meta_path, json.dumps(meta, ensure_ascii=False, indent=2))
    return {"status": "completed", "archive_path": str(archive_path), "diff": diff, "done": done}





def latest_done_archive(store: Store, session: str) -> Path | None:
    """Newest archive_* directory that has .done (Phase2 finished)."""
    hist = history_dir(store, session)
    dones: list[Path] = []
    for child in hist.glob("archive_*"):
        if child.is_dir() and (child / ".done").exists():
            dones.append(child)
    if not dones:
        return None
    return sorted(dones, key=lambda p: p.name)[-1]


def latest_archive_overview(
    store: Store,
    session: str,
    *,
    max_chars: int = 4000,
) -> tuple[str, str | None]:
    """Return (overview_text, archive_name) for resume/compact injection. Empty if none."""
    arch = latest_done_archive(store, session)
    if arch is None:
        return "", None
    ov = arch / ".overview.md"
    if not ov.exists():
        return "", arch.name
    try:
        text = ov.read_text(encoding="utf-8").strip()
    except OSError:
        return "", arch.name
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars].rstrip() + "\n..."
    return text, arch.name


def read_memory_diff(archive_path: str | Path) -> dict[str, Any]:
    """Read-only memory_diff for CLI/MCP."""
    path = Path(archive_path)
    if path.is_dir():
        diff_path = path / "memory_diff.json"
    else:
        diff_path = path
        if diff_path.name != "memory_diff.json" and diff_path.suffix == ".json":
            pass
        elif diff_path.exists() and diff_path.is_file() and "memory_diff" in diff_path.name:
            pass
        else:
            # allow passing archive dir as file-like path string ending without slash
            alt = Path(str(archive_path)) / "memory_diff.json"
            if alt.exists():
                diff_path = alt
    if not diff_path.exists():
        # try resolve relative to sessions home if given archive_NNN only — caller should pass full path
        raise FileNotFoundError(f"memory_diff not found: {archive_path}")
    data = json.loads(diff_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("memory_diff must be a JSON object")
    return data


def _inside(root: Path, candidate: Path) -> bool:
    """True only for a path under root after resolving links and '..'."""
    try:
        base = root.resolve()
        target = candidate.resolve()
    except OSError:
        return False
    return target == base or base in target.parents


def resolve_archive_path(store: Store, archive: str) -> Path:
    """Accept an archive under this store, never an arbitrary local path.

    Supported forms: an absolute path below memory/sessions, a path relative to
    the memory home, a path relative to memory/sessions, or session/archive_NNN.
    """
    raw = Path(archive)
    sessions_root = store.home / SESSIONS_ROOT
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend((store.home / raw, sessions_root / raw))
    parts = raw.parts
    if len(parts) >= 2 and ARCHIVE_RE.fullmatch(parts[-1]):
        candidates.append(sessions_root / _safe_session(parts[-2]) / "history" / parts[-1])
    elif len(parts) >= 2:
        candidates.append(sessions_root / _safe_session(parts[0]) / "history" / parts[1])
    for candidate in candidates:
        if not _inside(sessions_root, candidate) or not candidate.exists():
            continue
        path = candidate if candidate.is_dir() else candidate.parent if candidate.name == "memory_diff.json" else candidate
        return path if path.is_dir() else path.parent
    raise FileNotFoundError(f"archive not found: {archive}")


def pending_phase2_tasks(store: Store) -> list[dict[str, Any]]:
    queue_path = store.home / ".state" / "session_archive_queue.jsonl"
    if not queue_path.exists():
        return []
    items: list[dict[str, Any]] = []
    for line in queue_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, dict):
            continue
        ap = Path(str(row.get("archive_path") or ""))
        if ap.exists() and not (ap / ".done").exists():
            items.append(row)
    return items


def commit_sessions(
    store: Store,
    sessions: list[str],
    *,
    reason: str = "auto_commit",
) -> list[dict[str, Any]]:
    """Phase1 for each session; returns list of phase1 results."""
    results = []
    for session in sessions:
        results.append(phase1_commit(store, session, reason=reason))
    return results
