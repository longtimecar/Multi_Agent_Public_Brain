"""Hook handlers. These are what make capture and injection automatic.

Claude Code (and Codex / Gemini CLI / OpenCode with different event names) call
`membrain hook <event>` with a JSON payload on stdin and use stdout as context.
Capture writes raw episodes only — no model call, no latency on the user's turn.
"""

from __future__ import annotations

import json
import hashlib
import re
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .budget import estimate_tokens
from . import injection
from .agents import is_disabled
from .locking import file_lock
from .models import Episode
from .server import MemoryService
from .text import clean_turn

STATE_FILE = ".state/captured.json"

# DETACHED_PROCESS makes Windows open a visible console titled with the venv
# python path every time a background worker starts. CREATE_NO_WINDOW keeps
# the worker silent and still lets it outlive the hook.
_BACKGROUND_FLAGS = (subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == "nt" else 0


def _start_extractor(svc: MemoryService) -> None:
    """Run extraction separately so a slow model never blocks the hook."""
    log_path = svc.settings.home / ".state" / "extract.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["MEMBRAIN_HOME"] = str(svc.settings.home)
    with log_path.open("a", encoding="utf-8") as log:
        subprocess.Popen([sys.executable, "-m", "membrain.cli", "extract"], stdin=subprocess.DEVNULL,
                         stdout=log, stderr=subprocess.STDOUT, env=env, close_fds=True,
                         creationflags=_BACKGROUND_FLAGS, start_new_session=os.name != "nt")


def _start_session_archive_phase2(svc: MemoryService) -> None:
    """Background Phase2: abstract/overview + extract + memory_diff + .done."""
    log_path = svc.settings.home / ".state" / "session_archive.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["MEMBRAIN_HOME"] = str(svc.settings.home)
    with log_path.open("a", encoding="utf-8") as log:
        subprocess.Popen(
            [sys.executable, "-m", "membrain.cli", "session-archive-phase2"],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            close_fds=True,
            creationflags=_BACKGROUND_FLAGS,
            start_new_session=os.name != "nt",
        )


def _archive_then_extract(svc: MemoryService, sessions: list[str] | None = None, *, reason: str = "auto_commit") -> list[dict]:
    """Phase1 sync for sessions (or take_pending), then kick Phase2 worker."""
    from . import auto_commit, session_archive

    if sessions is None:
        bundle = auto_commit.commit_pending_with_archive(svc.store, reason=reason)
        phase1 = bundle.get("phase1") or []
    else:
        phase1 = session_archive.commit_sessions(svc.store, sessions, reason=reason)
    if any(item.get("archived") for item in phase1):
        _start_session_archive_phase2(svc)
    else:
        # No new archive; still run plain extract for leftover episodes.
        _start_extractor(svc)
    return phase1


def _start_codex_capture_verifier(svc: MemoryService, payload: dict[str, Any]) -> None:
    """Check transcript provenance off the prompt's critical path."""
    log_path = svc.settings.home / ".state" / "codex-capture.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["MEMBRAIN_HOME"] = str(svc.settings.home)
    with log_path.open("a", encoding="utf-8") as log:
        child = subprocess.Popen([sys.executable, "-m", "membrain.cli", "hook", "codex-verified-capture"],
                                 stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, env=env,
                                 close_fds=True, creationflags=_BACKGROUND_FLAGS, start_new_session=os.name != "nt")
        assert child.stdin is not None
        # Windows subprocess pipes may use a legacy code page. ASCII JSON
        # escapes preserve non-ASCII user names, paths and messages there.
        child.stdin.write(json.dumps(payload, ensure_ascii=True).encode("ascii"))
        child.stdin.close()

def _record_time(raw: Any) -> datetime:
    if isinstance(raw, str):
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def _read_stdin() -> dict[str, Any]:
    # Hook payloads are UTF-8 JSON. On Windows a redirected stdin can default
    # to a legacy code page, corrupting Chinese prompts and user paths.
    stream = getattr(sys.stdin, "buffer", None)
    try:
        raw = stream.read().decode("utf-8") if stream is not None else sys.stdin.read()
    except UnicodeDecodeError:
        return {}
    if not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _state_path(svc: MemoryService) -> Path:
    p = svc.settings.home / STATE_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _load_state(svc: MemoryService) -> dict[str, Any]:
    p = _state_path(svc)
    if not p.exists():
        return {"captured": {}}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"captured": {}}


def _save_state(svc: MemoryService, state: dict[str, Any]) -> None:
    _state_path(svc).write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")


def record_capture(svc: MemoryService, session: str, agent: str, turns: list[tuple]) -> int:
    """Append real transcript turns using each record's timestamp.

    Each turn is (role, text, uid, ts) or (role, text, uid, ts, ts_source);
    ts_source says whether ts is the turn's own log time or an estimate.
    """
    with file_lock(svc.settings.home / ".state" / "capture.lock"):
        return _record_capture_locked(svc, session, agent, turns)


def _take_pending_extract(svc: MemoryService) -> bool:
    with file_lock(svc.settings.home / ".state" / "capture.lock"):
        state = _load_state(svc)
        if not state.pop("pending_extract", False):
            return False
        _save_state(svc, state)
        return True


def _text_key(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def _index_episodes(svc: MemoryService, episodes: list[Episode]) -> None:
    """Make freshly captured turns searchable without a full reindex.

    Fail-open: a broken index must never cost the user their capture, and
    `.index/` is disposable — `membrain reindex` rebuilds it.
    """
    retriever = getattr(svc, "retriever", None)
    if retriever is None or not episodes:
        return
    try:
        retriever.sync_episodes(episodes)
    except Exception:
        pass


def record_prompt(svc: MemoryService, session: str, agent: str, text: str) -> bool:
    """Record the prompt the user just typed, before the agent starts replying.

    Capture on the Stop hook runs after the reply, so an agent that tries to
    save a fact during its reply cannot yet quote the turn. The prompt payload
    is the user's own input, so it is safe evidence. The later transcript pass
    skips any turn recorded here, which keeps one message as one episode.
    """
    text = clean_turn(text)
    if not text or not session or session == "unknown":
        return False
    disabled = getattr(svc, "disabled_agents", None)
    if disabled is not None and is_disabled(agent, disabled()):
        # The owner disconnected this agent: its turns stay out of the store
        # entirely instead of being hidden later.
        return False
    now = datetime.now(timezone.utc)
    with file_lock(svc.settings.home / ".state" / "capture.lock"):
        state = _load_state(svc)
        key = _text_key(text)
        # Two hook entries (project and global) can fire for one message.
        last = state.setdefault("last_prompt", {}).get(session)
        if last and last[0] == key and now.timestamp() - last[1] < 5:
            return False
        state["last_prompt"][session] = [key, now.timestamp()]
        state["pending_extract"] = True
        pending = state.setdefault("prompts", {}).setdefault(session, [])
        pending.append(key)
        del pending[:-200]
        uid = hashlib.sha256(f"prompt|{session}|{now.isoformat()}|{text}".encode("utf-8")).hexdigest()
        episode = Episode(id=f"ep_{uid[:16]}", ts=now, session=session, agent=agent,
                          role="user", text=text[:20000], ts_source="exact")
        svc.store.append_episodes([episode])
        _index_episodes(svc, [episode])
        _save_state(svc, state)
    return True


def _record_capture_locked(svc: MemoryService, session: str, agent: str, turns: list[tuple]) -> int:
    state = _load_state(svc)
    seen: list[str] = state["captured"].get(session, [])
    seen_set = set(seen)
    prompts: list[str] = state.get("prompts", {}).get(session, [])
    batch: list[Episode] = []
    project = None
    for turn in turns:
        # (role, text, uid, ts) or (role, text, uid, ts, ts_source). A caller that
        # does not declare accuracy gets "estimated", never a claim of precision.
        role, text, uid, timestamp = turn[0], turn[1], turn[2], turn[3]
        ts_source = turn[4] if len(turn) > 4 else "estimated"
        if not text.strip() or uid in seen_set:
            continue
        key = _text_key(text)
        if role == "user" and key in prompts:
            # Already recorded when the prompt was submitted.
            prompts.remove(key)
            seen.append(uid)
            seen_set.add(uid)
            continue
        batch.append(
            Episode(
                id=f"ep_{uid[:16]}",
                ts=timestamp,
                session=session,
                agent=agent,
                role=role,  # type: ignore[arg-type]
                text=text[:20000],
                project=project,
                ts_source=ts_source,  # type: ignore[arg-type]
            )
        )
        seen.append(uid)
        seen_set.add(uid)
    if batch:
        svc.store.append_episodes(batch)
        _index_episodes(svc, batch)
    state["captured"][session] = seen
    if "prompts" in state:
        state["prompts"][session] = prompts
    _save_state(svc, state)
    return len(batch)


def _transcript_turns(payload: dict[str, Any]) -> list[tuple[str, str, str, datetime]]:
    """Extract genuine user/assistant text, excluding tool and metadata records."""
    path = payload.get("transcript_path")
    if not path:
        print("membrain: capture event has no transcript_path; nothing captured", file=sys.stderr)
        return []
    if not Path(path).exists():
        # Silence here is the expensive failure: capture would appear to work
        # while storing nothing at all.
        print(f"membrain: transcript not found: {path}", file=sys.stderr)
        return []
    turns: list[tuple[str, str, str, datetime]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue
            role = rec.get("type")
            if (role not in ("user", "assistant") or rec.get("isMeta") or rec.get("isSidechain")
                    or rec.get("isApiErrorMessage") or rec.get("toolUseResult") is not None
                    or rec.get("sourceToolAssistantUUID")):
                continue
            msg = rec.get("message")
            if not isinstance(msg, dict) or msg.get("role") != role:
                continue
            content = msg.get("content")
            text = ""
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = "\n".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
            text = clean_turn(text)
            if not text:
                continue
            canonical = json.dumps(rec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            uid = str(rec.get("uuid") or hashlib.sha256(canonical.encode("utf-8")).hexdigest())
            # The transcript's own timestamp is the turn's real time; without it
            # _record_time falls back to "now", which is only an estimate.
            turns.append((role, text, uid, _record_time(rec.get("timestamp")),
                          "exact" if rec.get("timestamp") else "estimated"))
    return turns


def _codex_user_turn(payload: dict[str, Any]) -> tuple[str, str, str, datetime] | None:
    """Read a top-level Codex user-prompt hook, excluding app-added context.

    Codex also runs hooks for subagents, whose input may have been written by
    another agent. Those inputs must never become user evidence.
    """
    if payload.get("hook_event_name") != "UserPromptSubmit" or payload.get("agent_id") or payload.get("agent_type"):
        return None
    session, turn, text = payload.get("session_id"), payload.get("turn_id"), payload.get("prompt")
    if not all(isinstance(value, str) and value.strip() for value in (session, turn, text)):
        return None

    # Desktop annotations quote prior output; only the request after the
    # generated annotation preamble is the user's new message.
    if "<response-annotations" in text:
        _, closing, remainder = text.rpartition("</response-annotations>")
        marker = "## My request:"
        if not closing or not remainder.lstrip().startswith(marker):
            return None
        text = remainder.lstrip()[len(marker):]
    text = re.sub(r"<in-app-browser-context\b[^>]*>.*?</in-app-browser-context\s*>",
                  "", text, flags=re.IGNORECASE | re.DOTALL)
    if "<in-app-browser-context" in text.casefold() or text.lstrip().startswith("# AGENTS.md instructions"):
        return None
    if text.lstrip().startswith("## My request:"):
        text = text.lstrip()[len("## My request:"):]
    text = clean_turn(text)
    if not text:
        return None
    identity = json.dumps(["codex", session, turn, text], ensure_ascii=False, separators=(",", ":"))
    uid = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return "user", text, uid, datetime.now(timezone.utc)


def _codex_transcript_confirms_user(payload: dict[str, Any]) -> bool:
    """Fail closed unless the local desktop transcript identifies real user input.

    Hook input alone cannot distinguish every background agent from the owner.
    The transcript is not a stable Codex API, so unfamiliar formats are skipped.
    """
    path = payload.get("transcript_path")
    if not isinstance(path, str) or not path:
        return False
    try:
        with Path(path).open("rb") as fh:
            header = json.loads(fh.readline())
            metadata = header.get("payload", {})
            if (header.get("type") != "session_meta" or not isinstance(metadata, dict)
                    or metadata.get("thread_source") != "user"
                    or metadata.get("session_id") != payload.get("session_id")):
                return False
            # Only recent records can match the current turn. Discard the first
            # partial line after seeking into a large transcript.
            size = fh.seek(0, os.SEEK_END)
            if size > 8 * 1024 * 1024:
                fh.seek(-8 * 1024 * 1024, os.SEEK_END)
                fh.readline()
            else:
                fh.seek(0)
            for line in fh:
                try:
                    record = json.loads(line)
                except (ValueError, UnicodeDecodeError):
                    continue
                item = record.get("payload", {}) if isinstance(record, dict) else {}
                if record.get("type") != "response_item" or not isinstance(item, dict):
                    continue
                if item.get("type") != "message" or item.get("role") != "user":
                    continue
                origin = item.get("internal_chat_message_metadata_passthrough")
                if not isinstance(origin, dict) or origin.get("turn_id") != payload.get("turn_id"):
                    continue
                if origin.get("content_item_kinds") != ["user.text"]:
                    continue
                content = item.get("content")
                if (isinstance(content, list) and len(content) == 1 and isinstance(content[0], dict)
                        and content[0].get("type") == "input_text"
                        and content[0].get("text") == payload.get("prompt")):
                    return True
    except (OSError, ValueError, UnicodeDecodeError, AttributeError):
        return False
    return False


def _verify_codex_capture(svc: MemoryService, payload: dict[str, Any], timeout: float = 15.0) -> bool:
    turn = _codex_user_turn(payload)
    if not turn or not payload.get("transcript_path"):
        return False
    deadline = time.monotonic() + timeout
    while True:
        if _codex_transcript_confirms_user(payload):
            if record_capture(svc, str(payload["session_id"]), "codex", [turn]):
                _start_extractor(svc)
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.25)


def _session_start_context(svc: MemoryService, payload: dict[str, Any] | None = None) -> str:
    """Inject profile L0 + non-sensitive L1 overview. Never dumps sensitive beliefs."""
    from .overview import build_l1_overviews, make_abstract

    budget = min(svc.settings.session_token_budget, 2500)
    payload = payload or {}
    session = str(payload.get("session_id") or "unknown")
    facts = svc.store.live_facts()
    selected = []
    lines: list[str] = []
    profile = svc.settings.profile_path
    if profile.exists():
        text = re.sub(r"<!--.*?-->", "", profile.read_text(encoding="utf-8"), flags=re.DOTALL).strip()
        for line in text.splitlines():
            candidate = "<personal_memory>\n" + "\n".join(lines + [line]) + "\n</personal_memory>"
            if estimate_tokens(candidate) > budget:
                break
            lines.append(line)
    # Constraints + high-confidence goals as L0 abstracts (never sensitive dump)
    eligible = [
        f for f in facts
        if not f.sensitive and (
            f.type.value == "constraint"
            or (f.type.value == "goal" and f.confidence >= 0.6)
        )
    ]
    for fact in eligible:
        abstract = fact.abstract or make_abstract(fact.claim)
        line = f"- [L0] {abstract} ({fact.type.value})"
        candidate = "<personal_memory>\n" + "\n".join(lines + [line]) + "\n</personal_memory>"
        if estimate_tokens(candidate) <= budget:
            lines.append(line)
            selected.append(fact)
    # L1 overview of non-sensitive groups (capped).
    # Do NOT fingerprint L1 member ids into injection state: overview is a
    # summary layer; prompt-submit must still be able to inject L2 detail.
    overviews = build_l1_overviews(facts, include_sensitive=False)
    for row in overviews[:12]:
        line = f"- [L1 {row['type']}/{row['subject']}] ({row['count']}) {row['overview']}"
        candidate = "<personal_memory>\n" + "\n".join(lines + [line]) + "\n</personal_memory>"
        if estimate_tokens(candidate) > budget:
            break
        lines.append(line)
    # Gap 5: on resume/compact, attach latest completed archive L1 overview (half budget).
    source = str(payload.get("source") or "").strip().lower()
    if source in {"compact", "resume", "clear"}:
        try:
            from . import session_archive
            half = max(200, budget // 2)
            ov_text, arch_name = session_archive.latest_archive_overview(svc.store, session)
            if ov_text:
                header = f"- [archive L1 {arch_name}]"
                body_lines = [ln for ln in ov_text.splitlines() if ln.strip()][:40]
                chunk = header
                for ln in body_lines:
                    trial = chunk + "\n" + ln
                    candidate = "<personal_memory>\n" + "\n".join(lines + [trial]) + "\n</personal_memory>"
                    # Cap this archive block itself to half budget, and whole block to full budget.
                    if estimate_tokens(trial) > half or estimate_tokens(candidate) > budget:
                        break
                    chunk = trial
                if chunk != header:
                    lines.append(chunk)
        except Exception:
            pass
    # Gap 10: budgeted skills checklist (names + one-line summaries).
    if getattr(svc.settings, "skills_inject_session_start", True):
        try:
            from .skills_index import ensure_skills_dir, skills_injection_lines
            ensure_skills_dir(svc.settings.home)
            skill_lines, skill_notes = skills_injection_lines(
                svc.settings.home,
                budget_tokens=budget,
                existing_block="\n".join(lines),
            )
            if skill_lines:
                lines.extend(skill_lines)
        except Exception:
            pass
    block = "<personal_memory>\n" + "\n".join(lines) + "\n</personal_memory>" if lines else ""
    with file_lock(svc.settings.home / ".state" / "injection.lock"):
        state = {} if payload.get("source") in ("compact", "clear") else injection.load(svc.store, session)
        state.update({f.id: injection.fingerprint(f) for f in selected})
        injection.save(svc.store, session, state, "session-start", estimate_tokens(block), [f.id for f in selected])
    return block


def _note_auto_commit(svc: MemoryService, session: str, text: str) -> None:
    """Accumulate tokens; start extractor when threshold/idle policy fires."""
    try:
        from . import auto_commit
        auto_commit.note_activity(
            svc.store,
            session,
            text,
            threshold=svc.settings.auto_commit_token_threshold,
            idle_minutes=svc.settings.auto_commit_idle_minutes,
        )
        auto_commit.check_idle(
            svc.store,
            threshold=svc.settings.auto_commit_token_threshold,
            idle_minutes=svc.settings.auto_commit_idle_minutes,
        )
        pending = auto_commit.pending_sessions(svc.store)
        if pending:
            # Write before clearing: commit_pending_with_archive retires a
            # session only after its archive write returned, and leaves the
            # rest queued when one of them fails.
            result = auto_commit.commit_pending_with_archive(svc.store, reason="token_or_idle")
            if any(item.get("archived") for item in result["phase1"]):
                _start_session_archive_phase2(svc)
            else:
                _start_extractor(svc)
    except Exception as exc:  # never break the turn
        print(f"membrain auto-commit error: {exc}", file=sys.stderr)


def _prompt_context(svc: MemoryService, payload: dict[str, Any]) -> str:
    prompt = payload.get("prompt") or ""
    if not prompt.strip():
        return ""
    # Trivial turns (acknowledgements, slash commands) are not worth a lookup.
    if estimate_tokens(prompt.strip()) < 2:
        return ""
    if not payload.get("session_id"):
        return svc.retriever.recall(prompt, budget=min(svc.settings.turn_token_budget, 2000)).text
    session = str(payload["session_id"])
    with file_lock(svc.settings.home / ".state" / "injection.lock"):
        state = injection.load(svc.store, session)
        # Sensitive facts stay out of "already injected" blocking for topic turns:
        # fingerprints still track non-sensitive injects as before.
        unchanged = {fact.id for fact in svc.store.live_facts()
                     if (not fact.sensitive) and state.get(fact.id) == injection.fingerprint(fact)}
        result = svc.retriever.recall(prompt, budget=min(svc.settings.turn_token_budget, 2000), exclude_ids=unchanged)
        state.update({f.id: injection.fingerprint(f) for f in result.hits})
        injection.save(svc.store, session, state, "prompt-submit", result.tokens, [f.id for f in result.hits])
        return result.text


def run(event: str) -> int:
    svc = MemoryService()
    payload = _read_stdin()
    session = str(payload.get("session_id") or "unknown")
    agent = str(payload.get("agent") or "claude-code")

    try:
        if event == "session-start":
            block = _session_start_context(svc, payload)
            if block:
                print(block)
            _start_extractor(svc)
        elif event == "prompt-submit":
            try:
                record_prompt(svc, session, agent, str(payload.get("prompt") or ""))
                _note_auto_commit(svc, session, str(payload.get("prompt") or ""))
            except Exception as exc:  # capture must not stop recall
                print(f"membrain prompt capture error: {exc}", file=sys.stderr)
            block = _prompt_context(svc, payload)
            if block:
                print(block)
        elif event == "codex-prompt-submit":
            turn = _codex_user_turn(payload)
            if turn:
                _start_codex_capture_verifier(svc, payload)
                _note_auto_commit(svc, session, turn[1])
                # Recall against only the user's actual message, not the
                # desktop's annotation or browser-context preamble.
                block = _prompt_context(svc, {**payload, "prompt": turn[1]})
                if block:
                    print(block)
        elif event == "codex-verified-capture":
            _verify_codex_capture(svc, payload)
        elif event == "capture":
            n = record_capture(svc, session, agent, _transcript_turns(payload))
            print(json.dumps({"captured": n}), file=sys.stderr)
            # Prompts recorded at submit time leave n at 0, so the flag carries
            # them. Starting only when needed keeps idle turns at zero extra processes.
            if n or _take_pending_extract(svc):
                _start_extractor(svc)
        elif event in ("precompact", "auto-commit"):
            # Internal commit hook: token/idle policy or PreCompact from harness.
            # Does NOT modify any protected agent config; callers opt in.
            _note_auto_commit(svc, session, str(payload.get("prompt") or ""))
            from . import auto_commit, session_archive
            pending = auto_commit.pending_sessions(svc.store)
            # Queue first, retire after the write: a failed archive keeps the
            # session pending instead of dropping it.
            result = auto_commit.commit_pending_with_archive(svc.store, reason=event)
            phase1 = list(result["phase1"])
            # Force Phase1 for explicit precompact even below threshold.
            if event == "precompact" and session and session != "unknown" and session not in pending:
                phase1.append(session_archive.phase1_commit(svc.store, session, reason=event))
            if any(item.get("archived") for item in phase1):
                _start_session_archive_phase2(svc)
            elif phase1 or event == "precompact":
                _start_extractor(svc)
            print(json.dumps({"auto_commit": True, "pending": pending, "phase1": phase1,
                              "errors": result["errors"], "event": event}, ensure_ascii=False),
                  file=sys.stderr)
    except Exception as exc:  # a hook must never break the user's turn
        print(f"membrain hook error: {exc}", file=sys.stderr)
        return 0
    return 0
