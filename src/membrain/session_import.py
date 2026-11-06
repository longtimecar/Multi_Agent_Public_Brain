"""Batch import of local agent session transcripts into the membrain archive.

Read-only on every source. One session becomes one archive entry. The LLM
cleans and classifies each session (category + title + L0 abstract) and flags
noise; rule-based noise filters run first so junk never costs an LLM call.
Content-hash entry ids make reruns free. Each import writes its batch manifest
twice: `in_progress` before the first archive write, `completed` only after the
whole pass succeeds, so a crash always leaves a batch a rerun can resume instead
of half-written archive state with no record. Batch manifests reuse the
OpenViking import/batches/undo bookkeeping unchanged. Fact distillation is
opt-in and quote-verified. LLM failure is fail-open: the entry still lands with
rule-based fields and can be reclassified later.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import agents
from . import openviking_import as ov
from .archive import ArchiveEntry, CATEGORY_LABELS, content_hash, entry_id, make_abstract
from .config import local_date
from .llm import LLMError
from .models import Evidence, FactType
from .store import _atomic_write
from .text import clean_turn


def local_date_now():
    """Today on the operator's calendar."""
    return local_date(datetime.now(timezone.utc))

HOME = Path.home()
MAX_BODY = 120_000
MAX_TURN = 8_000
CALL_PAUSE = 3.0                   # seconds between model calls; the provider rate-limits
RATE_LIMIT_WAITS = (15.0, 30.0)
VALID_CATEGORIES = set(CATEGORY_LABELS)


def _decompress_zstd(blob: bytes) -> bytes:
    """Inflate a zstd frame or a concatenated sequence of frames.

    DeepSeek Harness writes one zstd frame per transcript chunk, so a session
    file is a long chain of frames, not a single one. Python 3.14 ships zstd
    in the standard library and its decompress() walks every frame; earlier
    versions need the optional `zstandard` package in streaming mode.

    Every decompression failure surfaces as RuntimeError so one corrupt file
    can never abort a whole scan — callers treat it as unreadable and move on.
    """
    try:  # 3.14+
        from compression import zstd  # type: ignore[import-not-found]
        return zstd.decompress(blob)
    except ImportError:
        pass
    except Exception as exc:  # a real frame error, not a missing module
        raise RuntimeError(f"zstd 数据无法解压: {exc}") from exc
    try:
        import io

        import zstandard  # type: ignore[import-not-found]
        with zstandard.ZstdDecompressor().stream_reader(
                io.BytesIO(blob), read_across_frames=True) as reader:
            return reader.read()
    except ImportError as exc:
        raise RuntimeError("读取 zstd 压缩的会话需要 Python 3.14 或安装 zstandard") from exc
    except Exception as exc:  # corrupt frame, truncated file, unknown descriptor
        raise RuntimeError(f"zstd 数据无法解压: {exc}") from exc

# Turns that carry no lasting information. Matched exactly after strip.
_NOISE_TURN = re.compile(
    r"^(?:hello|hi|hey|yo|你好|您好|哈喽|在吗|嗯+|好的?|okay?|ok|thanks?|谢谢|感谢|no response requested\.?)"
    r"[.!！？?~～\s]*$", re.I)


# A parsed turn: (role, text, time-or-None). Parsers emit this shape; RawSession
# splits the times out so downstream rendering keeps using (role, text).
ParsedTurn = tuple[str, str, "datetime | None"]


@dataclass
class RawSession:
    source: str
    scope: str
    origin: str
    date: str
    # (role, text) or (role, text, ts). The third slot carries the turn's own
    # timestamp when the agent's log has one; __post_init__ splits it out so the
    # rest of the module keeps working with plain (role, text) pairs.
    turns: list[tuple[str, str]] | list[tuple[str, str, datetime | None]] = field(default_factory=list)
    title: str = ""
    note_body: str | None = None
    # Per-turn times, aligned with `turns`. None = this source kept no per-turn
    # time, so the caller must fall back to session_ts and mark it estimated.
    turn_ts: list[datetime | None] | None = None
    # Session-level time (created/updated). Day granularity at best; used only
    # when a turn has no time of its own.
    session_ts: datetime | None = None

    def __post_init__(self) -> None:
        if any(len(t) > 2 for t in self.turns):
            plain: list[tuple[str, str]] = []
            stamps: list[datetime | None] = []
            for turn in self.turns:
                plain.append((turn[0], turn[1]))
                stamps.append(turn[2] if len(turn) > 2 else None)
            self.turns = plain
            self.turn_ts = stamps
        elif self.turn_ts is None:
            self.turn_ts = [None] * len(self.turns)

    def ts_for(self, index: int) -> tuple[datetime | None, bool]:
        """(timestamp, is_exact) for turn `index`: per-turn time wins."""
        stamps = self.turn_ts or []
        if index < len(stamps) and stamps[index] is not None:
            return stamps[index], True
        if self.session_ts is not None:
            return self.session_ts, False
        return None, False


# Log records name their time field differently per agent and per version, and
# store it either as an ISO string or as epoch seconds/milliseconds. Trying the
# common shapes beats binding each parser to one schema that will drift.
_TS_KEYS = ("timestamp", "ts", "time", "created_at", "createdAt", "created", "time_created", "date")


def _rec_ts(*candidates: Any) -> datetime | None:
    """First parseable aware timestamp among `candidates`, else None."""
    for raw in candidates:
        if isinstance(raw, datetime):
            return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
            except ValueError:
                continue
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        if isinstance(raw, (int, float)) and not isinstance(raw, bool) and raw > 0:
            # Epoch milliseconds above ~2001 in ms, seconds below. 1e11 separates
            # them for every plausible conversation timestamp.
            seconds = raw / 1000 if raw > 1e11 else raw
            try:
                return datetime.fromtimestamp(seconds, tz=timezone.utc)
            except (OverflowError, OSError, ValueError):
                continue
    return None


def _rec_ts_from(rec: Any) -> datetime | None:
    """Timestamp of one log record, probing the usual field names."""
    if not isinstance(rec, dict):
        return None
    timing = rec.get("timing") if isinstance(rec.get("timing"), dict) else {}
    return _rec_ts(*[rec.get(k) for k in _TS_KEYS], *[(timing or {}).get(k) for k in ("created", "first")])


def _clip(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 20].rstrip() + "\n\n…（已截断）"


def _day_of(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).date().isoformat()
    except OSError:
        return datetime.now(timezone.utc).date().isoformat()


def _local_day(ts: datetime | None) -> str:
    """Operator-calendar day of a timestamp, so session dates agree with recall."""
    if ts is None:
        return local_date_now().isoformat()
    return local_date(ts).isoformat()


# ---------------------------------------------------------------- adapters

def _parse_claude(path: Path) -> list[ParsedTurn]:
    turns: list[ParsedTurn] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("type") not in ("user", "assistant"):
            continue
        if rec.get("isSidechain") or rec.get("isMeta"):
            continue
        msg = rec.get("message") or {}
        role = msg.get("role") or rec.get("type")
        content = msg.get("content")
        text = ""
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "\n".join(p.get("text", "") for p in content
                             if isinstance(p, dict) and p.get("type") == "text")
        text = text.strip()
        if not text or text.startswith("<system-reminder>"):
            continue
        turns.append((str(role), _clip(text, MAX_TURN), _rec_ts_from(rec)))
    return turns


def _codex_user_text(raw: str) -> str:
    if "## My request:" in raw:
        raw = raw.split("## My request:", 1)[-1]
    raw = re.sub(r"<environment_context>.*?</environment_context>", "", raw, flags=re.S)
    raw = re.sub(r"<in-app-browser-context\b[^>]*>.*?</in-app-browser-context\s*>", "", raw, flags=re.S | re.I)
    return raw.strip()


def _parse_codex(path: Path) -> list[ParsedTurn]:
    turns: list[ParsedTurn] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("type") != "response_item":
            continue
        payload = rec.get("payload") or {}
        if payload.get("type") != "message":
            continue
        role = payload.get("role")
        parts = payload.get("content") or []
        texts = [str(part.get("text") or "") for part in parts
                 if isinstance(part, dict) and part.get("type") in ("input_text", "output_text", "text")]
        text = "\n".join(t for t in texts if t.strip()).strip()
        if not text or role == "developer":
            continue
        if role == "user":
            meta = (payload.get("internal_chat_message_metadata_passthrough") or {}).get("content_item_metadata") or []
            if meta and all((m or {}).get("provenance", {}).get("type") != "user" for m in meta if isinstance(m, dict)):
                continue
            text = _codex_user_text(text)
            if not text or text.startswith("# Files mentioned"):
                continue
        if role == "assistant" and len(text) < 40:
            continue
        turns.append((str(role), _clip(text, MAX_TURN), _rec_ts_from(rec)))
    return turns


def _parse_opencode(db_path: Path) -> list[RawSession]:
    out: list[RawSession] = []
    con = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        meta = {r["id"]: dict(r) for r in con.execute("SELECT id, time_created FROM session_v2")}
        buckets: dict[str, list[ParsedTurn]] = {}
        for r in con.execute("SELECT session_id, type, seq, data FROM session_message ORDER BY session_id, seq"):
            kind = r["type"]
            if kind not in ("user", "assistant"):
                continue
            try:
                data = json.loads(r["data"] or "{}")
            except ValueError:
                continue
            if kind == "user":
                text = str(data.get("text") or "").strip()
            else:
                text = "\n".join(str(p.get("text") or "") for p in (data.get("content") or [])
                                 if isinstance(p, dict) and p.get("type") == "text").strip()
            if not text:
                continue
            buckets.setdefault(r["session_id"], []).append(
                (kind, _clip(text, MAX_TURN), _rec_ts_from(data) or _rec_ts_from(r)))
        for sid, turns in buckets.items():
            created = (meta.get(sid) or {}).get("time_created") or 0
            session_ts = _rec_ts(created)
            day = (_local_day(session_ts) if session_ts else _day_of(db_path))
            out.append(RawSession("opencode", sid, f"{db_path.name}#{sid}", day, turns,
                                  session_ts=session_ts))
    finally:
        con.close()
    return out


def _collect_claude_code() -> list[RawSession]:
    root = HOME / ".claude" / "projects"
    if not root.is_dir():
        return []
    return [RawSession("claude-code", p.stem, p.as_posix(), _day_of(p), _parse_claude(p))
            for p in sorted(root.rglob("*.jsonl"))]


def _collect_codex() -> list[RawSession]:
    root = HOME / ".codex" / "sessions"
    if not root.is_dir():
        return []
    return [RawSession("codex", p.stem, p.as_posix(), _day_of(p), _parse_codex(p))
            for p in sorted(root.rglob("rollout-*.jsonl"))]


def _collect_opencode() -> list[RawSession]:
    candidates = [HOME / ".local" / "share" / "opencode" / "opencode.db"]
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        candidates.append(Path(xdg) / "opencode" / "opencode.db")
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        candidates.append(Path(local_appdata) / "opencode" / "opencode.db")
    candidates.append(HOME / "AppData" / "Local" / "opencode" / "opencode.db")
    for db in candidates:
        if db.is_file():
            return _parse_opencode(db)
    return []


_SYSREM = re.compile(r"<system-reminder\b[^>]*>.*?</system-reminder>", re.S)
_USER_TAG = re.compile(r"</?user_query[^>]*>")


def _parse_workbuddy(path: Path) -> list[ParsedTurn]:
    turns: list[ParsedTurn] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("type") != "message":
            continue
        role = rec.get("role")
        if role not in ("user", "assistant"):
            continue
        parts = rec.get("content") or []
        text = "\n".join(str(p.get("text") or "") for p in parts
                         if isinstance(p, dict) and p.get("type") in ("input_text", "output_text", "text")).strip()
        if role == "user":
            # injected context is not user-typed evidence; drop it, keep the query body
            text = clean_turn(_USER_TAG.sub("", _SYSREM.sub("", text))).strip()
        if not text:
            continue
        turns.append((str(role), _clip(text, MAX_TURN), _rec_ts_from(rec)))
    return turns


def _collect_workbuddy() -> list[RawSession]:
    root = HOME / ".workbuddy" / "projects"
    if not root.is_dir():
        return []
    return [RawSession("workbuddy", p.stem, p.as_posix(), _day_of(p), _parse_workbuddy(p))
            for p in sorted(root.rglob("*.jsonl"))]


def _dsh_content_text(parts: Any) -> str:
    """Join the plain-text parts of a dsh message content array."""
    if isinstance(parts, str):
        return parts.strip()
    if not isinstance(parts, list):
        return ""
    chunks = []
    for part in parts:
        if isinstance(part, dict) and part.get("type") in ("text", "input_text", "output_text"):
            chunks.append(str(part.get("text") or ""))
    return "\n".join(c for c in chunks if c.strip()).strip()


def _parse_dsh(path: Path) -> list[tuple[str, str]]:
    """Parse one session.v4.jsonl.zstd transcript (DeepSeek Harness)."""
    try:
        lines = _decompress_zstd(path.read_bytes()).decode("utf-8", errors="replace").splitlines()
    except (OSError, RuntimeError, ValueError):
        return []
    turns: list[ParsedTurn] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = rec.get("type")
        if kind not in ("user/message", "assistant/message"):
            continue
        role = "user" if kind == "user/message" else "assistant"
        message = (rec.get("data") or {}).get("message") or {}
        text = _dsh_content_text(message.get("content"))
        if role == "user":
            text = _USER_TAG.sub("", _SYSREM.sub("", text)).strip()
        if not text:
            continue
        turns.append((role, _clip(text, MAX_TURN),
                      _rec_ts_from(rec) or _rec_ts_from(rec.get("data")) or _rec_ts_from(message)))
    return turns


_DSH_CODEPOINT = re.compile(r"~([0-9A-Fa-f]{4})")


def _dsh_workspace(folder: str) -> str:
    """Decode a workspace folder name such as --D-DSH~0020OUTPUT--.

    Non-alphanumeric characters are stored as ~XXXX hex code points
    (~0020 = space); the dashes at both ends are just delimiters.
    """
    name = folder.strip("-")
    decoded = _DSH_CODEPOINT.sub(lambda m: chr(int(m.group(1), 16)), name)
    return decoded.replace("-", " ").strip() or name


def _collect_dsh() -> list[RawSession]:
    """DeepSeek Harness transcripts: ~/.dsh/sessions/<workspace>/<id>/session.v4.jsonl.zstd."""
    root = HOME / ".dsh" / "sessions"
    if not root.is_dir():
        return []
    out: list[RawSession] = []
    for archive in sorted(root.glob("*/*/session*.jsonl.zstd")):
        session_dir = archive.parent
        # Sub-agent transcripts duplicate their parent conversation; the top
        # level session keeps the full user-visible dialogue.
        try:
            head = json.loads(_decompress_zstd(archive.read_bytes()).decode("utf-8", errors="replace")
                              .splitlines()[0])
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError, IndexError):
            continue
        if not isinstance(head, dict) or head.get("origin") == "subagent":
            continue
        workspace = _dsh_workspace(root.name and session_dir.parent.name or "")
        turns = _parse_dsh(archive)
        if not turns:
            continue
        day = _day_of(archive)
        out.append(RawSession("dsh", session_dir.name, archive.as_posix(), day, turns,
                              title=f"[dsh] {workspace}" if workspace else ""))
    return out


def _collect_cursor() -> list[RawSession]:
    candidates = [
        HOME / "AppData" / "Roaming" / "Cursor" / "User" / "globalStorage" / "conversation-search.db",
        HOME / "Library" / "Application Support" / "Cursor" / "User" / "globalStorage" / "conversation-search.db",
        HOME / ".config" / "Cursor" / "User" / "globalStorage" / "conversation-search.db",
    ]
    out: list[RawSession] = []
    seen: set[Path] = set()
    for db in candidates:
        if not db.is_file():
            continue
        resolved = db.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        con = sqlite3.connect(resolved.as_uri() + "?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        try:
            rows = con.execute(
                "SELECT c.id AS cid, c.title AS ctitle, c.updated_at, f.c0 AS ftitle, f.c1 AS body "
                "FROM conversations c JOIN conversation_fts_content f ON f.id = c.fts_rowid").fetchall()
            for r in rows:
                body = (r["body"] or "").strip()
                if len(body) < 80:
                    continue
                title = (r["ctitle"] or r["ftitle"] or "").strip()
                session_ts = _rec_ts(r["updated_at"])
                day = _local_day(session_ts) if session_ts else _day_of(db)
                out.append(RawSession("cursor", str(r["cid"]), f"conversation-search.db#{r['cid']}", day,
                                      [], title=f"[cursor] {title[:60]}" if title else "",
                                      note_body="# 对话记录\n\n" + body, session_ts=session_ts))
        finally:
            con.close()
    return out


def _content_text(content: Any) -> str:
    """Pull plain text out of a chat message content field (string or blocks)."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                for key in ("text", "value", "content", "markdown"):
                    if isinstance(block.get(key), str):
                        parts.append(block[key])
                        break
        return "\n".join(p for p in parts if p).strip()
    if isinstance(content, dict):
        for key in ("text", "value", "content", "markdown"):
            if isinstance(content.get(key), str):
                return content[key].strip()
    return ""


def _walk_chat_turns(obj: Any, turns: list[ParsedTurn], seen: set[str]) -> None:
    """Tolerant walk for role/content message objects in VS Code family chat stores.

    The on-disk shape drifts between VS Code / Trae / Qoder versions, so we
    recognise messages structurally instead of binding to one schema.
    """
    if isinstance(obj, dict):
        role = obj.get("role")
        if isinstance(role, str) and role.lower() in ("user", "assistant", "human", "ai"):
            text = _content_text(obj.get("content") if obj.get("content") is not None
                                 else obj.get("text") if obj.get("text") is not None
                                 else obj.get("message"))
            if text:
                key = f"{role}|{' '.join(text.split())[:200]}"
                if key not in seen:
                    seen.add(key)
                    turns.append(("user" if role.lower() in ("user", "human") else "assistant",
                                  text[:8000], _rec_ts_from(obj)))
            return
        for value in obj.values():
            _walk_chat_turns(value, turns, seen)
    elif isinstance(obj, list):
        for value in obj:
            _walk_chat_turns(value, turns, seen)


def _chat_session_ts(meta: Any) -> datetime | None:
    """Session-level anchor for VS Code family stores: start time if kept."""
    if not isinstance(meta, dict):
        return None
    timing = meta.get("timing") if isinstance(meta.get("timing"), dict) else {}
    return (_rec_ts((timing or {}).get("created")) or _rec_ts(meta.get("lastMessageDate"))
            or _rec_ts_from(meta))


def _chat_day(meta: dict[str, Any], fallback: Path) -> str:
    ts = _chat_session_ts(meta)
    return _local_day(ts) if ts else _day_of(fallback)


def _read_chat_sessions(vscdb: Path) -> list[tuple[str, dict[str, Any], list[ParsedTurn]]]:
    """(session_id, meta, turns) for every non-empty chat session in one state.vscdb."""
    con = sqlite3.connect(vscdb.as_uri() + "?mode=ro", uri=True)
    out: list[tuple[str, dict[str, Any], list[ParsedTurn]]] = []
    try:
        row = con.execute("SELECT value FROM ItemTable WHERE key='chat.ChatSessionStore.index'").fetchone()
        if row is None:
            return out
        raw = row[0]
        index = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
        entries = index.get("entries") or {}
        for sid, meta in entries.items():
            turns: list[ParsedTurn] = []
            seen: set[str] = set()
            if not meta.get("isEmpty"):
                body_rows = con.execute(
                    "SELECT value FROM ItemTable WHERE key LIKE ? OR key LIKE ?",
                    (f"chat.ChatSessionStore.{sid}%", f"%{sid}%",)).fetchall()
                for body in body_rows:
                    blob = body[0]
                    try:
                        data = json.loads(blob.decode("utf-8", errors="replace") if isinstance(blob, bytes) else blob)
                    except (ValueError, AttributeError):
                        continue
                    _walk_chat_turns(data, turns, seen)
            if turns:
                out.append((str(sid), dict(meta), turns))
    finally:
        con.close()
    return out


def _collect_vscode_family() -> list[RawSession]:
    """VS Code / Trae / Qoder chat sessions (ChatSessionStore + chatSessions/*.json)."""
    roots = [
        ("vscode", HOME / "AppData" / "Roaming" / "Code"),
        ("trae", HOME / "AppData" / "Roaming" / "Trae"),
        ("qoder", HOME / "AppData" / "Roaming" / "Qoder"),
    ]
    out: list[RawSession] = []
    for source, root in roots:
        ws = root / "User" / "workspaceStorage"
        if not ws.is_dir():
            continue
        for vscdb in ws.glob("*/state.vscdb"):
            try:
                for sid, meta, turns in _read_chat_sessions(vscdb):
                    title = str(meta.get("title") or "").strip()
                    out.append(RawSession(source, f"{vscdb.parent.name}:{sid}",
                                          f"{vscdb}#{sid}", _chat_day(meta, vscdb), turns,
                                          title=f"[{source}] {title[:60]}" if title else "",
                                          session_ts=_chat_session_ts(meta)))
            except Exception:
                continue
        for session_json in ws.glob("*/chatSessions/*.json"):
            try:
                data = json.loads(session_json.read_text(encoding="utf-8", errors="replace"))
            except (OSError, ValueError):
                continue
            turns: list[ParsedTurn] = []
            seen: set[str] = set()
            _walk_chat_turns(data, turns, seen)
            if turns:
                out.append(RawSession(source, f"{session_json.parent.parent.name}:{session_json.stem}",
                                      session_json.as_posix(), _day_of(session_json), turns,
                                      title=f"[{source}] {session_json.stem[:60]}",
                                      session_ts=_chat_session_ts(data)))
    return out


def _collect_vscode() -> list[RawSession]:
    return [s for s in _collect_vscode_family() if s.source == "vscode"]


def _collect_trae() -> list[RawSession]:
    return [s for s in _collect_vscode_family() if s.source == "trae"]


_QODER_HOME_FOLDERS = (".qoder-cn", ".qoder")


def _parse_qoder(path: Path) -> list[ParsedTurn]:
    """Qoder transcripts: Claude-shaped records that also say who typed them.

    Most `type: user` records here are tool results flowing back into the
    conversation, not the owner speaking. Only `origin.kind == "human"` is the
    owner's own typed text, and evidence must be exactly that — capturing the
    rest would let model output become memory evidence.
    """
    turns: list[ParsedTurn] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(rec, dict) or rec.get("isSidechain"):
            continue
        kind = rec.get("type")
        if kind == "user":
            origin = rec.get("origin")
            if not (isinstance(origin, dict) and origin.get("kind") == "human"):
                continue
        elif kind != "assistant":
            continue
        msg = rec.get("message") or {}
        content = msg.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "\n".join(p.get("text", "") for p in content
                             if isinstance(p, dict) and p.get("type") == "text")
        else:
            continue
        text = text.strip()
        if not text or text.startswith("<system-reminder>"):
            continue
        turns.append((str(kind), _clip(text, MAX_TURN), _rec_ts_from(rec)))
    return turns


def _collect_qoder_transcripts() -> list[RawSession]:
    """`~/.qoder-cn/projects/<workspace>/<session>.jsonl` (and the .qoder build).

    The VS Code-fork layout under %APPDATA%/Qoder is what this collector used to
    look for, and it does not exist on this machine: Qoder keeps its transcripts
    in a home directory, the way Claude Code does. That is why a live Qoder
    conversation produced zero captured turns.
    """
    out: list[RawSession] = []
    for folder in _QODER_HOME_FOLDERS:
        root = HOME / folder / "projects"
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*/*.jsonl")):
            turns = _parse_qoder(path)
            if not turns:
                continue
            stamps = [t[2] for t in turns if t[2]]
            session_ts = max(stamps) if stamps else None
            day = _local_day(session_ts) if session_ts else _day_of(path)
            out.append(RawSession("qoder", path.stem, path.as_posix(), day, turns,
                                  title=f"[qoder] {path.stem[:40]}", session_ts=session_ts))
    return out


def _collect_qoder() -> list[RawSession]:
    transcripts = _collect_qoder_transcripts()
    forked = [s for s in _collect_vscode_family() if s.source == "qoder"]
    seen = {s.origin for s in transcripts}
    return transcripts + [s for s in forked if s.origin not in seen]


SOURCES: dict[str, Callable[[], list[RawSession]]] = {
    "claude-code": _collect_claude_code,
    "codex": _collect_codex,
    "opencode": _collect_opencode,
    "workbuddy": _collect_workbuddy,
    "cursor": _collect_cursor,
    "dsh": _collect_dsh,
    "vscode": _collect_vscode,
    "trae": _collect_trae,
    "qoder": _collect_qoder,
}


def collect(sources: list[str] | None = None) -> list[RawSession]:
    out: list[RawSession] = []
    for name in [s for s in SOURCES if sources is None or s in sources]:
        try:
            out.extend(SOURCES[name]())
        except Exception:  # one broken store must not block the rest (fail-open)
            continue
    return out


# ---------------------------------------------------------------- cleaning

def _clean_turns(turns: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return [(role, text) for role, text in turns if not _NOISE_TURN.match(text.strip())]


def _render_dialog(turns: list[tuple[str, str]]) -> str:
    lines = ["# 对话记录", ""]
    for role, text in turns:
        lines.append("## " + ("用户" if role == "user" else "助手"))
        lines.append("")
        lines.append(text)
        lines.append("")
    return _clip("\n".join(lines).strip(), MAX_BODY)


def _title(source: str, origin: str, turns: list[tuple[str, str]]) -> str:
    for role, text in turns:
        if role == "user":
            one = re.sub(r"\s+", " ", text.split("\n", 1)[0])[:72]
            return f"[{source}] {one or origin.rsplit('#', 1)[-1][:40]}"
    return f"[{source}] {origin.rsplit('#', 1)[-1][:40]}"


def _prepare(rs: RawSession) -> tuple[str, str, str, str] | None:
    """(body, entry_id, rule_title, content_hash) — None when the session is noise."""
    if rs.note_body is not None:
        body = _clip(rs.note_body, MAX_BODY)
        if len(body) < 80:
            return None
    else:
        turns = _clean_turns(rs.turns)
        body = _render_dialog(turns)
        if len(turns) < 2:
            return None
    chash = content_hash(body)
    rule_title = rs.title or _title(rs.source, rs.origin, rs.turns)
    return body, entry_id(rs.source, rs.scope, rs.origin, chash, rs.date), rule_title, chash


def _existing_entry(svc: Any, source: str, scope: str, origin: str, chash: str) -> str | None:
    """Same (source, scope, origin, content_hash) already imported — id may differ by batch day.

    Entry files are the source of truth (sqlite is derived), so scan files.
    """
    for entry in svc.archive.files.iter_entries():
        if (entry.source == source and entry.scope == scope
                and entry.origin == origin and entry.content_hash == chash):
            return entry.id
    return None


def _existing_id(svc: Any, rs: RawSession, chash: str) -> str | None:
    return _existing_entry(svc, rs.source, rs.scope, rs.origin, chash)


def _session_key(rs: RawSession) -> str:
    """Stable per-session key for selection, independent of file content."""
    return hashlib.sha256(f"{rs.source}|{rs.scope}|{rs.origin}".encode("utf-8")).hexdigest()[:16]


SKIPS_FILE = ".state/import-skips.json"


def _skips_path(svc: Any) -> Path:
    path = Path(svc.settings.home) / SKIPS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_skips(svc: Any) -> set[str]:
    path = _skips_path(svc)
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {str(x) for x in data if isinstance(x, str)}
    except (OSError, ValueError):
        return set()


def _save_skips(svc: Any, keys: set[str]) -> None:
    _atomic_write(_skips_path(svc), json.dumps(sorted(keys), ensure_ascii=False, indent=2))


def update_skips(svc: Any, *, add: list[str] | None = None, remove: list[str] | None = None,
                 clear: bool = False) -> set[str]:
    """Persist 'do not import' selections so they stop appearing as new."""
    keys: set[str] = set() if clear else load_skips(svc)
    keys.update(str(k) for k in (add or []))
    keys.difference_update(str(k) for k in (remove or []))
    _save_skips(svc, keys)
    return keys


_TITLE_ARTIFACT = re.compile(r"<command-(?:name|message|args)>|local-command-stdout|No response requested", re.I)


def _body_is_noise(body: str) -> bool:
    """True when the entry carries no substantive text (conservative).

    Dialog bodies are judged turn by turn; plain notes lose at most one leading
    H1 title line and are judged on the rest.
    """
    text = (body or "").strip()
    if not text:
        return True
    if re.search(r"^## (?:用户|助手)\s*$", text, flags=re.M):
        blocks = [b.strip() for b in re.split(r"^## (?:用户|助手)\s*$", text, flags=re.M)]
        texts = [b for b in blocks if b and not b.startswith("# 对话记录")]
    else:
        rest = re.sub(r"\A#\s+[^\n]*\n?", "", text).strip()
        texts = [rest] if rest else []
    return not [t for t in texts if not _NOISE_TURN.match(t) and len(t) >= 8]


def find_noise_entries(svc: Any) -> list[dict[str, Any]]:
    """Read-only D2.7 review: entries whose body/title are pure artifacts."""
    hidden = svc.archive.hidden_ids()
    out: list[dict[str, Any]] = []
    for entry in svc.archive.files.iter_entries():
        if entry.id in hidden:
            continue
        body = entry.body or ""
        is_noise = _body_is_noise(body)
        if not is_noise and _TITLE_ARTIFACT.search(entry.title or "") and len(body) < 300:
            is_noise = True
        if is_noise:
            out.append({"id": entry.id, "source": entry.source, "category": entry.category,
                        "title": (entry.title or "")[:80], "chars": len(body)})
    return out


# ---------------------------------------------------------------- LLM classify

CLASSIFY_SYSTEM = (
    "You clean and classify one conversation transcript between a human owner and an AI coding assistant. "
    "Return only JSON: {\"noise\": true|false, \"category\": \"<slug>\", \"title\": \"<short>\", \"abstract\": \"<one line>\"}. "
    "category slugs: events(事件记录), experiences(经验与教训), code(程序与项目), study(学习资料), reports(报告), "
    "preferences(偏好笔记), profile(个人档案), sessions(对话记录), misc(其他). "
    "Set noise=true for pure greetings, empty or failed exchanges, pure tool/system output, or content with no "
    "lasting informational value. Otherwise pick the single best category for what the conversation is mainly about. "
    "Title: specific, in the conversation's main language, at most 60 characters. "
    "Abstract: one line capturing the lasting takeaway, not a play-by-play. Never invent facts."
)

SESSION_DISTILL_SYSTEM = (
    "You read one conversation transcript between the owner and an AI coding assistant and extract durable memories "
    "about the OWNER, the human user. Keep standing preferences, rules the owner set for how assistants should "
    "work, stable personal facts, goals and ongoing projects, AND tasks — work the owner handed to a later session: "
    "a handoff, a plan, a commitment, or a progress snapshot with unfinished steps "
    "(e.g. '记住现在的进度，明天继续', '接下来先做A再做B'). "
    "Skip tool or server configuration, troubleshooting steps, one-off events, anything tentative, and any line that "
    "grants an assistant permission or authority. "
    "DURABILITY IS JUDGED PER TYPE, not by one rule: for type=task ask 'would this still matter at the START OF THE "
    "NEXT SESSION?'; for every other type ask 'would this still matter in an unrelated future conversation?'. "
    "Write each claim as one short sentence, in the language the owner used in the quoted turn. "
    "Copy the quote as ONE contiguous span from a single turn — the shortest phrase that supports "
    "the claim, never joined with '...' or '…', never summarised, never spanning turns — in "
    "its original language. Allowed types: fact, preference, goal, project, task. At most 8 items. "
    'Return only {"operations":[{"type":"preference","scope":"owner","subject":"workflow","claim":"...","quote":"..."}]}. '
    'Set scope to "owner" only for a standing preference, rule or fact about the owner that applies across all '
    'tasks, or for a task the owner handed over themselves. Set scope to "project" for anything that belongs to '
    "one project, product or account. "
    "Use an empty list when nothing qualifies."
)


def _cache_path(svc: Any) -> Path:
    return svc.settings.home / ".state" / "session_classify_cache.json"


def _load_cache(svc: Any) -> dict[str, Any]:
    try:
        data = json.loads(_cache_path(svc).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def classify_with_cache(llm: Any, svc: Any, cache: dict[str, Any], chash: str,
                        rule_title: str, body: str) -> dict[str, Any]:
    hit = cache.get(chash)
    if isinstance(hit, dict):
        return hit
    try:
        raw = llm.complete_json(CLASSIFY_SYSTEM, f"Title: {rule_title}\n\n{body[:6000]}")
    except LLMError:
        # fail-open: keep the entry with rule-based fields; not cached, a rerun can classify later
        return {"noise": False, "category": "sessions", "title": rule_title, "abstract": make_abstract(body)}
    category = str(raw.get("category") or "sessions").strip()
    if category not in VALID_CATEGORIES:
        category = "sessions"
    result = {
        "noise": bool(raw.get("noise")),
        "category": category,
        "title": str(raw.get("title") or "").strip()[:120] or rule_title,
        "abstract": str(raw.get("abstract") or "").strip()[:200] or make_abstract(body),
    }
    cache[chash] = result
    _atomic_write(_cache_path(svc), json.dumps(cache, ensure_ascii=False))
    return result


# ---------------------------------------------------------------- plan / apply

# Internal source key -> name shown in the dashboard. Keys without a spec
# here are shown as-is; agents.py is the single place that knows the names.
SOURCE_LABELS: dict[str, str] = {}
for _spec in agents.SPECS:
    SOURCE_LABELS[_spec.key] = _spec.name
SOURCE_LABELS.update({
    "dsh": "DeepSeek Harness",
    "claude-code": SOURCE_LABELS.get("claude-code", "Claude Code"),
    "openviking": "OpenViking",
    "openviking-import": "OpenViking 导入",
    "unknown-agent": "未知来源",
    "unknown": "未标记来源",
    "membrain-dashboard": "工作台录入",
})


def source_label(key: str) -> str:
    return SOURCE_LABELS.get(key, key)


def preview(svc: Any, sources: list[str] | None = None) -> dict[str, Any]:
    """What an import would do. Writes nothing.

    Per-session rows let the owner pick what to import and what to ignore;
    ignored keys persist in .state/import-skips.json until restored.
    """
    rows: dict[str, dict[str, int]] = {}
    skips = load_skips(svc)
    sessions = collect(sources)
    detail: list[dict[str, Any]] = []
    for rs in sessions:
        row = rows.setdefault(rs.source, {"source": rs.source, "label": source_label(rs.source),
                                          "found": 0, "already": 0, "noise": 0, "skipped": 0})
        row["found"] += 1
        key = _session_key(rs)
        prepared = _prepare(rs)
        title = rs.title or _title(rs.source, rs.origin, rs.turns)
        date = rs.date
        turns = len(rs.turns or [])
        if prepared is None:
            row["noise"] += 1
            status = "noise"
        else:
            _, eid, _, chash = prepared
            if svc.archive.files.exists(eid) or _existing_entry(svc, rs.source, rs.scope, rs.origin, chash):
                row["already"] += 1
                status = "already"
            elif key in skips:
                row["skipped"] += 1
                status = "skipped"
            else:
                status = "new"
        detail.append({"key": key, "source": rs.source, "label": source_label(rs.source),
                       "title": title[:80], "date": date, "turns": turns, "status": status})
    listed = [rows[name] for name in SOURCES if name in rows]
    return {
        "sources": listed,
        "sessions": detail[:500],
        "sessions_truncated": len(detail) > 500,
        "total": sum(r["found"] for r in listed),
        "already_imported": sum(r["already"] for r in listed),
        "new": sum(r["found"] - r["already"] - r["noise"] - r.get("skipped", 0) for r in listed),
        "noise": sum(r["noise"] for r in listed),
        "skipped": sum(r.get("skipped", 0) for r in listed),
    }


def _norm(text: str) -> str:
    return " ".join(text.casefold().split())


def _distill_entry(llm: Any, entry: ArchiveEntry) -> tuple[list[dict[str, str]], int]:
    for wait in RATE_LIMIT_WAITS:
        try:
            return ov.distill(llm, entry, system=SESSION_DISTILL_SYSTEM)
        except LLMError:
            time.sleep(wait)
    return ov.distill(llm, entry, system=SESSION_DISTILL_SYSTEM)


# ---------------------------------------------------------------- batch manifest

# A batch lands in imports/<batch>.json at three points: "in_progress" before
# the first archive write, again after every session, and once more when the pass
# ends. Only an error-free pass flips it to "completed"; a hard failure leaves
# "in_progress" so the next run adopts the same batch id and keeps going.
IMPORT_SOURCE = "session-import"
STATUS_RUNNING = "in_progress"
STATUS_DONE = "completed"
STATUS_PARTIAL = "partial"          # pass finished, but sessions or facts were lost

STEP_CLEAN = "清洗"
STEP_CLASSIFY = "分类"
STEP_WRITE = "归档写入"
STEP_SYNC = "archive 同步"
STEP_DISTILL = "事实提炼"
STEP_MANIFEST = "清单写入"

_COUNTERS = ("archived_new", "archive_existing", "noise_skipped", "facts_added",
             "facts_duplicate", "rejected", "not_selected", "skipped")


def _manifest_path(svc: Any, batch: str) -> Path:
    folder = Path(svc.settings.home) / "imports"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{batch}.json"


def load_manifest(svc: Any, batch: str) -> dict[str, Any] | None:
    try:
        data = json.loads(_manifest_path(svc, batch).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _save_manifest(svc: Any, manifest: dict[str, Any]) -> None:
    manifest["updated"] = datetime.now(timezone.utc).isoformat()
    _atomic_write(_manifest_path(svc, manifest["batch"]),
                  json.dumps(manifest, ensure_ascii=False, indent=2))


def pending_batches(svc: Any) -> list[dict[str, Any]]:
    """Imports that started and never finished (crash, kill, abort). Newest first.

    These are the batches a rerun may adopt. `partial` batches are deliberately
    excluded: their pass ended, so retrying them is a fresh batch, not a resume.
    """
    folder = Path(svc.settings.home) / "imports"
    if not folder.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for path in folder.glob("b_*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (isinstance(data, dict) and data.get("source") == IMPORT_SOURCE
                and data.get("status") == STATUS_RUNNING):
            out.append(data)
    return sorted(out, key=lambda m: str(m.get("created") or ""), reverse=True)


def _selection(sources: list[str] | None, only_set: set[str] | None) -> tuple[Any, Any]:
    return (sorted(sources) if sources is not None else None,
            sorted(only_set) if only_set is not None else None)


def _resume_target(svc: Any, sources: list[str] | None,
                   only_set: set[str] | None) -> dict[str, Any] | None:
    """Pending batch started with the same selection, if any.

    Same selection means the rerun is the continuation of that batch, not a
    different request; adopting it keeps one batch id over the whole set so
    undo still covers everything the two runs archived.
    """
    want = _selection(sources, only_set)
    for manifest in pending_batches(svc):
        if (manifest.get("sources"), manifest.get("only")) == want:
            return manifest
    return None


def _step_error(where: str, step: str, exc: BaseException) -> str:
    """`<session>｜<step>失败: <reason>` — every error says both."""
    detail = str(exc)[:160] or exc.__class__.__name__
    return f"{where}｜{step}失败: {detail}"


def apply_import(svc: Any, llm: Any, sources: list[str] | None = None,
                 distill_facts: bool = False, progress: Any = None,
                 only: list[str] | None = None, skip: list[str] | None = None,
                 unskip: list[str] | None = None, clear_skips: bool = False) -> dict[str, Any]:
    """Import sessions into the archive.

    only:  import just these session keys (None = everything eligible).
    skip / unskip / clear_skips: manage the persistent ignore list first;
    ignored sessions are neither imported nor offered again until restored.

    Crash safety: the batch manifest is written as `in_progress` (batch id,
    selection, every session key, the cursor) before the first archive write and
    refreshed after each session. A failure aborts the pass, leaves the manifest
    `in_progress` and re-raises naming the session and step, so the archive is
    never half-written with no record of why. A rerun with the same selection
    adopts that batch id and continues: entries already on disk are recognised by
    (source, scope, origin, content_hash), so finished sessions are counted as
    existing instead of being archived twice. Only an error-free pass flips the
    manifest to `completed`.
    """
    from datetime import date as _date

    if skip or unskip or clear_skips:
        update_skips(svc, add=skip, remove=unskip, clear=clear_skips)
    skips = load_skips(svc)
    only_set = set(only) if only is not None else None
    if only_set:
        # Picking a previously ignored session means it is wanted after all.
        update_skips(svc, remove=list(only_set))
        skips = load_skips(svc)
    cache = _load_cache(svc)
    sessions = collect(sources)
    session_keys = [_session_key(rs) for rs in sessions]

    previous = _resume_target(svc, sources, only_set)
    started = datetime.now(timezone.utc).isoformat()
    batch = (previous or {}).get("batch") or "b_" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    created = str((previous or {}).get("created") or started)
    # Kept across a resume so undo_one_batch still covers the earlier run's work.
    archive_ids: list[str] = [str(x) for x in (previous or {}).get("archive_ids") or []]
    fact_ids: list[str] = [str(x) for x in (previous or {}).get("fact_ids") or []]

    report: dict[str, Any] = {"batch": batch, "resumed": previous is not None,
                              "status": STATUS_RUNNING, "total": len(sessions),
                              "archived_new": 0, "archive_existing": 0, "noise_skipped": 0,
                              "facts_added": 0, "facts_duplicate": 0, "rejected": 0,
                              "not_selected": 0, "skipped": 0, "errors": []}
    done_keys: list[str] = []
    state = {"stage": STEP_CLEAN, "session": "", "cursor": 0}
    sources_spec, only_spec = _selection(sources, only_set)

    def snapshot(status: str) -> dict[str, Any]:
        done = set(done_keys)
        return {
            "batch": batch, "status": status, "stage": state["stage"], "source": IMPORT_SOURCE,
            "created": created, "sources": sources_spec, "only": only_spec,
            "total": len(sessions), "session_keys": session_keys, "done_keys": list(done_keys),
            "pending_keys": [k for k in session_keys if k not in done],
            "cursor": state["cursor"], "last_session": state["session"],
            **{name: report[name] for name in _COUNTERS},
            "archive_ids": list(archive_ids), "fact_ids": list(fact_ids),
            "errors": list(report["errors"]),
        }

    def checkpoint() -> None:
        """Persist progress; bookkeeping trouble must not mask the real failure."""
        try:
            _save_manifest(svc, snapshot(STATUS_RUNNING))
        except Exception as exc:
            report["errors"].append(_step_error(batch, "进度清单写入", exc))

    try:
        _save_manifest(svc, snapshot(STATUS_RUNNING))
    except Exception as exc:  # nothing is archived yet, so failing here is safe
        raise RuntimeError(_step_error(batch, STEP_MANIFEST, exc)) from exc

    entries: list[ArchiveEntry] = []
    try:
        for n, rs in enumerate(sessions, 1):
            if progress:
                progress({"stage": "cleaning", "done": n - 1, "total": len(sessions)})
            key = _session_key(rs)
            state["session"] = f"{rs.source}/{rs.scope}"
            state["stage"] = STEP_CLEAN
            try:
                if only_set is not None and key not in only_set:
                    report["not_selected"] += 1
                    continue
                if key in skips and (only_set is None or key not in only_set):
                    report["skipped"] += 1
                    continue
                prepared = _prepare(rs)
                if prepared is None:
                    report["noise_skipped"] += 1
                    continue
                body, eid, rule_title, chash = prepared
                if svc.archive.files.exists(eid) or _existing_entry(svc, rs.source, rs.scope, rs.origin, chash):
                    report["archive_existing"] += 1
                    continue
                state["stage"] = STEP_CLASSIFY
                cls = classify_with_cache(llm, svc, cache, chash, rule_title, body)
                if cls["noise"]:
                    report["noise_skipped"] += 1
                    continue
                entry = ArchiveEntry(id=eid, source=rs.source, category=cls["category"], title=cls["title"],
                                     abstract=cls["abstract"], body=body, date=rs.date, origin=rs.origin,
                                     scope=rs.scope, batch=batch, content_hash=chash)
                state["stage"] = STEP_WRITE
                svc.archive.files.write(entry)
                entries.append(entry)
                archive_ids.append(eid)
                report["archived_new"] += 1
                if CALL_PAUSE:
                    time.sleep(CALL_PAUSE)
            except Exception as exc:  # abort: stop before the archive drifts further
                report["errors"].append(_step_error(state["session"], state["stage"], exc))
                raise
            finally:
                done_keys.append(key)
                state["cursor"] = n
                checkpoint()

        state["stage"] = STEP_SYNC
        if progress:
            progress({"stage": "indexing", "done": 0, "total": max(1, len(entries))})
        svc.archive.sync(embed=True)

        if distill_facts and entries:
            claims = {_norm(f.claim) for f in svc.store.live_facts()}
            quotes = {(e.episode, e.quote) for f in svc.store.all_facts() for e in f.evidence}
            for n, entry in enumerate(entries, 1):
                if progress:
                    progress({"stage": "distilling", "done": n - 1, "total": len(entries)})
                state["stage"] = STEP_DISTILL
                state["session"] = f"{entry.source}/{entry.scope}"
                try:
                    kept, rejected = _distill_entry(llm, entry)
                    report["rejected"] += rejected
                    for op in kept:
                        if _norm(op["claim"]) in claims or (entry.id, op["quote"]) in quotes:
                            report["facts_duplicate"] += 1
                            continue
                        evidence = Evidence(episode=entry.id, quote=op["quote"], origin="imported",
                                            date=_date.fromisoformat(entry.date))
                        fact = svc.store.write_fact(svc.store.draft_fact(
                            type=FactType(op["type"]), subject=op["subject"], claim=op["claim"],
                            evidence=[evidence], source_agent="session-import", source_session=batch))
                        svc.retriever.sync_fact(fact)
                        claims.add(_norm(op["claim"]))
                        fact_ids.append(fact.id)
                        report["facts_added"] += 1
                except Exception as exc:
                    # Facts are additive: the archive entry is already committed,
                    # so one bad distillation is reported, not re-run from scratch.
                    report["errors"].append(_step_error(state["session"], STEP_DISTILL, exc))
            if fact_ids:
                svc.retriever.scan_conflicts()

        state["stage"] = STEP_MANIFEST
        report["status"] = STATUS_PARTIAL if report["errors"] else STATUS_DONE
        _save_manifest(svc, snapshot(report["status"]))
    except Exception as exc:
        # Keep the batch resumable and say exactly where it stopped.
        message = _step_error(state["session"] or batch, state["stage"], exc)
        if message not in report["errors"]:
            report["errors"].append(message)
        report["status"] = STATUS_RUNNING
        try:
            _save_manifest(svc, snapshot(STATUS_RUNNING))
        except Exception:  # the manifest on disk already says in_progress
            pass
        raise RuntimeError(message) from exc

    svc.store.append_event({"op": "import", "id": batch, "source": IMPORT_SOURCE})
    return report


def import_session_archive(svc: Any, llm: Any, archive_path: str | Path,
                           progress: Any = None) -> dict[str, Any]:
    """Auto-increment path: turn one internal two-phase session archive into a library note.

    Reads meta.json + messages.jsonl (never modifies them). Same clean → classify
    → dedupe → write flow as the batch importer; entry is keyed by
    (source=agent, scope=session, origin=archive uri, content_hash), so re-runs
    are free and duplicates with the batch importer cannot happen.
    Fail-open: any error is reported, never raised.
    """
    path = Path(archive_path)
    report: dict[str, Any] = {"archive": str(path), "archived_new": 0, "archive_existing": 0,
                              "noise_skipped": 0, "errors": []}
    try:
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta = {}
    session = str(meta.get("session") or path.parent.parent.name)
    uri = str(meta.get("archive_uri") or f"mb://user/{path.as_posix()}")
    agent = str(meta.get("agent") or "")
    turns: list[tuple[str, str]] = []
    first_day: str | None = None
    try:
        for line in (path / "messages.jsonl").read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if not agent and rec.get("agent"):
                agent = str(rec["agent"])
            role = str(rec.get("role") or "")
            text = str(rec.get("text") or "").strip()
            if role not in ("user", "assistant") or not text:
                continue
            if role == "user":
                text = _USER_TAG.sub("", _SYSREM.sub("", text)).strip()
            if not text:
                continue
            if not first_day:
                try:
                    first_day = datetime.fromisoformat(str(rec.get("ts")).replace("Z", "+00:00")).date().isoformat()
                except ValueError:
                    first_day = None
            turns.append((role, _clip(text, MAX_TURN)))
    except OSError as exc:
        report["errors"].append(str(exc)[:160])
        return report
    agent = agent or "unknown-agent"

    rs = RawSession(agent, session, uri, first_day or _day_of(path), turns)
    prepared = _prepare(rs)
    if prepared is None:
        report["noise_skipped"] = 1
        return report
    body, eid, rule_title, chash = prepared
    if svc.archive.files.exists(eid) or _existing_entry(svc, agent, session, uri, chash):
        report["archive_existing"] = 1
        return report
    cls = classify_with_cache(llm, svc, _load_cache(svc), chash, rule_title, body)
    if cls["noise"]:
        report["noise_skipped"] = 1
        return report
    entry = ArchiveEntry(id=eid, source=agent, category=cls["category"], title=cls["title"],
                         abstract=cls["abstract"], body=body, date=rs.date, origin=uri,
                         scope=session, batch="b_" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f"),
                         content_hash=chash)
    svc.archive.files.write(entry)
    svc.archive.sync(embed=True)
    report["archived_new"] = 1
    return report
