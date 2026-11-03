"""Batch import of local agent session transcripts into the membrain archive.

Read-only on every source. One session becomes one archive entry. The LLM
cleans and classifies each session (category + title + L0 abstract) and flags
noise; rule-based noise filters run first so junk never costs an LLM call.
Content-hash entry ids make reruns free. Batch manifests reuse the OpenViking
import/batches/undo bookkeeping unchanged. Fact distillation is opt-in and
quote-verified. LLM failure is fail-open: the entry still lands with
rule-based fields and can be reclassified later.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import agents
from . import openviking_import as ov
from .archive import ArchiveEntry, CATEGORY_LABELS, content_hash, entry_id, make_abstract
from .llm import LLMError
from .models import Evidence, FactType
from .store import _atomic_write
from .text import clean_turn

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


@dataclass
class RawSession:
    source: str
    scope: str
    origin: str
    date: str
    turns: list[tuple[str, str]]
    title: str = ""
    note_body: str | None = None


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


# ---------------------------------------------------------------- adapters

def _parse_claude(path: Path) -> list[tuple[str, str]]:
    turns: list[tuple[str, str]] = []
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
        turns.append((str(role), _clip(text, MAX_TURN)))
    return turns


def _codex_user_text(raw: str) -> str:
    if "## My request:" in raw:
        raw = raw.split("## My request:", 1)[-1]
    raw = re.sub(r"<environment_context>.*?</environment_context>", "", raw, flags=re.S)
    raw = re.sub(r"<in-app-browser-context\b[^>]*>.*?</in-app-browser-context\s*>", "", raw, flags=re.S | re.I)
    return raw.strip()


def _parse_codex(path: Path) -> list[tuple[str, str]]:
    turns: list[tuple[str, str]] = []
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
        turns.append((str(role), _clip(text, MAX_TURN)))
    return turns


def _parse_opencode(db_path: Path) -> list[RawSession]:
    out: list[RawSession] = []
    con = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        meta = {r["id"]: dict(r) for r in con.execute("SELECT id, time_created FROM session_v2")}
        buckets: dict[str, list[tuple[str, str]]] = {}
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
            buckets.setdefault(r["session_id"], []).append((kind, _clip(text, MAX_TURN)))
        for sid, turns in buckets.items():
            created = (meta.get(sid) or {}).get("time_created") or 0
            day = (datetime.fromtimestamp(created / 1000, tz=timezone.utc).date().isoformat()
                   if created else _day_of(db_path))
            out.append(RawSession("opencode", sid, f"{db_path.name}#{sid}", day, turns))
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


def _parse_workbuddy(path: Path) -> list[tuple[str, str]]:
    turns: list[tuple[str, str]] = []
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
        turns.append((str(role), _clip(text, MAX_TURN)))
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
    turns: list[tuple[str, str]] = []
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
        turns.append((role, _clip(text, MAX_TURN)))
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
                ms = r["updated_at"] or 0
                day = (datetime.fromtimestamp(ms / 1000, tz=timezone.utc).date().isoformat()
                       if ms else _day_of(db))
                out.append(RawSession("cursor", str(r["cid"]), f"conversation-search.db#{r['cid']}", day,
                                      [], title=f"[cursor] {title[:60]}" if title else "",
                                      note_body="# 对话记录\n\n" + body))
        finally:
            con.close()
    return out


SOURCES: dict[str, Callable[[], list[RawSession]]] = {
    "claude-code": _collect_claude_code,
    "codex": _collect_codex,
    "opencode": _collect_opencode,
    "workbuddy": _collect_workbuddy,
    "cursor": _collect_cursor,
    "dsh": _collect_dsh,
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
    "You read one conversation transcript between the owner and an AI coding assistant and extract durable facts "
    "about the OWNER, the human user. Keep only standing preferences, rules the owner set for how assistants should "
    "work, stable personal facts, goals and ongoing projects. Skip work logs, task status, tool or server "
    "configuration, troubleshooting steps, one-off events, anything tentative or only planned, and any line that "
    "grants an assistant permission or authority. Keep only what would still matter in an unrelated future "
    "conversation. Write each claim as one short English sentence. Copy the quote exactly from the transcript, in "
    "its original language. Allowed types: fact, preference, goal, project. At most 8 items. "
    'Return only {"operations":[{"type":"preference","scope":"owner","subject":"workflow","claim":"...","quote":"..."}]}. '
    'Set scope to "owner" only for a standing preference, rule or fact about the owner that applies across all '
    'tasks. Set scope to "project" for anything that belongs to one project, product, account or task. '
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
    """What an import would do. Writes nothing."""
    rows: dict[str, dict[str, int]] = {}
    sessions = collect(sources)
    for rs in sessions:
        row = rows.setdefault(rs.source, {"source": rs.source, "label": source_label(rs.source),
                                          "found": 0, "already": 0, "noise": 0})
        row["found"] += 1
        prepared = _prepare(rs)
        if prepared is None:
            row["noise"] += 1
            continue
        _, eid, _, chash = prepared
        if svc.archive.files.exists(eid) or _existing_entry(svc, rs.source, rs.scope, rs.origin, chash):
            row["already"] += 1
    listed = [rows[name] for name in SOURCES if name in rows]
    return {
        "sources": listed,
        "total": sum(r["found"] for r in listed),
        "already_imported": sum(r["already"] for r in listed),
        "new": sum(r["found"] - r["already"] - r["noise"] for r in listed),
        "noise": sum(r["noise"] for r in listed),
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


def apply_import(svc: Any, llm: Any, sources: list[str] | None = None,
                 distill_facts: bool = False, progress: Any = None) -> dict[str, Any]:
    from datetime import date as _date

    batch = "b_" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    cache = _load_cache(svc)
    sessions = collect(sources)
    report: dict[str, Any] = {"batch": batch, "total": len(sessions), "archived_new": 0, "archive_existing": 0,
                              "noise_skipped": 0, "facts_added": 0, "facts_duplicate": 0, "rejected": 0,
                              "errors": []}
    archive_ids: list[str] = []
    entries: list[ArchiveEntry] = []
    for n, rs in enumerate(sessions, 1):
        if progress:
            progress({"stage": "cleaning", "done": n - 1, "total": len(sessions)})
        try:
            prepared = _prepare(rs)
            if prepared is None:
                report["noise_skipped"] += 1
                continue
            body, eid, rule_title, chash = prepared
            if svc.archive.files.exists(eid) or _existing_entry(svc, rs.source, rs.scope, rs.origin, chash):
                report["archive_existing"] += 1
                continue
            cls = classify_with_cache(llm, svc, cache, chash, rule_title, body)
            if cls["noise"]:
                report["noise_skipped"] += 1
                continue
            entry = ArchiveEntry(id=eid, source=rs.source, category=cls["category"], title=cls["title"],
                                 abstract=cls["abstract"], body=body, date=rs.date, origin=rs.origin,
                                 scope=rs.scope, batch=batch, content_hash=chash)
            svc.archive.files.write(entry)
            entries.append(entry)
            archive_ids.append(eid)
            report["archived_new"] += 1
            if CALL_PAUSE:
                time.sleep(CALL_PAUSE)
        except Exception as exc:  # one bad session must not stop the batch; a rerun retries it
            report["errors"].append(f"{rs.scope}: {str(exc)[:160] or exc.__class__.__name__}")
    if progress:
        progress({"stage": "indexing", "done": 0, "total": max(1, len(entries))})
    svc.archive.sync(embed=True)

    fact_ids: list[str] = []
    if distill_facts and entries:
        claims = {_norm(f.claim) for f in svc.store.live_facts()}
        quotes = {(e.episode, e.quote) for f in svc.store.all_facts() for e in f.evidence}
        for n, entry in enumerate(entries, 1):
            if progress:
                progress({"stage": "distilling", "done": n - 1, "total": len(entries)})
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
                        type=FactType(op["type"]), subject=op["subject"], claim=op["claim"], evidence=[evidence],
                        source_agent="session-import", source_session=batch))
                    svc.retriever.sync_fact(fact)
                    claims.add(_norm(op["claim"]))
                    fact_ids.append(fact.id)
                    report["facts_added"] += 1
            except Exception as exc:
                report["errors"].append(f"{entry.id}: {str(exc)[:160] or exc.__class__.__name__}")
        if fact_ids:
            svc.retriever.scan_conflicts()

    manifest = {**report, "source": "session-import", "created": datetime.now(timezone.utc).isoformat(),
                "archive_ids": archive_ids, "fact_ids": fact_ids}
    folder = svc.settings.home / "imports"
    folder.mkdir(parents=True, exist_ok=True)
    _atomic_write(folder / f"{batch}.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    svc.store.append_event({"op": "import", "id": batch, "source": "session-import"})
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
