"""One-shot import of local Claude Code and Codex session JSONL into membrain archive."""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("MEMBRAIN_HOME", str(REPO / "memory"))

from membrain.archive import ArchiveEntry, content_hash, entry_id, make_abstract  # noqa: E402
from membrain.server import MemoryService  # noqa: E402

HOME = Path.home()
CLAUDE_ROOT = HOME / ".claude" / "projects"
CODEX_ROOT = HOME / ".codex" / "sessions"
BATCH = f"b_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
MAX_BODY = 120_000
MAX_TURN = 8_000


def _clip(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 20].rstrip() + "\n\n…（已截断）"


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
            text = "\n".join(
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
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
        texts: list[str] = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            if part.get("type") in ("input_text", "output_text", "text"):
                texts.append(str(part.get("text") or ""))
        text = "\n".join(t for t in texts if t.strip()).strip()
        if not text:
            continue
        if role == "developer":
            continue
        if role == "user":
            meta = (payload.get("internal_chat_message_metadata_passthrough") or {}).get("content_item_metadata") or []
            if meta and all((m or {}).get("provenance", {}).get("type") != "user" for m in meta if isinstance(m, dict)):
                continue
            text = _codex_user_text(text)
            if not text or text.startswith("# Files mentioned"):
                continue
        if role == "assistant":
            if len(text) < 40:
                continue
        turns.append((str(role), _clip(text, MAX_TURN)))
    return turns


def _render_dialog(turns: list[tuple[str, str]]) -> str:
    lines = ["# 对话记录", ""]
    for role, text in turns:
        label = "用户" if role == "user" else "助手"
        lines.append(f"## {label}")
        lines.append("")
        lines.append(text)
        lines.append("")
    body = "\n".join(lines).strip()
    return _clip(body, MAX_BODY)


def _title(source: str, path: Path, turns: list[tuple[str, str]]) -> str:
    for role, text in turns:
        if role == "user":
            one = re.sub(r"\s+", " ", text.split("\n", 1)[0])[:72]
            return f"[{source}] {one or path.stem}"
    return f"[{source}] {path.stem}"


def _import_file(svc: MemoryService, source: str, path: Path, parser) -> str:
    rel = path.as_posix()
    turns = parser(path)
    if not turns:
        return "skip_empty"
    body = _render_dialog(turns)
    chash = content_hash(body)
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    eid = entry_id(source, path.stem, rel, chash, day)
    if svc.archive.files.exists(eid):
        return "skip_exists"
    entry = ArchiveEntry(
        id=eid,
        source=source,
        category="sessions",
        title=_title(source, path, turns),
        abstract=make_abstract(body),
        body=body,
        date=day,
        origin=rel,
        scope=path.stem,
        batch=BATCH,
        content_hash=chash,
    )
    svc.archive.files.write(entry)
    svc.archive._put_rows([entry])
    try:
        svc.archive._put_vectors([entry])
    except Exception:
        pass
    svc.store.append_event({"op": "import", "id": BATCH, "source": "local-sessions"})
    return "imported"


def main() -> int:
    svc = MemoryService()
    stats = {"imported": 0, "skip_empty": 0, "skip_exists": 0}
    files: list[tuple[str, Path, object]] = []
    files.extend(("claude-code", p, _parse_claude) for p in sorted(CLAUDE_ROOT.rglob("*.jsonl")))
    files.extend(("codex", p, _parse_codex) for p in sorted(CODEX_ROOT.rglob("rollout-*.jsonl")))
    for source, path, parser in files:
        result = _import_file(svc, source, path, parser)
        stats[result] = stats.get(result, 0) + 1
    svc.archive.conn.commit()
    manifest = {
        "batch": BATCH,
        "source": "local-sessions",
        "created": datetime.now(timezone.utc).isoformat(),
        **stats,
        "total_files": len(files),
    }
    imports_dir = svc.settings.home / "imports"
    imports_dir.mkdir(parents=True, exist_ok=True)
    (imports_dir / f"{BATCH}.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
