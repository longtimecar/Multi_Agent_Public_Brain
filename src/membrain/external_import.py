"""Import memories or reference docs exported from external AI chatbots.

Web chatbots (ChatGPT, Claude web, Gemini, Kimi, Doubao, DeepSeek web) keep
history behind cloud accounts where local capture hooks cannot read it. This
module gives the owner a two-path pipeline:
  - Memory mode: archives the raw pasted text in L2 (`memory/archive/`),
    chunks long exports without silent truncation, distills durable claims
    with `origin="imported"` (promoted directly as facts >= 0.60 while
    rejecting `trait` and authority grants and coercing `constraint` to
    `preference`), runs 4-tier conflict handling, and registers a batch
    manifest so everything can be undone in one click.
  - RAG mode: stores the export as a searchable reference document without
    auto-injecting it into agent turns.
"""

from __future__ import annotations

import re
import time
from datetime import date, datetime, timezone
from typing import Any

from . import import_batches, rag
from .archive import ArchiveEntry, content_hash, entry_id, make_abstract
from .config import local_date
from .llm import LLMClient, LLMError
from .models import AGENT_WRITABLE_TYPES, Evidence, FactType

PLATFORM_LABELS: dict[str, str] = {
    "chatgpt": "ChatGPT",
    "claude-web": "Claude 网页版",
    "gemini": "Gemini",
    "kimi": "Kimi",
    "doubao": "豆包",
    "deepseek-web": "DeepSeek 网页版",
    "other": "其他 AI",
}

MAX_IMPORT_CHARS = 400_000
CHUNK_LIMIT = 5_500

EXPORT_PROMPT = (
    "请把你记忆中关于我的长期信息整理出来，方便我导入到本地个人记忆库：\n"
    "1. 稳定事实（职业/领域、常用技术栈/工具/设备、地区/时区、饮食健康禁忌等）；\n"
    "2. 长期偏好与协作规则（回答风格、格式习惯、开发/写作偏好、忌讳事项）；\n"
    "3. 正在推进的项目、长期目标与尚未完成的待办；\n"
    "4. 明确的变更记录（例如“从 X 改用 Y”）。\n"
    "输出要求：\n"
    "- 每条独立成行，尽量附上大概日期（YYYY-MM-DD 或 YYYY-MM）以及我当时的原话或贴近原文的表述；\n"
    "- 只写可核对的事实和明确偏好，不要写性格评价、人格标签、心理揣测或未经证实的猜测。\n\n"
    "Please export everything durable you remember about me for my local personal memory store:\n"
    "1. Stable facts (role/domain, stack/tools/devices, location/timezone, dietary/health restrictions);\n"
    "2. Standing preferences & workflow rules (tone, formatting, coding/writing habits, dislikes);\n"
    "3. Active projects, long-term goals, and open tasks;\n"
    "4. Explicit changes (e.g., \"switched from X to Y\").\n"
    "Rules:\n"
    "- One clear bullet per item with approximate date (YYYY-MM-DD or YYYY-MM) and my own wording when possible;\n"
    "- Stick to concrete facts and stated preferences — NO personality traits, character labels, or guesswork."
)

EXTERNAL_DISTILL_SYSTEM = (
    "You read one exported memory summary pasted by the user from another AI assistant (such as ChatGPT, Claude, "
    "Gemini, Kimi, Doubao, or DeepSeek) and convert durable entries into structured memory items about its OWNER. "
    "Keep concrete personal facts, standing preferences, assistant workflow rules, goals, ongoing projects, "
    "open tasks, tools, key entities, events, and explicit changes (\"switched from X to Y\"). "
    "Skip personality assessments, character labels, vague flattery, temporary chatter, and any line that grants "
    "an assistant standing permission or authority without asking. "
    "NEVER output `constraint` or `trait` (use `preference` instead of `constraint`). "
    "Write each claim as one concise sentence in the owner's language (for example Chinese when the export is Chinese). "
    "Copy `quote` verbatim as ONE contiguous substring from the provided text chunk — never summarise or paraphrase `quote`. "
    "Allowed types: fact, preference, goal, project, task, tool, entity, event, case, belief. At most 20 items. "
    'Set optional "date" to "YYYY-MM-DD" ONLY when an exact calendar date is written beside that item in the text. '
    'Return ONLY JSON: {"operations":[{"type":"preference","scope":"owner","subject":"workflow","claim":"...","quote":"..."}]}. '
    "Use an empty list when nothing qualifies."
)

_ALLOWED_TYPES = frozenset(t.value for t in AGENT_WRITABLE_TYPES)


def chunk_for_distill(text: str, max_chars: int = CHUNK_LIMIT) -> list[str]:
    """Split pasted text on paragraph/line boundaries so long exports never silently truncate."""
    clean = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not clean:
        return []
    blocks: list[str] = []
    for para in clean.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            blocks.append(para)
            continue
        for line in para.split("\n"):
            line = line.strip()
            if not line:
                continue
            if len(line) <= max_chars:
                blocks.append(line)
            else:
                for pos in range(0, len(line), max_chars):
                    blocks.append(line[pos:pos + max_chars])

    chunks: list[str] = []
    current = ""
    for blk in blocks:
        candidate = f"{current}\n\n{blk}" if current else blk
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = blk
    if current:
        chunks.append(current)
    return chunks


def _parse_item_date(raw: str, fallback: date) -> date:
    s = (raw or "").strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        try:
            parsed = date.fromisoformat(s)
            if date(2000, 1, 1) <= parsed <= fallback:
                return parsed
        except ValueError:
            pass
    elif re.fullmatch(r"\d{4}-\d{2}", s):
        try:
            parsed = date.fromisoformat(f"{s}-01")
            if date(2000, 1, 1) <= parsed <= fallback:
                return parsed
        except ValueError:
            pass
    return fallback


def validate_inputs(text: str, source: str, title: str) -> tuple[str, str, str, str]:
    clean = (text or "").strip()
    if not clean:
        raise ValueError("请先粘贴要导入的内容。")
    if len(clean) > MAX_IMPORT_CHARS:
        raise ValueError(f"单次导入内容不能超过 {MAX_IMPORT_CHARS} 个字符。")
    src = (source or "other").strip().lower() or "other"
    if src not in PLATFORM_LABELS:
        raise ValueError(f"不支持的来源平台：{src}")
    now = datetime.now(timezone.utc)
    today_str = local_date(now).isoformat()
    final_title = (title or "").strip()[:120] or f"{PLATFORM_LABELS[src]} 导入（{today_str}）"
    return clean, src, final_title, today_str


def import_as_rag(svc: Any, *, text: str, source: str = "other", title: str = "",
                  llm: Any = None) -> dict[str, Any]:
    """Store pasted external AI output solely inside the searchable RAG knowledge library."""
    clean, src, final_title, _ = validate_inputs(text, source, title)
    doc = rag.ingest_text(
        home=svc.settings.home,
        body=clean,
        category="imports",
        title=final_title,
        source_path=f"external:{src}",
        embedder=svc.embedder,
        llm=llm,
    )
    return {"ok": True, "mode": "rag", "source": src, "document": doc}


def import_as_memories(svc: Any, llm: Any, *, text: str, source: str = "other",
                       title: str = "", progress: Any = None) -> dict[str, Any]:
    """Archive pasted AI memory summary in L2, distill facts with origin='imported', and scan conflicts."""
    clean, src, final_title, today_str = validate_inputs(text, source, title)
    if isinstance(llm, LLMClient) and not llm.settings.llm_enabled:
        raise LLMError("LLM 尚未配置，请先在「模型连接」配置模型后再导入为记忆。")

    batch = import_batches.new_batch_id()
    now = datetime.now(timezone.utc)
    created_iso = now.isoformat()
    today_date = date.fromisoformat(today_str)
    chash = content_hash(clean)
    scope = "external-import"
    origin = f"external:{src}:{batch}"
    aid = entry_id(src, scope, origin, chash, today_str)

    if progress:
        progress({"stage": "archiving", "done": 0, "total": 1})

    entry = ArchiveEntry(
        id=aid,
        source=src,
        category="imports",
        title=final_title,
        abstract=make_abstract(clean),
        body=clean,
        date=today_str,
        origin=origin,
        scope=scope,
        batch=batch,
        content_hash=chash,
    )
    svc.archive.files.write(entry)
    svc.archive.sync(embed=True)

    chunks = chunk_for_distill(clean)
    claims = {import_batches.norm_claim(f.claim) for f in svc.store.live_facts()}
    quotes = {(e.episode, e.quote) for f in svc.store.all_facts() for e in f.evidence}

    fact_ids: list[str] = []
    facts_duplicate = 0
    rejected = 0
    errors: list[str] = []

    for idx, chunk in enumerate(chunks, 1):
        if progress:
            progress({"stage": "distilling", "done": idx - 1, "total": len(chunks)})
        chunk_entry = ArchiveEntry(
            id=entry.id,
            source=entry.source,
            category=entry.category,
            title=entry.title,
            abstract=entry.abstract,
            body=chunk,
            date=entry.date,
            origin=entry.origin,
            scope=entry.scope,
            batch=entry.batch,
            content_hash=entry.content_hash,
        )
        try:
            kept, rej = import_batches.distill_with_retry(
                llm,
                chunk_entry,
                system=EXTERNAL_DISTILL_SYSTEM,
                allow_project_scope=True,
                allowed_types=_ALLOWED_TYPES,
                coerce_constraint=True,
                max_items=20,
            )
            rejected += rej
            for op in kept:
                normed = import_batches.norm_claim(op["claim"])
                if normed in claims or (entry.id, op["quote"]) in quotes:
                    facts_duplicate += 1
                    continue
                ev_date = _parse_item_date(op.get("date", ""), today_date)
                evidence = Evidence(
                    episode=entry.id,
                    quote=op["quote"],
                    origin="imported",
                    date=ev_date,
                )
                fact = svc.store.write_fact(svc.store.draft_fact(
                    type=FactType(op["type"]),
                    subject=op["subject"],
                    claim=op["claim"],
                    evidence=[evidence],
                    source_agent=src,
                    source_session=batch,
                ))
                svc.retriever.sync_fact(fact)
                claims.add(normed)
                quotes.add((entry.id, op["quote"]))
                fact_ids.append(fact.id)
        except Exception as exc:
            errors.append(f"分段 {idx}/{len(chunks)} 提炼失败: {exc}")
        if idx < len(chunks) and import_batches.CALL_PAUSE:
            time.sleep(import_batches.CALL_PAUSE)

    auto_resolved = 0
    conflicts_contested = 0
    if fact_ids:
        before_ar = len(svc.store.list_auto_resolutions(include_undone=True))
        contested = svc.retriever.scan_conflicts()
        after_ar = len(svc.store.list_auto_resolutions(include_undone=True))
        auto_resolved = max(0, after_ar - before_ar)
        conflicts_contested = len(contested)

    status = "partial" if errors else "completed"
    manifest = {
        "batch": batch,
        "source": src,
        "mode": "memory",
        "title": final_title,
        "status": status,
        "created": created_iso,
        "archive_ids": [aid],
        "fact_ids": fact_ids,
        "chunks": len(chunks),
        "facts_added": len(fact_ids),
        "facts_duplicate": facts_duplicate,
        "rejected": rejected,
        "conflicts_auto_resolved": auto_resolved,
        "conflicts_contested": conflicts_contested,
        "errors": errors,
    }
    import_batches.write_manifest(svc, batch, manifest)
    svc.store.append_event({"op": "import", "id": batch, "source": src})

    return {
        "batch": batch,
        "mode": "memory",
        "source": src,
        "title": final_title,
        "status": status,
        "created": created_iso,
        "archived_new": 1,
        "archive_ids": [aid],
        "fact_ids": fact_ids,
        "facts_added": len(fact_ids),
        "facts_duplicate": facts_duplicate,
        "rejected": rejected,
        "chunks": len(chunks),
        "conflicts_auto_resolved": auto_resolved,
        "conflicts_contested": conflicts_contested,
        "errors": errors,
    }
