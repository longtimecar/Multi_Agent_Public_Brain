"""Session summaries: a derived view of one conversation.

`memory/summaries/<session>.md` holds three columns — what was done, what was
decided, what is still open. It exists because the store keeps turns while people
remember sessions: "昨晚那场我们干了什么" has no answer when memory is only
sentence-grained.

A summary is never evidence and never enters the facts index. If a model's
restatement could become a memory, the no-quote-no-write rule would be gone.
Facts still come only from extract.py over the user's own turns; a summary points
back at the episodes it was built from so a reader can go and check it.

Lives in `summaries/`, not `sessions/`: `memory/sessions/` already holds the
session archive's transcript notes, and mixing a derived view into the archive
tree would make it look like source material.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

from .llm import LLMError
from .models import Episode
from .store import Store, _atomic_write

SUMMARIES_DIRNAME = "summaries"
# Below this many user turns a session has nothing worth summarising.
MIN_TURNS = 2
_SAFE_ID = re.compile(r"[^A-Za-z0-9._-]+")

SUMMARY_SYSTEM = (
    "You summarise ONE finished conversation session between the owner and an AI assistant. "
    "Return only JSON: {\"done\":[...],\"decided\":[...],\"open\":[...]}. "
    "Each value is a list of at most 6 short Chinese sentences. "
    "done = what was actually accomplished in this session. "
    "decided = choices, rules or directions the owner settled on. "
    "open = work the owner handed to a later session: unfinished steps, commitments, "
    "things explicitly deferred ('明天继续', '回头再弄'). "
    "Report only what the conversation supports. Never invent progress, never restate "
    "the assistant's suggestions as the owner's decisions, and leave a list empty rather "
    "than padding it. Do not include secrets, tokens or absolute local paths."
)


@dataclass
class SummaryReport:
    written: list[str] = field(default_factory=list)
    skipped: int = 0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"written": self.written, "skipped": self.skipped, "errors": self.errors}


def summaries_dir(store: Store) -> Path:
    return store.home / SUMMARIES_DIRNAME


def summary_path(store: Store, session: str) -> Path:
    return summaries_dir(store) / f"{_SAFE_ID.sub('-', session).strip('-') or 'unknown'}.md"


def session_groups(store: Store) -> dict[str, list[Episode]]:
    """Episodes grouped by session, oldest first."""
    groups: dict[str, list[Episode]] = {}
    for ep in store.iter_episodes():
        groups.setdefault(ep.session, []).append(ep)
    for episodes in groups.values():
        episodes.sort(key=lambda e: e.ts)
    return groups


def _last_activity(episodes: list[Episode]) -> datetime:
    return max((ep.ts for ep in episodes), default=datetime.now(timezone.utc))


def idle_sessions(store: Store, idle_minutes: int = 60,
                  now: datetime | None = None) -> list[str]:
    """Sessions silent for at least `idle_minutes` — the "that one is over" signal.

    The store has no session-end event for every agent, so idleness is how a
    watcher notices a conversation finished.
    """
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(minutes=max(1, idle_minutes))
    return [session for session, episodes in session_groups(store).items()
            if _last_activity(episodes) <= cutoff]


def _render(session: str, episodes: list[Episode], sections: dict[str, list[str]],
            model_summarised: bool) -> str:
    users = [ep for ep in episodes if ep.role == "user"]
    start, end = episodes[0].ts, episodes[-1].ts
    meta = {
        "session": session,
        "agent": episodes[0].agent,
        "started": start.isoformat(),
        "ended": end.isoformat(),
        "day": episodes[0].local_date.isoformat(),
        "turns": len(episodes),
        "user_turns": len(users),
        # How much the timestamps can be trusted, carried up from the episodes.
        "ts_source": "exact" if all(ep.ts_source == "exact" for ep in episodes) else "estimated",
        "summarised_by": "model" if model_summarised else "structure-only",
        "episodes": [ep.id for ep in episodes],
        "derived": True,
    }
    lines = ["---", yaml.safe_dump(meta, sort_keys=False, allow_unicode=True, width=100).rstrip(), "---", ""]
    lines += [f"# 会话摘要 · {meta['day']}", ""]
    lines += [f"- 来源：{meta['agent']}",
              f"- 时间：{start.astimezone().strftime('%H:%M')} – {end.astimezone().strftime('%H:%M')}"
              f"（{len(users)} 轮用户发言）", ""]
    for title, key in (("做了什么", "done"), ("定了什么", "decided"), ("待办什么", "open")):
        lines += [f"## {title}", ""]
        items = [str(x).strip() for x in (sections.get(key) or []) if str(x).strip()]
        lines += [f"- {item}" for item in items] or ["- （无）"]
        lines.append("")
    if not model_summarised:
        lines += ["## 用户原话（未经模型归纳）", ""]
        lines += [f"- {ep.text.strip()[:300]}" for ep in users[-20:]]
        lines.append("")
    lines += ["---", "",
              "本文件是派生视图，不是证据，也不参与记忆检索。",
              f"它由上述 {len(episodes)} 个 episode 归纳而来；要核对请看原始回合。", ""]
    return "\n".join(lines)


def _ask_model(llm: Any, episodes: list[Episode]) -> dict[str, list[str]]:
    users = [ep for ep in episodes if ep.role == "user"][-40:]
    payload = json.dumps({
        "turns": [{"role": ep.role, "text": ep.text.strip()[:1200]}
                  for ep in episodes[-60:]],
        "owner_turns": [ep.text.strip()[:1200] for ep in users],
    }, ensure_ascii=False)
    response = llm.complete_json(SUMMARY_SYSTEM, payload)
    out: dict[str, list[str]] = {}
    for key in ("done", "decided", "open"):
        raw = response.get(key)
        out[key] = [str(x).strip()[:300] for x in raw if str(x).strip()][:6] if isinstance(raw, list) else []
    return out


def consolidate_session(svc: Any, session: str, llm: Any | None = None,
                        force: bool = False) -> Path | None:
    """Write (or refresh) the summary for one session. Returns the path written."""
    store: Store = svc.store
    episodes = session_groups(store).get(session) or []
    users = [ep for ep in episodes if ep.role == "user"]
    if len(users) < MIN_TURNS:
        return None
    path = summary_path(store, session)
    if path.exists() and not force:
        return None
    sections: dict[str, list[str]] = {"done": [], "decided": [], "open": []}
    model_summarised = False
    if llm is not None:
        try:
            sections = _ask_model(llm, episodes)
            model_summarised = True
        except (LLMError, ValueError, KeyError, TypeError):
            # No model, or a model that returned rubbish: fall back to the
            # owner's own words verbatim rather than fabricate a summary.
            sections = {"done": [], "decided": [], "open": []}
    _atomic_write(path, _render(session, episodes, sections, model_summarised))
    return path


def consolidate(svc: Any, session: str | None = None, llm: Any | None = None,
                idle_minutes: int = 60, limit: int = 10, force: bool = False) -> SummaryReport:
    """Summarise one session, or the sessions that have gone quiet."""
    report = SummaryReport()
    store: Store = svc.store
    if llm is None:
        # Build one here so both entry points (CLI and the watcher loop) get a
        # model when the owner configured one, and a structural-only summary
        # when they did not.
        from .llm import LLMClient

        llm = LLMClient(svc.settings) if svc.settings.llm_enabled else None
    targets = [session] if session else idle_sessions(store, idle_minutes)[:limit]
    for target in targets:
        try:
            written = consolidate_session(svc, target, llm=llm, force=force)
        except OSError as exc:
            report.errors.append(f"{target}: {exc}")
            continue
        if written is None:
            report.skipped += 1
        else:
            report.written.append(written.name)
    return report


def read_summary(store: Store, session: str) -> str | None:
    path = summary_path(store, session)
    return path.read_text(encoding="utf-8") if path.exists() else None


def list_summaries(store: Store) -> list[dict[str, Any]]:
    """Summaries newest first, with their front-matter for listing."""
    out: list[dict[str, Any]] = []
    directory = summaries_dir(store)
    if not directory.is_dir():
        return out
    for path in sorted(directory.glob("*.md"), reverse=True):
        try:
            text = path.read_text(encoding="utf-8")
            match = re.match(r"^---\n(.*?)\n---\n", text, re.S)
            meta = yaml.safe_load(match.group(1)) if match else {}
        except (OSError, yaml.YAMLError):
            continue
        if not isinstance(meta, dict):
            continue
        out.append({"file": path.name, "session": meta.get("session"), "agent": meta.get("agent"),
                    "day": meta.get("day"), "user_turns": meta.get("user_turns"),
                    "summarised_by": meta.get("summarised_by")})
    return out
