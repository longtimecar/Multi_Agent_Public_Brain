"""Local HTTP handler, route dispatch tables, and dashboard server lifecycle."""

from __future__ import annotations

import ipaddress
import json
import mimetypes
import os
import subprocess
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, unquote, urlparse

from pydantic import ValidationError

from .. import agents, archive, autostart, external_import, import_batches, session_import
from .. import agents as agent_names
from ..embeddings import EmbeddingError
from ..llm import LLMError
from ..models import Episode, Evidence, FactType, Status
from ..store import StoreError
from .admin_agent import _category_type
from .runtime import DashboardRuntime

WEB_DIR = Path(__file__).resolve().parents[1] / "web"
MAX_BODY_BYTES = 64_000
# Base64 inflates by ~1.37x; keep headroom over the 30MB file cap in rag.ingest_upload.
MAX_UPLOAD_BYTES = 48_000_000
_STATIC_PATHS = {"/", "/index.html", "/app.js", "/styles.css"}


def _fact_json(fact: Any) -> dict[str, Any]:
    return fact.model_dump(mode="json")


def _loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def make_handler(runtime: DashboardRuntime):
    class Handler(BaseHTTPRequestHandler):
        server_version = "MembrainDashboard/1.0"
        sys_version = ""

        def log_message(self, fmt: str, *args: Any) -> None:
            # Keep personal memory contents and local paths out of console logs.
            return

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' data:; style-src 'self'; "
                "script-src 'self'; connect-src 'self'; object-src 'none'; "
                "base-uri 'none'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, value: Any) -> None:
            self._send(
                status,
                json.dumps(value, ensure_ascii=False, default=str).encode("utf-8"),
                "application/json; charset=utf-8",
            )

        def _body(self, max_bytes: int = MAX_BODY_BYTES) -> dict[str, Any]:
            raw_len = self.headers.get("Content-Length", "0")
            try:
                length = int(raw_len)
            except ValueError as exc:
                raise ValueError("请求长度无效。") from exc
            if length <= 0 or length > max_bytes:
                raise ValueError(f"请求内容为空或超过 {max_bytes // 1000} KB。")
            value = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError("请求内容格式无效。")
            return value

        def _foreign_request(self) -> bool:
            """Whether this request did not come from the local dashboard page.

            Binding to loopback is not enough on its own. With DNS rebinding a
            web page on an attacker's domain can resolve that domain to
            127.0.0.1 and then read every memory over plain GET requests; the
            browser treats it as same-origin. Such a request still carries the
            attacker's name in Host, so every method, reads included, checks it.
            """
            host_header = self.headers.get("Host", "")
            host_name = urlparse("//" + host_header).hostname or ""
            origin = self.headers.get("Origin")
            return not _loopback(host_name) or (origin is not None and urlparse(origin).netloc != host_header)

        def _reject_foreign(self) -> None:
            # Drain the body before answering. Replying with unread bytes
            # still in the socket makes the client see a connection reset
            # instead of the 403, which read as a random test flake.
            try:
                length = int(self.headers.get("Content-Length", "0") or 0)
                if 0 < length <= MAX_BODY_BYTES:
                    self.rfile.read(length)
            except (ValueError, OSError):
                pass
            self._json(403, {"error": "仅允许从本机后台页面访问。"})

        # ----------------------------------------------------- query helpers
        def _profile(self) -> dict[str, Any]:
            path = runtime.service.settings.profile_path
            return {"exists": path.is_file(), "content": path.read_text(encoding="utf-8") if path.is_file() else ""}

        def _overview(self) -> dict[str, Any]:
            service = runtime.service
            facts = service.store.all_facts()
            episodes = list(service.store.iter_episodes())
            agents_map: dict[str, int] = {}
            for fact in facts:
                # Group by canonical key: one agent reaches the store as
                # "WorkBuddy" and "workbuddy", and counting both split every
                # per-source number in half. The module name is shadowed by the
                # `agents` counter, so it is imported under a local alias.
                label = agent_names.display_name(fact.source.agent)
                agents_map[label] = agents_map.get(label, 0) + 1
            recent = sorted(facts, key=lambda fact: (fact.created_at, fact.id), reverse=True)[:8]
            return {
                "stats": {
                    "total": len(facts),
                    "active": sum(fact.is_live for fact in facts),
                    "contested": sum(fact.status == Status.CONTESTED for fact in facts),
                    "superseded": sum(fact.status == Status.SUPERSEDED for fact in facts),
                    "retracted": sum(fact.status == Status.RETRACTED for fact in facts),
                    "episodes": len(episodes),
                    "userEpisodes": sum(episode.role == "user" for episode in episodes),
                },
                "agents": agents_map,
                "source_labels": session_import.SOURCE_LABELS,
                "recent": [_fact_json(fact) for fact in recent],
                "profile": self._profile(),
                "home": str(service.settings.home),
                "index": service.index.stats(),
            }

        def _facts(self, query: dict[str, list[str]]) -> dict[str, Any]:
            service = runtime.service
            status = query.get("status", [""])[0]
            fact_type = query.get("type", [""])[0]
            search = query.get("q", [""])[0].strip()
            disabled = service.disabled_agents()
            facts = [fact for fact in service.store.all_facts() if not agents.is_disabled(fact.source.agent, disabled)]
            # A search result is ordered by relevance. Sorting it by date again
            # throws that away: the best match for "怎么汇报进度" ended up below
            # a command that merely shares a word with it.
            by_relevance = False
            if search:
                direct = [
                    fact for fact in facts
                    if search.casefold() in " ".join((fact.subject, fact.claim, *(item.quote for item in fact.evidence))).casefold()
                ]
                ranked = []
                if status in ("", "active", "contested"):
                    ranked = service.retriever.recall(search, top_k=100).hits
                seen: set[str] = set()
                ordered = []
                for fact in [*ranked, *direct]:
                    if fact.id not in seen:
                        ordered.append(fact)
                        seen.add(fact.id)
                if not ordered:
                    # A category phrase with a qualifier ("我的工作偏好") matches
                    # no single memory, so similarity returns nothing and the
                    # page looks empty — while the bare "我的偏好" lists 100.
                    # Falling back to that kind of memory beats an empty page.
                    fallback = _category_type(search)
                    if fallback:
                        ordered = [fact for fact in facts if fact.type == fallback]
                facts = ordered
                by_relevance = bool(ranked)
            if status:
                facts = [fact for fact in facts if fact.status.value == status]
            if fact_type:
                facts = [fact for fact in facts if fact.type.value == fact_type]
            if not by_relevance:
                facts.sort(key=lambda fact: (fact.created_at, fact.id), reverse=True)
            limit_raw = query.get("limit", ["500"])[0]
            try:
                limit = max(1, min(1000, int(limit_raw)))
            except ValueError:
                limit = 500
            return {"items": [_fact_json(fact) for fact in facts[:limit]], "count": len(facts), "truncated": len(facts) > limit}

        def _episodes(self, query: dict[str, list[str]]) -> dict[str, Any]:
            limit_raw = query.get("limit", ["30"])[0]
            try:
                limit = max(1, min(100, int(limit_raw)))
            except ValueError:
                limit = 30
            disabled = runtime.service.disabled_agents()
            episodes = [
                episode for episode in runtime.service.store.iter_episodes()
                if not agents.is_disabled(episode.agent, disabled)
            ]
            episodes.sort(key=lambda episode: (episode.ts, episode.id), reverse=True)
            items = []
            for episode in episodes[:limit]:
                item = episode.model_dump(mode="json")
                # Show one row per agent: the stored spelling varies by writer
                # ("WorkBuddy" from hooks, "workbuddy" from the watcher).
                item["agent"] = agent_names.display_name(episode.agent)
                items.append(item)
            return {"items": items, "count": len(episodes)}

        def _archive(self, query: dict[str, list[str]]) -> dict[str, Any]:
            lib = runtime.service.archive
            text = query.get("q", [""])[0].strip()
            source, category = query.get("source", [""])[0], query.get("category", [""])[0]
            if text:
                items, notes = lib.search(text, source or None, category or None, top_k=20)
                return {
                    "items": items, "notes": notes,
                    "labels": archive.CATEGORY_LABELS, "source_labels": session_import.SOURCE_LABELS,
                }
            try:
                limit = max(1, min(500, int(query.get("limit", ["50"])[0])))
            except ValueError:
                limit = 50
            return {
                **lib.browse(source or None, category or None, limit=limit),
                "labels": archive.CATEGORY_LABELS,
                "source_labels": session_import.SOURCE_LABELS,
            }

        def _archive_entry(self, entry_id: str, query: dict[str, list[str]]) -> dict[str, Any]:
            def number(name: str, default: int) -> int:
                try:
                    return int(query.get(name, [str(default)])[0])
                except ValueError:
                    return default

            return runtime.service.archive.read(entry_id, number("offset", 0), number("limit", 4000))

        def _add_fact(self, data: dict[str, Any], old_id: str | None = None) -> dict[str, Any]:
            service = runtime.service
            claim = str(data.get("claim", "")).strip()
            quote = str(data.get("quote", "")).strip()
            subject = str(data.get("subject", "general")).strip()
            if not claim or len(claim) > 1000:
                raise ValueError("记忆内容不能为空，且最多 1000 个字符。")
            if not quote or len(quote) > 2000:
                raise ValueError("请填写你亲自输入的原话，最多 2000 个字符。")
            if not subject or len(subject) > 120:
                raise ValueError("主题不能为空，且最多 120 个字符。")

            if old_id:
                old = service.store.get_fact(old_id)
                fact_type = old.type
                subject = old.subject
            else:
                try:
                    fact_type = FactType(str(data.get("type", FactType.FACT.value)))
                except ValueError as exc:
                    raise ValueError("记忆类型无效。") from exc

            now = datetime.now(timezone.utc)
            session = f"dashboard-{uuid.uuid4().hex[:12]}"
            episode = Episode(
                id="", ts=now, session=session, agent="membrain-dashboard",
                role="user", text=quote, project=None, ts_source="exact",
            )
            saved_episode = service.store.append_episodes([episode])[0]
            evidence = [Evidence(episode=saved_episode.id, quote=quote, date=saved_episode.local_date, origin="stated")]
            replacement = service.store.draft_fact(
                type=fact_type,
                subject=subject,
                claim=claim,
                evidence=evidence,
                source_agent="membrain-dashboard",
                source_session=session,
                supersedes=old_id,
            )
            saved = service.store.write_fact(replacement)
            # The memory is on disk from here on. Everything below is derived
            # state (the lifecycle link and the index). Reporting a failure now
            # would make the caller retry a write that already landed, and the
            # retry adds a second copy of the same memory.
            index_warning = None
            try:
                if old_id:
                    service.store.supersede(old_id, saved.id, when=now.date())
                    service.retriever.sync_fact(service.store.get_fact(old_id))
                service.retriever.sync_fact(saved)
                service.retriever.scan_conflicts()
            except Exception as exc:  # derived state can be rebuilt; the fact cannot
                index_warning = f"{exc.__class__.__name__}: {exc}"
            result: dict[str, Any] = {
                "ok": True,
                "fact": _fact_json(service.store.get_fact(saved.id)),
                "episode": saved_episode.id,
            }
            if index_warning:
                result["index_warning"] = index_warning
            return result

        def _session_diffs(self, query: dict[str, list[str]]) -> dict[str, Any]:
            """List completed session archives that have memory_diff.json (gap 12)."""
            from .. import session_archive

            store = runtime.service.store
            root = store.home / session_archive.SESSIONS_ROOT
            only_skipped = (query.get("skipped_only") or ["0"])[0].lower() in {"1", "true", "yes"}
            q = (query.get("q") or [""])[0].strip().lower()
            items: list[dict[str, Any]] = []
            if root.exists():
                for session_dir in sorted(root.iterdir()):
                    if not session_dir.is_dir():
                        continue
                    low = session_dir.name.lower()
                    if low.startswith(("sim-", "sim_", "hook-switch-test", "test-", "test_", "fake-")):
                        continue  # dev fixtures are not real sessions
                    hist = session_dir / "history"
                    if not hist.is_dir():
                        continue
                    for arch in sorted(hist.glob("archive_*")):
                        diff_path = arch / "memory_diff.json"
                        if not diff_path.is_file():
                            continue
                        try:
                            diff = session_archive.read_memory_diff(arch)
                        except Exception:
                            continue
                        summary = diff.get("summary") or {}
                        skipped = diff.get("skipped_operations") or []
                        if only_skipped and not skipped:
                            continue
                        row = {
                            "session": session_dir.name,
                            "archive": arch.name,
                            "path": str(arch),
                            "uri": diff.get("archive_uri"),
                            "extracted_at": diff.get("extracted_at"),
                            "done": (arch / ".done").exists(),
                            "summary": summary,
                            "skipped_count": (
                                len(skipped) if isinstance(skipped, list)
                                else int(summary.get("total_skipped") or 0)
                            ),
                        }
                        blob = f"{row['session']} {row['archive']} {row.get('uri') or ''}".lower()
                        if q and q not in blob:
                            continue
                        items.append(row)
            items.sort(key=lambda r: (r.get("extracted_at") or "", r["session"], r["archive"]), reverse=True)
            return {"items": items, "total": len(items)}

        def _session_diff_detail(self, archive_key: str, query: dict[str, list[str]]) -> dict[str, Any]:
            """Read one memory_diff by session/archive or path."""
            from .. import session_archive

            key = unquote(archive_key).strip()
            path = session_archive.resolve_archive_path(runtime.service.store, key)
            diff = session_archive.read_memory_diff(path)
            skipped_only = (query.get("skipped_only") or ["0"])[0].lower() in {"1", "true", "yes"}
            if skipped_only:
                return {
                    "path": str(path),
                    "archive_uri": diff.get("archive_uri"),
                    "extracted_at": diff.get("extracted_at"),
                    "skipped_operations": diff.get("skipped_operations") or [],
                    "summary": diff.get("summary") or {},
                }
            return {"path": str(path), "diff": diff}

        # ----------------------------------------------------- GET dispatch
        def _get_agents(self, _q: dict[str, list[str]]) -> tuple[int, Any]:
            disabled = runtime.service.disabled_agents()
            items = agents.scan(runtime.service.settings.home)
            for item in items:
                item["isolated"] = item.get("key") in disabled
            return 200, {"agents": items}

        def _get_health(self, query: dict[str, list[str]]) -> tuple[int, Any]:
            # Asking for one kind must not pay for the duplicate
            # detector's embedding pass over every memory.
            kinds = {k.strip() for k in query.get("kinds", []) if k.strip()}
            return 200, runtime.health_report(refresh="refresh" in query, kinds=kinds or None)

        def _get_tasks(self, _q: dict[str, list[str]]) -> tuple[int, Any]:
            tasks = runtime.service.retriever.open_tasks(limit=50)
            return 200, {
                "tasks": [
                    {
                        "id": f.id,
                        "subject": f.subject,
                        "claim": f.claim,
                        "stated_on": f.valid_from.isoformat(),
                        "confidence": round(f.confidence, 2),
                    }
                    for f in tasks
                ],
                "count": len(tasks),
            }

        def _get_auto_resolutions(self, query: dict[str, list[str]]) -> tuple[int, Any]:
            inc = (query.get("include_undone") or ["0"])[0].lower() in {"1", "true", "yes"}
            return 200, runtime.list_auto_resolutions(include_undone=inc)

        def _serve_static(self, path: str) -> None:
            name = "index.html" if path in ("/", "/index.html") else path.lstrip("/")
            resolved = (WEB_DIR / name).resolve()
            if resolved.parent != WEB_DIR.resolve() or not resolved.is_file():
                self._json(404, {"error": "页面资源不存在。"})
                return
            content_type = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
            if resolved.suffix == ".js":
                content_type = "application/javascript"
            elif resolved.suffix == ".css":
                content_type = "text/css"
            charset = "; charset=utf-8" if content_type.startswith(("text/", "application/javascript")) else ""
            self._send(200, resolved.read_bytes(), content_type + charset)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if self._foreign_request():
                self._reject_foreign()
                return
            query = parse_qs(parsed.query)
            path = parsed.path
            exact_get: dict[str, Callable[[], tuple[int, Any]]] = {
                "/api/overview": lambda: (200, self._overview()),
                "/api/facts": lambda: (200, self._facts(query)),
                "/api/episodes": lambda: (200, self._episodes(query)),
                "/api/profile": lambda: (200, self._profile()),
                "/api/jobs/extract": lambda: (200, runtime.job()),
                "/api/model": lambda: (200, runtime.model_status()),
                "/api/autostart": lambda: (200, autostart.describe()),
                "/api/agents": lambda: self._get_agents(query),
                "/api/health": lambda: self._get_health(query),
                # The page polls this while a verification runs; it is a read,
                # so a GET must answer it.
                "/api/recall/verify/status": lambda: (200, runtime.recall_verify_status()),
                "/api/tasks": lambda: self._get_tasks(query),
                "/api/import/sessions/preview": lambda: (200, runtime.session_import_preview()),
                "/api/import/external/prompt": lambda: (200, {
                    "prompt": external_import.EXPORT_PROMPT,
                    "platforms": external_import.PLATFORM_LABELS,
                }),
                "/api/import/job": lambda: (200, runtime.import_job()),
                "/api/import/batches": lambda: (200, {"batches": import_batches.list_batches(runtime.service)}),
                "/api/conflicts/auto": lambda: self._get_auto_resolutions(query),
                "/api/rag/documents": lambda: (200, runtime.rag_list()),
                "/api/rag/search": lambda: (200, runtime.rag_search(query.get("q", [""])[0])),
                "/api/archive": lambda: (200, self._archive(query)),
                "/api/session-diffs": lambda: (200, self._session_diffs(query)),
                "/api/chat-sessions": lambda: (200, {"sessions": runtime.chat_sessions()}),
                "/api/watcher": lambda: (200, runtime.watcher_status()),
            }
            try:
                if path in exact_get:
                    code, payload = exact_get[path]()
                    self._json(code, payload)
                elif path.startswith("/api/facts/"):
                    fact_id = unquote(path.removeprefix("/api/facts/"))
                    self._json(200, _fact_json(runtime.service.store.get_fact(fact_id)))
                elif path.startswith("/api/rag/doc/"):
                    self._json(200, runtime.rag_document(unquote(path.removeprefix("/api/rag/doc/"))))
                elif path.startswith("/api/archive/"):
                    self._json(200, self._archive_entry(unquote(path.removeprefix("/api/archive/")), query))
                elif path.startswith("/api/session-diffs/"):
                    self._json(200, self._session_diff_detail(unquote(path.removeprefix("/api/session-diffs/")), query))
                elif path.startswith("/api/chat/"):
                    session_id = unquote(path.removeprefix("/api/chat/").strip("/"))
                    self._json(200, {"session_id": session_id, "messages": runtime.chat_history(session_id)})
                elif path in _STATIC_PATHS:
                    self._serve_static(path)
                else:
                    self._json(404, {"error": "没有找到这个地址。"})
            except StoreError as exc:
                self._json(404, {"error": str(exc)})
            except Exception:
                self._json(500, {"error": "读取失败，请刷新页面后重试。"})

        # ----------------------------------------------------- POST dispatch
        def _post_facts_resolve(self, data: dict[str, Any]) -> tuple[int, Any]:
            service = runtime.service
            keep = str(data.get("keep", "")).strip()
            drop = str(data.get("drop", "")).strip()
            if not keep or not drop or keep == drop:
                raise ValueError("请选择两条不同的记忆。")
            keep_fact = service.store.get_fact(keep)
            drop_fact = service.store.get_fact(drop)
            if keep_fact.status != Status.CONTESTED or drop_fact.status != Status.CONTESTED:
                raise ValueError("只能解决两条都已标记为冲突的记忆。")
            service.store.resolve(keep, drop)
            service.retriever.sync_fact(service.store.get_fact(keep))
            service.retriever.sync_fact(service.store.get_fact(drop))
            return 200, {
                "ok": True,
                "keep": _fact_json(service.store.get_fact(keep)),
                "drop": _fact_json(service.store.get_fact(drop)),
            }

        def _post_chat(self, data: dict[str, Any]) -> tuple[int, Any]:
            session_id = data.get("session_id")
            result = runtime.chat(
                str(data.get("message", "")),
                session_id if isinstance(session_id, str) else None,
            )
            return 200, result

        def _post_external_import(self, data: dict[str, Any]) -> tuple[int, Any]:
            mode = str(data.get("mode", "memory")).strip().lower() or "memory"
            text = data.get("text") if isinstance(data.get("text"), str) else ""
            source = str(data.get("source") or "other")
            title = str(data.get("title") or "")
            if mode == "rag":
                return 200, external_import.import_as_rag(
                    runtime.service, text=text, source=source, title=title, llm=runtime.llm_client,
                )
            if mode == "memory":
                job, started = runtime.start_external_import(text=text, source=source, title=title)
                return (202 if started else 409), job
            raise ValueError("导入模式只能是 memory 或 rag。")

        def _post_sessions_import(self, data: dict[str, Any]) -> tuple[int, Any]:
            def _keys(name: str) -> list[str] | None:
                value = data.get(name)
                if value is None:
                    return None
                if not isinstance(value, list):
                    raise ValueError(f"{name} 必须是数组。")
                return [str(v)[:80] for v in value]

            if data.get("skips_only"):
                session_import.update_skips(
                    runtime.service,
                    add=_keys("skip"),
                    remove=_keys("unskip"),
                    clear=bool(data.get("clear_skips")),
                )
                return 200, {"ok": True, "skipped": len(session_import.load_skips(runtime.service))}
            job, started = runtime.start_session_import(
                bool(data.get("distill_facts")),
                only=_keys("only"),
                skip=_keys("skip"),
                unskip=_keys("unskip"),
                clear_skips=bool(data.get("clear_skips")),
            )
            return (202 if started else 409), job

        def _post_import_undo(self, data: dict[str, Any]) -> tuple[int, Any]:
            result = import_batches.undo_batch(runtime.service, str(data.get("batch", "")))
            runtime._invalidate_health()
            return 200, result

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            try:
                if self._foreign_request():
                    self._reject_foreign()
                    return
                max_bytes = (
                    MAX_UPLOAD_BYTES if path == "/api/rag/upload"
                    else max(MAX_BODY_BYTES, 1_500_000) if path == "/api/import/external"
                    else MAX_BODY_BYTES
                )
                data = self._body(max_bytes)
                service = runtime.service
                exact_post: dict[str, Callable[[dict[str, Any]], tuple[int, Any]]] = {
                    "/api/rag/upload": lambda d: (201, runtime.rag_upload(d)),
                    "/api/rag/delete": lambda d: (200, runtime.rag_delete(d)),
                    "/api/facts": lambda d: (201, self._add_fact(d)),
                    "/api/facts/resolve": self._post_facts_resolve,
                    "/api/conflicts/auto/undo": lambda d: (200, runtime.undo_auto_resolution(str(d.get("resolution_id") or ""))),
                    "/api/jobs/extract": lambda _d: (lambda res: (202 if res[1] else 409, res[0]))(runtime.start_extract()),
                    "/api/chat": self._post_chat,
                    "/api/model/test": lambda _d: (200, runtime.test_model()),
                    "/api/model/config": lambda d: (200, runtime.configure_model(d)),
                    "/api/config/reload": lambda _d: (200, runtime.reload_config()),
                    "/api/health/action": lambda d: (200, runtime.health_action(d)),
                    "/api/recall/verify": lambda _d: (lambda res: (202 if res[1] else 409, res[0]))(runtime.start_recall_verify()),
                    "/api/recall/verify/status": lambda _d: (200, runtime.recall_verify_status()),
                    "/api/model/embedding/test": lambda _d: (200, runtime.test_embedding()),
                    "/api/model/embedding/config": lambda d: (200, runtime.configure_embedding(d)),
                    "/api/model/rerank/config": lambda d: (200, runtime.configure_rerank(d)),
                    "/api/model/rerank/test": lambda _d: (200, runtime.test_rerank()),
                    "/api/autostart": lambda d: (200, autostart.set_enabled(bool(d.get("enabled")))),
                    "/api/import/external": self._post_external_import,
                    "/api/import/sessions/start": self._post_sessions_import,
                    "/api/import/undo": self._post_import_undo,
                    "/api/reindex": lambda _d: (200, service.reindex_all()),
                }
                if path in exact_post:
                    code, payload = exact_post[path](data)
                    self._json(code, payload)
                elif path.startswith("/api/facts/") and path.endswith("/correct"):
                    fact_id = unquote(path.removeprefix("/api/facts/").removesuffix("/correct").strip("/"))
                    self._json(201, self._add_fact(data, old_id=fact_id))
                elif path.startswith("/api/facts/") and path.endswith("/retract"):
                    fact_id = unquote(path.removeprefix("/api/facts/").removesuffix("/retract").strip("/"))
                    reason = str(data.get("reason", "")).strip()
                    if not reason:
                        raise ValueError("请填写撤回原因。")
                    service.store.retract(fact_id, reason)
                    service.retriever.sync_fact(service.store.get_fact(fact_id))
                    self._json(200, {"ok": True, "fact": _fact_json(service.store.get_fact(fact_id))})
                elif path.startswith("/api/facts/") and path.endswith("/contest"):
                    fact_id = unquote(path.removeprefix("/api/facts/").removesuffix("/contest").strip("/"))
                    note = str(data.get("note", "")).strip() or "owner marked as contested"
                    service.store.contest([fact_id], note)
                    service.retriever.sync_fact(service.store.get_fact(fact_id))
                    self._json(200, {"ok": True, "fact": _fact_json(service.store.get_fact(fact_id))})
                elif path.startswith("/api/agents/") and path.endswith("/connect"):
                    key = unquote(path.removeprefix("/api/agents/").removesuffix("/connect").strip("/"))
                    result = agents.connect(key, service.settings.home)
                    if result.get("ok"):
                        # A connected agent is trusted again: lift any isolation.
                        service.enable_agent(key)
                    self._json(200, result)
                elif path.startswith("/api/agents/") and path.endswith("/disconnect"):
                    key = unquote(path.removeprefix("/api/agents/").removesuffix("/disconnect").strip("/"))
                    result = agents.disconnect(key)
                    service.disable_agent(key)
                    self._json(200, result)
                else:
                    self._json(404, {"error": "没有找到这个地址。"})
            except (ValueError, json.JSONDecodeError, UnicodeDecodeError, ValidationError) as exc:
                message = str(exc) if isinstance(exc, ValueError) and str(exc) else "请检查填写内容后重试。"
                self._json(400, {"error": message})
            except autostart.AutostartError as exc:
                self._json(400, {"error": str(exc)})
            except StoreError as exc:
                self._json(400, {"error": str(exc)})
            except LLMError as exc:
                self._json(502, {"error": str(exc)})
            except EmbeddingError as exc:
                self._json(502, {"error": str(exc)})
            except Exception:
                self._json(500, {"error": "保存失败，请刷新页面后重试。原有记忆不会被覆盖。"})

    return Handler


class _ExclusiveServer(ThreadingHTTPServer):
    # On Windows SO_REUSEADDR lets a second process bind a port that is already
    # serving, so two dashboards would run side by side. Refuse that instead.
    allow_reuse_address = os.name != "nt"


def running_dashboard(host: str, port: int) -> bool:
    """Whether a membrain dashboard already answers on this address."""
    import http.client

    try:
        conn = http.client.HTTPConnection(host, port, timeout=1.5)
        conn.request("GET", "/api/model")
        response = conn.getresponse()
        server = response.getheader("Server", "")
        conn.close()
        return "MembrainDashboard" in server
    except OSError:
        return False


def create_server(
    host: str = "127.0.0.1",
    port: int = 1966,
    runtime: DashboardRuntime | None = None,
) -> ThreadingHTTPServer:
    if not _loopback(host):
        raise ValueError("为了保护个人记忆，后台只允许绑定本机回环地址。")
    dashboard_runtime = runtime or DashboardRuntime()
    server = _ExclusiveServer((host, port), make_handler(dashboard_runtime))
    server.daemon_threads = True
    server.runtime = dashboard_runtime  # type: ignore[attr-defined]
    return server


def open_dashboard(host: str, port: int) -> None:
    import webbrowser

    webbrowser.open(f"http://{host}:{port}/")


def listening_pid(port: int) -> int | None:
    """The PID listening on `port` (None when nothing does).

    Parsed from netstat rather than psutil: the dashboard must not depend on an
    extra package to restart itself, and netstat ships with Windows.
    """
    if os.name != "nt":
        return None
    try:
        out = subprocess.run(
            ["netstat", "-ano", "-p", "TCP"],
            capture_output=True,
            timeout=8,
            creationflags=0x08000000,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = out.stdout.decode("utf-8", errors="replace") if isinstance(out.stdout, bytes) else (out.stdout or "")
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5 or not parts[0].upper().startswith("TCP"):
            continue
        local, state, pid = parts[1], parts[-2].upper(), parts[-1]
        if state != "LISTENING" or not local.endswith(f":{port}") or not pid.isdigit():
            continue
        return int(pid)
    return None


def stop_dashboard(port: int, timeout: float = 10.0) -> bool:
    """Stop the dashboard listening on `port`, so a relaunch really reloads code.

    Only a python process is killed: an unrelated program that happens to hold
    the port must not be terminated by a launcher click.
    """
    pid = listening_pid(port)
    if pid is None:
        return True
    try:
        listed = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            timeout=8,
            creationflags=0x08000000,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    raw = listed.stdout or b""
    name = raw.decode("utf-8", errors="replace").split(",")[0].strip('"').lower()
    if not name.startswith("python"):
        return False
    try:
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/F"],
            capture_output=True,
            timeout=10,
            creationflags=0x08000000,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if listening_pid(port) is None:
            return True
        time.sleep(0.2)
    return listening_pid(port) is None
