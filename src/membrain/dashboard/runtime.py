"""Dashboard runtime state, background jobs, model settings, and health orchestration."""

from __future__ import annotations

import json
import os
import queue as _queue
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .. import archive, external_import, health, session_import
from ..budget import estimate_tokens
from ..config import DEFAULT_ENV_FILE, IN_CHECKOUT, REPO_ROOT
from ..embeddings import Embedder, EmbeddingError
from ..llm import LLMClient, LLMError
from ..locking import file_lock
from ..models import Episode, Evidence, FactType
from ..server import MemoryService
from ..store import _atomic_write
from . import admin_agent

# Scanning embeds every live claim, so the dashboard caches the result instead of
# recomputing a 20-second report on every visit.
HEALTH_CACHE_SECONDS = 120.0


class DashboardRuntime:
    """Owns the shared memory service and background dashboard jobs."""

    def __init__(
        self,
        service: MemoryService | None = None,
        llm_client: Any | None = None,
        auto_extract_chat: bool = True,
    ) -> None:
        self.service = service or MemoryService()
        self.service.ensure_index()
        self.llm_client = llm_client or LLMClient(self.service.settings)
        self.auto_extract_chat = auto_extract_chat
        self._job_lock = threading.Lock()
        self._job: dict[str, Any] = {"status": "idle", "report": None, "error": None}
        self._chat_lock = threading.Lock()
        self._chats: dict[str, list[dict[str, str]]] = {}
        self._import_job: dict[str, Any] = {"status": "idle", "kind": None}
        self._import_lock = threading.Lock()
        self._recall_job_lock = threading.Lock()
        self._recall_job: dict[str, Any] = {"status": "idle", "report": None, "error": None}
        # Health scans embed every live claim (seconds on a real store); the
        # admin loop asks for the same report several times per task.
        self._health_cache: dict[str, Any] = {"at": 0.0, "data": None}
        # Auto-capture watcher: tails agent session stores so writes never
        # depend on a model calling a tool. Opt out with MEMBRAIN_WATCH=0.
        from ..watcher import Watcher

        self.watcher = Watcher(self.service, on_capture=self._watcher_extract)
        self.watcher.start()

    def _watcher_extract(self) -> None:
        """Kick extraction after a capture pass found new user turns."""
        if self.auto_extract_chat:
            self.start_extract()

    def watcher_status(self) -> dict[str, Any]:
        return self.watcher.status()

    # ------------------------------------------------------------- RAG library
    def rag_list(self) -> dict[str, Any]:
        from .. import rag

        return {"documents": rag.list_documents(self.service.settings.home)}

    def rag_search(self, query: str, top_k: int = 8) -> dict[str, Any]:
        from .. import rag

        hits = rag.search(self.service.settings.home, query, top_k=top_k, embedder=self.service.embedder)
        return {"query": query, "hits": hits}

    def rag_document(self, doc_id: str) -> dict[str, Any]:
        from .. import rag

        try:
            return rag.get_document(self.service.settings.home, doc_id)
        except KeyError as exc:
            raise ValueError("没有找到这篇资料。") from exc

    def rag_upload(self, data: dict[str, Any]) -> dict[str, Any]:
        from .. import rag

        filename = str(data.get("filename", "")).strip()
        payload = str(data.get("data", "")).strip()
        category = str(data.get("category", "docs")).strip() or "docs"
        if not filename or not payload:
            raise ValueError("缺少文件名或文件内容。")
        return rag.ingest_upload(
            home=self.service.settings.home,
            filename=filename,
            data_b64=payload,
            category=category,
            embedder=self.service.embedder,
            llm=self.llm_client,
        )

    def rag_delete(self, data: dict[str, Any]) -> dict[str, Any]:
        from .. import rag

        doc_id = str(data.get("id", "")).strip()
        if not doc_id:
            raise ValueError("缺少资料 id。")
        ok = rag.delete_document(self.service.settings.home, doc_id)
        return {"ok": ok}

    # ------------------------------------------------------------- Imports
    def import_job(self) -> dict[str, Any]:
        with self._import_lock:
            return dict(self._import_job)

    def session_import_preview(self) -> dict[str, Any]:
        return {
            **session_import.preview(self.service),
            "labels": archive.CATEGORY_LABELS,
            "source_labels": session_import.SOURCE_LABELS,
        }

    def start_session_import(
        self,
        distill_facts: bool = False,
        only: list[str] | None = None,
        skip: list[str] | None = None,
        unskip: list[str] | None = None,
        clear_skips: bool = False,
    ) -> tuple[dict[str, Any], bool]:
        with self._import_lock:
            if self._import_job.get("status") == "running":
                if self._import_job.get("kind") != "sessions":
                    raise ValueError("已有其他导入任务正在运行，请等待它完成。")
                return dict(self._import_job), False
            job_id = uuid.uuid4().hex
            self._import_job = {
                "id": job_id,
                "kind": "sessions",
                "status": "running",
                "progress": {"stage": "preparing"},
            }

        def update(**fields: Any) -> None:
            with self._import_lock:
                if self._import_job.get("id") == job_id:
                    self._import_job = {**self._import_job, **fields}

        def run() -> None:
            try:
                report = session_import.apply_import(
                    self.service,
                    self.llm_client,
                    distill_facts=distill_facts,
                    only=only,
                    skip=skip,
                    unskip=unskip,
                    clear_skips=clear_skips,
                    progress=lambda p: update(progress=p),
                )
                update(status="done", report=report, progress=None)
            except Exception as exc:  # the page shows a recoverable message
                update(status="error", error=str(exc) or exc.__class__.__name__, progress=None)

        threading.Thread(target=run, name="membrain-dashboard-session-import", daemon=True).start()
        return self.import_job(), True

    def start_external_import(
        self,
        *,
        text: str,
        source: str = "other",
        title: str = "",
    ) -> tuple[dict[str, Any], bool]:
        """Validate pasted external AI text and distill durable memories in background."""
        clean, src, final_title, _ = external_import.validate_inputs(text, source, title)
        if isinstance(self.llm_client, LLMClient) and not self.llm_client.settings.llm_enabled:
            raise LLMError("LLM 尚未配置，请先在「模型连接」配置模型后再导入为记忆。")
        with self._import_lock:
            if self._import_job.get("status") == "running":
                if self._import_job.get("kind") != "external":
                    raise ValueError("已有其他导入任务正在运行，请等待它完成。")
                return dict(self._import_job), False
            job_id = uuid.uuid4().hex
            self._import_job = {
                "id": job_id,
                "kind": "external",
                "status": "running",
                "progress": {"stage": "preparing"},
            }

        def update(**fields: Any) -> None:
            with self._import_lock:
                if self._import_job.get("id") == job_id:
                    self._import_job = {**self._import_job, **fields}

        def run() -> None:
            try:
                report = external_import.import_as_memories(
                    self.service,
                    self.llm_client,
                    text=clean,
                    source=src,
                    title=final_title,
                    progress=lambda p: update(progress=p),
                )
                self._invalidate_health()
                update(status="done", report=report, progress=None)
            except Exception as exc:  # the page shows a recoverable message
                update(status="error", error=str(exc) or exc.__class__.__name__, progress=None)

        threading.Thread(target=run, name="membrain-dashboard-external-import", daemon=True).start()
        return self.import_job(), True

    def list_auto_resolutions(self, include_undone: bool = False) -> dict[str, Any]:
        return {"items": self.service.store.list_auto_resolutions(include_undone=include_undone)}

    def undo_auto_resolution(self, resolution_id: str) -> dict[str, Any]:
        result = self.service.store.undo_auto_resolve(resolution_id)
        for fid in (result["keep_id"], result["drop_id"]):
            self.service.retriever.sync_fact(self.service.store.get_fact(fid))
        self._invalidate_health()
        return result

    # ------------------------------------------------------------- Models & config
    @staticmethod
    def _endpoint(base_url: str | None) -> str:
        """Show the endpoint without leaking the path prefix beyond the API root."""
        if not base_url:
            return "—"
        parsed = urlsplit(base_url)
        host = parsed.hostname or ""
        try:
            port = parsed.port
        except ValueError:
            port = None
        if port:
            host = f"{host}:{port}"
        return f"{parsed.scheme}://{host}{parsed.path}" if parsed.scheme else parsed.path

    def model_status(self) -> dict[str, Any]:
        settings = self.service.settings
        return {
            "configured": settings.llm_enabled,
            "format": settings.llm_api_format,
            "model": settings.llm_model or "—",
            "base_url": self._endpoint(settings.llm_base_url),
            "api_key_set": bool(settings.llm_api_key),
            "timeout": settings.llm_timeout,
            "embedding": self.service.settings.describe()["embeddings"],
            "embedding_configured": settings.embeddings_enabled,
            "embedding_model": settings.embed_model or "—",
            "embedding_base_url": self._endpoint(settings.embed_base_url),
            "embedding_api_key_set": bool(settings.embed_api_key),
            "embedding_timeout": settings.embed_timeout,
            # Rerank is a separate model from the embedding one: embeddings turn
            # text into vectors for the first pass, a reranker rescores the
            # shortlist. One endpoint cannot do both.
            "rerank_configured": bool(settings.rerank_base_url and settings.rerank_model),
            "rerank_enabled": bool(settings.use_rerank_default),
            "rerank_provider": settings.rerank_provider,
            "rerank_model": settings.rerank_model or "—",
            "rerank_base_url": self._endpoint(settings.rerank_base_url),
            "rerank_api_key_set": bool(settings.rerank_api_key),
            "rerank_timeout": settings.rerank_timeout,
            **self._recall_baseline_status(),
        }

    def _recall_baseline_status(self) -> dict[str, Any]:
        """Baseline info for the recall-quality card, plus a stale-model flag."""
        baseline_path = self.service.settings.home / "state" / "recall-baseline.json"
        try:
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"recall_baseline": None, "recall_baseline_stale": False}
        stale = bool(baseline.get("embed_model")) and baseline.get("embed_model") != self.service.settings.embed_model
        return {"recall_baseline": baseline, "recall_baseline_stale": stale}

    def start_recall_verify(self) -> tuple[dict[str, Any], bool]:
        """Run the calibration script in the background and store its verdict."""
        with self._recall_job_lock:
            if self._recall_job["status"] == "running":
                return dict(self._recall_job), False
            job_id = uuid.uuid4().hex
            self._recall_job = {"id": job_id, "status": "running", "report": None, "error": None}
        root = REPO_ROOT if IN_CHECKOUT else Path.cwd()
        script = root / "scripts" / "calibrate.py"
        baseline_path = self.service.settings.home / "state" / "recall-baseline.json"
        log_path = self.service.settings.home / ".state" / "recall-verify.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        def run() -> None:
            try:
                if not script.is_file():
                    raise RuntimeError(f"calibrate.py 未找到（{script}）。从源码部署时才提供召回验证。")
                env = os.environ.copy()
                env["MEMBRAIN_HOME"] = str(self.service.settings.home)
                started = time.monotonic()
                proc = subprocess.Popen(
                    [sys.executable, "-u", str(script), "--baseline-path", str(baseline_path), "--strict"],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace", env=env,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                out_q: "_queue.Queue[str | None]" = _queue.Queue()

                def _pump() -> None:
                    try:
                        for raw in proc.stdout or ():
                            out_q.put(raw.rstrip("\n"))
                    finally:
                        out_q.put(None)  # end-of-stream sentinel

                threading.Thread(target=_pump, daemon=True).start()
                lines: list[str] = []
                deadline = started + 600
                last_line = ""
                while True:
                    if time.monotonic() > deadline:
                        proc.kill()
                        raise RuntimeError("验证超时（10 分钟），已终止。")
                    try:
                        item = out_q.get(timeout=2)
                    except _queue.Empty:
                        item = "<tick>"  # silent: refresh elapsed only
                    if item is None:
                        break
                    if item != "<tick>":
                        if item:
                            lines.append(item)
                            last_line = item
                    with self._recall_job_lock:
                        if self._recall_job.get("id") == job_id:
                            self._recall_job["progress"] = {
                                "line": last_line[:200],
                                "elapsed": int(time.monotonic() - started),
                            }
                code = proc.wait(timeout=30)
                report = {"exit_code": code, "output": "\n".join(lines)[-4000:], "error": "", "ok": code == 0}
                with self._recall_job_lock:
                    self._recall_job = {
                        "id": job_id, "status": "done", "report": report, "error": None, "progress": None,
                    }
            except Exception as exc:
                with self._recall_job_lock:
                    self._recall_job = {
                        "id": job_id, "status": "error", "report": None, "error": str(exc), "progress": None,
                    }

        threading.Thread(target=run, daemon=True).start()
        return dict(self._recall_job), True

    def recall_verify_status(self) -> dict[str, Any]:
        with self._recall_job_lock:
            return dict(self._recall_job)

    def _write_env(self, env_path: Path, updates: dict[str, str]) -> None:
        """Merge key/value pairs into the .env file, keeping the rest intact."""
        lock_path = env_path.parent / ".state" / "dashboard-env.lock"
        with file_lock(lock_path):
            lines = env_path.read_text(encoding="utf-8-sig").splitlines() if env_path.is_file() else []
            remaining = dict(updates)
            output: list[str] = []
            for line in lines:
                content = line.strip()
                prefix = "export " if content.startswith("export ") else ""
                key, separator, _ = (content[len(prefix):].partition("=") if content and not content.startswith("#") else ("", "", ""))
                if separator and key.strip() in remaining:
                    name = key.strip()
                    value = remaining.pop(name)
                    output.append(f'{prefix}{name}="{value}"' if value else f"{prefix}{name}=")
                else:
                    output.append(line)
            for name, value in remaining.items():
                output.append(f'{name}="{value}"' if value else f"{name}=")
            _atomic_write(env_path, "\n".join(output).rstrip() + "\n")

    @staticmethod
    def _validated_url(raw: str, label: str) -> str:
        if not raw or len(raw) > 1000:
            raise ValueError(f"请填写有效的{label}。")
        parsed = urlsplit(raw)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username
                or parsed.password or parsed.query or parsed.fragment):
            raise ValueError(f"{label}须为不含账号、密钥和查询参数的 HTTP 或 HTTPS 地址。")
        return raw

    @staticmethod
    def _validated_model(raw: str, label: str) -> str:
        if not raw or len(raw) > 200 or any(char in raw for char in '\r\n"'):
            raise ValueError(f"{label}不能为空，且最多 200 个字符。")
        return raw

    @staticmethod
    def _validated_key(raw: Any) -> str:
        if not isinstance(raw, str) or len(raw) > 4000 or any(char in raw for char in '\r\n"\\'):
            raise ValueError("API Key 格式无效。")
        return raw

    @staticmethod
    def _validated_timeout(raw: Any, fallback: float) -> float:
        try:
            timeout = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("超时须为 1 到 300 秒。") from exc
        if not 1 <= timeout <= 300:
            raise ValueError("超时须为 1 到 300 秒。")
        return timeout

    def configure_model(self, data: dict[str, Any]) -> dict[str, Any]:
        """Save the LLM settings. The embedding settings are saved separately."""
        settings = self.service.settings
        api_format = str(data.get("format", "")).strip().lower()
        base_url = str(data.get("base_url", "")).strip()
        model = str(data.get("model", "")).strip()
        key_input = data.get("api_key", "")
        clear_key = data.get("clear_api_key") is True
        if api_format not in {"openai", "anthropic"}:
            raise ValueError("请选择 OpenAI 兼容或 Anthropic 接口格式。")
        base_url = self._validated_url(base_url, "模型接口地址")
        model = self._validated_model(model, "模型名称")
        key_input = self._validated_key(key_input)
        timeout = self._validated_timeout(data.get("timeout", settings.llm_timeout), settings.llm_timeout)

        api_key = None if clear_key else (key_input or settings.llm_api_key)
        env_path = settings.env_file or DEFAULT_ENV_FILE
        self._write_env(env_path, {
            "MEMBRAIN_LLM_BASE_URL": base_url,
            "MEMBRAIN_LLM_API_FORMAT": api_format,
            "MEMBRAIN_LLM_MODEL": model,
            "MEMBRAIN_LLM_TIMEOUT": str(timeout),
            "MEMBRAIN_LLM_API_KEY": api_key or "",
        })
        updated = replace(
            settings, llm_base_url=base_url, llm_api_format=api_format,
            llm_model=model, llm_api_key=api_key, llm_timeout=timeout, env_file=env_path,
        )
        self.service.settings = updated
        self.llm_client = LLMClient(updated)
        return self.model_status()

    def configure_embedding(self, data: dict[str, Any]) -> dict[str, Any]:
        """Save the embedding endpoint. An empty address disables embeddings."""
        settings = self.service.settings
        base_url = str(data.get("base_url", "")).strip()
        model = str(data.get("model", "")).strip()
        key_input = self._validated_key(data.get("api_key", ""))
        clear_key = data.get("clear_api_key") is True
        timeout = self._validated_timeout(data.get("timeout", settings.embed_timeout), settings.embed_timeout)

        if not base_url:
            # Turning embeddings off is a valid action: retrieval falls back to
            # full-text search.
            api_key = None
        else:
            base_url = self._validated_url(base_url, "Embedding 接口地址")
            model = self._validated_model(model, "Embedding 模型名称")
            api_key = None if clear_key else (key_input or settings.embed_api_key)

        env_path = settings.env_file or DEFAULT_ENV_FILE
        self._write_env(env_path, {
            "MEMBRAIN_EMBED_BASE_URL": base_url,
            "MEMBRAIN_EMBED_MODEL": model,
            "MEMBRAIN_EMBED_TIMEOUT": str(timeout),
            "MEMBRAIN_EMBED_API_KEY": api_key or "",
        })
        updated = replace(
            settings, embed_base_url=base_url or None, embed_model=model or settings.embed_model,
            embed_api_key=api_key, embed_timeout=timeout, env_file=env_path,
        )
        self.service.settings = updated
        # The retriever and the archive keep their own reference to the
        # embedder, so all three have to be swapped together — otherwise the
        # new endpoint would only be used by half the code paths.
        embedder = Embedder(updated)
        self.service.embedder = embedder
        self.service.retriever.embedder = embedder
        self.service.archive.embedder = embedder
        return self.model_status()

    def configure_rerank(self, data: dict[str, Any]) -> dict[str, Any]:
        """Save the rerank endpoint. An empty address turns reranking off.

        Reranking only changes the order of candidates that retrieval already
        found, so turning it off costs recall nothing — the ranking just falls
        back to the lexical + semantic blend.
        """
        settings = self.service.settings
        base_url = str(data.get("base_url", "")).strip()
        model = str(data.get("model", "")).strip()
        key_input = self._validated_key(data.get("api_key", ""))
        clear_key = data.get("clear_api_key") is True
        timeout = self._validated_timeout(data.get("timeout", settings.rerank_timeout), settings.rerank_timeout)
        wanted = data.get("enabled")
        enabled = True if wanted is None else bool(wanted)

        if not base_url or not model:
            base_url, model = "", ""
            api_key = None
            enabled = False
        else:
            base_url = self._validated_url(base_url, "重排接口地址")
            model = self._validated_model(model, "重排模型名称")
            api_key = None if clear_key else (key_input or settings.rerank_api_key)

        env_path = settings.env_file or DEFAULT_ENV_FILE
        self._write_env(env_path, {
            "MEMBRAIN_RERANK_PROVIDER": "openai" if base_url else "heuristic",
            "MEMBRAIN_RERANK_BASE_URL": base_url,
            "MEMBRAIN_RERANK_MODEL": model,
            "MEMBRAIN_RERANK_TIMEOUT": str(timeout),
            "MEMBRAIN_RERANK_API_KEY": api_key or "",
            "MEMBRAIN_USE_RERANK": "1" if enabled else "0",
        })
        updated = replace(
            settings,
            rerank_provider="openai" if base_url else "heuristic",
            rerank_base_url=base_url or None,
            rerank_model=model or None,
            rerank_api_key=api_key,
            rerank_timeout=timeout,
            use_rerank_default=enabled,
            env_file=env_path,
        )
        self.service.settings = updated
        # The retriever reads thresholds and rerank settings off its own copy.
        self.service.retriever.settings = updated
        return self.model_status()

    def test_rerank(self) -> dict[str, Any]:
        """Call the rerank endpoint once with a known-good pair of documents."""
        settings = self.service.settings
        if not (settings.rerank_base_url and settings.rerank_model):
            raise ValueError("重排尚未配置。填写接口地址和模型名称后即可启用。")
        from ..rerank import _openai_compatible_scores

        documents = ["喜欢喝咖啡", "用 PySide6 写了一个番茄钟", "要求回复简短、结论先行"]
        started = time.monotonic()
        scores = _openai_compatible_scores(
            "我平时喝什么",
            documents,
            base_url=settings.rerank_base_url,
            api_key=settings.rerank_api_key,
            model=settings.rerank_model,
            timeout=settings.rerank_timeout,
        )
        best = max(range(len(scores)), key=lambda i: scores[i]) if scores else 0
        return {
            "elapsed_ms": int((time.monotonic() - started) * 1000),
            "scores": [round(float(s), 4) for s in scores],
            "best": documents[best] if documents else "",
            "documents": documents,
        }

    def test_embedding(self) -> dict[str, Any]:
        """Call the embedding endpoint once and report the vector size."""
        settings = self.service.settings
        if not settings.embeddings_enabled:
            raise EmbeddingError("Embedding 尚未配置。填写接口地址后即可启用语义检索。")
        started = time.monotonic()
        vectors = self.service.embedder.embed(["membrain connectivity check"])
        if vectors is None or len(vectors) == 0:
            raise EmbeddingError("Embedding 接口没有返回向量。")
        return {
            "ok": True,
            "dimensions": int(vectors.shape[1]),
            "elapsed_ms": round((time.monotonic() - started) * 1000),
            "model": settings.embed_model,
        }

    def test_model(self) -> dict[str, Any]:
        if not self.service.settings.llm_enabled:
            raise LLMError("LLM 尚未配置。请在 .env 中设置模型地址和模型名。")
        started = time.monotonic()
        answer = self.llm_client.complete_text(
            "You are a connectivity check. Reply with a short confirmation only.",
            "Reply with OK.",
        )
        return {
            "ok": True,
            "answer": answer[:200],
            "elapsed_ms": round((time.monotonic() - started) * 1000),
            "model": self.service.settings.llm_model,
        }

    HOT_FIELDS = (
        "llm_base_url", "llm_model", "llm_api_key", "llm_api_format", "llm_timeout",
        "embed_base_url", "embed_api_key", "embed_model", "embed_timeout",
        "semantic_floor", "use_intent_default", "use_rerank_default",
    )

    def reload_config(self) -> dict[str, Any]:
        """Re-read .env into the running runtime without a restart.

        Only model/embedding/threshold settings hot-reload. The memory home and
        other structural paths stay as they were: switching those safely means
        restarting the dashboard, because the store and index hold open files.
        """
        from ..config import load_settings

        new_settings = load_settings()
        old_settings = self.service.settings
        changed: dict[str, dict[str, str]] = {}
        for field in self.HOT_FIELDS:
            old_value = getattr(old_settings, field, None)
            new_value = getattr(new_settings, field, None)
            secret = "api_key" in field
            if (old_value or "") != (new_value or ""):
                changed[field] = {
                    "from": "(已设置)" if secret and old_value else (old_value or ""),
                    "to": "(已设置)" if secret and new_value else (new_value or ""),
                }
        # Swap the shared settings reference: every holder (LLM client, embedder,
        # retriever) reads fields off it per call, so no object rebuild is needed.
        self.service.settings = new_settings
        self.llm_client.settings = new_settings
        self.service.embedder.settings = new_settings
        self.service.retriever.settings = new_settings
        self.service.archive.settings = new_settings
        return {
            "ok": True,
            "changed": changed,
            "changed_count": len(changed),
            "note": "记忆目录等结构性配置不参与热加载，仍需重启。",
        }

    # ------------------------------------------------------------- Health
    def _invalidate_health(self) -> None:
        self._health_cache = {"at": 0.0, "data": None}

    def health_report(self, refresh: bool = False, kinds: set[str] | None = None) -> dict[str, Any]:
        cached = self._health_cache
        if (not refresh and cached["data"] is not None
                and (time.monotonic() - cached["at"]) < HEALTH_CACHE_SECONDS):
            data = dict(cached["data"])
            if kinds:
                data["findings"] = [f for f in data["findings"] if f["kind"] in kinds]
            return data
        service = self.service
        scanner = health.HealthScanner(service, lambda fact: service.retriever.recall_days_since_use(fact))
        findings = scanner.scan(kinds)
        dismissed = health.DismissalStore(service.settings.home / "state" / "health-dismissed.json").load()
        visible = [finding for finding in findings if finding.fingerprint not in dismissed]
        visible.sort(key=lambda finding: (-finding.severity, finding.kind))
        score = max(0, 100 - sum(finding.severity * 3 for finding in visible))
        # Not `kinds`: that name is the requested-kind filter above, and
        # shadowing it made the cache check think a full scan was partial.
        counts: dict[str, int] = {}
        for finding in visible:
            counts[finding.kind] = counts.get(finding.kind, 0) + 1
        result = {
            "score": score,
            "findings": [finding.as_dict() for finding in visible],
            "counts": counts,
            "dismissed": len(findings) - len(visible),
            "scanned_at": datetime.now(timezone.utc).isoformat(),
        }
        if kinds is None:
            # Only a full scan may seed the shared cache; a partial one would
            # hide the kinds it did not look for.
            self._health_cache = {"at": time.monotonic(), "data": result}
        return dict(result)

    def health_action(self, data: dict[str, Any]) -> dict[str, Any]:
        action = str(data.get("action", "")).strip().lower()
        fingerprint = str(data.get("fingerprint", "")).strip()
        if action == "dismiss":
            if not fingerprint:
                raise ValueError("dismiss 需要 finding 的 fingerprint。")
            store_path = self.service.settings.home / "state" / "health-dismissed.json"
            dismissed = health.DismissalStore(store_path).load()
            dismissed.add(fingerprint)
            health.DismissalStore(store_path).save(dismissed)
            return {"ok": True, "action": "dismiss"}
        if action not in {"supersede", "retract"}:
            raise ValueError("action 只支持 dismiss / supersede / retract。")
        keep_id = str(data.get("keep_id", "")).strip()
        drop_id = str(data.get("drop_id", "")).strip()
        now = datetime.now(timezone.utc)
        if action == "supersede":
            if not keep_id or not drop_id or keep_id == drop_id:
                raise ValueError("合并替代需要 keep_id 和 drop_id 两条不同记忆。")
            keep = self.service.store.get_fact(keep_id)
            drop = self.service.store.get_fact(drop_id)
            if FactType.CONSTRAINT in {keep.type, drop.type}:
                raise ValueError("硬约束只能由用户在记忆详情中手动处理。")
            # Two near-duplicates often quote the same turn; a Fact may cite an
            # episode only once, so collapse per episode and keep the fuller quote.
            merged_evidence: dict[str, Evidence] = {}
            for item in [*keep.evidence, *drop.evidence]:
                existing = merged_evidence.get(item.episode)
                if existing is None or len(item.quote) > len(existing.quote):
                    merged_evidence[item.episode] = item
            replacement = self.service.store.draft_fact(
                type=keep.type,
                subject=keep.subject,
                claim=keep.claim,
                evidence=list(merged_evidence.values()),
                source_agent="membrain-dashboard",
                source_session="health-center",
                supersedes=keep.id,
                created_at=min(keep.created_at, drop.created_at),
            )
            saved = self.service.store.write_fact(replacement)
            # Both sides have to retire into the merged record. Leaving `keep`
            # live created a new exact duplicate of the merged fact, so a merge
            # never reduced the duplicate count.
            self.service.store.supersede(keep.id, saved.id, when=now.date())
            self.service.store.supersede(drop.id, saved.id, when=now.date())
            for fact_id in (saved.id, keep.id, drop.id):
                self.service.retriever.sync_fact(self.service.store.get_fact(fact_id))
            self._invalidate_health()
            return {
                "ok": True,
                "action": "supersede",
                "kept": saved.id,
                "dropped": drop.id,
                "merged_from": [keep.id, drop.id],
                "note": "两条证据已合并保留在新记忆中。",
            }
        if not drop_id:
            raise ValueError("撤回需要 drop_id。")
        target = self.service.store.get_fact(drop_id)
        if target.type == FactType.CONSTRAINT:
            raise ValueError("硬约束只能由用户在记忆详情中手动处理。")
        self.service.store.retract(drop_id, f"健康中心撤回：{fingerprint or 'owner confirmed'}")
        self.service.retriever.sync_fact(self.service.store.get_fact(drop_id))
        return {"ok": True, "action": "retract", "dropped": drop_id}

    # ------------------------------------------------------------- Chat & admin loop
    def chat_history(self, session_id: str) -> list[dict[str, str]]:
        with self._chat_lock:
            cached = self._chats.get(session_id)
        if cached is not None:
            return list(cached)
        turns = [
            episode for episode in self.service.store.iter_episodes()
            if episode.session == session_id
            and episode.agent == "membrain-dashboard"
            and episode.role in ("user", "assistant")
        ]
        turns.sort(key=lambda episode: (episode.ts, episode.id))
        return [{"role": episode.role, "content": episode.text} for episode in turns[-20:]]

    def chat_sessions(self) -> list[dict[str, Any]]:
        """Past dashboard chat sessions, newest first. Titles come from the first user turn."""
        sessions: dict[str, dict[str, Any]] = {}
        for episode in self.service.store.iter_episodes():
            if episode.agent != "membrain-dashboard" or episode.role not in ("user", "assistant"):
                continue
            row = sessions.setdefault(
                episode.session,
                {"session_id": episode.session, "title": "", "last_ts": "", "turns": 0},
            )
            row["turns"] += 1
            ts = episode.ts.isoformat()
            if ts > row["last_ts"]:
                row["last_ts"] = ts
            if episode.role == "user" and not row["title"]:
                row["title"] = " ".join(episode.text.split())[:40] or "（无标题会话）"
        rows = sorted(sessions.values(), key=lambda row: row["last_ts"], reverse=True)
        return rows[:50]

    def _admin_visible(self, fact: Any) -> bool:
        return admin_agent.admin_visible(fact)

    def _admin_row(self, fact: Any) -> dict[str, Any]:
        return admin_agent.admin_row(fact)

    def _admin_execute(
        self,
        action: str,
        args: dict[str, Any],
        message: str,
        episode: Episode,
        session: str,
    ) -> tuple[bool, dict[str, Any]]:
        return admin_agent.execute_action(self, action, args, message, episode, session)

    @staticmethod
    def _partial_answer(trace: list[dict[str, Any]]) -> str:
        return admin_agent.partial_answer(trace)

    def _admin_loop(
        self,
        message: str,
        history: list[dict[str, str]],
        recall: Any,
        episode: Episode,
        session: str,
        library: list[dict[str, Any]] | None = None,
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any] | None]:
        return admin_agent.run_loop(self, message, history, recall, episode, session, library=library)

    def chat(self, message: str, session_id: str | None = None) -> dict[str, Any]:
        message = message.strip()
        if not message:
            raise ValueError("请输入消息。")
        if len(message) > 20000:
            raise ValueError("消息最多 20000 个字符。")
        session = session_id.strip() if isinstance(session_id, str) and session_id.strip() else uuid.uuid4().hex
        with self._chat_lock:
            history = list(self._chats.get(session, []))

        now = datetime.now(timezone.utc)
        episode = Episode(
            id=f"ep_chat_{uuid.uuid4().hex[:16]}",
            ts=now,
            session=session,
            agent="membrain-dashboard",
            role="user",
            text=message,
            project=None,
        )
        saved_episode = self.service.store.append_episodes([episode])[0]
        recall = self.service.retriever.recall(
            message,
            budget=min(self.service.settings.turn_token_budget, 2000),
        )
        started = time.monotonic()
        estimated_input = estimate_tokens(message)
        library: list[dict[str, Any]] = []
        library_note: str | None = None
        try:
            library = self.rag_search(message, top_k=3)["hits"]
        except Exception as exc:  # an unreadable library must not block the answer
            library_note = f"{exc.__class__.__name__}: {exc}"
        try:
            answer, actions, memory_action = self._admin_loop(
                message, history, recall, saved_episode, session, library=library,
            )
        except LLMError:
            if self.auto_extract_chat:
                self.start_extract()
            raise
        except Exception as exc:
            if self.auto_extract_chat:
                self.start_extract()
            raise LLMError("模型请求失败") from exc
        elapsed_ms = round((time.monotonic() - started) * 1000)
        assistant_episode = Episode(
            id=f"ep_chat_{uuid.uuid4().hex[:16]}",
            ts=datetime.now(timezone.utc),
            session=session,
            agent="membrain-dashboard",
            role="assistant",
            text=answer,
            project=None,
        )
        self.service.store.append_episodes([assistant_episode])
        with self._chat_lock:
            updated = [*history, {"role": "user", "content": message},
                       {"role": "assistant", "content": answer}][-20:]
            self._chats[session] = updated
        if self.auto_extract_chat:
            self.start_extract()
        return {
            "session_id": session,
            "episode_id": saved_episode.id,
            "answer": answer,
            "memory_action": memory_action,
            "actions": actions,
            "memory": [
                {
                    "id": fact.id,
                    "claim": fact.claim,
                    "subject": fact.subject,
                    "status": fact.status.value,
                    "confidence": fact.confidence,
                }
                for fact in recall.hits
            ],
            "memory_tokens": recall.tokens,
            "memory_mode": recall.mode,
            "memory_notes": recall.notes,
            "library": [{"id": hit.get("id"), "title": hit.get("title")} for hit in library],
            "library_note": library_note,
            "model": self.service.settings.llm_model or "—",
            "elapsed_ms": elapsed_ms,
            "estimated_input_tokens": estimated_input,
            "estimated_output_tokens": estimate_tokens(answer),
        }

    # ------------------------------------------------------------- Extraction job
    def start_extract(self) -> tuple[dict[str, Any], bool]:
        with self._job_lock:
            if self._job["status"] == "running":
                return dict(self._job), False
            job_id = uuid.uuid4().hex
            self._job = {"id": job_id, "status": "running", "report": None, "error": None, "warnings": []}

        def run() -> None:
            try:
                from ..extract import Extractor
                from ..llm import LLMClient as _LLMClient

                report = Extractor(
                    self.service.store,
                    self.service.retriever,
                    _LLMClient(self.service.settings),
                ).run()
                with self._job_lock:
                    # A session that fails its verbatim-quote check three times
                    # is skipped, not a failed pass: the run still wrote facts
                    # and the backlog moved. Marking the whole job "error" is
                    # why every scheduled run looked broken while it worked.
                    failed_outright = bool(report.errors) and report.sessions == 0
                    self._job = {
                        "id": job_id,
                        "status": "error" if failed_outright else "done",
                        "report": report.as_dict(),
                        "error": "; ".join(report.errors) if failed_outright else None,
                        "warnings": report.errors[:20],
                    }
            except Exception as exc:  # the browser gets a recoverable job error
                with self._job_lock:
                    self._job = {
                        "id": job_id,
                        "status": "error",
                        "report": None,
                        "error": str(exc) or exc.__class__.__name__,
                    }

        threading.Thread(target=run, name="membrain-dashboard-extract", daemon=True).start()
        return self.job(), True

    def job(self) -> dict[str, Any]:
        with self._job_lock:
            return dict(self._job)
