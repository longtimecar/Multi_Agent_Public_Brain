"""Auto-capture watcher and dashboard chat-session listing."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import membrain.watcher as watcher
from membrain.session_import import RawSession
from membrain.store import Store


def _fake_service(tmp_path):
    home = tmp_path / "memory"
    return SimpleNamespace(settings=SimpleNamespace(home=home), store=Store(home),
                           disabled_agents=lambda: set())


def test_watch_once_records_user_turns_with_stable_uids(tmp_path, monkeypatch):
    svc = _fake_service(tmp_path)
    sessions = [
        RawSession("claude-code", "s1", "s1.jsonl", "2026-11-04",
                   [("user", "帮我写一封邮件"), ("assistant", "好的"), ("user", "记住我喜欢喝茶")]),
    ]
    monkeypatch.setattr(watcher.session_import, "collect", lambda sources=None: sessions)
    first = watcher.watch_once(svc)
    assert first["captured"] == 2
    assert len(list(svc.store.iter_episodes())) == 2
    # second pass is idempotent: stable uids dedupe
    second = watcher.watch_once(svc)
    assert second["captured"] == 0
    assert len(list(svc.store.iter_episodes())) == 2


def test_watch_once_skips_agent_turns_and_empty(tmp_path, monkeypatch):
    svc = _fake_service(tmp_path)
    sessions = [
        RawSession("codex", "s2", "s2.jsonl", "2026-11-04", [("assistant", "只有助手")]),
    ]
    monkeypatch.setattr(watcher.session_import, "collect", lambda sources=None: sessions)
    stats = watcher.watch_once(svc)
    assert stats["captured"] == 0
    assert list(svc.store.iter_episodes()) == []


def test_watcher_status_and_on_capture(tmp_path, monkeypatch):
    svc = _fake_service(tmp_path)
    fired = []
    w = watcher.Watcher(svc, on_capture=lambda: fired.append(1))
    monkeypatch.setattr(watcher.session_import, "collect", lambda sources=None: [
        RawSession("workbuddy", "s3", "s3.jsonl", "2026-11-04", [("user", "你好，帮我查个资料")]),
    ])
    w.run_once()
    status = w.status()
    assert status["runs"] == 1
    assert status["total_captured"] == 1
    assert fired == [1]
    assert status["last_run"]["captured"] == 1


def test_chat_sessions_lists_dashboard_sessions(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMBRAIN_HOME", str(tmp_path / "memory"))
    from membrain.dashboard import DashboardRuntime
    from membrain.models import Episode
    from membrain.server import MemoryService

    service = MemoryService()
    service.store.append_episodes([
        Episode(id="ep_a", ts=datetime(2026, 11, 4, 10, 0, tzinfo=timezone.utc), session="chat-1",
                agent="membrain-dashboard", role="user", text="第一条问题"),
        Episode(id="ep_b", ts=datetime(2026, 11, 4, 10, 1, tzinfo=timezone.utc), session="chat-1",
                agent="membrain-dashboard", role="assistant", text="回答"),
        Episode(id="ep_c", ts=datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc), session="chat-2",
                agent="membrain-dashboard", role="user", text="第二条问题"),
        Episode(id="ep_d", ts=datetime(2026, 11, 4, 9, 0, tzinfo=timezone.utc), session="chat-0",
                agent="claude-code", role="user", text="不是后台聊天"),
    ])
    runtime = DashboardRuntime(service=service, auto_extract_chat=False)
    try:
        rows = runtime.chat_sessions()
    finally:
        runtime.watcher.stop()
    assert [r["session_id"] for r in rows] == ["chat-2", "chat-1"]
    assert rows[0]["title"] == "第二条问题"
    assert rows[1]["turns"] == 2


def test_extract_skips_one_off_task_specs():
    from membrain.extract import _one_off_task_spec

    assert _one_off_task_spec("写一封请同事反馈初稿的邮件，交付可直接复制的邮件：主题行，第一段说明事项在走到哪一步")
    assert not _one_off_task_spec("我不吃辣")
    assert not _one_off_task_spec("写周报模板，以后每次都用这个格式")  # durable wording rescues
