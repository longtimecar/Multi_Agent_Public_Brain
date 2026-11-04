"""Session import pipeline: noise gate, LLM classify, dedupe, undo, fail-open."""

import json
import sqlite3
from pathlib import Path

from membrain import openviking_import as ov
from membrain import session_import as si
from membrain.llm import FakeLLMClient, LLMError
from membrain.server import MemoryService

GOOD_TURNS = [
    ("user", "帮我把配置文件里的超时改成 30 秒，并解释一下为什么默认是 10 秒。"),
    ("assistant", "已改好。默认 10 秒是为了避免长连接占用；改成 30 秒适合慢网络，但要注意挂起请求会更久才报错。"
                  "另外我把改动写进了配置说明文档，方便以后回滚时对照。"),
]
NOISE_TURNS = [("user", "hello"), ("assistant", "No response requested.")]


def _claude_file(root: Path, name: str, turns: list[tuple[str, str]]) -> Path:
    path = root / ".claude" / "projects" / "demo" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"type": role, "message": {"role": role, "content": text}}) for role, text in turns]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _setup(tmp_path, monkeypatch):
    from test_core import make_settings
    monkeypatch.setattr(si, "CALL_PAUSE", 0)
    monkeypatch.setattr(ov, "CALL_PAUSE", 0)
    agent_home = tmp_path / "agenthome"
    monkeypatch.setattr(si, "HOME", agent_home)
    return MemoryService(make_settings(tmp_path / "home")), agent_home


def test_session_import_classify_noise_dedupe_and_undo(tmp_path, monkeypatch):
    svc, agent_home = _setup(tmp_path, monkeypatch)
    _claude_file(agent_home, "good.jsonl", GOOD_TURNS)
    _claude_file(agent_home, "noise.jsonl", NOISE_TURNS)
    llm = FakeLLMClient([{"noise": False, "category": "code", "title": "调整超时配置",
                          "abstract": "超时改为 30 秒并讨论默认值取舍"}])

    preview = si.preview(svc)
    rows = {r["source"]: r for r in preview["sources"]}
    assert rows["claude-code"]["found"] == 2 and rows["claude-code"]["noise"] == 1

    report = si.apply_import(svc, llm)
    assert report["archived_new"] == 1 and report["noise_skipped"] == 1
    assert len(llm.calls) == 1  # the noise session never cost an LLM call

    items = svc.archive.browse("claude-code")["items"]
    assert len(items) == 1 and items[0]["category"] == "code" and items[0]["title"] == "调整超时配置"

    again = si.apply_import(svc, llm)
    assert again["archived_new"] == 0 and again["archive_existing"] == 1

    undone = ov.undo_batch(svc, report["batch"])
    assert undone == {"facts_retracted": 0, "archive_hidden": 1}


def test_session_import_llm_failure_fails_open(tmp_path, monkeypatch):
    svc, agent_home = _setup(tmp_path, monkeypatch)
    _claude_file(agent_home, "good.jsonl", GOOD_TURNS)
    report = si.apply_import(svc, FakeLLMClient([LLMError("boom")]))
    assert report["archived_new"] == 1 and not report["errors"]
    items = svc.archive.browse("claude-code")["items"]
    assert items and items[0]["category"] == "sessions"


def test_existing_origin_hash_dedupe_and_noise_finder(tmp_path, monkeypatch):
    from membrain.archive import ArchiveEntry, content_hash, entry_id
    svc, agent_home = _setup(tmp_path, monkeypatch)
    _claude_file(agent_home, "good.jsonl", GOOD_TURNS)
    rs = si.collect(["claude-code"])[0]
    body, _eid, _title, chash = si._prepare(rs)
    old_id = entry_id("claude-code", rs.scope, rs.origin, chash, "2000-01-01")
    svc.archive.files.write(ArchiveEntry(id=old_id, source="claude-code", category="sessions", title="old import",
                                         abstract="old", body=body, date="2000-01-01", origin=rs.origin,
                                         scope=rs.scope, batch="b_20000101000000000000", content_hash=chash))
    report = si.apply_import(svc, FakeLLMClient([]))
    assert report["archived_new"] == 0 and report["archive_existing"] == 1

    junk_body = "# 对话记录\n\n## 用户\n\nhello\n\n## 劅手\n\nNo response requested."
    junk_id = entry_id("claude-code", "junk", "junk.jsonl", content_hash(junk_body), "2000-01-01")
    svc.archive.files.write(ArchiveEntry(id=junk_id, source="claude-code", category="sessions",
                                         title="[claude-code] <command-name>/model</command-name>", abstract="hello",
                                         body=junk_body, date="2000-01-01", origin="junk.jsonl", scope="junk",
                                         batch="b_20000101000000000000", content_hash=content_hash(junk_body)))
    cands = si.find_noise_entries(svc)
    assert [c["id"] for c in cands] == [junk_id]


def test_workbuddy_and_cursor_adapters(tmp_path, monkeypatch):
    svc, agent_home = _setup(tmp_path, monkeypatch)
    proj = agent_home / ".workbuddy" / "projects" / "ws1"
    proj.mkdir(parents=True)
    recs = [
        json.dumps({"type": "session-meta", "sessionId": "s1"}),
        json.dumps({"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "<system-reminder data-role=\"x\">SECRET INJECTED</system-reminder>\n"
                                           "<user_query>帮我把部署脚本改成幂等的，重复执行不出错。</user_query>"}]}),
        json.dumps({"type": "message", "role": "assistant", "content": [
            {"type": "output_text", "text": "已改成幂等：先检测已有资源再增量补齐，重复执行不会重复创建，并加了 dry-run 开关方便验证。"}]}),
        json.dumps({"type": "function_call", "name": "x"}),
    ]
    (proj / "s1.jsonl").write_text("\n".join(recs), encoding="utf-8")

    db = agent_home / "AppData" / "Roaming" / "Cursor" / "User" / "globalStorage" / "conversation-search.db"
    db.parent.mkdir(parents=True)
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE conversations (fts_rowid INTEGER, source TEXT, scope TEXT, id TEXT, title TEXT,"
                " branches TEXT, updated_at INTEGER, is_archived INTEGER, root_fingerprint TEXT, cache_fingerprint TEXT)")
    con.execute("CREATE TABLE conversation_fts_content (id INTEGER, c0 TEXT, c1 TEXT, c2 TEXT)")
    con.execute("INSERT INTO conversations VALUES (1, 'local', '', 'conv-1', 'cursor 会话', '', 1793603882778, 0, 'x', NULL)")
    cursor_body = ('这是 Cursor 里一场完整对话的正文内容，讨论了如何给项目加 CI 流水线，并决定用 GitHub Actions 跑单测和 lint。'
                   '随后又细化了缓存策略：依赖缓存按 lock 哈希失效，构建产物保留 7 天，失败时自动回退到上次绿色构建，保证主干始终可发布。')
    con.execute("INSERT INTO conversation_fts_content VALUES (1, 'cursor 会话', ?, '')", (cursor_body,))
    con.commit()
    con.close()

    sessions = si.collect()
    by = {s.source: s for s in sessions}
    assert set(by) == {"workbuddy", "cursor"}
    assert all("SECRET INJECTED" not in t for _, t in by["workbuddy"].turns)
    assert any("幂等" in t for _, t in by["workbuddy"].turns)
    assert "CI 流水线" in (by["cursor"].note_body or "")

    report = si.apply_import(svc, FakeLLMClient([
        {"noise": False, "category": "code", "title": "幂等部署脚本", "abstract": "部署脚本改幂等"},
        {"noise": False, "category": "code", "title": "CI 流水线", "abstract": "给项目加 CI"},
    ]))
    assert report["archived_new"] == 2 and report["noise_skipped"] == 0
    tree = svc.archive.browse()["tree"]
    assert "workbuddy" in tree and "cursor" in tree


def test_import_session_archive_auto_increment(tmp_path, monkeypatch):
    svc, _agent_home = _setup(tmp_path, monkeypatch)
    arch = svc.settings.home / "sessions" / "sess-auto" / "history" / "archive_001"
    arch.mkdir(parents=True)
    meta = {"agent": "grok-bot-cli", "session": "sess-auto",
            "archive_uri": "mb://user/sessions/sess-auto/history/archive_001"}
    (arch / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    msgs = "\n".join([
        json.dumps({"role": "user", "ts": "2026-11-03T08:00:00+00:00",
                    "text": "<system-reminder data-role=\"x\">INJECTED SECRET</system-reminder>"
                            "<user_query>部署脚本要幂等，重复执行不出错。</user_query>"}),
        json.dumps({"role": "assistant", "ts": "2026-11-03T08:01:00+00:00",
                    "text": "已实现幂等部署：先检测后增量，重复执行零副作用，并附了回滚步骤。"}),
    ])
    (arch / "messages.jsonl").write_text(msgs, encoding="utf-8")

    llm = FakeLLMClient([{"noise": False, "category": "code", "title": "幂等部署脚本", "abstract": "部署改幂等"}])
    report = si.import_session_archive(svc, llm, arch)
    assert report["archived_new"] == 1 and not report["errors"]
    items = svc.archive.browse("grok-bot-cli")["items"]
    assert items and items[0]["title"] == "幂等部署脚本"
    assert all("INJECTED SECRET" not in (i.get("title") or "") for i in items)

    again = si.import_session_archive(svc, FakeLLMClient([]), arch)
    assert again["archive_existing"] == 1

    noise_arch = svc.settings.home / "sessions" / "sess-noise" / "history" / "archive_001"
    noise_arch.mkdir(parents=True)
    (noise_arch / "meta.json").write_text(json.dumps({"agent": "grok-bot-cli", "session": "sess-noise",
                                                      "archive_uri": "mb://user/sessions/sess-noise/x"}), encoding="utf-8")
    (noise_arch / "messages.jsonl").write_text(
        json.dumps({"role": "user", "ts": "2026-11-03T08:10:00+00:00", "text": "hello"}), encoding="utf-8")
    r2 = si.import_session_archive(svc, FakeLLMClient([]), noise_arch)
    assert r2["noise_skipped"] == 1 and len(llm.calls) == 1  # noise never cost an LLM call


def test_session_import_opencode_adapter(tmp_path, monkeypatch):
    svc, agent_home = _setup(tmp_path, monkeypatch)
    db = agent_home / ".local" / "share" / "opencode" / "opencode.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE session_v2 (id TEXT, time_created INTEGER)")
    con.execute("CREATE TABLE session_message (session_id TEXT, type TEXT, seq INTEGER, data TEXT)")
    con.execute("INSERT INTO session_v2 VALUES ('ses_1', 1792641318217)")
    con.execute("INSERT INTO session_message VALUES ('ses_1', 'user', 1, ?)",
                (json.dumps({"text": "帮我重构这个函数，把重复的校验抽成公共函数。"}),))
    con.execute("INSERT INTO session_message VALUES ('ses_1', 'assistant', 2, ?)",
                (json.dumps({"content": [{"type": "text",
                                          "text": "已经抽取成 validate_input 公共函数，两处调用点都改完了，并补了单测覆盖空入参。"}]}),))
    con.commit()
    con.close()
    sessions = si.collect()
    assert [s.source for s in sessions] == ["opencode"]
    assert sessions[0].scope == "ses_1" and len(sessions[0].turns) == 2
    report = si.apply_import(svc, FakeLLMClient([{"noise": False, "category": "code",
                                                  "title": "重构校验函数", "abstract": "重复校验抽成公共函数"}]))
    assert report["archived_new"] == 1
    assert svc.archive.browse("opencode")["items"][0]["title"] == "重构校验函数"


def test_selective_import_and_persistent_skips(tmp_path, monkeypatch):
    svc, agent_home = _setup(tmp_path, monkeypatch)
    _claude_file(agent_home, "keep.jsonl", GOOD_TURNS)
    _claude_file(agent_home, "drop.jsonl", [("user", "换个话题，帮我规划周末的行程安排并给出三个备选方案。"),
                                            ("assistant", "好的，以下是三个备选方案……（内容足够长不是噪声）" + "详"*20)])
    preview = si.preview(svc)
    sessions = [s for s in preview["sessions"] if s["status"] == "new"]
    assert len(sessions) == 2
    keep_key = next(s["key"] for s in sessions if "超时" in s["title"] or s["turns"] == 2)
    drop_key = next(s["key"] for s in sessions if s["key"] != keep_key)

    llm = FakeLLMClient([{"noise": False, "category": "life", "title": "周末行程", "abstract": "周末行程三个方案"},
                         {"noise": False, "category": "code", "title": "调整超时配置", "abstract": "超时改为 30 秒"}])
    # 只导入勾选的一条
    report = si.apply_import(svc, llm, only=[keep_key])
    assert report["archived_new"] == 1 and report["not_selected"] == 1

    # 忽略另一条：持久化后 preview 不再给 new
    si.update_skips(svc, add=[drop_key])
    preview2 = si.preview(svc)
    statuses = {s["key"]: s["status"] for s in preview2["sessions"]}
    assert statuses[drop_key] == "skipped"
    assert preview2["new"] == 0 and preview2["skipped"] == 1

    # 显式选中被忽略的会话 = 恢复并导入
    llm2 = FakeLLMClient([{"noise": False, "category": "life", "title": "周末行程", "abstract": "周末行程三个方案"}])
    report2 = si.apply_import(svc, llm2, only=[drop_key])
    assert report2["archived_new"] == 1
    assert drop_key not in si.load_skips(svc)


def test_clear_skips_restores_everything(tmp_path, monkeypatch):
    svc, agent_home = _setup(tmp_path, monkeypatch)
    _claude_file(agent_home, "keep.jsonl", GOOD_TURNS)
    preview = si.preview(svc)
    key = preview["sessions"][0]["key"]
    si.update_skips(svc, add=[key])
    assert si.load_skips(svc) == {key}
    si.update_skips(svc, clear=True)
    assert si.load_skips(svc) == set()


def test_selective_import_and_persistent_skips(tmp_path, monkeypatch):
    svc, agent_home = _setup(tmp_path, monkeypatch)
    _claude_file(agent_home, "keep.jsonl", GOOD_TURNS)
    _claude_file(agent_home, "drop.jsonl", [("user", "换个话题，帮我规划周末的行程安排并给出三个备选方案。"),
                                            ("assistant", "好的，以下是三个备选方案……（内容足够长不是噪声）" + "详"*20)])
    preview = si.preview(svc)
    sessions = [s for s in preview["sessions"] if s["status"] == "new"]
    assert len(sessions) == 2
    keep_key = next(s["key"] for s in sessions if "超时" in s["title"] or s["turns"] == 2)
    drop_key = next(s["key"] for s in sessions if s["key"] != keep_key)

    llm = FakeLLMClient([{"noise": False, "category": "life", "title": "周末行程", "abstract": "周末行程三个方案"},
                         {"noise": False, "category": "code", "title": "调整超时配置", "abstract": "超时改为 30 秒"}])
    # 只导入勾选的一条
    report = si.apply_import(svc, llm, only=[keep_key])
    assert report["archived_new"] == 1 and report["not_selected"] == 1

    # 忽略另一条：持久化后 preview 不再给 new
    si.update_skips(svc, add=[drop_key])
    preview2 = si.preview(svc)
    statuses = {s["key"]: s["status"] for s in preview2["sessions"]}
    assert statuses[drop_key] == "skipped"
    assert preview2["new"] == 0 and preview2["skipped"] == 1

    # 显式选中被忽略的会话 = 恢复并导入
    llm2 = FakeLLMClient([{"noise": False, "category": "life", "title": "周末行程", "abstract": "周末行程三个方案"}])
    report2 = si.apply_import(svc, llm2, only=[drop_key])
    assert report2["archived_new"] == 1
    assert drop_key not in si.load_skips(svc)


def test_clear_skips_restores_everything(tmp_path, monkeypatch):
    svc, agent_home = _setup(tmp_path, monkeypatch)
    _claude_file(agent_home, "keep.jsonl", GOOD_TURNS)
    preview = si.preview(svc)
    key = preview["sessions"][0]["key"]
    si.update_skips(svc, add=[key])
    assert si.load_skips(svc) == {key}
    si.update_skips(svc, clear=True)
    assert si.load_skips(svc) == set()
