from __future__ import annotations

import json
import os
import threading
import time
from datetime import date, datetime, timedelta, timezone
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from membrain.config import Settings, load_dotenv
from membrain.dashboard import DashboardRuntime, create_server
from membrain.llm import FakeLLMClient
from membrain.models import Episode, Evidence, FactType
from membrain.server import MemoryService


@pytest.fixture
def dashboard(tmp_path):
    settings = Settings(
        home=tmp_path / "memory",
        embed_base_url=None,
        embed_api_key=None,
        embed_model="text-embedding-3-small",
        embed_timeout=1,
        turn_token_budget=1500,
        session_token_budget=2500,
    )
    service = MemoryService(settings)
    runtime = DashboardRuntime(service)
    server = create_server("127.0.0.1", 0, runtime)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def request(path: str, payload: dict | None = None, headers: dict | None = None):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request_headers = {"Content-Type": "application/json", **(headers or {})}
        request_obj = Request(base + path, data=data, headers=request_headers)
        try:
            with urlopen(request_obj, timeout=3) as response:
                content = response.read()
                return response.status, response.headers, content
        except HTTPError as exc:
            return exc.code, exc.headers, exc.read()

    yield request, service

    server.shutdown()
    server.server_close()
    thread.join(timeout=2)
    service.index.close()


def test_dashboard_serves_real_overview_and_static_app(dashboard):
    request, service = dashboard
    status, headers, html = request("/")
    assert status == 200
    assert b"membrain" in html and b"memory-list" in html
    assert headers["Cache-Control"] == "no-store"
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]

    status, _, raw = request("/api/overview")
    assert status == 200
    overview = json.loads(raw)
    assert overview["stats"]["total"] == 0
    assert overview["stats"]["episodes"] == 0

    status, _, raw = request("/api/import/preview")
    assert status == 200
    preview = json.loads(raw)
    assert preview["total"] == 0 and preview["configured"] is False
    assert preview["root"] == ""
    assert preview["source_labels"]["openviking"] == "OpenViking"
    assert preview["source_labels"]["unknown-agent"] == "未知来源"

    status, _, raw = request("/api/import/start", {})
    assert status == 400
    assert "未配置" in json.loads(raw)["error"]

    sessions = service.settings.home / "sessions"
    assert not sessions.exists()
    status, _, raw = request("/api/session-diffs")
    assert status == 200 and json.loads(raw)["items"] == []
    assert not sessions.exists()  # a read-only listing must not create a directory


def test_dashboard_active_count_includes_contested_memories(dashboard):
    request, _ = dashboard
    status, _, raw = request("/api/facts", {
        "type": "fact", "subject": "counts", "claim": "Counts stay live while contested.",
        "quote": "I am not sure whether this still holds.",
    })
    assert status == 201
    fact_id = json.loads(raw)["fact"]["id"]
    assert request(f"/api/facts/{fact_id}/contest", {"note": "Needs review"})[0] == 200
    overview = json.loads(request("/api/overview")[2])
    assert overview["stats"]["total"] == 1
    assert overview["stats"]["active"] == 1
    assert overview["stats"]["contested"] == 1


def test_dashboard_does_not_silently_overlap_import_jobs(tmp_path):
    from test_core import make_settings

    service = MemoryService(make_settings(tmp_path))
    runtime = DashboardRuntime(service, llm_client=FakeLLMClient([]), auto_extract_chat=False)
    try:
        runtime._import_job = {"status": "running", "kind": "sessions"}
        with pytest.raises(ValueError, match="Agent 会话导入"):
            runtime.start_import()
        runtime._import_job = {"status": "running", "kind": "openviking"}
        with pytest.raises(ValueError, match="OpenViking"):
            runtime.start_session_import()
    finally:
        service.index.close()


def test_dashboard_add_correct_and_retract_keep_original_fact_immutable(dashboard):
    request, service = dashboard
    status, _, raw = request("/api/facts", {
        "type": "preference",
        "subject": "food",
        "claim": "The user avoids spicy food.",
        "quote": "我平时不吃辣。",
    })
    assert status == 201
    created = json.loads(raw)["fact"]
    original_path = service.store.fact_path(created["id"])
    original_contents = original_path.read_text(encoding="utf-8")
    assert created["evidence"][0]["quote"] == "我平时不吃辣。"
    assert created["source"]["agent"] == "membrain-dashboard"

    status, _, raw = request(f"/api/facts/{created['id']}/correct", {
        "claim": "The user now eats spicy food.",
        "quote": "更正：我现在可以吃辣了。",
    })
    assert status == 201
    replacement = json.loads(raw)["fact"]
    assert replacement["evidence"][0]["quote"] == "更正：我现在可以吃辣了。"
    assert service.store.get_fact(created["id"]).status.value == "superseded"
    assert original_path.read_text(encoding="utf-8") == original_contents

    status, _, raw = request(f"/api/facts/{replacement['id']}/retract", {"reason": "用户撤回这条测试记忆"})
    assert status == 200
    assert json.loads(raw)["fact"]["status"] == "retracted"
    assert original_path.read_text(encoding="utf-8") == original_contents


def test_dashboard_library_import_preview_start_browse_read_and_undo(tmp_path, monkeypatch):
    import time
    from membrain import openviking_import as ov
    from test_archive_import import good_ops, make_tree
    from test_core import make_settings

    monkeypatch.setattr(ov, "CALL_PAUSE", 0)
    service = MemoryService(make_settings(tmp_path))
    runtime = DashboardRuntime(service, FakeLLMClient([good_ops()]))
    runtime.import_root = make_tree(tmp_path)
    server = create_server("127.0.0.1", 0, runtime)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def call(path, payload=None):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        try:
            with urlopen(Request(base + path, data=data, headers={"Content-Type": "application/json"}), timeout=5) as r:
                return r.status, json.loads(r.read())
        except HTTPError as exc:
            return exc.code, json.loads(exc.read())

    try:
        status, preview = call("/api/import/preview")
        assert status == 200 and preview["total"] == 6 and preview["labels"]["events"]
        assert not (service.settings.home / "archive").exists()  # a preview writes nothing
        assert call("/api/import/start", {})[0] == 202
        for _ in range(100):
            job = call("/api/import/job")[1]
            if job["status"] != "running":
                break
            time.sleep(0.1)
        assert job["status"] == "done" and job["report"]["facts_added"] == 2 and job["report"]["archived_new"] == 6
        tree = call("/api/archive")[1]["tree"]
        assert tree["openviking"]["events"] == 1 and tree["openviking"]["preferences"] == 1
        found = call("/api/archive?q=" + "%E7%94%B5%E6%B1%A0")[1]["items"]  # 电池
        assert found and found[0]["category"] == "events"
        entry = call("/api/archive/" + found[0]["id"])[1]
        assert "电池" in entry["text"] and entry["total_chars"] > 0
        assert call("/api/archive/a_20260101_zzzzzz")[0] == 404
        batch = call("/api/import/batches")[1]["batches"][0]
        assert batch["facts"] == 2 and not batch["undone"]
        assert call("/api/import/undo", {"batch": batch["batch"]})[1] == {"facts_retracted": 2, "archive_hidden": 6}
        assert call("/api/import/undo", {"batch": batch["batch"]})[0] == 400
        assert call("/api/archive?q=" + "%E7%94%B5%E6%B1%A0")[1]["items"] == []
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        service.index.close()


def test_dashboard_rejects_non_loopback_bind(dashboard):
    _, service = dashboard
    with pytest.raises(ValueError, match="本机回环地址"):
        create_server("0.0.0.0", 0, DashboardRuntime(service))


def test_dashboard_rejects_cross_site_memory_writes(dashboard):
    request, service = dashboard
    status, _, raw = request("/api/facts", {}, {"Origin": "http://outside.example"})
    assert status == 403
    assert json.loads(raw)["error"]
    assert service.store.all_facts() == []


def test_dashboard_chat_and_model_status_keep_assistant_out_of_memory_evidence(tmp_path):
    settings = Settings(
        home=tmp_path / "memory",
        embed_base_url=None,
        embed_api_key=None,
        embed_model="text-embedding-3-small",
        embed_timeout=1,
        turn_token_budget=1500,
        session_token_budget=2500,
        llm_base_url="https://user:password@example.test/v1?token=private",
        llm_model="fake-chat",
    )
    service = MemoryService(settings)
    quote = "I am building the Membrain shared personal memory project."
    seed_episode = service.store.append_episodes([Episode(
        id="", ts=datetime.now(timezone.utc), session="seed-session", agent="test-agent",
        role="user", text=quote, project=None,
    )])[0]
    fact = service.store.draft_fact(
        type=FactType.PROJECT,
        subject="membrain",
        claim="The user is building the Membrain shared personal memory project.",
        evidence=[Evidence(episode=seed_episode.id, quote=quote, date=seed_episode.ts.date(), origin="stated")],
        source_agent="test-agent",
        source_session="seed-session",
    )
    stored_fact = service.store.write_fact(fact)
    service.retriever.sync_fact(stored_fact)
    fake = FakeLLMClient(["连接正常", {"action": "reply", "args": {"text": "你好，这里是个人 Agent。"}}])
    runtime = DashboardRuntime(service, llm_client=fake, auto_extract_chat=False)
    server = create_server("127.0.0.1", 0, runtime)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def request(path: str, payload: dict | None = None):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request_obj = Request(base + path, data=data, headers={"Content-Type": "application/json"})
        with urlopen(request_obj, timeout=3) as response:
            return response.status, json.loads(response.read())

    try:
        status, model = request("/api/model")
        assert status == 200 and model["configured"] and model["model"] == "fake-chat"
        assert "password" not in model["base_url"] and "private" not in model["base_url"]
        status, test_result = request("/api/model/test", {})
        assert status == 200 and test_result["ok"]
        status, result = request("/api/chat", {
            "session_id": "dashboard-test-session",
            "message": "Tell me about the Membrain shared personal memory project.",
        })
        assert status == 200 and result["answer"] == "你好，这里是个人 Agent。"
        assert result["session_id"] == "dashboard-test-session" and result["episode_id"]
        assert [item["id"] for item in result["memory"]] == [stored_fact.id]
        assert "The user is building the Membrain" in fake.calls[-1][1]
        episodes = [episode for episode in service.store.iter_episodes()
                    if episode.session == result["session_id"]]
        assert [episode.role for episode in episodes] == ["user", "assistant"]
        status, history = request(f"/api/chat/{result['session_id']}")
        assert status == 200 and len(history["messages"]) == 2
        assert history["messages"][1]["role"] == "assistant"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        service.index.close()


def test_dashboard_chat_corrects_a_memory_only_from_matching_user_evidence(tmp_path):
    from test_core import make_settings, write

    service = MemoryService(make_settings(tmp_path))
    fact = write(service.store, "Prefers tea", "drink", FactType.PREFERENCE, quote="I prefer tea.")
    service.retriever.sync_fact(fact)
    fake = FakeLLMClient([
        {"action": "correct", "args": {"memory_id": fact.id, "replacement_claim": "Prefers coffee"}},
        {"action": "reply", "args": {"text": "已更正：现在更喜欢咖啡。"}},
    ])
    runtime = DashboardRuntime(service, llm_client=fake, auto_extract_chat=False)
    try:
        result = runtime.chat(
            "Correct this memory: I no longer drink tea; I prefer coffee.",
            session_id="chat-correction",
        )
        assert result["memory_action"]["action"] == "correct"
        assert result["memory_action"]["old_claim"] == "Prefers tea"
        assert result["answer"] == "已更正：现在更喜欢咖啡。"
        assert service.store.get_fact(fact.id).status.value == "superseded"
        replacement = service.store.get_fact(result["memory_action"]["replacement_id"])
        assert replacement.claim == "Prefers coffee" and replacement.is_live
        assert replacement.evidence[0].episode == result["episode_id"]
        assert len(fake.calls) == 2  # one action turn + the final reply turn
    finally:
        service.index.close()


def test_dashboard_chat_retracts_a_memory_and_does_not_guess_on_ambiguity(tmp_path):
    from test_core import make_settings, write

    service = MemoryService(make_settings(tmp_path))
    fact = write(service.store, "Prefers tea", "drink", FactType.PREFERENCE, quote="I like tea.")
    service.retriever.sync_fact(fact)
    runtime = DashboardRuntime(service, llm_client=FakeLLMClient([
        {"action": "reply", "args": {"text": "目标不唯一，请给出要撤回的记忆原句。"}},
    ]), auto_extract_chat=False)
    try:
        ambiguous = runtime.chat("Correct this memory: tea.", session_id="chat-ambiguous")
        assert "目标不唯一" in ambiguous["answer"]
        assert ambiguous["memory_action"] is None
        assert service.store.get_fact(fact.id).is_live

        # the retract run reuses the same scriptable client with a fresh script
        runtime.llm_client.responses.extend([
            {"action": "retract", "args": {"memory_id": fact.id}},
            {"action": "reply", "args": {"text": "已撤回该记忆。"}},
        ])
        removed = runtime.chat("Retract this memory: I no longer like tea.", session_id="chat-retract")
        assert removed["memory_action"]["action"] == "retract"
        assert service.store.get_fact(fact.id).status.value == "retracted"
    finally:
        service.index.close()


def test_dashboard_general_preference_question_supplies_saved_memory_to_model(tmp_path):
    from test_core import make_settings, write

    service = MemoryService(make_settings(tmp_path))
    fact = write(service.store, "Likes eating chili peppers", "dietary", FactType.PREFERENCE,
                 quote="I like eating chili peppers.")
    service.retriever.sync_fact(fact)
    fake = FakeLLMClient([{"action": "reply", "args": {"text": "You have said that you like chili peppers."}}])
    runtime = DashboardRuntime(service, llm_client=fake, auto_extract_chat=False)
    try:
        result = runtime.chat("查询一下我的记忆，我平时爱干嘛", session_id="test-overview")
        assert [item["id"] for item in result["memory"]] == [fact.id]
        assert "Likes eating chili peppers" in fake.calls[-1][1]
        assert result["memory_mode"] == "overview"
    finally:
        service.index.close()


def test_dashboard_saves_model_config_without_returning_the_key(tmp_path):
    env_file = tmp_path / ".env"
    settings = Settings(
        home=tmp_path / "memory",
        embed_base_url=None,
        embed_api_key=None,
        embed_model="text-embedding-3-small",
        embed_timeout=1,
        turn_token_budget=1500,
        session_token_budget=2500,
        env_file=env_file,
    )
    service = MemoryService(settings)
    runtime = DashboardRuntime(service, llm_client=FakeLLMClient([]), auto_extract_chat=False)
    server = create_server("127.0.0.1", 0, runtime)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    try:
        request_obj = Request(
            base + "/api/model/config",
            data=json.dumps({
                "format": "openai",
                "base_url": "https://api.example.test/v1",
                "model": "private-model",
                "api_key": "test-secret-value",
                "timeout": 45,
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request_obj, timeout=3) as response:
            payload = json.loads(response.read())
        assert payload["configured"] and payload["api_key_set"]
        assert "test-secret-value" not in json.dumps(payload)
        assert runtime.service.settings.llm_model == "private-model"
        values: dict[str, str] = {}
        assert load_dotenv(env_file, values)
        assert values["MEMBRAIN_LLM_API_KEY"] == "test-secret-value"
        assert values["MEMBRAIN_LLM_TIMEOUT"] == "45.0"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        service.index.close()


def test_dashboard_admin_records_a_memory_only_from_verbatim_user_words(tmp_path):
    from test_core import make_settings, write

    service = MemoryService(make_settings(tmp_path))
    fact = write(service.store, "Prefers tea", "drink", FactType.PREFERENCE, quote="I prefer tea.")
    service.retriever.sync_fact(fact)
    runtime = DashboardRuntime(service, llm_client=FakeLLMClient([
        {"action": "record", "args": {"type": "preference", "subject": "drink",
                                       "claim": "Prefers coffee now", "quote": "I actually prefer coffee now."}},
        {"action": "search", "args": {"query": "coffee", "top_k": 3}},
        {"action": "reply", "args": {"text": "已记录并检索确认。"}},
    ]), auto_extract_chat=False)
    try:
        message = "我改主意了，I actually prefer coffee now. 不是茶了。"
        result = runtime.chat(message, session_id="admin-record")
        assert result["memory_action"]["action"] == "record"
        assert result["memory_action"]["claim"] == "Prefers coffee now"
        stored = service.store.get_fact(result["memory_action"]["memory_id"])
        assert stored.is_live and stored.evidence[0].quote == "I actually prefer coffee now."
        assert result["actions"][0]["ok"] is True
        # the fabricated quote was rejected by the verbatim check on the first attempt? No:
        # the first attempt quoted real words; now verify a fabricated one is rejected.
        runtime.llm_client.responses.extend([
            {"action": "record", "args": {"type": "fact", "subject": "food",
                                           "claim": "Hates spicy food", "quote": "I hate spicy food."}},
            {"action": "reply", "args": {"text": "没有记录，因为你没有说过这句话。"}},
        ])
        rejected = runtime.chat("我最近饮食正常。", session_id="admin-record-fake")
        assert rejected["actions"][0]["ok"] is False
        assert "逐字" in rejected["actions"][0]["result"]["error"]
        assert rejected["memory_action"] is None
    finally:
        service.index.close()


def test_dashboard_admin_reports_empty_store_and_replies(tmp_path):
    from test_core import make_settings

    service = MemoryService(make_settings(tmp_path))
    runtime = DashboardRuntime(service, llm_client=FakeLLMClient([
        {"action": "search", "args": {"query": "preference", "top_k": 5}},
        {"action": "stats"},
        {"action": "reply", "args": {"text": "记忆库目前是空的；你正常对话后记忆会逐步积累。"}},
    ]), auto_extract_chat=False)
    try:
        result = runtime.chat("帮我管理记忆", session_id="admin-empty")
        assert result["answer"].startswith("记忆库目前是空的")
        assert result["actions"][0]["result"]["results"] == []
        assert result["actions"][1]["result"]["visible_live_memories"] == 0
        assert len(result["actions"]) == 2
    finally:
        service.index.close()


def test_dashboard_chat_starts_background_extraction(tmp_path, monkeypatch):
    settings = Settings(
        home=tmp_path / "memory",
        embed_base_url=None,
        embed_api_key=None,
        embed_model="text-embedding-3-small",
        embed_timeout=1,
        turn_token_budget=1500,
        session_token_budget=2500,
        llm_base_url="https://example.test/v1",
        llm_model="fake-chat",
    )
    service = MemoryService(settings)
    runtime = DashboardRuntime(service, llm_client=FakeLLMClient([
        {"action": "reply", "args": {"text": "好的。"}}]), auto_extract_chat=True)
    started: list[int] = []
    monkeypatch.setattr(runtime, "start_extract", lambda: started.append(1) or ({"status": "running"}, True))
    runtime.chat("我正在做一个跨 Agent 个人记忆项目。", session_id="extract-chat")
    assert started == [1]
    assert [episode.role for episode in service.store.iter_episodes()] == ["user", "assistant"]
    service.index.close()


def test_agent_isolation_hides_recall_and_list_until_reconnected(tmp_path):
    from test_core import make_settings

    service = MemoryService(make_settings(tmp_path))

    def write_from(agent, claim):
        fact = service.store.draft_fact(
            type=FactType.PREFERENCE, subject="drink", claim=claim,
            evidence=[Evidence(episode=f"ep_{agent}_{claim[:6]}", quote=claim,
                               date=datetime.now(timezone.utc).date(), origin="stated")],
            source_agent=agent, source_session=f"{agent}-session",
        )
        stored = service.store.write_fact(fact)
        service.retriever.sync_fact(stored)
        return stored

    cursor_fact = write_from("cursor", "Cursor user drinks tea")
    claude_fact = write_from("claude-code", "Claude user drinks coffee")

    assert service.disabled_agents() == set()
    hits = service.retriever.recall("drinks").hits
    assert {fact.id for fact in hits} >= {cursor_fact.id, claude_fact.id}

    service.disable_agent("cursor")
    assert service.disabled_agents() == {"cursor"}
    assert (service.settings.home / "state" / "disabled-agents.json").is_file()
    assert cursor_fact.id not in {fact.id for fact in service.retriever.recall("drinks").hits}
    assert claude_fact.id in {fact.id for fact in service.retriever.recall("drinks").hits}
    # the raw store keeps the data; only recall and listing hide it
    assert service.store.get_fact(cursor_fact.id).is_live

    service.enable_agent("cursor")
    assert cursor_fact.id in {fact.id for fact in service.retriever.recall("drinks").hits}
    service.index.close()


def test_disabled_agent_captures_are_refused(tmp_path):
    from test_core import make_settings

    service = MemoryService(make_settings(tmp_path))
    service.disable_agent("codex")
    from membrain.hooks import record_prompt
    assert record_prompt(service, "codex-session", "codex", "user typed this") is False
    assert not list(service.store.iter_episodes())
    service.enable_agent("codex")
    assert record_prompt(service, "codex-session", "codex", "user typed this") is True
    assert list(service.store.iter_episodes())
    service.index.close()


def test_health_scan_reports_duplicates_noise_and_forget(tmp_path):
    from test_core import make_settings

    service = MemoryService(make_settings(tmp_path))
    runtime = DashboardRuntime(service, llm_client=FakeLLMClient([]), auto_extract_chat=False)

    def write_old(agent, claim, ftype, days):
        fact = service.store.draft_fact(
            type=ftype, subject="drink", claim=claim,
            evidence=[Evidence(episode=f"ep_{agent}_{days}", quote=quote_of(claim),
                               date=datetime.now(timezone.utc).date(), origin="stated")],
            source_agent=agent, source_session=f"{agent}-s",
        )
        # age the fact by rewriting created_at through a direct store write
        stored = service.store.write_fact(fact)
        stored = stored.model_copy(update={"created_at": datetime.now(timezone.utc) - timedelta(days=days)})
        path = service.store.fact_path(stored.id)
        path.write_text(service.store.render(stored), encoding="utf-8")
        service.retriever.sync_fact(service.store.get_fact(stored.id))
        return stored

    def quote_of(claim):
        return f"用户说过：{claim}"

    a = write_old("cursor", "Cursor drinks oolong tea", FactType.FACT, 120)
    b = write_old("workbuddy", "Cursor drinks oolong tea!", FactType.FACT, 100)
    noisy = write_old("codex", "maybe临时噪声x", FactType.FACT, 120)
    write_old("workbuddy", "Prefers oolong tea", FactType.PREFERENCE, 400)

    report = runtime.health_report()
    kinds = {finding["kind"] for finding in report["findings"]}
    assert "duplicate" in kinds, report["findings"]
    assert report["score"] < 100
    duplicate = next(f for f in report["findings"] if f["kind"] == "duplicate")
    assert a.id in duplicate["fact_ids"] and b.id in duplicate["fact_ids"]
    service.index.close()


def test_health_supersede_merges_evidence_and_retracts(tmp_path):
    from test_core import make_settings

    service = MemoryService(make_settings(tmp_path))
    runtime = DashboardRuntime(service, llm_client=FakeLLMClient([]), auto_extract_chat=False)

    def write_from(agent, claim):
        fact = service.store.draft_fact(
            type=FactType.FACT, subject="tool", claim=claim,
            evidence=[Evidence(episode=f"ep_{agent}", quote=claim,
                               date=datetime.now(timezone.utc).date(), origin="stated")],
            source_agent=agent, source_session=f"{agent}-s",
        )
        return service.store.write_fact(fact)

    a = write_from("cursor", "Uses Cursor daily")
    b = write_from("codex", "Uses Cursor every day")
    result = runtime.health_action({
        "action": "supersede", "keep_id": a.id, "drop_id": b.id,
        "fingerprint": "test-dup",
    })
    assert result["ok"] and result["action"] == "supersede"
    merged = service.store.get_fact(result["kept"])
    assert len(merged.evidence) == 2
    assert service.store.get_fact(b.id).status.value == "superseded"

    retracted = runtime.health_action({
        "action": "retract", "drop_id": a.id, "fingerprint": "test-retract",
    })
    assert retracted["ok"]
    assert service.store.get_fact(a.id).status.value == "retracted"
    service.index.close()


def test_rag_upload_accepts_bodies_over_64kb_and_lists_document(dashboard):
    import base64

    request, service = dashboard
    body = ("段落内容。" * 12000)[:90_000]  # ~90KB UTF-8 text, over the 64KB generic cap
    payload = {"filename": "big-note.txt", "data": base64.b64encode(body.encode("utf-8")).decode("ascii")}
    status, _, raw = request("/api/rag/upload", payload)
    assert status == 201, raw[:200]
    doc = json.loads(raw)
    assert doc["chunks"] >= 1

    status, _, raw = request("/api/rag/documents")
    assert status == 200
    docs = json.loads(raw)["documents"]
    assert any(d["id"] == doc["id"] for d in docs)

    status, _, raw = request("/api/rag/doc/" + doc["id"])
    assert status == 200
    detail = json.loads(raw)
    assert detail["chunk_list"] and detail["chunk_list"][0]["text"]

    status, _, raw = request("/api/rag/delete", {"id": doc["id"]})
    assert status == 200 and json.loads(raw)["ok"] is True
    _, _, raw = request("/api/rag/documents")
    assert not any(d["id"] == doc["id"] for d in json.loads(raw)["documents"])


def test_rag_upload_accepts_bodies_over_64kb_and_lists_document(dashboard):
    import base64

    request, service = dashboard
    body = ("段落内容。" * 12000)[:90_000]  # ~90KB UTF-8 text, over the 64KB generic cap
    payload = {"filename": "big-note.txt", "data": base64.b64encode(body.encode("utf-8")).decode("ascii")}
    status, _, raw = request("/api/rag/upload", payload)
    assert status == 201, raw[:200]
    doc = json.loads(raw)
    assert doc["chunks"] >= 1

    status, _, raw = request("/api/rag/documents")
    assert status == 200
    docs = json.loads(raw)["documents"]
    assert any(d["id"] == doc["id"] for d in docs)

    status, _, raw = request("/api/rag/doc/" + doc["id"])
    assert status == 200
    detail = json.loads(raw)
    assert detail["chunk_list"] and detail["chunk_list"][0]["text"]

    status, _, raw = request("/api/rag/delete", {"id": doc["id"]})
    assert status == 200 and json.loads(raw)["ok"] is True
    _, _, raw = request("/api/rag/documents")
    assert not any(d["id"] == doc["id"] for d in json.loads(raw)["documents"])


def test_admin_cleans_a_duplicate_pair_and_both_sides_retire(tmp_path):
    """Merging must leave exactly one live memory, not create a third.

    The old merge retired only the dropped side, so the kept memory stayed live
    next to the merged record they were identical: merging never reduced the
    duplicate count.
    """
    from test_core import make_settings, write

    service = MemoryService(make_settings(tmp_path))
    keep = write(service.store, "Prefers tea in the morning", "drink", FactType.PREFERENCE)
    # A different day keeps the evidence episodes distinct, the way two real
    # conversations would be.
    drop = write(service.store, "Prefers tea in the morning.", "drink", FactType.PREFERENCE, days_ago=1)
    service.retriever.sync_fact(keep)
    service.retriever.sync_fact(drop)
    runtime = DashboardRuntime(service, llm_client=FakeLLMClient([
        {"action": "health_scan", "args": {"kinds": ["duplicate"]}},
        {"action": "health_resolve", "args": {"decisions": [
            {"op": "merge", "keep_id": keep.id, "drop_id": drop.id}]}},
        {"action": "reply", "args": {"text": "已合并 1 组重复记忆。"}},
    ]), auto_extract_chat=False)
    try:
        result = runtime.chat("把重复的记忆清理一下", session_id="admin-health")
        scan = result["actions"][0]["result"]
        assert scan["findings"], scan
        assert scan["findings"][0]["facts"], "finding must carry the facts to decide on"

        merged = result["actions"][1]["result"]
        assert merged["applied"] == 1, merged

        assert not service.store.get_fact(keep.id).is_live
        assert not service.store.get_fact(drop.id).is_live
        live = service.store.live_facts()
        assert len(live) == 1
        assert len(live[0].evidence) == 2, "合并必须保留双方证据"

        report = runtime.health_report(refresh=True)
        assert not [f for f in report["findings"] if f["kind"] == "duplicate"]
    finally:
        service.index.close()


def test_admin_refuses_to_merge_a_constraint(tmp_path):
    from test_core import make_settings, write

    service = MemoryService(make_settings(tmp_path))
    hard = write(service.store, "Never shares the home address", "privacy", FactType.CONSTRAINT)
    other = write(service.store, "Never shares the home address.", "privacy", FactType.PREFERENCE)
    service.retriever.sync_fact(hard)
    service.retriever.sync_fact(other)
    runtime = DashboardRuntime(service, llm_client=FakeLLMClient([
        {"action": "health_merge", "args": {"keep_id": other.id, "drop_id": hard.id}},
        {"action": "reply", "args": {"text": "硬约束需要你手动处理。"}},
    ]), auto_extract_chat=False)
    try:
        result = runtime.chat("合并这两条", session_id="admin-health-constraint")
        assert result["actions"][0]["ok"] is False
        assert "硬约束" in result["actions"][0]["result"]["error"]
        assert service.store.get_fact(hard.id).is_live
    finally:
        service.index.close()


def test_health_report_is_cached_between_calls(tmp_path, monkeypatch):
    from test_core import make_settings, write

    service = MemoryService(make_settings(tmp_path))
    write(service.store, "Prefers tea", "drink", FactType.PREFERENCE)
    runtime = DashboardRuntime(service, auto_extract_chat=False)
    try:
        calls = {"n": 0}
        original = runtime.health_report

        def counted(refresh: bool = False):
            calls["n"] += 1
            return original(refresh=refresh)

        first = runtime.health_report()
        second = runtime.health_report()
        assert first["scanned_at"] == second["scanned_at"], "120 秒内应复用缓存"
    finally:
        service.index.close()


def test_admin_cleans_a_duplicate_pair_and_both_sides_retire(tmp_path):
    """Merging must leave exactly one live memory, not create a third.

    The old merge retired only the dropped side, so the kept memory stayed live
    next to the merged record they were identical: merging never reduced the
    duplicate count.
    """
    from test_core import make_settings, write

    service = MemoryService(make_settings(tmp_path))
    keep = write(service.store, "Prefers tea in the morning", "drink", FactType.PREFERENCE)
    # A different day keeps the evidence episodes distinct, the way two real
    # conversations would be.
    drop = write(service.store, "Prefers tea in the morning.", "drink", FactType.PREFERENCE, days_ago=1)
    service.retriever.sync_fact(keep)
    service.retriever.sync_fact(drop)
    runtime = DashboardRuntime(service, llm_client=FakeLLMClient([
        {"action": "health_scan", "args": {"kinds": ["duplicate"]}},
        {"action": "health_resolve", "args": {"decisions": [
            {"op": "merge", "keep_id": keep.id, "drop_id": drop.id}]}},
        {"action": "reply", "args": {"text": "已合并 1 组重复记忆。"}},
    ]), auto_extract_chat=False)
    try:
        result = runtime.chat("把重复的记忆清理一下", session_id="admin-health")
        scan = result["actions"][0]["result"]
        assert scan["findings"], scan
        assert scan["findings"][0]["facts"], "finding must carry the facts to decide on"

        merged = result["actions"][1]["result"]
        assert merged["applied"] == 1, merged

        assert not service.store.get_fact(keep.id).is_live
        assert not service.store.get_fact(drop.id).is_live
        live = service.store.live_facts()
        assert len(live) == 1
        assert len(live[0].evidence) == 2, "合并必须保留双方证据"

        report = runtime.health_report(refresh=True)
        assert not [f for f in report["findings"] if f["kind"] == "duplicate"]
    finally:
        service.index.close()


def test_admin_refuses_to_merge_a_constraint(tmp_path):
    from test_core import make_settings, write

    service = MemoryService(make_settings(tmp_path))
    hard = write(service.store, "Never shares the home address", "privacy", FactType.CONSTRAINT)
    other = write(service.store, "Never shares the home address.", "privacy", FactType.PREFERENCE)
    service.retriever.sync_fact(hard)
    service.retriever.sync_fact(other)
    runtime = DashboardRuntime(service, llm_client=FakeLLMClient([
        {"action": "health_merge", "args": {"keep_id": other.id, "drop_id": hard.id}},
        {"action": "reply", "args": {"text": "硬约束需要你手动处理。"}},
    ]), auto_extract_chat=False)
    try:
        result = runtime.chat("合并这两条", session_id="admin-health-constraint")
        assert result["actions"][0]["ok"] is False
        assert "硬约束" in result["actions"][0]["result"]["error"]
        assert service.store.get_fact(hard.id).is_live
    finally:
        service.index.close()


def test_health_report_is_cached_between_calls(tmp_path, monkeypatch):
    from test_core import make_settings, write

    service = MemoryService(make_settings(tmp_path))
    write(service.store, "Prefers tea", "drink", FactType.PREFERENCE)
    runtime = DashboardRuntime(service, auto_extract_chat=False)
    try:
        calls = {"n": 0}
        original = runtime.health_report

        def counted(refresh: bool = False):
            calls["n"] += 1
            return original(refresh=refresh)

        first = runtime.health_report()
        second = runtime.health_report()
        assert first["scanned_at"] == second["scanned_at"], "120 秒内应复用缓存"
    finally:
        service.index.close()


def test_merge_collapses_evidence_that_cites_the_same_turn(tmp_path):
    """One turn may cite a memory only once, so merging must not repeat it."""
    from test_core import make_settings, write

    service = MemoryService(make_settings(tmp_path))
    a = write(service.store, "Prefers tea in the morning", "drink", FactType.PREFERENCE)
    b = write(service.store, "Prefers tea in the morning.", "drink", FactType.PREFERENCE)
    service.retriever.sync_fact(a)
    service.retriever.sync_fact(b)
    runtime = DashboardRuntime(service, auto_extract_chat=False)
    try:
        merged = runtime.health_action({"action": "supersede", "keep_id": a.id, "drop_id": b.id})
        assert merged["ok"] is True
        saved = service.store.get_fact(merged["kept"])
        assert len(saved.evidence) == 1, "同一回合只能被引用一次"
    finally:
        service.index.close()


def test_merge_collapses_evidence_that_cites_the_same_turn(tmp_path):
    """One turn may cite a memory only once, so merging must not repeat it."""
    from test_core import make_settings, write

    service = MemoryService(make_settings(tmp_path))
    a = write(service.store, "Prefers tea in the morning", "drink", FactType.PREFERENCE)
    b = write(service.store, "Prefers tea in the morning.", "drink", FactType.PREFERENCE)
    service.retriever.sync_fact(a)
    service.retriever.sync_fact(b)
    runtime = DashboardRuntime(service, auto_extract_chat=False)
    try:
        merged = runtime.health_action({"action": "supersede", "keep_id": a.id, "drop_id": b.id})
        assert merged["ok"] is True
        saved = service.store.get_fact(merged["kept"])
        assert len(saved.evidence) == 1, "同一回合只能被引用一次"
    finally:
        service.index.close()


def test_dashboard_fact_write_survives_index_sync_failure(dashboard, monkeypatch):
    request, service = dashboard

    def boom(_fact):
        raise RuntimeError("index is down")

    monkeypatch.setattr(service.retriever, "sync_fact", boom)
    status, _, raw = request("/api/facts", {"type": "preference", "subject": "drink",
                                            "claim": "Likes tea", "quote": "我喜欢喝茶"})
    assert status == 201, raw[:200]
    body = json.loads(raw)
    assert body["ok"] is True
    assert "index_warning" in body and "index is down" in body["index_warning"]
    assert service.store.get_fact(body["fact"]["id"]).claim == "Likes tea"


def test_merging_a_duplicate_keeps_the_earlier_write_date(tmp_path):
    """A merge continues a memory; it must not look newly created today.

    The list shows when a memory originates, and the detail shows when it was
    first written. Stamping the merge with "now" made an old statement look
    fabricated the next day.
    """
    from datetime import timedelta

    from membrain.store import _atomic_write
    from test_core import make_settings, write

    service = MemoryService(make_settings(tmp_path))
    runtime = DashboardRuntime(service, auto_extract_chat=False)
    old = write(service.store, "Prefers tea in the morning", "drink", FactType.PREFERENCE)
    new = write(service.store, "Prefers tea in the morning.", "drink", FactType.PREFERENCE, days_ago=1)
    long_ago = datetime.now(timezone.utc) - timedelta(days=2)
    rewritten = service.store.get_fact(old.id).model_copy(update={"created_at": long_ago})
    _atomic_write(service.store.fact_path(old.id), service.store.render(rewritten))
    try:
        merged = runtime.health_action({"action": "supersede", "keep_id": old.id, "drop_id": new.id})
        saved = service.store.get_fact(merged["kept"])
        assert saved.created_at == long_ago, "合并必须沿用最早的首次记录时间"
    finally:
        service.index.close()


def test_health_flags_a_command_stored_as_a_preference_and_the_agent_retracts_it(tmp_path):
    """「把重复的清理掉」被存成偏好，健康报告要点名，Agent 自己撤回。

    这是本轮真实事故：后台聊天里的一句操作指令变成了 preference。人工删不算
    修复——系统要能自己发现并按规则处理。
    """
    from test_core import make_settings

    service = MemoryService(make_settings(tmp_path))
    runtime = DashboardRuntime(service, auto_extract_chat=False)
    turn = Episode(id="e1", ts=datetime.now(timezone.utc), session="s", agent="workbuddy",
                   role="user", text="把记忆库里重复的和冲突的都清理掉，不用问我，直接处理完告诉我结果。")
    service.store.append_episodes([turn])
    fact = service.store.draft_fact(
        type=FactType.PREFERENCE, subject="memory-maintenance",
        claim="Automatically clears duplicates and conflicts without asking",
        evidence=[Evidence(episode="e1", quote="不用问我，直接处理完告诉我结果",
                           date=turn.local_date, origin="stated")],
        source_agent="workbuddy",
    )
    stored = service.store.write_fact(fact)
    service.retriever.sync_fact(stored)
    try:
        report = runtime.health_report(refresh=True)
        findings = [f for f in report["findings"] if f["kind"] == "command_misfiled"]
        assert findings and stored.id in findings[0]["fact_ids"]

        runtime.llm_client = FakeLLMClient([
            {"action": "health_scan", "args": {"kinds": ["command_misfiled"]}},
            {"action": "health_resolve", "args": {"decisions": [
                {"op": "retract", "memory_id": stored.id, "reason": "这是指令，不是偏好"}]}},
            {"action": "reply", "args": {"text": "已撤回 1 条指令误存为偏好的记忆。"}},
        ])
        result = runtime.chat("把指令误存成偏好的记忆清理掉", session_id="admin-misfiled")
        applied = result["actions"][1]["result"]
        assert applied["applied"] == 1, applied
        assert service.store.get_fact(stored.id).status.value == "retracted"
    finally:
        service.index.close()


def test_health_retract_refuses_a_memory_the_report_does_not_name(tmp_path):
    from test_core import make_settings, write

    service = MemoryService(make_settings(tmp_path))
    runtime = DashboardRuntime(service, auto_extract_chat=False)
    unrelated = write(service.store, "Prefers tea", "drink", FactType.PREFERENCE)
    try:
        ok, result = runtime._admin_execute(
            "health_retract", {"memory_id": unrelated.id}, "清理", 
            Episode(id="e1", ts=datetime.now(timezone.utc), session="s", agent="user",
                    role="user", text="清理"), "s")
        assert ok is False
        assert "健康报告" in result["error"]
        assert service.store.get_fact(unrelated.id).is_live
    finally:
        service.index.close()


def test_health_flags_a_command_stored_as_a_preference_and_the_agent_retracts_it(tmp_path):
    """「把重复的清理掉」被存成偏好，健康报告要点名，Agent 自己撤回。

    这是本轮真实事故：后台聊天里的一句操作指令变成了 preference。人工删不算
    修复——系统要能自己发现并按规则处理。
    """
    from test_core import make_settings

    service = MemoryService(make_settings(tmp_path))
    runtime = DashboardRuntime(service, auto_extract_chat=False)
    turn = Episode(id="e1", ts=datetime.now(timezone.utc), session="s", agent="workbuddy",
                   role="user", text="把记忆库里重复的和冲突的都清理掉，不用问我，直接处理完告诉我结果。")
    service.store.append_episodes([turn])
    fact = service.store.draft_fact(
        type=FactType.PREFERENCE, subject="memory-maintenance",
        claim="Automatically clears duplicates and conflicts without asking",
        evidence=[Evidence(episode="e1", quote="不用问我，直接处理完告诉我结果",
                           date=turn.local_date, origin="stated")],
        source_agent="workbuddy",
    )
    stored = service.store.write_fact(fact)
    service.retriever.sync_fact(stored)
    try:
        report = runtime.health_report(refresh=True)
        findings = [f for f in report["findings"] if f["kind"] == "command_misfiled"]
        assert findings and stored.id in findings[0]["fact_ids"]

        runtime.llm_client = FakeLLMClient([
            {"action": "health_scan", "args": {"kinds": ["command_misfiled"]}},
            {"action": "health_resolve", "args": {"decisions": [
                {"op": "retract", "memory_id": stored.id, "reason": "这是指令，不是偏好"}]}},
            {"action": "reply", "args": {"text": "已撤回 1 条指令误存为偏好的记忆。"}},
        ])
        result = runtime.chat("把指令误存成偏好的记忆清理掉", session_id="admin-misfiled")
        applied = result["actions"][1]["result"]
        assert applied["applied"] == 1, applied
        assert service.store.get_fact(stored.id).status.value == "retracted"
    finally:
        service.index.close()


def test_health_retract_refuses_a_memory_the_report_does_not_name(tmp_path):
    from test_core import make_settings, write

    service = MemoryService(make_settings(tmp_path))
    runtime = DashboardRuntime(service, auto_extract_chat=False)
    unrelated = write(service.store, "Prefers tea", "drink", FactType.PREFERENCE)
    try:
        ok, result = runtime._admin_execute(
            "health_retract", {"memory_id": unrelated.id}, "清理", 
            Episode(id="e1", ts=datetime.now(timezone.utc), session="s", agent="user",
                    role="user", text="清理"), "s")
        assert ok is False
        assert "健康报告" in result["error"]
        assert service.store.get_fact(unrelated.id).is_live
    finally:
        service.index.close()


def test_health_clean_applies_the_safe_policy_in_one_call(tmp_path):
    """一条动作完成：撤回命令误存、合并重复，约束类一律跳过。"""
    from test_core import make_settings, write

    service = MemoryService(make_settings(tmp_path))
    runtime = DashboardRuntime(service, auto_extract_chat=False)
    turn = Episode(id="e1", ts=datetime.now(timezone.utc), session="s", agent="workbuddy",
                   role="user", text="把记忆库里重复的都清理掉，不用问我。")
    service.store.append_episodes([turn])
    misfiled = service.store.write_fact(service.store.draft_fact(
        type=FactType.PREFERENCE, subject="memory",
        claim="Automatically clears duplicates",
        evidence=[Evidence(episode="e1", quote="把记忆库里重复的都清理掉", date=turn.local_date,
                           origin="stated")], source_agent="workbuddy"))
    a = write(service.store, "Prefers tea in the morning", "drink", FactType.PREFERENCE)
    b = write(service.store, "Prefers tea in the morning.", "drink", FactType.PREFERENCE, days_ago=1)
    try:
        ok, result = runtime._admin_execute(
            "health_clean", {"kinds": ["command_misfiled", "duplicate"]}, "清理", turn, "s")
        assert ok is True, result
        assert result["retracted"] == 1, result
        assert result["merged"] >= 1, result
        assert service.store.get_fact(misfiled.id).status.value == "retracted"
        assert service.store.get_fact(a.id).status.value != "active"
        live = [f for f in service.store.live_facts() if f.subject == "drink"]
        assert len(live) == 1, "重复合并后只应剩一条"
    finally:
        service.index.close()


def test_health_clean_applies_the_safe_policy_in_one_call(tmp_path):
    """一条动作完成：撤回命令误存、合并重复，约束类一律跳过。"""
    from test_core import make_settings, write

    service = MemoryService(make_settings(tmp_path))
    runtime = DashboardRuntime(service, auto_extract_chat=False)
    turn = Episode(id="e1", ts=datetime.now(timezone.utc), session="s", agent="workbuddy",
                   role="user", text="把记忆库里重复的都清理掉，不用问我。")
    service.store.append_episodes([turn])
    misfiled = service.store.write_fact(service.store.draft_fact(
        type=FactType.PREFERENCE, subject="memory",
        claim="Automatically clears duplicates",
        evidence=[Evidence(episode="e1", quote="把记忆库里重复的都清理掉", date=turn.local_date,
                           origin="stated")], source_agent="workbuddy"))
    a = write(service.store, "Prefers tea in the morning", "drink", FactType.PREFERENCE)
    b = write(service.store, "Prefers tea in the morning.", "drink", FactType.PREFERENCE, days_ago=1)
    try:
        ok, result = runtime._admin_execute(
            "health_clean", {"kinds": ["command_misfiled", "duplicate"]}, "清理", turn, "s")
        assert ok is True, result
        assert result["retracted"] == 1, result
        assert result["merged"] >= 1, result
        assert service.store.get_fact(misfiled.id).status.value == "retracted"
        assert service.store.get_fact(a.id).status.value != "active"
        live = [f for f in service.store.live_facts() if f.subject == "drink"]
        assert len(live) == 1, "重复合并后只应剩一条"
    finally:
        service.index.close()


def test_partial_answer_reports_what_actually_happened():
    """清理成功了，最后一步模型出错，不能回答"没能完成"。"""
    trace = [
        {"action": "health_scan", "ok": True, "result": {"findings": []}},
        {"action": "health_clean", "ok": True, "result": {"retracted": 2, "merged": 5}},
    ]
    text = DashboardRuntime._partial_answer(trace)
    assert "撤回 2" in text and "合并 5" in text
    assert "没能完成" not in text


def test_partial_answer_is_honest_when_nothing_worked():
    text = DashboardRuntime._partial_answer([{"action": "health_scan", "ok": True, "result": {}}])
    assert "没能完成" in text


def test_partial_answer_reports_what_actually_happened():
    """清理成功了，最后一步模型出错，不能回答"没能完成"。"""
    trace = [
        {"action": "health_scan", "ok": True, "result": {"findings": []}},
        {"action": "health_clean", "ok": True, "result": {"retracted": 2, "merged": 5}},
    ]
    text = DashboardRuntime._partial_answer(trace)
    assert "撤回 2" in text and "合并 5" in text
    assert "没能完成" not in text


def test_partial_answer_is_honest_when_nothing_worked():
    text = DashboardRuntime._partial_answer([{"action": "health_scan", "ok": True, "result": {}}])
    assert "没能完成" in text


@pytest.mark.skipif(os.name != "nt", reason="netstat/taskkill 只在 Windows 上验证")
def test_the_launcher_can_actually_replace_a_running_dashboard():
    """点启动脚本要真的换掉旧进程。

    之前重复启动只是打开一个浏览器页签，于是"关掉再开"永远跑的还是旧代码：
    界面看起来正常，改动一条都没生效。这条测试起一个真进程、按端口找到它、
    再把端口收回来。
    """
    import socket
    import subprocess
    import sys

    from membrain.dashboard import listening_pid, stop_dashboard

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    code = (
        "import http.server, socketserver\n"
        "class H(http.server.BaseHTTPRequestHandler):\n"
        "    def do_GET(self):\n"
        "        self.send_response(200); self.send_header('Server','MembrainDashboard')\n"
        "        self.end_headers(); self.wfile.write(b'{}')\n"
        "    def log_message(self, *a): pass\n"
        "socketserver.TCPServer.allow_reuse_address = True\n"
        f"with socketserver.TCPServer(('127.0.0.1', {port}), H) as s: s.serve_forever()\n"
    )
    child = subprocess.Popen([sys.executable, "-c", code])
    try:
        for _ in range(50):
            if listening_pid(port):
                break
            time.sleep(0.1)
        # 不比对 PID：解释器可能再起一层子进程，端口归属才是事实。
        assert listening_pid(port) is not None
        assert stop_dashboard(port) is True
        assert listening_pid(port) is None
        for _ in range(50):
            if child.poll() is not None:
                break
            time.sleep(0.1)
        assert child.poll() is not None, "旧进程必须真的退出，而不只是端口改了归属"
    finally:
        if child.poll() is None:
            child.kill()


@pytest.mark.parametrize("query,expected", [
    ("我的工作偏好", FactType.PREFERENCE),
    ("我的任务", FactType.TASK),
    ("我的目标", FactType.GOAL),
    ("我的项目", FactType.PROJECT),
    ("服务器端口", None),
])
def test_a_category_phrase_is_recognised_for_the_empty_search_fallback(query, expected):
    """「我的工作偏好」搜不出单条记忆，但用户问的是一整类。

    没有这个兜底，搜索框会给出一个空页面——而「我的偏好」却能列出上百条，
    同一个意思两种结果。
    """
    from membrain.dashboard import _category_type

    assert _category_type(query) is expected


@pytest.mark.parametrize("query,expected", [
    ("我的工作偏好", FactType.PREFERENCE),
    ("我的任务", FactType.TASK),
    ("我的目标", FactType.GOAL),
    ("我的项目", FactType.PROJECT),
    ("服务器端口", None),
])
def test_a_category_phrase_is_recognised_for_the_empty_search_fallback(query, expected):
    """「我的工作偏好」搜不出单条记忆，但用户问的是一整类。

    没有这个兜底，搜索框会给出一个空页面——而「我的偏好」却能列出上百条，
    同一个意思两种结果。
    """
    from membrain.dashboard import _category_type

    assert _category_type(query) is expected


def test_recall_verify_status_answers_a_get(dashboard):
    """页面用 GET 轮询验证进度；只在 POST 里处理会让按钮五次重试后失败。"""
    request, _ = dashboard
    status, _, raw = request("/api/recall/verify/status")
    assert status == 200, raw[:160]
    assert "status" in json.loads(raw)


def test_health_endpoint_honours_the_kind_filter_over_get(dashboard):
    """带 kinds 查询时只跑那一类检测：全量扫描要嵌入所有记忆，页面上就是几十秒。"""
    request, service = dashboard
    turn = Episode(id="e-cmd", ts=datetime.now(timezone.utc), session="s", agent="workbuddy",
                   role="user", text="把记忆库里重复的都清理掉，不用问我。")
    service.store.append_episodes([turn])
    fact = service.store.write_fact(service.store.draft_fact(
        type=FactType.PREFERENCE, subject="memory",
        claim="Automatically clears duplicates",
        evidence=[Evidence(episode="e-cmd", quote="把记忆库里重复的都清理掉",
                           date=turn.local_date, origin="stated")],
        source_agent="workbuddy"))
    service.retriever.sync_fact(fact)

    status, _, raw = request("/api/health?kinds=command_misfiled")
    assert status == 200, raw[:160]
    kinds = [finding["kind"] for finding in json.loads(raw)["findings"]]
    assert kinds == ["command_misfiled"], kinds

    status, _, raw = request("/api/health?kinds=duplicate")
    assert status == 200
    assert json.loads(raw)["findings"] == [], "只要重复时不该顺带返回指令误存"


def test_recall_verify_status_answers_a_get(dashboard):
    """页面用 GET 轮询验证进度；只在 POST 里处理会让按钮五次重试后失败。"""
    request, _ = dashboard
    status, _, raw = request("/api/recall/verify/status")
    assert status == 200, raw[:160]
    assert "status" in json.loads(raw)


def test_health_endpoint_honours_the_kind_filter_over_get(dashboard):
    """带 kinds 查询时只跑那一类检测：全量扫描要嵌入所有记忆，页面上就是几十秒。"""
    request, service = dashboard
    turn = Episode(id="e-cmd", ts=datetime.now(timezone.utc), session="s", agent="workbuddy",
                   role="user", text="把记忆库里重复的都清理掉，不用问我。")
    service.store.append_episodes([turn])
    fact = service.store.write_fact(service.store.draft_fact(
        type=FactType.PREFERENCE, subject="memory",
        claim="Automatically clears duplicates",
        evidence=[Evidence(episode="e-cmd", quote="把记忆库里重复的都清理掉",
                           date=turn.local_date, origin="stated")],
        source_agent="workbuddy"))
    service.retriever.sync_fact(fact)

    status, _, raw = request("/api/health?kinds=command_misfiled")
    assert status == 200, raw[:160]
    kinds = [finding["kind"] for finding in json.loads(raw)["findings"]]
    assert kinds == ["command_misfiled"], kinds

    status, _, raw = request("/api/health?kinds=duplicate")
    assert status == 200
    assert json.loads(raw)["findings"] == [], "只要重复时不该顺带返回指令误存"


def test_rerank_config_is_saved_and_applied_without_a_restart(tmp_path):
    """重排模型要能在界面里自己填，保存即生效。

    它是和 Embedding 不同的另一个模型：Embedding 把文本变成向量做初筛，
    重排再给候选打分。两者地址和模型名都不一样，不能共用一个输入框。
    """
    from dataclasses import replace

    from test_core import make_settings

    service = MemoryService(make_settings(tmp_path))
    env_file = tmp_path / "rerank.env"
    service.settings = replace(service.settings, env_file=env_file)
    runtime = DashboardRuntime(service, auto_extract_chat=False)
    try:
        saved = runtime.configure_rerank({
            "base_url": "https://api.siliconflow.cn/v1",
            "model": "BAAI/bge-reranker-v2-m3",
            "api_key": "sk-test-key",
            "timeout": 15,
        })
        assert saved["rerank_configured"] is True
        assert saved["rerank_enabled"] is True
        assert saved["rerank_provider"] == "openai"
        assert saved["rerank_api_key_set"] is True

        written = env_file.read_text(encoding="utf-8")
        assert 'MEMBRAIN_USE_RERANK="1"' in written
        assert '"BAAI/bge-reranker-v2-m3"' in written
        # 检索器必须换成同一份设置，否则只有一半代码用新模型
        assert service.retriever.settings.rerank_model == "BAAI/bge-reranker-v2-m3"

        off = runtime.configure_rerank({"base_url": "", "model": ""})
        assert off["rerank_configured"] is False
        assert off["rerank_enabled"] is False
        assert off["rerank_provider"] == "heuristic"
    finally:
        service.index.close()


def test_rerank_config_is_saved_and_applied_without_a_restart(tmp_path):
    """重排模型要能在界面里自己填，保存即生效。

    它是和 Embedding 不同的另一个模型：Embedding 把文本变成向量做初筛，
    重排再给候选打分。两者地址和模型名都不一样，不能共用一个输入框。
    """
    from dataclasses import replace

    from test_core import make_settings

    service = MemoryService(make_settings(tmp_path))
    env_file = tmp_path / "rerank.env"
    service.settings = replace(service.settings, env_file=env_file)
    runtime = DashboardRuntime(service, auto_extract_chat=False)
    try:
        saved = runtime.configure_rerank({
            "base_url": "https://api.siliconflow.cn/v1",
            "model": "BAAI/bge-reranker-v2-m3",
            "api_key": "sk-test-key",
            "timeout": 15,
        })
        assert saved["rerank_configured"] is True
        assert saved["rerank_enabled"] is True
        assert saved["rerank_provider"] == "openai"
        assert saved["rerank_api_key_set"] is True

        written = env_file.read_text(encoding="utf-8")
        assert 'MEMBRAIN_USE_RERANK="1"' in written
        assert '"BAAI/bge-reranker-v2-m3"' in written
        # 检索器必须换成同一份设置，否则只有一半代码用新模型
        assert service.retriever.settings.rerank_model == "BAAI/bge-reranker-v2-m3"

        off = runtime.configure_rerank({"base_url": "", "model": ""})
        assert off["rerank_configured"] is False
        assert off["rerank_enabled"] is False
        assert off["rerank_provider"] == "heuristic"
    finally:
        service.index.close()
