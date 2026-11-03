from __future__ import annotations

import pytest

from membrain import openviking_import as ov
from membrain.archive import make_abstract, tokens
from membrain.llm import FakeLLMClient
from membrain.server import MemoryService
from membrain.store import StoreError
from test_core import make_settings


def make_tree(tmp_path):
    base = tmp_path / "src" / "default" / "memories"
    notes = {
        "profile.md": "# <USER>\n简介。\n<!-- MEMORY_FIELDS\n{\"memory_type\": \"profile\"}\n-->",
        "preferences/user/协作偏好.md": "- 不用 emoji，喜欢简洁纯文字\n- 周六日不安排任务提醒",
        "events/2026/10/14/EC 固件问题.md": "# Summary\nEC 向 ACPI 误报电池状态，导致无法识别电池。",
        "experiences/2026-10-25-自启动修复.md": "# 自启动修复\n问题出在服务没有随宿主退出。",
        "entities/程序/02-backend.md": "# 02-backend 报告\n模块地图和命令清单。",
        "entities/学习/dhcp snooping.md": "DHCP Snooping 让交换机只信任合法的 DHCP 服务器。",
        "identity.md": "template",
        ".abstract.md": "skip me",
        "events/empty.md": "<!-- MEMORY_FIELDS\n{}\n-->",
    }
    for rel, text in notes.items():
        path = base / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return tmp_path / "src"


def good_ops():
    return {"operations": [
        {"type": "preference", "subject": "style", "claim": "Prefers plain text without emoji",
         "quote": "不用 emoji，喜欢简洁纯文字"},
        {"type": "preference", "subject": "reminders", "claim": "No task reminders on weekends",
         "quote": "周六日不安排任务提醒"},
        {"type": "fact", "subject": "x", "claim": "Invented", "quote": "这句话不在笔记里"},
        {"type": "preference", "subject": "agent", "claim": "Agent may act without asking",
         "quote": "周六日不安排任务提醒"},
        {"type": "trait", "subject": "x", "claim": "Is shy", "quote": "不用 emoji"},
    ]}


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(ov, "CALL_PAUSE", 0)
    monkeypatch.setattr(ov, "RATE_LIMIT_WAITS", (0, 0))


@pytest.fixture
def svc(tmp_path):
    service = MemoryService(make_settings(tmp_path))
    yield service
    service.index.close()


def test_plan_classifies_by_folder_and_skips_templates(tmp_path):
    plan = ov.build_plan(make_tree(tmp_path))
    cats = {p.rel.replace("\\", "/"): p.entry.category for p in plan.entries}
    assert cats == {"profile.md": "profile", "preferences/user/协作偏好.md": "preferences",
                    "events/2026/10/14/EC 固件问题.md": "events", "experiences/2026-10-25-自启动修复.md": "experiences",
                    "entities/程序/02-backend.md": "code", "entities/学习/dhcp snooping.md": "study"}
    assert plan.skipped == {"模板或摘要文件": 2, "内容为空": 1}
    only = [p.rel.replace("\\", "/") for p in plan.entries if p.atomize]
    assert only == ["preferences/user/协作偏好.md"]
    by_rel = {p.rel.replace("\\", "/"): p.entry for p in plan.entries}
    assert by_rel["events/2026/10/14/EC 固件问题.md"].date == "2026-10-14"
    assert by_rel["experiences/2026-10-25-自启动修复.md"].date == "2026-10-25"
    assert by_rel["events/2026/10/14/EC 固件问题.md"].title == "EC 固件问题"  # generic "Summary" heading
    assert "MEMORY_FIELDS" not in by_rel["profile.md"].body


def test_preview_writes_nothing(svc, tmp_path):
    result = ov.preview(svc, make_tree(tmp_path))
    assert result["total"] == 6 and result["new"] == 6 and result["to_distill"] == 1
    assert result["configured"] and result["available"]
    assert not (svc.settings.home / "archive").exists() and not list(svc.store.facts_dir.glob("f_*.md"))


def test_import_requires_an_explicit_source_root(svc, tmp_path):
    empty = ov.preview(svc)
    assert empty["total"] == 0 and empty["configured"] is False
    assert empty["root"] == "" and empty["available"] is False

    missing = ov.preview(svc, tmp_path / "not-configured")
    assert missing["total"] == 0 and missing["configured"] is True
    assert missing["available"] is False

    with pytest.raises(StoreError, match="未配置 OpenViking"):
        ov.apply_import(svc, FakeLLMClient([]))
    assert svc.store.all_facts() == [] and not (svc.settings.home / "archive").exists()


def test_apply_archives_everything_and_distills_only_verified_facts(svc, tmp_path):
    llm = FakeLLMClient([good_ops()])
    report = ov.apply_import(svc, llm, make_tree(tmp_path))
    assert report["archived_new"] == 6 and report["facts_added"] == 2 and report["rejected"] == 3
    assert len(llm.calls) == 1  # only the preference note went to the model
    facts = svc.store.live_facts()
    assert {f.claim for f in facts} == {"Prefers plain text without emoji", "No task reminders on weekends"}
    fact = facts[0]
    assert fact.evidence[0].origin == "imported" and fact.confidence >= 0.6
    assert fact.source.agent == "openviking-import" and fact.source.session == report["batch"]
    assert svc.archive.files.exists(fact.evidence[0].episode)  # evidence points at the archived note
    results, _ = svc.archive.search("电池状态", top_k=3)
    assert results and results[0]["category"] == "events"
    assert svc.archive.browse()["tree"]["openviking"]["code"] == 1
    assert svc.archive.read(results[0]["id"])["text"].startswith("# Summary")


def test_rerun_is_idempotent_and_retries_after_a_model_failure(svc, tmp_path):
    root = make_tree(tmp_path)
    first = ov.apply_import(svc, FakeLLMClient([RuntimeError("model down")]), root)
    assert first["facts_added"] == 0 and len(first["errors"]) == 1 and first["archived_new"] == 6
    second = ov.apply_import(svc, FakeLLMClient([good_ops()]), root)
    assert second["archived_new"] == 0 and second["archive_existing"] == 6 and second["facts_added"] == 2
    third = ov.apply_import(svc, FakeLLMClient([good_ops()]), root)
    assert third["facts_added"] == 0 and third["facts_duplicate"] == 2
    assert len(svc.store.live_facts()) == 2


def test_undo_retracts_the_batch_without_deleting_files(svc, tmp_path):
    report = ov.apply_import(svc, FakeLLMClient([good_ops()]), make_tree(tmp_path))
    assert ov.list_batches(svc)[0]["facts"] == 2
    result = ov.undo_batch(svc, report["batch"])
    assert result == {"facts_retracted": 2, "archive_hidden": 6}
    assert svc.store.live_facts() == [] and svc.archive.search("电池状态")[0] == []
    assert len(list(svc.store.facts_dir.glob("f_*.md"))) == 2 and svc.archive.files.count() == 6
    assert ov.list_batches(svc)[0]["undone"]
    with pytest.raises(StoreError):
        ov.undo_batch(svc, report["batch"])
    with pytest.raises(StoreError):
        ov.undo_batch(svc, "../escape")


def test_index_rebuilds_from_files_and_ledger(svc, tmp_path):
    report = ov.apply_import(svc, FakeLLMClient([good_ops()]), make_tree(tmp_path))
    svc.archive.retract(next(iter(svc.archive.files.iter_entries())).id, "test")
    stats = svc.reindex_all()
    assert stats["archive"]["entries"] == 5
    assert report["batch"] in {b["batch"] for b in ov.list_batches(svc)}


def test_changed_note_replaces_the_old_version_in_search(svc, tmp_path):
    root = make_tree(tmp_path)
    ov.apply_import(svc, FakeLLMClient([{"operations": []}]), root)
    note = root / "default" / "memories" / "experiences" / "2026-10-25-自启动修复.md"
    note.write_text("# 自启动修复\n新的结论：改用计划任务。", encoding="utf-8")
    ov.apply_import(svc, FakeLLMClient([{"operations": []}]), root)
    hits, _ = svc.archive.search("自启动修复")
    assert len([h for h in hits if h["category"] == "experiences"]) == 1
    assert "计划任务" in svc.archive.read(next(h for h in hits if h["category"] == "experiences")["id"])["text"]


def test_rate_limited_call_is_retried_once(svc, tmp_path):
    from membrain.llm import LLMError
    llm = FakeLLMClient([LLMError("LLM request failed"), good_ops()])
    report = ov.apply_import(svc, llm, make_tree(tmp_path))
    assert report["errors"] == [] and report["facts_added"] == 2 and len(llm.calls) == 2


def test_project_scoped_claims_are_not_imported_as_facts(svc, tmp_path):
    ops = {"operations": [
        {"type": "preference", "scope": "owner", "subject": "style", "claim": "Prefers plain text without emoji",
         "quote": "不用 emoji，喜欢简洁纯文字"},
        {"type": "preference", "scope": "project", "subject": "reminders", "claim": "Monitor stars twice a day",
         "quote": "周六日不安排任务提醒"}]}
    report = ov.apply_import(svc, FakeLLMClient([ops]), make_tree(tmp_path))
    assert report["facts_added"] == 1 and report["rejected"] == 1
    assert [f.claim for f in svc.store.live_facts()] == ["Prefers plain text without emoji"]


def test_missing_operations_list_means_nothing_qualified(svc, tmp_path):
    report = ov.apply_import(svc, FakeLLMClient([{}]), make_tree(tmp_path))
    assert report["errors"] == [] and report["facts_added"] == 0


class TopicEmbedder:
    """Battery, weather and everything else point three different ways."""
    enabled = True
    model = "topic-3"

    def embed(self, texts):
        import numpy as np
        return np.array([[1.0, 0.0, 0.0] if "电池" in t else [0.0, 0.0, 1.0] if "天气" in t else [0.0, 1.0, 0.0]
                         for t in texts], dtype=np.float32)

    embed_queries = embed


def test_text_only_match_needs_most_of_the_query_once_meaning_is_available(svc, tmp_path):
    svc.archive.embedder = TopicEmbedder()
    ov.apply_import(svc, FakeLLMClient([{"operations": []}]), make_tree(tmp_path))
    svc.archive.sync()
    # "状态" appears in the battery note, but the weather question is about something else.
    assert svc.archive.search("巴黎天气状态预报")[0] == []
    hits, _ = svc.archive.search("电池状态")
    assert hits and hits[0]["category"] == "events" and hits[0]["similarity"] == 1.0


def test_mcp_archive_tools_search_browse_and_read(svc, tmp_path):
    from membrain.server import build_server
    ov.apply_import(svc, FakeLLMClient([good_ops()]), make_tree(tmp_path))
    server = build_server(svc)
    tool = lambda name: server._tool_manager.get_tool(name).fn  # noqa: E731
    found = tool("search_archive")(query="电池状态")["results"]
    assert found and found[0]["category"] == "events"
    assert tool("browse_archive")()["tree"]["openviking"]["study"] == 1
    listed = tool("browse_archive")(source="openviking", category="code")
    assert listed["overview"]["count"] == 1 and listed["items"][0]["title"].startswith("02-backend")
    assert "电池" in tool("read_archive")(entry_id=found[0]["id"])["text"]
    assert tool("read_archive")(entry_id="a_20260101_zzzzzz")["ok"] is False
    assert tool("read_archive")(entry_id="../../../etc/passwd")["ok"] is False


def test_chinese_tokens_and_abstract():
    assert tokens("电池状态 abc") == ["电池", "池状", "状态", "abc"]
    assert make_abstract("# 标题\n第一句。\n<!-- x -->\n第二句。") == "第一句。 第二句。"
