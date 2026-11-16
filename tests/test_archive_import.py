from __future__ import annotations

from datetime import date

import pytest

from membrain import import_batches
from membrain.archive import ArchiveEntry, content_hash, entry_id, make_abstract, tokens
from membrain.llm import FakeLLMClient, LLMError
from membrain.models import Evidence, FactType
from membrain.server import MemoryService
from membrain.store import StoreError
from test_core import make_settings


def _sample_notes() -> list[tuple[str, str, str, str]]:
    return [
        ("preferences", "协作偏好", "2026-10-20", "- 不用 emoji，喜欢简洁纯文字\n- 周六日不安排任务提醒"),
        ("events", "EC 固件问题", "2026-10-14", "# Summary\nEC 向 ACPI 误报电池状态，导致无法识别电池。"),
        ("experiences", "自启动修复", "2026-10-25", "# 自启动修复\n问题出在服务没有随宿主退出。"),
        ("code", "02-backend 报告", "2026-10-26", "# 02-backend 报告\n模块地图和命令清单。"),
    ]


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


def seed_archive_batch(svc: MemoryService, llm: FakeLLMClient | None = None) -> dict:
    batch = import_batches.new_batch_id()
    archive_ids: list[str] = []
    fact_ids: list[str] = []
    llm_client = llm or FakeLLMClient([good_ops()])
    rejected = 0
    for cat, title, day, body in _sample_notes():
        aid = entry_id("notes", "default", f"{cat}/{title}", content_hash(body), day)
        entry = ArchiveEntry(
            id=aid, source="notes", category=cat, title=title,
            abstract=make_abstract(body), body=body, date=day,
            origin=f"{cat}/{title}.md", scope="default", batch=batch,
            content_hash=content_hash(body),
        )
        svc.archive.files.write(entry)
        archive_ids.append(aid)
        if cat == "preferences":
            kept, rej = import_batches.distill_with_retry(llm_client, entry)
            rejected += rej
            for op in kept:
                ev = Evidence(episode=entry.id, quote=op["quote"], origin="imported",
                              date=date.fromisoformat(day))
                fact = svc.store.write_fact(svc.store.draft_fact(
                    type=FactType(op["type"]), subject=op["subject"], claim=op["claim"],
                    evidence=[ev], source_agent="notes-import", source_session=batch))
                svc.retriever.sync_fact(fact)
                fact_ids.append(fact.id)
    svc.archive.sync(embed=True)
    report = {"batch": batch, "source": "notes", "created": "2026-10-26T00:00:00+00:00",
              "archive_ids": archive_ids, "fact_ids": fact_ids, "errors": [], "rejected": rejected}
    import_batches.write_manifest(svc, batch, report)
    svc.store.append_event({"op": "import", "id": batch, "source": "notes"})
    return report


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(import_batches, "CALL_PAUSE", 0)
    monkeypatch.setattr(import_batches, "RATE_LIMIT_WAITS", (0, 0))


@pytest.fixture
def svc(tmp_path):
    service = MemoryService(make_settings(tmp_path))
    yield service
    service.index.close()


def test_distill_verifies_quotes_authority_and_traits(svc):
    entry = ArchiveEntry(
        id="a_20261020_test01", source="notes", category="preferences", title="协作偏好",
        abstract="偏好", body="- 不用 emoji，喜欢简洁纯文字\n- 周六日不安排任务提醒",
        date="2026-10-20", origin="pref.md", scope="default", batch="b_1", content_hash="h1",
    )
    kept, rejected = import_batches.distill(FakeLLMClient([good_ops()]), entry)
    assert len(kept) == 2 and rejected == 3
    assert {item["claim"] for item in kept} == {"Prefers plain text without emoji", "No task reminders on weekends"}


def test_archive_browse_search_and_distill_facts(svc):
    report = seed_archive_batch(svc)
    assert len(report["archive_ids"]) == 4 and len(report["fact_ids"]) == 2
    facts = svc.store.live_facts()
    assert {f.claim for f in facts} == {"Prefers plain text without emoji", "No task reminders on weekends"}
    fact = facts[0]
    assert fact.evidence[0].origin == "imported" and fact.confidence >= 0.6
    assert svc.archive.files.exists(fact.evidence[0].episode)
    results, _ = svc.archive.search("电池状态", top_k=3)
    assert results and results[0]["category"] == "events"
    assert svc.archive.browse()["tree"]["notes"]["code"] == 1
    assert svc.archive.read(results[0]["id"])["text"].startswith("# Summary")


def test_undo_retracts_the_batch_without_deleting_files(svc):
    report = seed_archive_batch(svc)
    assert import_batches.list_batches(svc)[0]["facts"] == 2
    result = import_batches.undo_batch(svc, report["batch"])
    assert result == {"facts_retracted": 2, "archive_hidden": 4}
    assert svc.store.live_facts() == [] and svc.archive.search("电池状态")[0] == []
    assert len(list(svc.store.facts_dir.glob("f_*.md"))) == 2 and svc.archive.files.count() == 4
    assert import_batches.list_batches(svc)[0]["undone"]
    with pytest.raises(StoreError):
        import_batches.undo_batch(svc, report["batch"])
    with pytest.raises(StoreError):
        import_batches.undo_batch(svc, "../escape")


def test_index_rebuilds_from_files_and_ledger(svc):
    report = seed_archive_batch(svc)
    svc.archive.retract(next(iter(svc.archive.files.iter_entries())).id, "test")
    stats = svc.reindex_all()
    assert stats["archive"]["entries"] == 3
    assert report["batch"] in {b["batch"] for b in import_batches.list_batches(svc)}


def test_changed_note_replaces_the_old_version_in_search(svc):
    aid1 = entry_id("notes", "default", "experiences/autostart", content_hash("old"), "2026-10-25")
    svc.archive.files.write(ArchiveEntry(
        id=aid1, source="notes", category="experiences", title="自启动修复",
        abstract="old", body="# 自启动修复\n旧结论。", date="2026-10-25",
        origin="experiences/autostart", scope="default", batch="b_20261025000000000001",
        content_hash=content_hash("old"),
    ))
    aid2 = entry_id("notes", "default", "experiences/autostart", content_hash("new"), "2026-10-25")
    svc.archive.files.write(ArchiveEntry(
        id=aid2, source="notes", category="experiences", title="自启动修复",
        abstract="new", body="# 自启动修复\n新的结论：改用计划任务。", date="2026-10-25",
        origin="experiences/autostart", scope="default", batch="b_20261025000000000002",
        content_hash=content_hash("new"),
    ))
    svc.archive.sync(embed=False)
    hits, _ = svc.archive.search("自启动修复")
    assert len([h for h in hits if h["category"] == "experiences"]) == 1
    assert "计划任务" in svc.archive.read(next(h for h in hits if h["category"] == "experiences")["id"])["text"]


def test_rate_limited_call_is_retried_once():
    entry = ArchiveEntry(
        id="a_20261020_test02", source="notes", category="preferences", title="协作偏好",
        abstract="偏好", body="- 不用 emoji，喜欢简洁纯文字\n- 周六日不安排任务提醒",
        date="2026-10-20", origin="pref.md", scope="default", batch="b_1", content_hash="h2",
    )
    llm = FakeLLMClient([LLMError("LLM request failed"), good_ops()])
    kept, rejected = import_batches.distill_with_retry(llm, entry)
    assert len(kept) == 2 and rejected == 3 and len(llm.calls) == 2


def test_project_scoped_claims_are_filtered_unless_allowed():
    entry = ArchiveEntry(
        id="a_20261020_test03", source="notes", category="preferences", title="协作偏好",
        abstract="偏好", body="- 不用 emoji，喜欢简洁纯文字\n- 周六日不安排任务提醒",
        date="2026-10-20", origin="pref.md", scope="default", batch="b_1", content_hash="h3",
    )
    ops = {"operations": [
        {"type": "preference", "scope": "owner", "subject": "style", "claim": "Prefers plain text without emoji",
         "quote": "不用 emoji，喜欢简洁纯文字"},
        {"type": "preference", "scope": "project", "subject": "reminders", "claim": "Monitor stars twice a day",
         "quote": "周六日不安排任务提醒"}]}
    kept, rejected = import_batches.distill(FakeLLMClient([ops]), entry)
    assert len(kept) == 1 and rejected == 1
    assert kept[0]["claim"] == "Prefers plain text without emoji"


def test_missing_operations_list_means_nothing_qualified():
    entry = ArchiveEntry(
        id="a_20261020_test04", source="notes", category="preferences", title="空",
        abstract="空", body="nothing", date="2026-10-20", origin="x", scope="default",
        batch="b_1", content_hash="h4",
    )
    kept, rejected = import_batches.distill(FakeLLMClient([{}]), entry)
    assert kept == [] and rejected == 0


class TopicEmbedder:
    """Battery, weather and everything else point three different ways."""
    enabled = True
    model = "topic-3"

    def embed(self, texts):
        import numpy as np
        return np.array([[1.0, 0.0, 0.0] if "电池" in t else [0.0, 0.0, 1.0] if "天气" in t else [0.0, 1.0, 0.0]
                         for t in texts], dtype=np.float32)

    embed_queries = embed


def test_text_only_match_needs_most_of_the_query_once_meaning_is_available(svc):
    svc.archive.embedder = TopicEmbedder()
    seed_archive_batch(svc, llm=FakeLLMClient([{"operations": []}]))
    svc.archive.sync()
    # "状态" appears in the battery note, but the weather question is about something else.
    assert svc.archive.search("巴黎天气状态预报")[0] == []
    hits, _ = svc.archive.search("电池状态")
    assert hits and hits[0]["category"] == "events" and hits[0]["similarity"] == 1.0


def test_mcp_archive_tools_search_browse_and_read(svc):
    from membrain.server import build_server
    seed_archive_batch(svc)
    server = build_server(svc)
    tool = lambda name: server._tool_manager.get_tool(name).fn  # noqa: E731
    found = tool("search_archive")(query="电池状态")["results"]
    assert found and found[0]["category"] == "events"
    assert tool("browse_archive")()["tree"]["notes"]["code"] == 1
    listed = tool("browse_archive")(source="notes", category="code")
    assert listed["overview"]["count"] == 1 and listed["items"][0]["title"].startswith("02-backend")
    assert "电池" in tool("read_archive")(entry_id=found[0]["id"])["text"]
    assert tool("read_archive")(entry_id="a_20260101_zzzzzz")["ok"] is False
    assert tool("read_archive")(entry_id="../../../etc/passwd")["ok"] is False


def test_chinese_tokens_and_abstract():
    assert tokens("电池状态 abc") == ["电池", "池状", "状态", "abc"]
    assert make_abstract("# 标题\n第一句。\n<!-- x -->\n第二句。") == "第一句。 第二句。"
