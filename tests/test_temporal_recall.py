"""Temporal recall: the "记住进度，明天继续" chain, end to end.

Every test here exists because a real question failed on 2026-11-05: the owner
said "记住现在的进度，明天继续" one evening and could not ask it back the next
day. The four layers that broke were capture time, extraction, recall and
service availability; these pin each of them.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import pytest

from membrain import consolidate, session_import
from membrain.config import local_date as local_date_of
from membrain.extract import Extractor, _is_one_off_spec, _is_task_handoff
from membrain.index import Index
from membrain.models import Episode, FactType
from membrain.retrieval import Retriever
from membrain.store import Store
from membrain.text import parse_time_window
from test_core import FakeEmbedder, make_settings


def ep(eid, text, ts, role="user", session="s1", agent="test", ts_source="exact"):
    return Episode(id=eid, ts=ts, session=session, agent=agent, role=role, text=text,
                   ts_source=ts_source)


class _Svc:
    """The minimum hooks/watcher/health need: settings, store, retriever."""

    def __init__(self, settings, store, retriever):
        self.settings = settings
        self.store = store
        self.retriever = retriever

    def disabled_agents(self):
        return set()


@pytest.fixture()
def rig(tmp_path):
    settings = make_settings(tmp_path)
    store = Store(settings.home)
    index = Index(settings.db_path)
    retriever = Retriever(store, index, FakeEmbedder(), settings)
    return settings, store, index, retriever, _Svc(settings, store, retriever)


# ------------------------------------------------------- layer 1: real time

def test_episode_rejects_naive_timestamp():
    with pytest.raises(Exception):
        Episode(id="e", ts=datetime(2026, 11, 4, 23, 30), session="s", agent="a",
                role="user", text="t")


def test_ts_source_defaults_to_estimated():
    """A capture path that forgets to declare accuracy must not claim precision."""
    e = Episode(id="e", ts=datetime(2026, 11, 4, tzinfo=timezone.utc), session="s",
                agent="a", role="user", text="t")
    assert e.ts_source == "estimated"
    assert e.time_is_exact is False


def test_episodes_shard_by_local_day_not_utc(rig, monkeypatch):
    """23:30Z on Nov 4 is 07:30 on Nov 5 in UTC+8 — it belongs to Nov 5.

    Filing it under the UTC day makes a one-day-wide "昨晚" window look in the
    wrong file and miss completely.
    """
    monkeypatch.setenv("MEMBRAIN_TZ", "Asia/Shanghai")
    _, store, _, _, _ = rig
    ts = datetime(2026, 11, 4, 23, 30, tzinfo=timezone.utc)
    store.append_episodes([ep("e1", "记住现在的进度，明天继续", ts)])
    written = list(store.episodes_dir.glob("*.jsonl"))
    assert [p.stem for p in written] == ["2026-11-05"]


def test_watcher_keeps_original_turn_time(rig, monkeypatch):
    """The regression that made every time-based recall a lie."""
    from membrain import watcher

    _, store, _, _, svc = rig
    said_at = datetime(2026, 11, 4, 22, 15, tzinfo=timezone.utc)
    session = session_import.RawSession(
        "claude-code", "sess-1", "/tmp/sess-1.jsonl", "2026-11-04",
        [("user", "记住现在的进度，明天继续", said_at), ("assistant", "好", said_at)],
    )
    monkeypatch.setattr(session_import, "collect", lambda *a, **k: [session])
    stats = watcher.watch_once(svc)
    assert stats["captured"] == 1
    assert stats["exact_ts"] == 1 and stats["estimated_ts"] == 0
    stored = list(store.iter_episodes())
    assert len(stored) == 1
    assert stored[0].ts == said_at, "capture time must not replace the turn's own time"
    assert stored[0].ts_source == "exact"


def test_watcher_marks_estimated_when_source_keeps_no_turn_time(rig, monkeypatch):
    from membrain import watcher

    _, store, _, _, svc = rig
    session_ts = datetime(2026, 11, 4, 9, 0, tzinfo=timezone.utc)
    session = session_import.RawSession(
        "qoder", "sess-2", "/tmp/sess-2", "2026-11-04",
        [("user", "继续优化 membrain", None)], session_ts=session_ts,
    )
    monkeypatch.setattr(session_import, "collect", lambda *a, **k: [session])
    stats = watcher.watch_once(svc)
    assert stats["estimated_ts"] == 1 and stats["exact_ts"] == 0
    stored = list(store.iter_episodes())[0]
    assert stored.ts == session_ts
    assert stored.ts_source == "estimated"


def test_rec_ts_accepts_iso_and_epoch_shapes():
    iso = session_import._rec_ts("2026-11-04T22:15:00Z")
    assert iso == datetime(2026, 11, 4, 22, 15, tzinfo=timezone.utc)
    ms = session_import._rec_ts(1793988900000)
    assert ms is not None and ms.tzinfo is not None
    assert session_import._rec_ts(None) is None
    assert session_import._rec_ts("not a time") is None


# ------------------------------------------------- layer 3: reading the time

@pytest.mark.parametrize("phrase,expected", [
    ("我昨晚说今天的任务是什么", (1, 1)),
    ("昨天", (1, 1)),
    ("前天晚上", (2, 2)),
    ("今天", (0, 0)),
    ("3天前", (3, 3)),
])
def test_parse_time_window(phrase, expected):
    today = date(2026, 11, 5)
    window = parse_time_window(phrase, today)
    assert window == (today - timedelta(days=expected[0]), today - timedelta(days=expected[1]))


def test_time_window_is_always_ordered():
    """"上周" spans 7..13 days back; start must not come after end."""
    today = date(2026, 11, 5)
    for phrase in ("上周", "最近", "上个月", "本周"):
        start, end = parse_time_window(phrase, today)
        assert start <= end, phrase


def test_no_time_phrase_means_no_filter():
    assert parse_time_window("我喜欢喝 tea") is None
    assert parse_time_window("") is None


def test_recall_demotes_facts_outside_the_asked_window(rig):
    """Relevance still leads; the window breaks ties instead of hard-filtering."""
    from membrain.models import Evidence

    _, store, _, retriever, _ = rig
    today = date.today()
    # Wording stays inside FakeEmbedder's "work/memory" topic so both facts are
    # genuinely retrievable and the only difference between them is the date.
    inside = store.write_fact(store.draft_fact(
        type=FactType.TASK, subject="membrain", claim="Resumes memory work next session",
        evidence=[Evidence(episode="e1", quote="记住进度", date=today - timedelta(days=1),
                           origin="stated")],
        source_agent="test", source_session="s1"))
    outside = store.write_fact(store.draft_fact(
        type=FactType.TASK, subject="membrain", claim="Resumes memory work next session again",
        evidence=[Evidence(episode="e2", quote="记住进度", date=today - timedelta(days=60),
                           origin="stated")],
        source_agent="test", source_session="s1"))
    retriever.sync_fact(inside)
    retriever.sync_fact(outside)
    result = retriever.recall("我昨晚说的 memory work")
    assert result.time_window == (today - timedelta(days=1), today - timedelta(days=1))
    ids = [f.id for f in result.hits]
    assert inside.id in ids
    if outside.id in ids:
        assert ids.index(inside.id) < ids.index(outside.id), "in-window evidence ranks first"


def test_prompt_offers_task_and_stops_ignoring_it(rig):
    from membrain.llm import FakeLLMClient

    _, store, _, retriever, _ = rig
    system, _ = Extractor(store, retriever, FakeLLMClient([]))._prompt("{}")
    assert "Ignore tasks" not in system
    allowed = system.split("Allowed fact types are exactly:")[1].split(".")[0]
    assert "task" in allowed
    # The two durability tests must both be present, one per type family.
    assert "START OF THE NEXT SESSION" in system
    assert "a month from now" in system


# --------------------------------------- layer 3: falling back to raw turns

def test_recall_falls_back_to_raw_turns_when_facts_are_thin(rig):
    _, store, _, retriever, _ = rig
    ts = datetime(2026, 11, 4, 22, 15, tzinfo=timezone.utc)
    store.append_episodes([ep("e1", "记住现在的进度，明天继续优化 membrain", ts)])
    retriever.sync_episodes(store.iter_episodes())
    result = retriever.recall("进度 明天继续 优化 membrain")
    assert result.episode_hits, "the raw turn must be findable when no fact covers it"
    assert result.episode_hits[0]["origin"] == "episode"
    assert any("来自会话记录" in n for n in result.notes)


def test_episode_fallback_matches_chinese(rig):
    """unicode61 tokenizes a CJK run as one token; the term scan must cover it."""
    _, store, _, retriever, _ = rig
    # Spoken yesterday, asked about as "昨天": the window and the CJK matcher
    # have to agree, or this returns nothing for two separate reasons at once.
    ts = datetime.now(timezone.utc) - timedelta(days=1)
    store.append_episodes([ep("e1", "记住现在的进度，明天继续优化记忆系统", ts)])
    retriever.sync_episodes(store.iter_episodes())
    result = retriever.recall("我昨天说的进度是什么")
    assert result.episode_hits
    assert "进度" in result.episode_hits[0]["text"]


def test_episode_fallback_respects_the_time_window(rig):
    _, store, _, retriever, _ = rig
    store.append_episodes([
        ep("old", "记住进度，明天继续", datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)),
    ])
    retriever.sync_episodes(store.iter_episodes())
    result = retriever.recall("我昨晚说的进度")
    assert result.time_window is not None
    assert not any(h["episode"] == "old" for h in result.episode_hits)


def test_in_window_fact_is_not_diluted_by_raw_turns(rig):
    """A correct fact from the asked-about day is a complete answer.

    Loosely-matching raw turns next to it invite answering from the wrong one,
    which is what happened in the first acceptance run.
    """
    from membrain.models import Evidence

    _, store, _, retriever, _ = rig
    today = date.today()
    yesterday = datetime.now(timezone.utc) - timedelta(days=1)
    store.append_episodes([
        ep("e1", "今天把解析器和 models 都改完了", yesterday),
        ep("e2", "记住现在的进度，明天继续", yesterday + timedelta(minutes=1)),
    ])
    fact = store.write_fact(store.draft_fact(
        type=FactType.TASK, subject="work-handoff",
        claim="Resumes memory work next session",
        evidence=[Evidence(episode="e2", quote="记住现在的进度，明天继续",
                           date=local_date_of(yesterday), origin="stated")],
        source_agent="test", source_session="s1"))
    retriever.sync_fact(fact)
    retriever.sync_episodes(store.iter_episodes())
    result = retriever.recall("我昨天说的 memory work 是什么")
    assert fact.id in [f.id for f in result.hits]
    assert result.episode_hits == [], "an in-window fact already answers it"


def test_out_of_window_facts_trigger_the_fallback(rig):
    """Facts exist but none from the asked-about day: go read the raw turns."""
    from membrain.models import Evidence

    _, store, _, retriever, _ = rig
    store.append_episodes([ep("e1", "记住现在的进度，明天继续",
                              datetime.now(timezone.utc) - timedelta(days=1))])
    fact = store.write_fact(store.draft_fact(
        type=FactType.TASK, subject="work-handoff", claim="Old work plan from long ago",
        evidence=[Evidence(episode="e9", quote="记住进度",
                           date=date.today() - timedelta(days=90), origin="stated")],
        source_agent="test", source_session="s1"))
    retriever.sync_fact(fact)
    retriever.sync_episodes(store.iter_episodes())
    # "work" pulls the stale fact in; "进度" is what the raw turn actually says.
    result = retriever.recall("我昨天说的 work 进度")
    assert fact.id in [f.id for f in result.hits], "the stale fact is still retrievable"
    assert result.episode_hits, "stale facts must not block the raw-turn fallback"


def test_type_question_is_answered_by_lookup_not_similarity(rig):
    """"我昨晚说的任务是什么" names a type and a day, not a topic.

    The stored claim says 时间戳修复; nothing in it resembles 任务. Text
    similarity cannot bridge that, so the question is answered structurally.
    """
    from membrain.models import Evidence

    _, store, _, retriever, _ = rig
    today = date.today()
    fact = store.write_fact(store.draft_fact(
        type=FactType.TASK, subject="software-development",
        claim="明天继续完成时间戳修复工作",
        evidence=[Evidence(episode="e1", quote="记住现在的进度，明天继续",
                           date=today - timedelta(days=1), origin="stated")],
        source_agent="test", source_session="s1"))
    retriever.sync_fact(fact)
    result = retriever.recall("我昨晚说今天的任务是什么")
    assert fact.id in [f.id for f in result.hits], "a type+day question must find its task"
    assert result.hits[0].id == fact.id


def test_type_question_excludes_finished_tasks(rig):
    """A closed task is still a memory, but it is not an answer to "my task".

    Asserted on the lookup itself: FakeEmbedder gives every topic-less text the
    same vector, so an end-to-end recall would "match" this fact semantically
    for reasons that have nothing to do with the rule under test.
    """
    from membrain.models import Evidence

    _, store, _, retriever, _ = rig
    today = date.today()
    done = store.write_fact(store.draft_fact(
        type=FactType.TASK, subject="software-development", claim="已经做完的那件事",
        evidence=[Evidence(episode="e1", quote="记住进度", date=today - timedelta(days=1),
                           origin="stated")],
        source_agent="test", source_session="s1"))
    window = (today - timedelta(days=1), today - timedelta(days=1))
    live = {f.id: f for f in store.live_facts()}
    assert done.id in retriever._typed_window_hits("我昨晚的任务是什么", live, window)

    store.close_task(done.id)
    assert store.get_fact(done.id).closed_at is not None
    live = {f.id: f for f in store.live_facts()}
    assert done.id not in retriever._typed_window_hits("我昨晚的任务是什么", live, window)
    assert retriever.open_tasks() == []


def test_type_question_finds_an_english_claim_after_reindex(rig):
    """The extractor's language choice must not decide whether a task is findable.

    A run of the live acceptance check produced an English claim for a Chinese
    handoff and recall came back empty; the type+window lookup exists precisely
    so wording and language cannot break this question. Exercised through
    reindex(), which is the path a real store takes.
    """
    from membrain.models import Evidence

    _, store, _, retriever, _ = rig
    today = date.today()
    fact = store.write_fact(store.draft_fact(
        type=FactType.TASK, subject="code-progress",
        claim="Has finished parser and models; tomorrow will complete the timestamp fix",
        evidence=[Evidence(episode="e1", quote="记住现在的进度，明天继续",
                           date=today - timedelta(days=1), origin="stated")],
        source_agent="test", source_session="s1"))
    retriever.reindex()
    result = retriever.recall("我昨晚说今天的任务是什么")
    assert fact.id in [f.id for f in result.hits], "an English claim must still answer a Chinese question"
    assert result.time_window == (today - timedelta(days=1), today - timedelta(days=1))


def test_semantic_floor_cannot_veto_a_type_and_day_match(rig):
    """Cosine similarity must not overrule a structured answer.

    The claim sits in FakeEmbedder's "dietary" topic and the question in its
    "work" topic, so the semantic score is far below the floor — exactly the
    shape that made a correct task vanish when the extractor phrased the claim
    in a different language from the question.
    """
    from membrain.models import Evidence

    _, store, _, retriever, _ = rig
    today = date.today()
    fact = store.write_fact(store.draft_fact(
        type=FactType.TASK, subject="dietary", claim="Finish the spicy food review",
        evidence=[Evidence(episode="e1", quote="记住现在的进度，明天继续",
                           date=today - timedelta(days=1), origin="stated")],
        source_agent="test", source_session="s1"))
    retriever.sync_fact(fact)
    result = retriever.recall("我昨晚说今天的任务是什么 work")
    assert result.time_window == (today - timedelta(days=1), today - timedelta(days=1))
    assert fact.id in [f.id for f in result.hits], "the semantic floor vetoed a structured match"


def test_type_question_without_a_time_phrase_still_lists_tasks(rig):
    from membrain.models import Evidence

    _, store, _, retriever, _ = rig
    fact = store.write_fact(store.draft_fact(
        type=FactType.TASK, subject="software-development", claim="待办：修完时间戳",
        evidence=[Evidence(episode="e1", quote="记住进度", date=date.today() - timedelta(days=9),
                           origin="stated")],
        source_agent="test", source_session="s1"))
    retriever.sync_fact(fact)
    result = retriever.recall("我的待办有哪些")
    assert result.time_window is None
    assert fact.id in [f.id for f in result.hits]


def test_episode_fallback_hides_disconnected_agents(rig):
    _, store, _, retriever, _ = rig
    store.append_episodes([ep("e1", "记住进度，明天继续", datetime.now(timezone.utc), agent="Cursor")])
    retriever.sync_episodes(store.iter_episodes())
    retriever.disabled_agents = {"cursor"}
    result = retriever.recall("记住进度 明天继续")
    assert not result.episode_hits


# ------------------------------------------- layer 2: tasks get stored at all

def test_gate_keeps_a_turn_that_is_both_spec_and_handoff(rig):
    """The old gate judged raw text before the type, so this turn was unreachable."""
    both = "帮我写一份交接文档，记住进度明天继续"
    assert _is_one_off_spec(both)
    assert _is_task_handoff(both)


def test_extract_stores_a_task_handoff(rig):
    _, store, _, retriever, _ = rig
    from membrain.llm import FakeLLMClient
    said = "记住现在的进度，明天继续"
    store.append_episodes([ep("e1", said, datetime(2026, 11, 4, 22, 15, tzinfo=timezone.utc))])
    llm = FakeLLMClient([{"operations": [{
        "op": "add", "type": "task", "subject": "work-handoff",
        "claim": "Resumes the current work next session", "episode": "e1",
        "quote": said, "durable": True}]}])
    report = Extractor(store, retriever, llm).run("s1")
    assert report.facts_added == 1, report.errors
    fact = store.live_facts()[0]
    assert fact.type == FactType.TASK
    assert fact.evidence[0].quote == said


def test_extract_rejects_task_without_handoff_wording(rig):
    """The gate is not just "type==task"; the turn must actually hand work over."""
    _, store, _, retriever, _ = rig
    from membrain.llm import FakeLLMClient
    said = "我不吃辣"
    store.append_episodes([ep("e1", said, datetime(2026, 11, 4, tzinfo=timezone.utc))])
    llm = FakeLLMClient([{"operations": [{
        "op": "add", "type": "task", "subject": "x", "claim": "Y",
        "episode": "e1", "quote": said, "durable": True}]}])
    report = Extractor(store, retriever, llm).run("s1")
    assert report.facts_added == 0
    assert report.skipped >= 1


def test_session_distill_whitelist_allows_task():
    from membrain import openviking_import as ov
    assert "task" in ov._TYPES
    assert "task" in session_import.SESSION_DISTILL_SYSTEM
    assert "Skip work logs, task status" not in session_import.SESSION_DISTILL_SYSTEM


# ------------------------------------------------------- closing a task

def test_open_tasks_lists_only_unclosed(rig):
    _, store, _, retriever, _ = rig
    from membrain.models import Evidence
    fact = store.write_fact(store.draft_fact(
        type=FactType.TASK, subject="membrain", claim="Finishes the timestamp fix",
        evidence=[Evidence(episode="e1", quote="记住进度", date=date(2026, 11, 4), origin="stated")],
        source_agent="test", source_session="s1"))
    assert [f.id for f in retriever.open_tasks()] == [fact.id]
    store.close_task(fact.id)
    assert store.get_fact(fact.id).closed_at is not None
    assert retriever.open_tasks() == []
    store.reopen_task(fact.id)
    assert store.get_fact(fact.id).closed_at is None
    assert [f.id for f in retriever.open_tasks()] == [fact.id]


def test_close_task_is_append_only(rig):
    """Closing goes through the ledger; the fact file is never rewritten."""
    _, store, _, _, _ = rig
    from membrain.models import Evidence
    fact = store.write_fact(store.draft_fact(
        type=FactType.TASK, subject="membrain", claim="Finishes the timestamp fix",
        evidence=[Evidence(episode="e1", quote="记住进度", date=date(2026, 11, 4), origin="stated")],
        source_agent="test", source_session="s1"))
    path = store.fact_path(fact.id)
    before = path.read_text(encoding="utf-8")
    store.close_task(fact.id)
    assert path.read_text(encoding="utf-8") == before, "fact file must not change"
    assert any(json.loads(line)["op"] == "close"
               for line in (store.home / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
               if line.strip())


def test_close_task_rejects_non_task(rig):
    _, store, _, _, _ = rig
    from membrain.models import Evidence
    from membrain.store import StoreError
    fact = store.write_fact(store.draft_fact(
        type=FactType.PREFERENCE, subject="dietary", claim="Does not eat spicy food",
        evidence=[Evidence(episode="e1", quote="我不吃辣", date=date(2026, 11, 4), origin="stated")],
        source_agent="test", source_session="s1"))
    with pytest.raises(StoreError):
        store.close_task(fact.id)


def test_disconnecting_an_agent_hides_facts_under_every_spelling(rig):
    """The isolation leak, closed.

    The watcher labelled captured facts "WorkBuddy" while hooks and the relay
    protocol wrote "workbuddy". Isolation compared those exactly, so after the
    owner disconnected the agent, every memory captured through the watcher
    stayed visible in recall.
    """
    from membrain.models import Evidence

    _, store, _, retriever, _ = rig
    today = date.today()
    watched = store.write_fact(store.draft_fact(
        type=FactType.PREFERENCE, subject="style", claim="Prefers terse memory work summaries",
        evidence=[Evidence(episode="e1", quote="要简短", date=today, origin="stated")],
        source_agent="WorkBuddy", source_session="s1"))
    hooked = store.write_fact(store.draft_fact(
        type=FactType.PREFERENCE, subject="style", claim="Prefers terse memory work notes too",
        evidence=[Evidence(episode="e2", quote="要简短", date=today, origin="stated")],
        source_agent="workbuddy", source_session="s2"))
    retriever.sync_fact(watched)
    retriever.sync_fact(hooked)
    assert len(retriever.recall("terse memory work").hits) == 2, "both are visible while connected"

    retriever.disabled_agents = {"workbuddy"}
    ids = [f.id for f in retriever.recall("terse memory work").hits]
    assert watched.id not in ids, "the watcher-labelled fact leaked past isolation"
    assert hooked.id not in ids


def test_watcher_stores_the_canonical_agent_key(rig, monkeypatch):
    """New captures must not recreate the two-identity problem.

    Labels belong in agents.SPECS for display; stored data uses the same key
    hooks and the relay protocol use, so disconnecting an agent hides all of it.
    """
    from membrain import watcher

    _, store, _, _, svc = rig
    session = session_import.RawSession(
        "claude-code", "s-x", "/tmp/s-x.jsonl", "2026-11-04",
        [("user", "记住进度，明天继续", datetime(2026, 11, 4, 22, 0, tzinfo=timezone.utc))])
    monkeypatch.setattr(session_import, "collect", lambda *a, **k: [session])
    watcher.watch_once(svc)
    assert [ep.agent for ep in store.iter_episodes()] == ["claude-code"]


def test_open_task_is_never_a_forget_candidate(rig):
    """Unfinished work must not retire because nobody mentioned it."""
    from membrain.health import HealthScanner
    from membrain.models import Evidence

    _, store, _, _, svc = rig
    old = date.today() - timedelta(days=200)
    store.write_fact(store.draft_fact(
        type=FactType.TASK, subject="membrain", claim="Still open after 200 days",
        evidence=[Evidence(episode="e1", quote="记住进度", date=old, origin="stated")],
        source_agent="test", source_session="s1"))
    findings = HealthScanner(svc, since_use=lambda fact: 200).scan()
    forget = [f for f in findings if f.kind == "forget"]
    assert not forget, [f.summary for f in forget]


def test_closed_task_can_become_a_forget_candidate(rig):
    """Once the owner says it is done, the task decays like other short-lived types."""
    from membrain.health import HealthScanner
    from membrain.models import Evidence

    _, store, _, _, svc = rig
    old = date.today() - timedelta(days=200)
    fact = store.write_fact(store.draft_fact(
        type=FactType.TASK, subject="membrain", claim="Finished long ago",
        evidence=[Evidence(episode="e1", quote="记住进度", date=old, origin="stated")],
        source_agent="test", source_session="s1"))
    store.close_task(fact.id)
    findings = HealthScanner(svc, since_use=lambda f: 200).scan()
    assert any(f.kind == "forget" and fact.id in f.fact_ids for f in findings)


# ------------------------------------------------------- layer 2: summaries

def test_consolidate_writes_a_derived_summary(rig):
    _, store, _, _, svc = rig
    base = datetime(2026, 11, 4, 22, 0, tzinfo=timezone.utc)
    store.append_episodes([
        ep("e1", "记住现在的进度，明天继续", base),
        ep("a1", "好，我记下了", base + timedelta(minutes=1), role="assistant"),
        ep("e2", "先把时间戳修完", base + timedelta(minutes=2)),
    ])
    report = consolidate.consolidate(svc, session="s1")
    assert report.written, report.errors
    text = consolidate.read_summary(store, "s1")
    assert text is not None
    for heading in ("做了什么", "定了什么", "待办什么"):
        assert heading in text
    # It points back at its sources and declares itself derived.
    assert "e1" in text and "derived: true" in text


def test_summary_is_not_evidence_and_not_indexed(rig):
    """A model's restatement must never become a memory."""
    _, store, index, retriever, svc = rig
    base = datetime(2026, 11, 4, 22, 0, tzinfo=timezone.utc)
    store.append_episodes([ep("e1", "第一句话", base), ep("e2", "第二句话", base)])
    consolidate.consolidate(svc, session="s1")
    assert index.stats()["facts"] == 0, "a summary must not create facts"
    assert store.live_facts() == []


def test_summary_skips_trivial_sessions(rig):
    _, store, _, _, svc = rig
    store.append_episodes([ep("e1", "嗯", datetime.now(timezone.utc))])
    report = consolidate.consolidate(svc, session="s1")
    assert report.written == []


def test_idle_sessions_detects_quiet_conversations(rig):
    _, store, _, _, _ = rig
    now = datetime(2026, 11, 5, 12, 0, tzinfo=timezone.utc)
    store.append_episodes([ep("e1", "早前的对话", now - timedelta(hours=5)),
                           ep("e2", "早前的回复", now - timedelta(hours=5), role="assistant"),
                           ep("e3", "刚刚的对话", now - timedelta(minutes=5), session="s2")])
    idle = consolidate.idle_sessions(store, idle_minutes=60, now=now)
    assert "s1" in idle and "s2" not in idle


def _beverage(store, claim, episode_id):
    from membrain.models import Evidence

    return store.write_fact(store.draft_fact(
        type=FactType.PREFERENCE, subject="beverage", claim=claim,
        evidence=[Evidence(episode=episode_id, quote=claim, date=date.today(), origin="stated")],
        source_agent="test", source_session="s1"))


def test_related_but_distinct_memories_are_not_duplicates(rig):
    """「爱喝可乐」和「爱喝咖啡」实测相似度 0.90，但它们不是同一条。

    阈值定在 0.88 时，健康页把这类"同品类不同事实"报成重复；照着合并会把
    真实偏好揉成一句。这里的两个数字来自真实库的测量。
    """
    import numpy as np

    from membrain.health import HealthScanner

    _, store, _, _, svc = rig
    cola = _beverage(store, "Likes drinking cola", "e1")
    coffee = _beverage(store, "Loves drinking coffee", "e2")
    scanner = HealthScanner(svc, since_use=lambda fact: 0)

    def matrix(similarity: float):
        return np.array([[1.0, 0.0],
                         [similarity, (1 - similarity ** 2) ** 0.5]], dtype=np.float32)

    scanner._claim_matrix = lambda facts: matrix(0.90)
    assert scanner._find_duplicates([cola, coffee]) == [], "不同事实不能报重复"

    scanner._claim_matrix = lambda facts: matrix(0.97)
    assert scanner._find_duplicates([cola, coffee]), "同一句话的改写应该被识别"


def test_health_reuses_indexed_vectors_instead_of_embedding_again(rig):
    """全量健康扫描原先把每条记忆重新嵌入一次，这才是它要几十秒的原因。"""
    from membrain.health import HealthScanner

    _, store, index, retriever, _ = rig
    fact = _beverage(store, "Likes drinking cola", "e1")
    retriever.sync_fact(fact)

    class _NoNetwork:
        enabled = True
        model = retriever.embedder.model

        def embed(self, *args, **kwargs):
            raise AssertionError("索引里已有向量，扫描不应再请求嵌入")

    class _Service:
        def __init__(self):
            self.store, self.retriever, self.index = store, retriever, index
            self.embedder = _NoNetwork()

        def disabled_agents(self):
            return set()

    matrix = HealthScanner(_Service(), since_use=lambda f: 0)._claim_matrix([fact])
    assert matrix is not None and len(matrix) == 1


def _beverage(store, claim, episode_id):
    from membrain.models import Evidence

    return store.write_fact(store.draft_fact(
        type=FactType.PREFERENCE, subject="beverage", claim=claim,
        evidence=[Evidence(episode=episode_id, quote=claim, date=date.today(), origin="stated")],
        source_agent="test", source_session="s1"))


def test_related_but_distinct_memories_are_not_duplicates(rig):
    """「爱喝可乐」和「爱喝咖啡」实测相似度 0.90，但它们不是同一条。

    阈值定在 0.88 时，健康页把这类"同品类不同事实"报成重复；照着合并会把
    真实偏好揉成一句。这里的两个数字来自真实库的测量。
    """
    import numpy as np

    from membrain.health import HealthScanner

    _, store, _, _, svc = rig
    cola = _beverage(store, "Likes drinking cola", "e1")
    coffee = _beverage(store, "Loves drinking coffee", "e2")
    scanner = HealthScanner(svc, since_use=lambda fact: 0)

    def matrix(similarity: float):
        return np.array([[1.0, 0.0],
                         [similarity, (1 - similarity ** 2) ** 0.5]], dtype=np.float32)

    scanner._claim_matrix = lambda facts: matrix(0.90)
    assert scanner._find_duplicates([cola, coffee]) == [], "不同事实不能报重复"

    scanner._claim_matrix = lambda facts: matrix(0.97)
    assert scanner._find_duplicates([cola, coffee]), "同一句话的改写应该被识别"


def test_health_reuses_indexed_vectors_instead_of_embedding_again(rig):
    """全量健康扫描原先把每条记忆重新嵌入一次，这才是它要几十秒的原因。"""
    from membrain.health import HealthScanner

    _, store, index, retriever, _ = rig
    fact = _beverage(store, "Likes drinking cola", "e1")
    retriever.sync_fact(fact)

    class _NoNetwork:
        enabled = True
        model = retriever.embedder.model

        def embed(self, *args, **kwargs):
            raise AssertionError("索引里已有向量，扫描不应再请求嵌入")

    class _Service:
        def __init__(self):
            self.store, self.retriever, self.index = store, retriever, index
            self.embedder = _NoNetwork()

        def disabled_agents(self):
            return set()

    matrix = HealthScanner(_Service(), since_use=lambda f: 0)._claim_matrix([fact])
    assert matrix is not None and len(matrix) == 1
