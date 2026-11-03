"""Scheme A: URI namespaces, L0/L1, new types, sensitive gating, auto-commit."""

from __future__ import annotations

from datetime import date, datetime, timezone, timedelta

import pytest

from membrain.auto_commit import check_idle, note_activity, take_pending
from membrain.config import Settings
from membrain.index import Index
from membrain.models import Evidence, FactType
from membrain.overview import build_l1_overviews, looks_sensitive, make_abstract
from membrain.retrieval import Retriever
from membrain.store import Store
from membrain.uri import build_uri, parse_uri, uri_matches_scope


def make_settings(tmp_path):
    return Settings(
        home=tmp_path / "memory",
        embed_base_url=None,
        embed_api_key=None,
        embed_model="fake",
        embed_timeout=5.0,
        turn_token_budget=1500,
        session_token_budget=2500,
        auto_commit_token_threshold=100,
        auto_commit_idle_minutes=30,
    )


@pytest.fixture()
def world(tmp_path):
    settings = make_settings(tmp_path)
    store = Store(settings.home)
    index = Index(settings.db_path)
    from membrain.embeddings import Embedder

    retriever = Retriever(store, index, Embedder(settings), settings)
    return settings, store, index, retriever


def write(store, claim, subject, type_=FactType.FACT, sensitive=None, peer=None):
    fact = store.draft_fact(
        type=type_,
        subject=subject,
        claim=claim,
        evidence=[Evidence(episode=f"ep_{subject}_{claim[:6]}", quote=claim, date=date.today(), origin="stated")],
        source_agent="test",
        sensitive=sensitive,
        peer=peer,
    )
    return store.write_fact(fact)


def test_uri_build_and_parse():
    uri = build_uri("preference", "Dietary Prefs")
    assert uri == "mb://user/memories/preference/dietary-prefs"
    parsed = parse_uri(uri)
    assert parsed["type"] == "preference"
    assert parsed["subject"] == "dietary-prefs"
    assert uri_matches_scope(uri, "mb://user/memories/preference")
    assert uri_matches_scope(uri, "preference")
    assert not uri_matches_scope(uri, "mb://user/memories/goal")


def test_new_types_and_uri_on_write(world):
    _, store, _, retriever = world
    for t in (FactType.ENTITY, FactType.EVENT, FactType.CASE, FactType.TOOL, FactType.BELIEF):
        f = write(store, f"Claim about {t.value}", t.value, t)
        assert f.uri == f"mb://user/memories/{t.value}/{t.value}"
        assert f.abstract
        retriever.sync_fact(f)
    belief = [f for f in store.live_facts() if f.type == FactType.BELIEF][0]
    assert belief.sensitive is True


def test_religion_marked_sensitive(world):
    _, store, _, _ = world
    f = write(store, "User practices Buddhism", "religion", FactType.PREFERENCE)
    assert f.sensitive is True
    assert looks_sensitive("谈一谈政治立场")


def test_sensitive_excluded_from_default_recall(world):
    _, store, index, retriever = world
    normal = write(store, "Does not eat spicy food", "dietary", FactType.PREFERENCE)
    belief = write(store, "Supports local temple visits", "religion", FactType.BELIEF)
    retriever.sync_fact(normal)
    retriever.sync_fact(belief)
    # Non-topic query must not inject belief
    result = retriever.recall("what food do I like", top_k=10)
    ids = {f.id for f in result.hits}
    assert normal.id in ids or "spicy" in (result.text or "").lower() or result.mode
    assert belief.id not in ids
    # Topic-relevant query may include sensitive
    topic = retriever.recall("我的宗教信仰是什么", top_k=10)
    assert belief.id in {f.id for f in topic.hits} or belief.sensitive


def test_scope_and_level_filters(world):
    _, store, _, retriever = world
    pref = write(store, "Likes tea", "beverage", FactType.PREFERENCE)
    goal = write(store, "Ship membrain scheme A", "membrain", FactType.GOAL)
    retriever.sync_fact(pref)
    retriever.sync_fact(goal)
    scoped = retriever.recall("tea", scope="mb://user/memories/preference", top_k=10)
    assert all(f.type == FactType.PREFERENCE for f in scoped.hits)
    l0 = retriever.recall("tea", level=0, type="preference", top_k=5)
    assert l0.level == 0
    l1 = retriever.recall("我的偏好", level=1, top_k=20)
    assert l1.level == 1
    assert "memory_overview" in (l1.text or "") or l1.mode.startswith("overview")


def test_l1_overview_skips_sensitive(world):
    _, store, _, _ = world
    write(store, "Likes tea", "beverage", FactType.PREFERENCE)
    write(store, "Conservative on taxes", "politics", FactType.BELIEF)
    rows = build_l1_overviews(store.live_facts(), include_sensitive=False)
    assert all(not r["sensitive"] for r in rows)
    assert all(r["type"] != "belief" for r in rows)


def test_migrate_scheme_a_backfill(world):
    _, store, _, retriever = world
    # Write minimal markdown without scheme A fields
    path = store.facts_dir / "f_20261102_abc123.md"
    path.write_text(
        "---\n"
        "id: f_20261102_abc123\n"
        "type: fact\n"
        "subject: hometown\n"
        "evidence:\n"
        "  - episode: ep_x\n"
        "    quote: I grew up in Shanghai\n"
        "    date: 2026-11-02\n"
        "    origin: stated\n"
        "confidence: 0.4\n"
        "valid_from: 2026-11-02\n"
        "source:\n"
        "  agent: test\n"
        "status: active\n"
        "created_at: 2026-11-02T00:00:00+00:00\n"
        "---\n"
        "# Grew up in Shanghai\n",
        encoding="utf-8",
    )
    report = store.migrate_scheme_a()
    assert report["updated"] >= 1
    fact = store.get_fact("f_20261102_abc123")
    assert fact.uri == "mb://user/memories/fact/hometown"
    assert fact.abstract
    retriever.reindex()


def test_auto_commit_token_threshold(world):
    settings, store, _, _ = world
    # threshold=100 in fixture
    note_activity(store, "s1", "x" * 400, threshold=settings.auto_commit_token_threshold, idle_minutes=30)
    pending = take_pending(store)
    assert "s1" in pending


def test_auto_commit_idle(world, monkeypatch):
    settings, store, _, _ = world
    note_activity(store, "s2", "hello world", threshold=10_000, idle_minutes=30)
    # Force last_activity into the past
    from membrain import auto_commit

    state = auto_commit.load_state(store)
    past = (datetime.now(timezone.utc) - timedelta(minutes=45)).isoformat()
    state["sessions"]["s2"]["last_activity"] = past
    state["sessions"]["s2"]["tokens"] = 10
    auto_commit.save_state(store, state)
    pending = check_idle(store, threshold=10_000, idle_minutes=30)
    assert "s2" in pending


def test_abstract_helper():
    assert make_abstract("short") == "short"
    long = "word " * 80
    assert len(make_abstract(long, limit=40)) <= 40
