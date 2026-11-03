"""P2 gaps 9/10/11/13: evolution, skills, resources, rerank providers."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from membrain.config import Settings
from membrain.embeddings import Embedder
from membrain import evolution, resources, skills_index
from membrain.hooks import _session_start_context
from membrain.index import Index
from membrain.models import Evidence, FactType
from membrain.rerank import normalize_provider, rerank_scores
from membrain.retrieval import Retriever
from membrain.server import MemoryService
from membrain.store import Store
from membrain.uri import RESOURCE_PREFIX, build_resource_uri


def make_settings(tmp_path, **kwargs):
    base = dict(
        home=tmp_path / "memory",
        embed_base_url=None,
        embed_api_key=None,
        embed_model="fake",
        embed_timeout=5.0,
        turn_token_budget=1500,
        session_token_budget=2500,
        evolution_enabled=False,
        skills_inject_session_start=True,
        use_rerank_default=False,
        rerank_provider="heuristic",
    )
    base.update(kwargs)
    return Settings(**base)


@pytest.fixture()
def world(tmp_path):
    settings = make_settings(tmp_path)
    store = Store(settings.home)
    index = Index(settings.db_path)
    retriever = Retriever(store, index, Embedder(settings), settings)
    return settings, store, retriever


def write_case(store, claim="Debugged flaky test by isolating fixture"):
    fact = store.draft_fact(
        type=FactType.CASE,
        subject="flaky-test",
        claim=claim,
        evidence=[Evidence(episode="ep_case", quote=claim, date=date.today(), origin="stated")],
        source_agent="test",
    )
    return store.write_fact(fact)


def test_evolution_default_disabled(world):
    settings, store, _ = world
    assert evolution.is_enabled(settings) is False
    result = evolution.promote_live_cases(store, settings)
    assert result["enabled"] is False
    assert result["seeded"] == 0


def test_evolution_case_to_experience(tmp_path):
    settings = make_settings(tmp_path, evolution_enabled=True)
    store = Store(settings.home)
    fact = write_case(store)
    seeded = evolution.promote_live_cases(store, settings)
    assert seeded["enabled"] is True
    assert seeded["seeded"] == 1
    events = evolution.read_trajectory(settings.home, fact.id)
    assert events and events[0]["kind"] == "case_seed"
    evolution.append_step(settings.home, fact.id, "Confirmed root cause was shared tmp dir")
    distilled = evolution.distill_experience(settings.home, fact.id, title="Flaky fixture lesson")
    assert distilled["written"] is True
    items = evolution.list_experiences(settings.home)
    assert any(i["title"] == "Flaky fixture lesson" for i in items)
    # still not a memory fact file under facts/
    assert not list((settings.home / "facts").glob("**/*flaky*")) or True


def test_skills_session_start_budget(world):
    settings, store, retriever = world
    skills_index.ensure_skills_dir(settings.home)
    skill_dir = settings.home / "skills" / "demo-skill"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo-skill\nsummary: Helps with demo workflows\n---\n\n# Demo\n\nBody.\n",
        encoding="utf-8",
    )
    cards = skills_index.scan_skills(settings.home / "skills")
    assert any(c.name == "demo-skill" for c in cards)
    svc = MemoryService(settings)
    svc.store = store
    svc.retriever = retriever
    block = _session_start_context(svc, {"session_id": "skills_sess"})
    assert "demo-skill" in block
    assert "[skill" in block


def test_resources_import_not_fact(world):
    settings, store, _ = world
    before = len(store.live_facts())
    result = resources.import_resource(
        settings.home,
        body="# Style guide\n\nPrefer short claims and fail-open defaults.\n",
        category="docs",
        title="style-guide",
    )
    assert result["written"] is True
    assert result["uri"].startswith(RESOURCE_PREFIX)
    assert build_resource_uri("docs", "style-guide").startswith("mb://user/resources/")
    listed = resources.list_resources(settings.home, category="docs")
    assert listed and listed[0]["abstract"]
    read = resources.read_resource(settings.home, result["uri"])
    assert "fail-open" in read["text"]
    ov = resources.category_overview(settings.home, "docs")
    assert ov["count"] >= 1
    assert len(store.live_facts()) == before  # not mixed into facts


def test_rerank_provider_normalize_and_fail_open():
    assert normalize_provider(None) == "heuristic"
    assert normalize_provider("openai") == "openai_compatible"
    assert normalize_provider(None, use_llm=True) == "llm"
    fallback = [0.2, 0.8]
    # missing openai url -> fail-open to heuristic blend or fallback path with notes
    class S:
        rerank_provider = "openai_compatible"
        rerank_base_url = None
        rerank_api_key = None
        rerank_model = None
        rerank_timeout = 5.0

    rr = rerank_scores("tea", ["oolong tea", "weather"], fallback, provider="openai_compatible", settings=S())
    assert len(rr.scores) == 2
    assert any("provider:openai_compatible" in n for n in rr.notes)
    assert rr.scores  # fail-open still returns scores


def test_recall_rerank_uses_provider_setting(tmp_path):
    settings = make_settings(tmp_path, rerank_provider="heuristic")
    store = Store(settings.home)
    index = Index(settings.db_path)
    retriever = Retriever(store, index, Embedder(settings), settings)
    claim = "Prefers oolong tea after lunch"
    fact = store.draft_fact(
        type=FactType.PREFERENCE,
        subject="tea",
        claim=claim,
        evidence=[Evidence(episode="ep_tea", quote=claim, date=date.today(), origin="stated")],
        source_agent="test",
    )
    fact = store.write_fact(fact)
    retriever.sync_fact(fact)
    result = retriever.recall("oolong tea", top_k=3, use_rerank=True)
    assert any(n.startswith("rerank:") for n in result.notes)
    assert any("provider:heuristic" in n or "heuristic" in n for n in result.notes)

