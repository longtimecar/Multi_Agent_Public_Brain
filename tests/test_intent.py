"""OV gap 2: IntentAnalyzer → TypedQuery (fail-open, default off)."""

from __future__ import annotations

import pytest

from membrain.config import Settings
from membrain.embeddings import Embedder
from membrain.index import Index
from membrain.intent import IntentAnalyzer, TypedQuery, looks_like_chitchat
from membrain.llm import FakeLLMClient, LLMError
from membrain.models import Evidence, FactType
from membrain.overview import make_abstract
from membrain.retrieval import Retriever
from membrain.store import Store


def make_settings(tmp_path):
    return Settings(
        home=tmp_path / "memory",
        embed_base_url=None,
        embed_api_key=None,
        embed_model="fake",
        embed_timeout=5.0,
        turn_token_budget=1500,
        session_token_budget=2500,
    )


@pytest.fixture()
def world(tmp_path):
    settings = make_settings(tmp_path)
    store = Store(settings.home)
    index = Index(settings.db_path)
    retriever = Retriever(store, index, Embedder(settings), settings)
    return settings, store, retriever


def write(store, claim, subject, type_=FactType.PREFERENCE):
    fact = store.draft_fact(
        type=type_,
        subject=subject,
        claim=claim,
        evidence=[Evidence(episode=f"ep_{subject}", quote=claim, date=__import__("datetime").date.today(), origin="stated")],
        source_agent="test",
    )
    return store.write_fact(fact)


def test_chitchat_zero_queries_with_llm():
    llm = FakeLLMClient([{"reasoning": "greeting", "queries": []}])
    plan = IntentAnalyzer(llm).analyze("你好")
    assert plan.queries == []
    assert looks_like_chitchat("hi")


def test_preference_emits_memory_query():
    llm = FakeLLMClient([{
        "reasoning": "ask prefs",
        "queries": [{"query": "用户的饮食偏好", "context_type": "memory", "intent": "prefs", "priority": 2}],
    }])
    plan = IntentAnalyzer(llm).analyze("我平时爱吃什么")
    assert len(plan.queries) >= 1
    assert plan.queries[0].context_type == "memory"


def test_llm_failure_fallback_original_query():
    llm = FakeLLMClient([LLMError("boom")])
    plan = IntentAnalyzer(llm).analyze("我的项目进度")
    assert plan.fallback is True
    assert len(plan.queries) == 1
    assert plan.queries[0].query == "我的项目进度"


def test_recall_use_intent_default_off_unchanged(world):
    _, store, retriever = world
    f = write(store, "Likes oolong tea", "beverage")
    retriever.sync_fact(f)
    a = retriever.recall("tea", top_k=5)
    b = retriever.recall("tea", top_k=5, use_intent=False)
    assert a.mode == b.mode


def test_recall_use_intent_chitchat_skips(world):
    _, store, retriever = world
    f = write(store, "Likes oolong tea", "beverage")
    retriever.sync_fact(f)
    llm = FakeLLMClient([{"reasoning": "hi", "queries": []}])
    result = retriever.recall("你好", top_k=5, use_intent=True, llm=llm)
    assert result.mode == "intent-skip"
    assert result.hits == []
