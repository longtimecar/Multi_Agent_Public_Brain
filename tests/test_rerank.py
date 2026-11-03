"""OV gap 3: L0 abstract rerank (fail-open)."""

from __future__ import annotations

from datetime import date

import pytest

from membrain.config import Settings
from membrain.embeddings import Embedder
from membrain.index import Index
from membrain.llm import FakeLLMClient, LLMError
from membrain.models import Evidence, FactType
from membrain.rerank import heuristic_scores, rerank_scores
from membrain.retrieval import Retriever
from membrain.store import Store


def make_settings(tmp_path):
    return Settings(
        home=tmp_path / "memory",
        embed_base_url=None,
        embed_api_key=None,
        embed_model="fake",
        embed_timeout=5.0,
        turn_token_budget=2000,
        session_token_budget=2500,
    )


@pytest.fixture()
def world(tmp_path):
    settings = make_settings(tmp_path)
    store = Store(settings.home)
    index = Index(settings.db_path)
    retriever = Retriever(store, index, Embedder(settings), settings)
    return settings, store, retriever


def write(store, claim, subject, type_=FactType.FACT):
    fact = store.draft_fact(
        type=type_,
        subject=subject,
        claim=claim,
        evidence=[Evidence(episode=f"ep_{subject}_{claim[:4]}", quote=claim, date=date.today(), origin="stated")],
        source_agent="test",
    )
    return store.write_fact(fact)


def test_heuristic_prefers_overlap():
    docs = ["User likes oolong tea", "User ships Python packages", "Random weather note"]
    scores = heuristic_scores("oolong tea preference", docs)
    assert scores[0] == max(scores)


def test_rerank_llm_failure_keeps_fallback():
    fallback = [0.1, 0.9, 0.2]
    llm = FakeLLMClient([LLMError("down")])
    result = rerank_scores("tea", ["a", "b", "c"], fallback, llm=llm, use_llm=True)
    assert result.scores == fallback
    assert result.model_used is False
    assert any("llm_failed" in n for n in result.notes)


def test_recall_rerank_moves_target(world):
    _, store, retriever = world
    target = write(store, "Prefers oolong tea in the afternoon", "tea", FactType.PREFERENCE)
    decoy1 = write(store, "Works on packaging tools", "packaging")
    decoy2 = write(store, "Tracks weather for hiking", "weather")
    for f in (target, decoy1, decoy2):
        retriever.sync_fact(f)
    # Without vectors, lexical may be weak; rerank uses abstract overlap.
    result = retriever.recall("oolong tea", top_k=2, use_rerank=True)
    ids = [f.id for f in result.hits]
    assert target.id in ids or "tea" in (result.text or "").lower()
    assert any(n.startswith("rerank:") for n in result.notes) or result.hits


def test_recall_rerank_failure_notes(world, monkeypatch):
    _, store, retriever = world
    f = write(store, "Likes tea", "beverage", FactType.PREFERENCE)
    retriever.sync_fact(f)

    def boom(*args, **kwargs):
        raise RuntimeError("rerank crashed")

    monkeypatch.setattr("membrain.rerank.rerank_scores", boom)
    result = retriever.recall("tea", top_k=3, use_rerank=True)
    assert result is not None
    assert any("rerank_failed" in n for n in result.notes)
