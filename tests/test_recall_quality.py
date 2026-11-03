from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from membrain.config import QWEN_RETRIEVAL_INSTRUCTION, load_settings
from membrain.embeddings import Embedder, EmbeddingError
from membrain.hooks import _prompt_context
from membrain.models import FactType
from membrain.retrieval import Retriever
from test_core import FakeEmbedder, make_settings, world, write


def test_query_instruction_is_model_specific_and_configurable(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMBRAIN_ENV_FILE", str(tmp_path / "missing"))
    monkeypatch.setenv("MEMBRAIN_EMBED_MODEL", "Qwen/Qwen3-Embedding-8B")
    monkeypatch.delenv("MEMBRAIN_EMBED_QUERY_INSTRUCTION", raising=False)
    assert load_settings().query_instruction == QWEN_RETRIEVAL_INSTRUCTION
    monkeypatch.setenv("MEMBRAIN_EMBED_QUERY_INSTRUCTION", "Find relevant personal facts")
    assert load_settings().query_instruction == "Find relevant personal facts"
    monkeypatch.delenv("MEMBRAIN_EMBED_QUERY_INSTRUCTION")
    monkeypatch.setenv("MEMBRAIN_EMBED_MODEL", "another-embedding-model")
    assert load_settings().query_instruction == ""


@pytest.mark.parametrize("value,expected", [("0.61", 0.61), ("bad", 0.41), ("nan", 0.41), ("2", 0.41)])
def test_semantic_floor_configuration(tmp_path, monkeypatch, value, expected):
    monkeypatch.setenv("MEMBRAIN_ENV_FILE", str(tmp_path / "missing"))
    monkeypatch.setenv("MEMBRAIN_SEMANTIC_FLOOR", value)
    assert load_settings().semantic_floor == expected


def test_query_cache_keys_include_instruction_and_facts_remain_unprefixed(tmp_path, monkeypatch):
    settings = dataclasses.replace(make_settings(tmp_path, embed=True), embed_model="Qwen3-Embedding-test")
    embedder = Embedder(settings)
    calls = []

    def post(texts):
        calls.append(list(texts))
        return [[1.0, float(i + 1)] for i in range(len(texts))]

    monkeypatch.setattr(embedder, "_post", post)
    embedder.embed(["same text"])
    embedder.embed_queries(["same text"])
    embedder.embed_queries(["same text"])
    assert calls == [
        ["same text"],
        [f"Instruct: {QWEN_RETRIEVAL_INSTRUCTION}\nQuery: same text"],
    ]
    embedder.settings = dataclasses.replace(settings, embed_query_instruction="A different task")
    embedder.embed_queries(["same text"])
    assert calls[-1] == ["Instruct: A different task\nQuery: same text"]
    embedder.settings = dataclasses.replace(settings, embed_model="other-model")
    embedder.embed_queries(["same text"])
    assert len(calls) == 3  # the unprefixed input was already cached


def test_similarity_floor_filters_semantic_and_lexical_matches(world):
    settings, store, index, _ = world
    settings = dataclasses.replace(settings, semantic_floor=0.5)

    class ScoredEmbedder:
        enabled = True
        model = "scores"

        def embed(self, texts):
            scores = [0.499 if "lower" in t else 0.5 for t in texts]
            return np.array([[s, (1.0 - s * s) ** 0.5] for s in scores], dtype=np.float32)

        def embed_queries(self, texts):
            return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)

    retriever = Retriever(store, index, ScoredEmbedder(), settings)
    below = write(store, "lower needle", "low")
    equal = write(store, "equal needle", "high")
    retriever.sync_fact(below)
    retriever.sync_fact(equal)
    result = retriever.recall("needle")
    assert result.mode == "hybrid"
    assert [f.id for f in result.hits] == [equal.id]


def test_short_prompt_uses_token_estimate_and_skips_only_trivial_input(tmp_path):
    calls = []
    def recall(prompt, budget):
        calls.append((prompt, budget))
        return SimpleNamespace(text="found")

    svc = SimpleNamespace(settings=make_settings(tmp_path), retriever=SimpleNamespace(recall=recall))
    assert _prompt_context(svc, {"prompt": " "}) == ""
    assert _prompt_context(svc, {"prompt": "ok"}) == ""
    assert _prompt_context(svc, {"prompt": "饮食"}) == "found"
    assert _prompt_context(svc, {"prompt": "food?"}) == "found"
    assert calls == [("饮食", 1500), ("food?", 1500)]


def test_unrelated_semantic_fact_is_not_injected(world):
    settings, store, index, _ = world
    retriever = Retriever(store, index, FakeEmbedder(), settings)
    write(store, "Does not eat spicy food", "dietary", FactType.PREFERENCE)
    retriever.reindex()
    result = retriever.recall("memory project")
    assert result.mode == "hybrid"
    assert result.hits == [] and result.text == ""


def test_unvectorized_fact_retains_lexical_fallback(world):
    settings, store, index, _ = world
    retriever = Retriever(store, index, FakeEmbedder(), settings)
    indexed = write(store, "Works on an agent memory project", "projects")
    retriever.sync_fact(indexed)
    lexical_only = write(store, "Does not eat spicy food", "dietary", FactType.PREFERENCE)
    index.upsert(lexical_only, store.fact_path(lexical_only.id).read_text(encoding="utf-8"))
    index.commit()
    result = retriever.recall("spicy food")
    assert result.mode == "hybrid"
    assert lexical_only.id in {fact.id for fact in result.hits}


def test_embedding_failure_retains_lexical_fallback(world):
    settings, store, index, _ = world
    fake = FakeEmbedder()
    retriever = Retriever(store, index, fake, settings)
    fact = write(store, "Does not eat spicy food", "dietary", FactType.PREFERENCE)
    retriever.sync_fact(fact)

    class FailingEmbedder(FakeEmbedder):
        def embed_queries(self, texts):
            raise EmbeddingError("unavailable")

    failing = Retriever(store, index, FailingEmbedder(), settings)
    result = failing.recall("spicy food")
    assert result.mode == "lexical"
    assert fact.id in {hit.id for hit in result.hits}
    assert "embedding request failed; lexical only" in result.notes


@pytest.mark.parametrize("query", ["查询一下我的记忆，我平时爱干嘛", "介绍一下我的喜好"])
def test_general_preference_question_recalls_saved_preferences(world, query):
    settings, store, index, _ = world
    retriever = Retriever(store, index, FakeEmbedder(), settings)
    preference = write(store, "Likes eating chili peppers", "dietary", FactType.PREFERENCE)
    project = write(store, "Works on a memory project", "projects", FactType.PROJECT)
    retriever.reindex()
    result = retriever.recall(query)
    assert preference.id in {f.id for f in result.hits}
    assert project.id not in {f.id for f in result.hits}
    assert result.mode == "overview"


def test_general_memory_overview_respects_lifecycle_exclusions_and_budget(world):
    settings, store, index, _ = world
    retriever = Retriever(store, index, FakeEmbedder(), settings)
    preference = write(store, "Likes eating chili peppers", "dietary", FactType.PREFERENCE)
    project = write(store, "Works on a memory project", "projects", FactType.PROJECT)
    expired = write(store, "An old preference", "old", FactType.PREFERENCE)
    store.retract(expired.id, "user correction")
    retriever.reindex()
    result = retriever.recall("查看我所有记忆", exclude_ids={project.id}, top_k=1, budget=1500)
    assert [f.id for f in result.hits] == [preference.id]
    assert result.tokens <= 1500


def test_specific_question_about_memory_project_does_not_request_an_overview(world):
    settings, store, index, _ = world
    retriever = Retriever(store, index, FakeEmbedder(), settings)
    write(store, "Works on a memory project", "projects", FactType.PROJECT)
    retriever.reindex()
    assert retriever.recall("我的记忆项目是什么").mode == "hybrid"
