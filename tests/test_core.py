"""End-to-end checks for the seven properties the design claims.

Uses a fake embedder so semantic recall is testable without an API call: the
"semantic" space is a hand-built topic vector, which is enough to prove the
hybrid path fuses lexical and vector results correctly.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import date, timedelta

import numpy as np
import pytest

from membrain.config import Settings, load_dotenv, load_settings
from membrain.confidence import compute_confidence, is_promoted, render_hedged
from membrain.models import Evidence, Fact, FactType, Status
from membrain.retrieval import Retriever
from membrain.store import Store


def make_settings(tmp_path, embed: bool = False):
    return Settings(
        home=tmp_path / "memory",
        embed_base_url="http://fake" if embed else None,
        embed_api_key=None,
        embed_model="fake-32",
        embed_timeout=5.0,
        turn_token_budget=1500,
        session_token_budget=2500,
    )


class FakeEmbedder:
    """Topic vector: dietary / work / style. Enough to exercise hybrid fusion."""

    enabled = True
    model = "fake-32"

    def __init__(self) -> None:
        self.calls = 0

    @staticmethod
    def _vec(text: str) -> list[float]:
        t = text.lower()
        v = np.zeros(8, dtype=np.float32)
        if any(k in t for k in ("spicy", "辣", "diet", "food", "吃", "饮食")):
            v[0] = 1.0
        if any(k in t for k in ("project", "memory", "work", "agent")):
            v[1] = 1.0
        if any(k in t for k in ("terse", "short", "style", "em dash")):
            v[2] = 1.0
        v[3] = 0.05  # floor so nothing is exactly orthogonal
        return v.tolist()

    def embed(self, texts):
        self.calls += 1
        return np.vstack([np.asarray(self._vec(t), dtype=np.float32) / (np.linalg.norm(self._vec(t)) + 1e-9) for t in texts])

    def embed_queries(self, texts):
        return self.embed(texts)


@pytest.fixture()
def world(tmp_path):
    settings = make_settings(tmp_path)
    store = Store(settings.home)
    from membrain.index import Index

    index = Index(settings.db_path)
    from membrain.embeddings import Embedder

    retriever = Retriever(store, index, Embedder(settings), settings)
    return settings, store, index, retriever


def write(store, claim, subject, type_=FactType.FACT, days_ago=0, quote=None, origin="stated"):
    when = date.today() - timedelta(days=days_ago)
    fact = store.draft_fact(
        type=type_,
        subject=subject,
        claim=claim,
        evidence=[Evidence(episode=f"ep_{claim[:8]}_{days_ago}", quote=quote or claim, date=when, origin=origin)],
        source_agent="test",
    )
    return store.write_fact(fact)


# 1. Sharing: one store, any client -----------------------------------------

def test_fact_roundtrips_through_disk(world):
    _, store, _, _ = world
    written = write(store, "Does not eat spicy food", "dietary", FactType.PREFERENCE)
    reloaded = Store(store.home).get_fact(written.id)
    assert reloaded.claim == written.claim
    assert reloaded.evidence[0].quote == written.evidence[0].quote
    assert reloaded.source.agent == "test"


# 2. Append-only: nothing is overwritten -------------------------------------

def test_write_is_immutable(world):
    _, store, _, _ = world
    first = write(store, "Uses Rust", "language")
    with pytest.raises(Exception):
        store.write_fact(first)  # same id twice must be refused
    second = write(store, "Uses Python", "language")
    store.supersede(first.id, second.id)
    assert store.get_fact(first.id).status == Status.SUPERSEDED
    assert store.get_fact(first.id).valid_to is not None
    assert store.get_fact(second.id).status == Status.ACTIVE


def test_ledger_is_a_fold_not_a_rewrite(world):
    _, store, _, _ = world
    fact = write(store, "Uses Rust", "language")
    store.retract(fact.id, "user said wrong")
    events = store._ledger_state()[fact.id]
    assert [e["op"] for e in events] == ["retract"]
    # The canonical file is untouched: only the ledger changed.
    assert "Uses Rust" in store.fact_path(fact.id).read_text(encoding="utf-8")


# 3. Confidence is computed, not asserted ------------------------------------

def _single(quote, type_=FactType.PREFERENCE, origin="stated"):
    ev = [Evidence(episode="e1", quote=quote, date=date.today(), origin=origin)]
    return Fact(id="f_20260101_abcdef", type=type_, subject="style", claim="A claim",
                evidence=ev, valid_from=date.today())


def test_single_hedged_episode_is_capped_and_hedged():
    fact = _single("也许我不用破折号", FactType.CONSTRAINT)
    assert compute_confidence(fact) <= 0.40
    assert not is_promoted(fact)


def test_single_inferred_episode_is_capped_and_hedged():
    fact = _single("I never use em dashes", FactType.FACT, origin="inferred")
    assert compute_confidence(fact) <= 0.40
    assert not is_promoted(fact)


@pytest.mark.parametrize("quote", ["我对花生过敏", "I am allergic to peanuts.", "I work as a backend engineer."])
@pytest.mark.parametrize("type_", [FactType.PREFERENCE, FactType.FACT, FactType.GOAL])
def test_single_plain_statement_is_promoted(quote, type_):
    fact = _single(quote, type_)
    assert compute_confidence(fact) >= 0.60
    assert is_promoted(fact)


@pytest.mark.parametrize("quote", ["也许我下周要去参加比赛", "Maybe I will compete next week", "if it rains I stay home"])
def test_single_hedged_statement_is_not_promoted(quote):
    assert not is_promoted(_single(quote, FactType.FACT))


def test_imported_rule_that_mentions_uncertainty_is_still_promoted():
    rule = "遇到不认识的、不确定的、或可能已经过时的问题，必须先联网检索"
    assert is_promoted(_single(rule, FactType.PREFERENCE, origin="imported"))


def test_widen_to_clause_restores_a_dropped_hedge():
    from membrain.text import widen_to_clause
    text = "我平时爱喝点小酒，我不喜欢咖啡那样子，也许我下周需要去参加一个比武大赛"
    assert widen_to_clause(text, "参加一个比武大赛") == "也许我下周需要去参加一个比武大赛"
    # A plain clause next to a hedged one stays untouched.
    assert widen_to_clause(text, "我不喜欢咖啡那样子") == "我不喜欢咖啡那样子"


def test_single_stated_observation_can_be_reported_as_a_past_statement(world):
    _, store, _, _ = world
    fact = write(store, "Likes eating chili peppers", "dietary", FactType.PREFERENCE,
                 quote="Maybe I like eating chili peppers.")
    note = render_hedged(fact)
    assert "unconfirmed" in note.lower()
    assert "previous" in note.lower()
    assert "verify before relying" not in note.lower()


def test_same_day_corroboration_does_not_promote():
    today = date.today()
    ev = [Evidence(episode=f"e{i}", quote="q", date=today) for i in range(5)]
    fact = Fact(
        id="f_20260101_abcdef",
        type=FactType.PREFERENCE,
        subject="dietary",
        claim="Hates spicy food",
        evidence=ev,
        valid_from=today,
    )
    burst = compute_confidence(fact)
    spread = [
        Evidence(episode=f"s{i}", quote="q", date=today - timedelta(days=7 * i))
        for i in range(3)
    ]
    fact_spread = fact.model_copy(update={"evidence": spread})
    assert burst < compute_confidence(fact_spread)


# 4. Retrieval fuses lexical + semantic --------------------------------------

def test_lexical_recall_finds_overlap(world):
    _, store, _, retriever = world
    write(store, "Does not eat spicy food", "dietary", FactType.PREFERENCE)
    retriever.reindex()
    result = retriever.recall("spicy food")
    assert any("spicy" in f.claim for f in result.hits)
    assert result.mode == "lexical"


def test_semantic_recall_without_keyword_overlap(world):
    settings, store, index, _ = world
    from membrain.embeddings import Embedder
    from membrain.index import Index

    fake = FakeEmbedder()
    retriever = Retriever(store, index, fake, settings)  # type: ignore[arg-type]
    write(store, "Does not eat spicy food", "dietary", FactType.PREFERENCE)
    write(store, "Works on an agent memory project", "projects", FactType.PROJECT)
    retriever.reindex()
    result = retriever.recall("有什么饮食上的注意事项吗")
    # The query shares no words with the fact, so only the vector arm can rank
    # it first. Checking membership alone would pass even if ranking were noise.
    assert result.hits and "spicy" in result.hits[0].claim, result.text
    assert result.mode == "hybrid"


# 5. Budget: never exceed the cap, never truncate mid-item ------------------

def test_injection_respects_budget(world):
    _, store, _, retriever = world
    for i in range(60):
        write(store, f"Preference number {i} about a wide range of daily habits", f"topic{i}")
    retriever.reindex()
    result = retriever.recall("preference", top_k=50, budget=200)
    assert result.tokens <= 200
    assert result.text.count("\n-") == len(result.hits)


# 6. Conflicts surface, never auto-resolve ----------------------------------

def test_contested_surfaces_both_and_hedges(world):
    _, store, _, retriever = world
    a = write(store, "Does not eat spicy food", "dietary", FactType.PREFERENCE)
    b = write(store, "Loves spicy food", "dietary", FactType.PREFERENCE)
    store.contest([a.id, b.id], "direct contradiction")
    retriever.reindex()
    assert store.get_fact(a.id).status == Status.CONTESTED
    assert store.get_fact(b.id).status == Status.CONTESTED
    result = retriever.recall("spicy")
    claims = " ".join(f.claim for f in result.hits)
    assert "Does not eat spicy food" in claims and "Loves spicy food" in claims
    assert "CONTESTED" in result.text


def test_resolution_is_explicit(world):
    _, store, _, retriever = world
    a = write(store, "Does not eat spicy food", "dietary", FactType.PREFERENCE)
    b = write(store, "Loves spicy food", "dietary", FactType.PREFERENCE)
    store.contest([a.id, b.id], "conflict")
    store.resolve(keep=b.id, drop=a.id)
    assert store.get_fact(a.id).status == Status.RETRACTED
    assert store.get_fact(b.id).status == Status.ACTIVE


# 7. Durability: the index is disposable ------------------------------------

def test_index_rebuilds_from_markdown(world):
    settings, store, index, retriever = world
    write(store, "Does not eat spicy food", "dietary", FactType.PREFERENCE)
    retriever.reindex()
    assert index.stats()["facts"] == 1

    index.close()
    settings.db_path.unlink()

    from membrain.index import Index

    fresh = Index(settings.db_path)
    fresh_retriever = Retriever(store, fresh, retriever.embedder, settings)
    assert fresh_retriever.reindex()["facts"] == 1
    assert any("spicy" in f.claim for f in fresh_retriever.recall("spicy").hits)


# 8. Evidence discipline ----------------------------------------------------

def test_evidence_quote_is_required(world):
    _, store, _, _ = world
    with pytest.raises(Exception):
        store.draft_fact(
            type=FactType.FACT,
            subject="x",
            claim="c",
            evidence=[Evidence(episode="e", quote="  ", date=date.today())],
        )


def test_duplicate_episode_evidence_rejected():
    with pytest.raises(Exception):
        Fact(
            id="f_20260101_abcdef",
            type=FactType.FACT,
            subject="x",
            claim="c",
            evidence=[
                Evidence(episode="same", quote="a", date=date.today()),
                Evidence(episode="same", quote="b", date=date.today()),
            ],
            valid_from=date.today(),
        )


# 9. Conflict detection is mechanical and non-resolving ---------------------

def test_detects_negation_conflict(world):
    _, store, _, retriever = world
    a = write(store, "Does not eat spicy food", "dietary", FactType.PREFERENCE)
    b = write(store, "Eats spicy food often", "dietary", FactType.PREFERENCE)
    found = retriever.scan_conflicts()
    assert {c.a.id for c in found} == {a.id} or {c.b.id for c in found} == {a.id}
    assert store.get_fact(a.id).status == Status.CONTESTED
    assert store.get_fact(b.id).status == Status.CONTESTED


def test_no_false_positive_on_unrelated_subjects(world):
    _, store, _, retriever = world
    write(store, "Does not eat spicy food", "dietary", FactType.PREFERENCE)
    write(store, "Likes Go for services", "language", FactType.PREFERENCE)
    assert retriever.scan_conflicts() == []


def test_detection_never_picks_a_winner(world):
    _, store, _, retriever = world
    a = write(store, "Does not eat spicy food", "dietary", FactType.PREFERENCE)
    b = write(store, "Eats spicy food", "dietary", FactType.PREFERENCE)
    retriever.scan_conflicts()
    # Both still live and contested: no automatic supersession.
    for fid in (a.id, b.id):
        f = store.get_fact(fid)
        assert f.status == Status.CONTESTED and f.valid_to is None


def test_scan_is_idempotent(world):
    _, store, _, retriever = world
    write(store, "Does not eat spicy food", "dietary", FactType.PREFERENCE)
    write(store, "Eats spicy food", "dietary", FactType.PREFERENCE)
    first = retriever.scan_conflicts()
    second = retriever.scan_conflicts()
    assert len(first) >= 1 and second == []


# 10. Credentials come from .env, never from code ---------------------------

def test_dotenv_parsing(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\n"
        "\n"
        "PLAIN=value\n"
        "export EXPORTED=yes\n"
        'DOUBLE="has spaces # not a comment"\n'
        "SINGLE='single'\n"
        "INLINE=abc # trailing comment\n"
        "URL=https://api.example.com/v1#frag\n"
        "EMPTY=\n"
        "not a pair\n",
        encoding="utf-8",
    )
    environ: dict[str, str] = {}
    assert load_dotenv(env_file, environ) is True
    assert environ["PLAIN"] == "value"
    assert environ["EXPORTED"] == "yes"
    assert environ["DOUBLE"] == "has spaces # not a comment"
    assert environ["SINGLE"] == "single"
    assert environ["INLINE"] == "abc"
    assert environ["URL"] == "https://api.example.com/v1#frag"
    assert environ["EMPTY"] == ""
    assert "not a pair" not in environ


def test_real_environment_overrides_dotenv(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("KEY=from_file\nBLANK=from_file\n", encoding="utf-8")
    environ = {"KEY": "from_env", "BLANK": ""}
    load_dotenv(env_file, environ)
    assert environ["KEY"] == "from_env"
    # An empty variable counts as unset, so the file fills it.
    assert environ["BLANK"] == "from_file"


def test_missing_dotenv_is_not_an_error(tmp_path):
    environ: dict[str, str] = {}
    assert load_dotenv(tmp_path / "absent.env", environ) is False
    assert environ == {}


def test_settings_read_credentials_from_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "MEMBRAIN_EMBED_BASE_URL=https://embed.example/v1\n"
        "MEMBRAIN_EMBED_API_KEY=embed-secret\n"
        "MEMBRAIN_EMBED_MODEL=bge-m3\n"
        "MEMBRAIN_LLM_BASE_URL=https://llm.example\n"
        "MEMBRAIN_LLM_API_KEY=llm-secret\n"
        "MEMBRAIN_LLM_MODEL=some-model\n"
        "MEMBRAIN_LLM_API_FORMAT=Anthropic\n"
        f"MEMBRAIN_OPENVIKING_ROOT={(tmp_path / 'openviking').as_posix()}\n",
        encoding="utf-8",
    )
    # Blank each key through monkeypatch: blank counts as unset, so the file
    # fills it, and monkeypatch restores the real environment afterwards.
    for key in ("MEMBRAIN_EMBED_BASE_URL", "MEMBRAIN_EMBED_API_KEY", "MEMBRAIN_EMBED_MODEL",
                "MEMBRAIN_LLM_BASE_URL", "MEMBRAIN_LLM_API_KEY", "MEMBRAIN_LLM_MODEL",
                "MEMBRAIN_LLM_API_FORMAT", "MEMBRAIN_OPENVIKING_ROOT"):
        monkeypatch.setenv(key, "")
    monkeypatch.setenv("MEMBRAIN_ENV_FILE", str(env_file))
    s = load_settings()
    assert s.env_file == env_file
    assert s.embeddings_enabled and s.embed_model == "bge-m3"
    assert s.embed_api_key == "embed-secret"
    assert s.llm_enabled and s.llm_api_format == "anthropic"
    assert s.llm_api_key == "llm-secret"
    assert s.openviking_root == (tmp_path / "openviking").resolve()


def test_describe_never_prints_keys(tmp_path):
    s = dataclasses.replace(
        make_settings(tmp_path, embed=True),
        embed_api_key="embed-secret",
        llm_base_url="https://llm.example",
        llm_api_key="llm-secret",
        llm_model="some-model",
    )
    text = json.dumps(s.describe())
    assert "embed-secret" not in text and "llm-secret" not in text
    assert "fake-32" in text and "some-model" in text


# 11. Facts written after the first index build still get vectors -----------

def test_semantic_recall_for_fact_written_after_index_build(world):
    settings, store, index, _ = world
    retriever = Retriever(store, index, FakeEmbedder(), settings)  # type: ignore[arg-type]
    retriever.reindex()  # empty store: no vectors yet, nothing recorded
    # Facts written afterwards are indexed one by one, the way every real write is.
    for fact in (
        write(store, "Works on an agent memory project", "projects", FactType.PROJECT),
        write(store, "Does not eat spicy food", "dietary", FactType.PREFERENCE),
    ):
        retriever.sync_fact(fact)
    result = retriever.recall("有什么饮食上的注意事项吗")
    assert result.mode == "hybrid", result.notes
    assert result.hits and "spicy" in result.hits[0].claim


def test_model_change_falls_back_until_reindex(world):
    settings, store, index, _ = world
    fake = FakeEmbedder()
    retriever = Retriever(store, index, fake, settings)  # type: ignore[arg-type]
    fact = write(store, "Does not eat spicy food", "dietary", FactType.PREFERENCE)
    retriever.sync_fact(fact)

    switched = FakeEmbedder()
    switched.model = "other-model"
    other = Retriever(store, index, switched, settings)  # type: ignore[arg-type]
    other.sync_fact(fact)  # must not mix vector spaces
    result = other.recall("饮食")
    assert result.mode == "lexical"
    assert any("different embedding model" in n for n in result.notes)

    other.reindex()
    assert other.recall("有什么饮食上的注意事项吗").mode == "hybrid"


def test_embeddings_url_accepts_both_base_url_forms():
    from membrain.embeddings import embeddings_url

    expected = "https://api.example.com/v1/embeddings"
    for base in (
        "https://api.example.com/v1",
        "https://api.example.com/v1/",
        "https://api.example.com/v1/embeddings",
        "https://api.example.com/v1/embeddings/",
    ):
        assert embeddings_url(base) == expected


def test_one_agent_is_stored_under_one_spelling(world):
    """钩子写 "WorkBuddy"、采集写 "workbuddy"，同一个 Agent 被算成两个来源。

    存储只认一个规范名，展示层再映射回可读名称；否则每个来源的统计都被劈成两半。
    """
    from datetime import datetime, timezone

    from membrain.models import Episode

    _, store, _, _ = world
    when = datetime(2026, 1, 2, 9, 0, tzinfo=timezone.utc)
    store.append_episodes([
        Episode(id="", ts=when, session="s1", agent="WorkBuddy", role="user", text="一"),
        Episode(id="", ts=when, session="s2", agent="workbuddy", role="user", text="二"),
        Episode(id="", ts=when, session="s3", agent="Claude Code", role="user", text="三"),
    ])
    agents = sorted(episode.agent for episode in store.iter_episodes())
    assert agents == ["claude-code", "workbuddy", "workbuddy"]
