"""P1 gaps 4/5/6/7/8/12: search modes, resume overview, context quotas, peer, merge table, diff board."""

from __future__ import annotations

import json
import threading
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import pytest

from membrain.config import Settings
from membrain.context_pack import pack_by_type_quotas
from membrain.dashboard import DashboardRuntime, create_server
from membrain.embeddings import Embedder
from membrain.hooks import _session_start_context
from membrain.index import Index
from membrain.llm import FakeLLMClient
from membrain.merge_policy import preferred_operation, rule_for, table_as_list
from membrain.models import Episode, Evidence, FactType
from membrain.peer import normalize_peer, peer_allows
from membrain.retrieval import Retriever
from membrain.search_mode import resolve_intent_rerank
from membrain.server import MemoryService
from membrain.session_archive import latest_archive_overview, phase1_commit, phase2_process
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


def write(store, claim, subject, type_=FactType.PREFERENCE, peer=None):
    fact = store.draft_fact(
        type=type_,
        subject=subject,
        claim=claim,
        evidence=[Evidence(episode=f"ep_{subject}", quote=claim, date=date.today(), origin="stated")],
        source_agent="test",
        peer=peer,
    )
    return store.write_fact(fact)


def test_resolve_modes_quick_thinking():
    intent, rerank, notes = resolve_intent_rerank("quick", use_intent=None, use_rerank=None)
    assert intent is False and rerank is False
    assert any("quick" in n for n in notes)
    intent, rerank, notes = resolve_intent_rerank("thinking", use_intent=None, use_rerank=None)
    assert intent is True and rerank is True


def test_recall_quick_skips_intent(world):
    _, store, retriever = world
    f = write(store, "Likes oolong tea", "beverage")
    retriever.sync_fact(f)
    llm = FakeLLMClient([{"reasoning": "hi", "queries": []}])
    quick = retriever.recall("tea", top_k=5, mode="quick", llm=llm)
    assert "search_mode:quick" in quick.notes
    assert any(h.id == f.id for h in quick.hits) or "oolong" in (quick.text or "").lower()
    thinking = retriever.recall("你好", top_k=5, mode="thinking", llm=llm)
    assert thinking.mode == "intent-skip"


def test_context_mode_notes_and_budget(world):
    _, store, retriever = world
    for i in range(5):
        retriever.sync_fact(write(store, f"Pref {i} spicy food", f"food{i}", FactType.PREFERENCE))
    for i in range(5):
        retriever.sync_fact(write(store, f"Entity note {i}", f"ent{i}", FactType.ENTITY))
    result = retriever.recall("spicy entity", top_k=8, mode="context", purpose="chat")
    assert result.mode == "context"
    assert any("context_quotas" in n for n in result.notes)
    pref_hits = [h for h in result.hits if h.type == FactType.PREFERENCE]
    assert len(pref_hits) <= 2


def test_pack_by_type_quotas_budget():
    class F:
        def __init__(self, id, type_value, subject, claim, confidence=0.9, uri=None, sensitive=False):
            self.id = id
            self.type = type("T", (), {"value": type_value})()
            self.subject = subject
            self.claim = claim
            self.confidence = confidence
            self.uri = uri
            self.sensitive = sensitive

    ranked = [(F(f"id{i}", "preference", "food", "x" * 40), 1.0 - i * 0.01) for i in range(10)]
    chosen, text, tokens, notes = pack_by_type_quotas(ranked, quotas={"preference": 2}, top_k=8, budget=80)
    assert len(chosen) <= 2


def test_peer_scope_actor_filters(world):
    _, store, retriever = world
    a = write(store, "Project A uses Rust", "proj-a", FactType.PROJECT, peer="repo-a")
    b = write(store, "Project B uses Go", "proj-b", FactType.PROJECT, peer="repo-b")
    for f in (a, b):
        retriever.sync_fact(f)
    cross = retriever.recall("Project uses", top_k=5, peer="repo-a", peer_scope="all")
    assert a.id in {h.id for h in cross.hits}
    actor = retriever.recall("Project uses", top_k=5, peer="repo-a", peer_scope="actor")
    ids_actor = {h.id for h in actor.hits}
    assert a.id in ids_actor
    assert b.id not in ids_actor


def test_peer_normalize_and_allows():
    assert normalize_peer("git@github.com:Org/Repo.git") == "github.com-org-repo"
    assert peer_allows("mb://user/memories/fact/x?peer=repo-a", None, peer="repo-a", peer_scope="actor")
    assert not peer_allows("mb://user/memories/fact/x?peer=repo-b", None, peer="repo-a", peer_scope="actor")
    assert peer_allows("mb://user/memories/fact/x", None, peer="repo-a", peer_scope="actor")


def test_merge_policy_table():
    rows = table_as_list()
    assert len(rows) >= 8
    assert rule_for("belief").strategy == "never_auto_merge"
    assert preferred_operation("preference", explicit_change=True) == "supersede"
    assert preferred_operation("event") == "add"


def test_resume_injects_archive_overview(world):
    settings, store, retriever = world
    now = datetime.now(timezone.utc)
    for i, text in enumerate(["I prefer green tea", "Remember green tea please"]):
        store.append_episodes([Episode(
            id=f"ep_resume_{i}", ts=now, session="resume_sess", agent="test", role="user", text=text,
        )])
    p1 = phase1_commit(store, "resume_sess", reason="test")
    assert p1.get("archived")
    arch = Path(p1["archive_path"])
    phase2_process(store, arch, extractor_factory=None)
    ov, name = latest_archive_overview(store, "resume_sess")
    assert name
    assert ov.strip()
    svc = MemoryService(settings)
    svc.store = store
    svc.retriever = retriever
    block = _session_start_context(svc, {"session_id": "resume_sess", "source": "resume"})
    assert "archive L1" in block


@pytest.fixture()
def dashboard(tmp_path):
    settings = make_settings(tmp_path)
    service = MemoryService(settings)
    runtime = DashboardRuntime(service)
    server = create_server("127.0.0.1", 0, runtime)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def request(path: str):
        req = Request(base + path, headers={"Accept": "application/json"})
        # Correctness test, not a speed test; avoid load-induced timeout flakes.
        with urlopen(req, timeout=30) as response:
            return response.status, json.loads(response.read().decode("utf-8"))

    yield request, service
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)
    service.index.close()


def test_dashboard_session_diffs_api(dashboard):
    request, service = dashboard
    store = service.store
    now = datetime.now(timezone.utc)
    store.append_episodes([Episode(id="ep_diff_1", ts=now, session="diff_sess", agent="test", role="user", text="I like matcha")])
    p1 = phase1_commit(store, "diff_sess", reason="test")
    arch = Path(p1["archive_path"])
    phase2_process(store, arch, extractor_factory=None)
    status, payload = request("/api/session-diffs")
    assert status == 200
    assert payload["total"] >= 1
    key = f"diff_sess/{arch.name}"
    status, detail = request(f"/api/session-diffs/{key}")
    assert status == 200
    assert "diff" in detail
    assert "skipped_operations" in detail["diff"]
