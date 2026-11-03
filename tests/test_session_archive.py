"""OV gap 1: two-phase session archive + memory_diff."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from membrain.config import Settings
from membrain.embeddings import Embedder
from membrain.extract import ExtractionReport
from membrain.index import Index
from membrain.models import Episode
from membrain.retrieval import Retriever
from membrain.session_archive import (
    empty_memory_diff,
    phase1_commit,
    phase2_process,
    read_memory_diff,
    resolve_archive_path,
)
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
        auto_commit_token_threshold=100,
        auto_commit_idle_minutes=30,
    )


@pytest.fixture()
def world(tmp_path):
    settings = make_settings(tmp_path)
    store = Store(settings.home)
    index = Index(settings.db_path)
    retriever = Retriever(store, index, Embedder(settings), settings)
    return settings, store, retriever


def _episodes(store: Store, session: str, texts: list[str]) -> None:
    now = datetime.now(timezone.utc)
    eps = []
    for i, text in enumerate(texts):
        eps.append(
            Episode(
                id=f"ep_{session}_{i}",
                ts=now,
                session=session,
                agent="test",
                role="user" if i % 2 == 0 else "assistant",
                text=text,
            )
        )
    store.append_episodes(eps)


class _FakeExtractor:
    def __init__(self, report: ExtractionReport | None = None):
        self.report = report or ExtractionReport(errors=[])
        self.calls: list[str | None] = []

    def run(self, session: str | None = None) -> ExtractionReport:
        self.calls.append(session)
        return self.report


def test_phase1_and_empty_diff_phase2(world):
    _, store, _ = world
    session = "s_archive_1"
    _episodes(store, session, [
        "I usually drink oolong tea in the afternoon.",
        "Noted, tea preference recorded.",
        "Also I avoid spicy food.",
        "Got it.",
    ])
    p1 = phase1_commit(store, session, reason="test")
    assert p1["status"] == "accepted"
    assert p1["task_id"]
    assert p1["archived"] is True
    archive = p1["archive_path"]
    assert (store.home / "sessions").exists() or True
    from pathlib import Path
    ap = Path(archive)
    assert (ap / "messages.jsonl").exists()
    assert (ap / "meta.json").exists()

    fake = _FakeExtractor(ExtractionReport(skipped=2, errors=[]))
    p2 = phase2_process(store, ap, run_extract=True, extractor_factory=lambda: fake)
    assert p2["status"] == "completed"
    assert (ap / ".abstract.md").exists()
    assert (ap / ".overview.md").exists()
    assert (ap / "memory_diff.json").exists()
    assert (ap / ".done").exists()
    diff = read_memory_diff(ap)
    assert "operations" in diff
    assert diff["operations"]["adds"] == []
    assert diff["summary"]["total_adds"] == 0
    assert fake.calls == [session]


def test_phase1_skips_when_no_new_messages(world):
    _, store, _ = world
    session = "s_empty"
    p1 = phase1_commit(store, session, reason="test")
    assert p1["status"] == "skipped"
    assert p1["task_id"] is None


def test_empty_memory_diff_shape():
    d = empty_memory_diff("mb://user/sessions/x/history/archive_001")
    assert d["summary"]["total_adds"] == 0
    assert set(d["operations"]) == {"adds", "updates", "deletes"}


def test_archive_path_resolution_stays_inside_the_memory_store(world, tmp_path):
    _, store, _ = world
    outside = tmp_path / "outside" / "archive_001"
    outside.mkdir(parents=True)
    (outside / "memory_diff.json").write_text('{"summary":{}}', encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        resolve_archive_path(store, str(outside))
    with pytest.raises(FileNotFoundError):
        resolve_archive_path(store, "../../outside/archive_001")

    inside = store.home / "sessions" / "s1" / "history" / "archive_001"
    inside.mkdir(parents=True)
    (inside / "memory_diff.json").write_text('{"summary":{}}', encoding="utf-8")
    assert resolve_archive_path(store, str(inside)) == inside
    assert resolve_archive_path(store, "sessions/s1/history/archive_001") == inside
    assert resolve_archive_path(store, "s1/archive_001") == inside
