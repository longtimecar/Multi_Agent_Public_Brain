"""Auto-commit queue: write first, clear after, never twice."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from membrain import auto_commit, session_archive
from membrain.config import Settings
from membrain.models import Episode
from membrain.store import Store


@pytest.fixture()
def store(tmp_path):
    settings = Settings(
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
    return Store(settings.home)


def _episodes(store: Store, session: str, texts: list[str]) -> None:
    now = datetime.now(timezone.utc)
    store.append_episodes([
        Episode(
            id=f"ep_{session}_{i}",
            ts=now,
            session=session,
            agent="test",
            role="user" if i % 2 == 0 else "assistant",
            text=text,
        )
        for i, text in enumerate(texts)
    ])


def _archive_dirs(store: Store, session: str) -> list[Path]:
    hist = store.home / "sessions" / session / "history"
    return sorted(p for p in hist.glob("archive_*") if p.is_dir())


def _mark_pending(store: Store, session: str, *, tokens: int = 400) -> None:
    auto_commit.note_activity(store, session, "x" * tokens, threshold=100, idle_minutes=30)
    assert session in auto_commit.pending_sessions(store)


def test_commit_failure_keeps_pending_and_next_run_succeeds(store, monkeypatch):
    session = "s_retry"
    _episodes(store, session, ["I drink oolong tea.", "Noted."])
    _mark_pending(store, session)

    def boom(*args, **kwargs):
        raise OSError("disk full")

    real = session_archive.phase1_commit
    monkeypatch.setattr(session_archive, "phase1_commit", boom)
    failed = auto_commit.commit_pending_with_archive(store, reason="test")

    # Nothing was written, so the session must still be queued and the reason readable.
    assert [e["session"] for e in failed["errors"]] == [session]
    assert "disk full" in failed["errors"][0]["error"]
    assert session in auto_commit.pending_sessions(store)
    assert _archive_dirs(store, session) == []
    state = auto_commit.load_state(store)
    assert "disk full" in state["sessions"][session]["last_error"]

    # The next run (with the write working again) still has the turn to commit.
    monkeypatch.setattr(session_archive, "phase1_commit", real)
    retried = auto_commit.commit_pending_with_archive(store, reason="test")

    assert retried["errors"] == []
    assert retried["phase1"][0]["status"] == "accepted"
    assert auto_commit.pending_sessions(store) == []
    assert len(_archive_dirs(store, session)) == 1
    assert "last_error" not in auto_commit.load_state(store)["sessions"][session]


def test_repeated_commit_of_same_batch_creates_one_archive(store):
    session = "s_dedupe"
    _episodes(store, session, ["I avoid spicy food.", "Got it."])
    _mark_pending(store, session)

    first = auto_commit.commit_pending_with_archive(store, reason="test")
    second = auto_commit.commit_pending_with_archive(store, reason="test")

    assert first["phase1"][0]["status"] == "accepted"
    assert second["phase1"] == [] and second["committed"] == []
    assert len(_archive_dirs(store, session)) == 1

    # Simulate a crash between the successful write and the queue clear: the
    # session is pending again but the batch fingerprint says it was committed.
    state = auto_commit.load_state(store)
    state["sessions"][session]["pending"] = True
    state["sessions"][session].pop("committed_fingerprint")
    auto_commit.save_state(store, state)
    third = auto_commit.commit_pending_with_archive(store, reason="test")

    assert third["errors"] == []
    assert third["phase1"][0]["status"] == "skipped"
    assert third["phase1"][0]["reason"] == "no_new_messages"
    assert len(_archive_dirs(store, session)) == 1

    # And the fingerprint short-circuit refuses to even talk to the archive.
    state = auto_commit.load_state(store)
    state["sessions"][session]["pending"] = True
    auto_commit.save_state(store, state)
    fourth = auto_commit.commit_pending_with_archive(store, reason="test")
    assert fourth["phase1"][0]["status"] == "already_committed"
    assert len(_archive_dirs(store, session)) == 1


def test_one_failing_session_does_not_block_the_batch(store, monkeypatch):
    _episodes(store, "s_good", ["I like jasmine tea."])
    _episodes(store, "s_bad", ["I like black tea."])
    _mark_pending(store, "s_good")
    _mark_pending(store, "s_bad")

    real = session_archive.phase1_commit

    def flaky(_store, session, *, reason="auto_commit", agent=None):
        if session == "s_bad":
            raise RuntimeError("archive lock lost")
        return real(_store, session, reason=reason, agent=agent)

    monkeypatch.setattr(session_archive, "phase1_commit", flaky)
    result = auto_commit.commit_pending_with_archive(store, reason="test")

    assert result["committed"] == ["s_good"]
    assert [e["session"] for e in result["errors"]] == ["s_bad"]
    assert auto_commit.pending_sessions(store) == ["s_bad"]
    assert len(_archive_dirs(store, "s_good")) == 1
    assert _archive_dirs(store, "s_bad") == []
