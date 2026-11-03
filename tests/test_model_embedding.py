"""Tests for the embedding configuration entry point and agent detection fixes."""

from __future__ import annotations

import json

import pytest

from membrain import agents
from membrain.config import Settings
from membrain.dashboard import DashboardRuntime
from membrain.server import MemoryService


NO_CMD = lambda _name: None  # noqa: E731


def _runtime(tmp_path) -> DashboardRuntime:
    settings = Settings(
        home=tmp_path / "memory",
        embed_base_url=None,
        embed_api_key=None,
        embed_model="text-embedding-3-small",
        embed_timeout=20,
        llm_base_url=None,
        llm_api_key=None,
        llm_model=None,
        turn_token_budget=1500,
        session_token_budget=2500,
        env_file=tmp_path / ".env",
    )
    return DashboardRuntime(MemoryService(settings))


# --------------------------------------------------------------- embedding


def test_embedding_status_is_reported_separately_from_the_llm(tmp_path):
    runtime = _runtime(tmp_path)
    status = runtime.model_status()
    assert status["configured"] is False          # no LLM configured
    assert status["embedding_configured"] is False  # no embedding configured
    assert status["embedding_model"] == "text-embedding-3-small"
    assert status["embedding_base_url"] == "—"
    assert status["embedding_api_key_set"] is False


def test_saving_embedding_config_writes_env_and_updates_settings(tmp_path):
    runtime = _runtime(tmp_path)
    status = runtime.configure_embedding({
        "base_url": "https://api.example.com/v1",
        "model": "bge-m3",
        "api_key": "secret-key-value",
        "timeout": 25,
    })
    assert status["embedding_configured"] is True
    assert status["embedding_model"] == "bge-m3"
    assert status["embedding_api_key_set"] is True
    assert status["embedding_timeout"] == 25

    env = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "MEMBRAIN_EMBED_BASE_URL" in env and "bge-m3" in env
    assert "secret-key-value" in env

    # The running service must see the new endpoint, not just the file.
    assert runtime.service.settings.embed_base_url == "https://api.example.com/v1"
    assert runtime.service.embedder.settings.embed_model == "bge-m3"


def test_retriever_and_archive_share_the_replaced_embedder(tmp_path):
    """A half-swapped embedder would use the old endpoint on some paths."""
    runtime = _runtime(tmp_path)
    runtime.configure_embedding({"base_url": "https://api.example.com/v1", "model": "bge-m3"})
    embedder = runtime.service.embedder
    assert runtime.service.retriever.embedder is embedder
    assert runtime.service.archive.embedder is embedder


def test_empty_embedding_address_disables_embeddings(tmp_path):
    runtime = _runtime(tmp_path)
    runtime.configure_embedding({"base_url": "https://api.example.com/v1", "model": "bge-m3"})
    status = runtime.configure_embedding({"base_url": "", "model": "", "timeout": 20})
    assert status["embedding_configured"] is False
    assert runtime.service.settings.embed_base_url is None


def test_embedding_config_rejects_unsafe_addresses(tmp_path):
    runtime = _runtime(tmp_path)
    for bad in ("ftp://example.com/v1", "http://user:pass@example.com/v1",
                "https://example.com/v1?key=secret", "not-a-url"):
        with pytest.raises(ValueError):
            runtime.configure_embedding({"base_url": bad, "model": "bge-m3"})


def test_embedding_config_rejects_an_empty_model(tmp_path):
    runtime = _runtime(tmp_path)
    with pytest.raises(ValueError):
        runtime.configure_embedding({"base_url": "https://api.example.com/v1", "model": ""})


def test_embedding_config_rejects_an_out_of_range_timeout(tmp_path):
    runtime = _runtime(tmp_path)
    for bad in (0, 301, -5, "abc"):
        with pytest.raises(ValueError):
            runtime.configure_embedding({"base_url": "https://api.example.com/v1",
                                         "model": "bge-m3", "timeout": bad})


def test_testing_embedding_without_a_configuration_explains_itself(tmp_path):
    runtime = _runtime(tmp_path)
    with pytest.raises(Exception) as excinfo:
        runtime.test_embedding()
    assert "Embedding" in str(excinfo.value)


def test_clearing_the_embedding_key_keeps_the_endpoint(tmp_path):
    runtime = _runtime(tmp_path)
    runtime.configure_embedding({"base_url": "https://api.example.com/v1", "model": "bge-m3",
                                 "api_key": "secret-key-value"})
    status = runtime.configure_embedding({"base_url": "https://api.example.com/v1", "model": "bge-m3",
                                          "clear_api_key": True})
    assert status["embedding_api_key_set"] is False
    assert status["embedding_configured"] is True


# ------------------------------------------------------------------ agents


def _scan(tmp_path, registry=(), which=NO_CMD):
    return {i["key"]: i for i in agents.scan(tmp_path / "memory", home=tmp_path,
                                             appdata=tmp_path / "Roaming",
                                             which=which, registry=lambda: list(registry))}


def test_deepseek_and_doubao_are_recognised_as_real_agents(tmp_path):
    roaming = tmp_path / "Roaming"
    (roaming / "DeepSeek Bot").mkdir(parents=True)
    (roaming / "Doubao").mkdir(parents=True)
    items = _scan(tmp_path)
    assert items["deepseek"]["installed"] and items["deepseek"]["status"] == "not_connected"
    assert items["doubao"]["installed"] and items["doubao"]["status"] == "not_connected"


def test_deepseek_harness_is_found_in_the_local_app_data_folder(tmp_path):
    # %LOCALAPPDATA% sits next to %APPDATA%, not inside it.
    local = tmp_path / "Local"
    (local / "@deepseek-aidsh-desktop-updater").mkdir(parents=True)
    items = _scan(tmp_path)
    assert items["deepseek-harness"]["installed"]


def test_trae_and_qoder_use_their_real_folder_names(tmp_path):
    roaming = tmp_path / "Roaming"
    (roaming / "Trae").mkdir(parents=True)
    (roaming / "com.qoder.app.stable").mkdir(parents=True)
    items = _scan(tmp_path)
    assert items["trae"]["installed"] and items["qoder"]["installed"]


def test_a_version_suffix_does_not_create_a_second_entry(tmp_path):
    """The uninstall list carries both "Kimi Desktop" and "Kimi Desktop 1.2"."""
    items = _scan(tmp_path, registry=["Kimi Desktop 1.2.3", "Kimi Desktop"])
    others = [i for i in items.values() if i["key"].startswith("other:")]
    assert [i["name"] for i in others] == ["Kimi Desktop"]


def test_a_versioned_name_of_a_known_agent_is_not_reported_twice(tmp_path):
    items = _scan(tmp_path, registry=["DeepSeek Bot 0.2.3", "DeepSeek Bot"])
    others = [i for i in items.values() if i["key"].startswith("other:")]
    assert others == []


def test_an_npm_scope_folder_is_not_reported_as_an_application(tmp_path):
    roaming = tmp_path / "Roaming"
    (roaming / "@deepseek-ai").mkdir(parents=True)
    items = _scan(tmp_path, registry=["@deepseek-ai"])
    others = [i for i in items.values() if i["key"].startswith("other:")]
    assert others == []


def test_known_agents_do_not_reappear_as_generic_entries(tmp_path):
    items = _scan(tmp_path, registry=["DeepSeek Bot", "DeepSeek Harness 0.2.1-rc.3", "Doubao"])
    others = [i["name"] for i in items.values() if i["key"].startswith("other:")]
    assert others == []


def test_unknown_agent_names_are_still_reported(tmp_path):
    items = _scan(tmp_path, registry=["Kimi Desktop", "Notepad++"])
    others = [i["name"] for i in items.values() if i["key"].startswith("other:")]
    assert others == ["Kimi Desktop"]


# ------------------------------------------------------ deepseek harness (dsh)


def _zstd_blob(frames: list[dict]) -> bytes:
    """Build a concatenated multi-frame zstd blob the way dsh writes it."""
    import io

    import zstandard
    cctx = zstandard.ZstdCompressor()
    buf = io.BytesIO()
    for frame in frames:
        buf.write(cctx.compress((json.dumps(frame) + "\n").encode("utf-8")))
    return buf.getvalue()


def _dsh_tree(tmp_path, frames: list[dict], workspace="--D-Doc~0020notes--", session="session-abc") -> None:
    root = tmp_path / ".dsh" / "sessions" / workspace / session
    root.mkdir(parents=True)
    (root / "session.v4.jsonl.zstd").write_bytes(_zstd_blob(frames))


def _dsh_sessions(tmp_path):
    from membrain import session_import
    real_home = session_import.HOME
    session_import.HOME = tmp_path  # the collector resolves ~/.dsh against HOME
    try:
        return session_import._collect_dsh()
    finally:
        session_import.HOME = real_home


def test_dsh_collects_the_main_session_and_skips_subagents(tmp_path):
    _dsh_tree(tmp_path, [
        {"type": "session", "id": "s1", "origin": "subagent"},
        {"type": "user/message", "data": {"message": {"role": "user",
         "content": [{"type": "text", "text": "帮我审一下这段配置"}]}}},
        {"type": "assistant/message", "data": {"message": {"role": "assistant",
         "content": [{"type": "reasoning", "text": "thinking"},
                     {"type": "text", "text": "配置没问题"}]}}},
    ], session="sub-uuid-1")
    _dsh_tree(tmp_path, [
        {"type": "session", "id": "s2"},
        {"type": "user/message", "data": {"message": {"role": "user",
         "content": [{"type": "text", "text": "顶层提问"}]}}},
        {"type": "assistant/message", "data": {"message": {"role": "assistant",
         "content": [{"type": "text", "text": "顶层回答"}]}}},
    ], session="session-main")
    sessions = _dsh_sessions(tmp_path)
    assert [s.scope for s in sessions] == ["session-main"]
    assert sessions[0].source == "dsh"
    assert sessions[0].turns == [("user", "顶层提问"), ("assistant", "顶层回答")]


def test_dsh_workspace_name_decodes_unicode_and_spaces(tmp_path):
    _dsh_tree(tmp_path, [
        {"type": "session", "id": "s1"},
        {"type": "user/message", "data": {"message": {"role": "user",
         "content": [{"type": "text", "text": "hi there, this is a real question about indexes"}]}}},
    ])
    sessions = _dsh_sessions(tmp_path)
    assert len(sessions) == 1
    # --D-Doc~0020notes-- → D Doc notes (the ~0020 was a space code point)
    assert "D Doc notes" in sessions[0].title


def test_dsh_empty_and_unreadable_sessions_are_skipped(tmp_path):
    _dsh_tree(tmp_path, [{"type": "session", "id": "s1"}], session="session-empty")
    root = tmp_path / ".dsh" / "sessions" / "--D-Doc~0020notes--" / "session-broken"
    root.mkdir(parents=True)
    (root / "session.v4.jsonl.zstd").write_bytes(b"not zstd at all")
    assert _dsh_sessions(tmp_path) == []


def test_dsh_injected_context_is_not_user_evidence(tmp_path):
    _dsh_tree(tmp_path, [
        {"type": "session", "id": "s1"},
        {"type": "user/message", "data": {"message": {"role": "user",
         "content": [{"type": "text", "text": "<system-reminder>injected profile</system-reminder>real question"}]}}},
    ])
    sessions = _dsh_sessions(tmp_path)
    assert sessions[0].turns == [("user", "real question")]

