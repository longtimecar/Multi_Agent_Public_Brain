from __future__ import annotations

import json
import io
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from membrain.hooks import _codex_user_turn, _transcript_turns, clean_turn, record_capture
from membrain.store import Store

FIXTURE = Path(__file__).parent / "fixtures" / "transcript.jsonl"


def test_transcript_keeps_real_turns_and_record_timestamps(tmp_path):
    turns = _transcript_turns({"transcript_path": str(FIXTURE)})
    assert [turn[1] for turn in turns] == ["I prefer tea.", "Understood.", "我平时喝茶。"]
    assert turns[0][3].isoformat() == "2026-10-12T11:22:33+00:00"
    svc = SimpleNamespace(settings=SimpleNamespace(home=tmp_path / "memory"), store=Store(tmp_path / "memory"))
    assert record_capture(svc, "session", "test", turns) == 3
    assert record_capture(svc, "session", "test", turns) == 0
    episodes = list(svc.store.iter_episodes())
    assert [e.ts.isoformat() for e in episodes] == [turn[3].isoformat() for turn in turns]


def test_fallback_ids_are_stable_across_processes():
    code = "import json,sys; from membrain.hooks import _transcript_turns; print(json.dumps([x[2] for x in _transcript_turns({'transcript_path':sys.argv[1]})]))"
    first = subprocess.check_output([sys.executable, "-c", code, str(FIXTURE)], text=True)
    second = subprocess.check_output([sys.executable, "-c", code, str(FIXTURE)], text=True)
    assert json.loads(first) == json.loads(second)


def test_existing_episode_filter_removes_injected_material():
    assert clean_turn("real <personal_memory>fake</personal_memory> text") == "real  text"
    assert clean_turn("<system-reminder>incomplete") == ""
    assert clean_turn("/compact") == ""


def test_prompt_is_recorded_once_even_when_the_transcript_pass_sees_it_later(tmp_path):
    from membrain.hooks import _load_state as hooks_state, _save_state as save_state, record_prompt
    svc = SimpleNamespace(settings=SimpleNamespace(home=tmp_path / "memory"), store=Store(tmp_path / "memory"))
    assert record_prompt(svc, "session", "claude-code", "我对花生过敏")
    assert not record_prompt(svc, "session", "claude-code", "   ")
    assert not record_prompt(svc, "unknown", "claude-code", "no session id")
    # The Stop-hook transcript pass delivers the same message, with surrounding whitespace.
    from datetime import datetime, timezone
    turns = [("user", "我对花生过敏 ", "uuid-1", datetime.now(timezone.utc)),
             ("assistant", "收到。", "uuid-2", datetime.now(timezone.utc))]
    assert record_capture(svc, "session", "claude-code", turns) == 1
    assert record_capture(svc, "session", "claude-code", turns) == 0
    texts = [(e.role, e.text) for e in svc.store.iter_episodes()]
    assert texts == [("user", "我对花生过敏"), ("assistant", "收到。")]
    # The same message arriving twice at once (two hook entries) is one episode.
    assert not record_prompt(svc, "session", "claude-code", "我对花生过敏")
    assert len(list(svc.store.iter_episodes())) == 2
    # A repeat in a later turn is a new episode, not a lost one.
    state = hooks_state(svc)
    state["last_prompt"]["session"][1] -= 60
    save_state(svc, state)
    assert record_prompt(svc, "session", "claude-code", "我对花生过敏")
    assert len(list(svc.store.iter_episodes())) == 3


def test_extractor_is_started_as_detached_process(tmp_path, monkeypatch):
    from membrain.hooks import _start_extractor
    calls = []
    monkeypatch.setattr("membrain.hooks.subprocess.Popen", lambda args, **kw: calls.append((args, kw)))
    svc = SimpleNamespace(settings=SimpleNamespace(home=tmp_path / "memory"))
    _start_extractor(svc)
    args, options = calls[0]
    assert args[-3:] == ["-m", "membrain.cli", "extract"]
    assert options["env"]["MEMBRAIN_HOME"] == str(svc.settings.home)
    assert options["close_fds"] is True


def test_hook_stdin_reads_utf8_even_if_windows_text_encoding_is_gbk(monkeypatch):
    from membrain.hooks import _read_stdin

    payload = {"prompt": "我喜欢茶", "transcript_path": "C:/用户/对话.jsonl"}
    source = io.BytesIO(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(source, encoding="gbk"))
    assert _read_stdin() == payload


def test_codex_verifier_receives_ascii_json_for_windows_pipe(tmp_path, monkeypatch):
    from membrain.hooks import _start_codex_capture_verifier

    class Pipe(io.BytesIO):
        def close(self):
            pass

    pipe = Pipe()
    monkeypatch.setattr("membrain.hooks.subprocess.Popen", lambda *args, **kwargs: SimpleNamespace(stdin=pipe))
    svc = SimpleNamespace(settings=SimpleNamespace(home=tmp_path / "memory"))
    payload = {"prompt": "我喜欢茶", "transcript_path": "C:/用户/对话.jsonl"}
    _start_codex_capture_verifier(svc, payload)
    assert all(byte < 128 for byte in pipe.getvalue())
    assert json.loads(pipe.getvalue()) == payload


def test_codex_user_turn_excludes_generated_context_and_is_stable():
    payload = {
        "hook_event_name": "UserPromptSubmit", "session_id": "session-1", "turn_id": "turn-1",
        "prompt": ('<in-app-browser-context source="ambient">not user text</in-app-browser-context>\n'
                   '# Response annotations:\n<response-annotations>assistant text ## My request: false</response-annotations>\n'
                   '## My request:\nI prefer tea. <memory>injected text</memory>'),
    }
    turn = _codex_user_turn(payload)
    assert turn is not None and turn[:2] == ("user", "I prefer tea.")
    assert _codex_user_turn(payload)[2] == turn[2]
    assert _codex_user_turn({**payload, "turn_id": "turn-2"})[2] != turn[2]
    assert _codex_user_turn({**payload, "agent_id": "subagent"}) is None
    assert _codex_user_turn({**payload, "hook_event_name": "Stop"}) is None
    assert _codex_user_turn({**payload, "turn_id": ""}) is None
    assert _codex_user_turn({**payload, "prompt": "<response-annotations>quoted output</response-annotations>"}) is None
    assert _codex_user_turn({**payload, "prompt": "<response-annotations>quoted ## My request: output"}) is None


def test_codex_prompt_hook_verifies_user_origin_before_capture(tmp_path, monkeypatch, capsys):
    from membrain import hooks

    svc = SimpleNamespace(settings=SimpleNamespace(home=tmp_path / "memory"), store=Store(tmp_path / "memory"))
    transcript = tmp_path / "codex-session.jsonl"
    payload = {"hook_event_name": "UserPromptSubmit", "session_id": "session-1", "turn_id": "turn-1",
               "prompt": "I enjoy hiking.", "transcript_path": str(transcript)}
    started = []
    scheduled = []
    recalled = []
    monkeypatch.setattr(hooks, "MemoryService", lambda: svc)
    monkeypatch.setattr(hooks, "_read_stdin", lambda: payload)
    monkeypatch.setattr(hooks, "_start_extractor", lambda _: started.append(True))
    monkeypatch.setattr(hooks, "_start_codex_capture_verifier", lambda _, data: scheduled.append(data))
    monkeypatch.setattr(hooks, "_prompt_context", lambda _, data: recalled.append(data["prompt"]) or "")

    assert hooks.run("codex-prompt-submit") == 0
    assert hooks.run("codex-prompt-submit") == 0
    assert list(svc.store.iter_episodes()) == []
    assert len(scheduled) == 2
    assert hooks._verify_codex_capture(svc, payload, timeout=0) is False
    records = [
        {"type": "session_meta", "payload": {"session_id": "session-1", "thread_source": "user"}},
        {"type": "response_item", "payload": {"type": "message", "role": "user",
            "internal_chat_message_metadata_passthrough": {"turn_id": "turn-1", "content_item_kinds": ["user.text"]},
            "content": [{"type": "input_text", "text": payload["prompt"]}]}},
    ]
    transcript.write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")
    assert hooks._verify_codex_capture(svc, payload, timeout=0) is True
    assert hooks._verify_codex_capture(svc, payload, timeout=0) is True
    episodes = list(svc.store.iter_episodes())
    assert [(e.agent, e.role, e.text) for e in episodes] == [("codex", "user", "I enjoy hiking.")]
    assert started == [True]
    assert recalled == ["I enjoy hiking.", "I enjoy hiking."]
    assert capsys.readouterr().out == ""


def test_codex_capture_rejects_subagents_and_unverified_input(tmp_path):
    from membrain.hooks import _codex_transcript_confirms_user

    transcript = tmp_path / "session.jsonl"
    payload = {"session_id": "session", "turn_id": "turn", "prompt": "I prefer tea.",
               "transcript_path": str(transcript)}
    header = {"type": "session_meta", "payload": {"session_id": "session", "thread_source": "user"}}
    item = {"type": "response_item", "payload": {"type": "message", "role": "user",
        "internal_chat_message_metadata_passthrough": {"turn_id": "turn", "content_item_kinds": ["user.text"]},
        "content": [{"type": "input_text", "text": payload["prompt"]}]}}

    def write():
        transcript.write_text("\n".join(json.dumps(row) for row in (header, item)) + "\n", encoding="utf-8")

    write()
    assert _codex_transcript_confirms_user(payload)
    header["payload"]["thread_source"] = "subagent"
    write()
    assert not _codex_transcript_confirms_user(payload)
    header["payload"]["thread_source"] = "user"
    item["payload"]["internal_chat_message_metadata_passthrough"]["content_item_kinds"] = ["agent.text"]
    write()
    assert not _codex_transcript_confirms_user(payload)
    item["payload"]["internal_chat_message_metadata_passthrough"]["content_item_kinds"] = ["user.text"]
    item["payload"]["content"][0]["text"] = "different text"
    write()
    assert not _codex_transcript_confirms_user(payload)
