"""Thin harness adapter: normalize Cursor/WorkBuddy/OpenCode payloads → membrain hooks.

Usage:
  python harness_hook.py <agent> <event>

Events (harness-facing): session-start | prompt-submit | capture | precompact | auto-commit
Stdout is harness-shaped:
  - cursor session-start → {"additional_context": "..."}
  - cursor beforeSubmitPrompt/prompt-submit → {"continue": true}
  - cursor preCompact → {"user_message": optional}
  - workbuddy/claude-like → raw injection text (same as scripts/hook.py)
  - json mode (--json) → always emit a JSON object with keys context/continue

Never raises into the agent turn (exit 0).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("MEMBRAIN_HOME", str(REPO / "memory"))

_venv = REPO / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
if _venv.exists() and Path(sys.executable).resolve() != _venv.resolve():
    if os.environ.get("_MEMBRAIN_HARNESS_REEXEC") != "1":
        os.environ["_MEMBRAIN_HARNESS_REEXEC"] = "1"
        os.execv(str(_venv), [str(_venv), str(Path(__file__).resolve()), *sys.argv[1:]])

sys.path.insert(0, str(REPO / "src"))

AGENT_ALIASES = {
    "cursor": "cursor",
    "opencode": "opencode",
    "workbuddy": "workbuddy",
    "claude": "claude-code",
    "claude-code": "claude-code",
    "codex": "codex",
}

EVENT_ALIASES = {
    "session-start": "session-start",
    "sessionstart": "session-start",
    "session_start": "session-start",
    "prompt-submit": "prompt-submit",
    "promptsubmit": "prompt-submit",
    "beforeSubmitPrompt": "prompt-submit",
    "beforesubmitprompt": "prompt-submit",
    "userpromptsubmit": "prompt-submit",
    "capture": "capture",
    "stop": "capture",
    "sessionend": "capture",
    "session-end": "capture",
    "precompact": "precompact",
    "pre-compact": "precompact",
    "auto-commit": "auto-commit",
    "autocommit": "auto-commit",
}


def _read_stdin() -> dict:
    stream = getattr(sys.stdin, "buffer", None)
    try:
        raw = stream.read().decode("utf-8") if stream is not None else sys.stdin.read()
    except UnicodeDecodeError:
        return {}
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _normalize(agent: str, event: str, payload: dict) -> tuple[str, dict]:
    out = dict(payload)
    out["agent"] = agent
    # Cursor uses conversation_id / session_id interchangeably
    sid = (
        out.get("session_id")
        or out.get("sessionID")
        or out.get("conversation_id")
        or out.get("sessionId")
    )
    if sid:
        out["session_id"] = str(sid)
    # prompt fields
    prompt = out.get("prompt") or out.get("text") or out.get("user_message") or ""
    if isinstance(prompt, dict):
        prompt = prompt.get("text") or ""
    if not prompt and isinstance(out.get("message"), dict):
        prompt = out["message"].get("content") or out["message"].get("text") or ""
    if prompt and "prompt" not in out:
        out["prompt"] = str(prompt)
    # Cursor preCompact has no prompt; still useful for auto-commit
    if event == "precompact" and not out.get("prompt"):
        out["prompt"] = ""
    return event, out


def _run_membrain(event: str, payload: dict) -> str:
    """Call hooks.run with patched stdin; capture printed context."""
    import io
    from membrain import hooks

    old_stdin = sys.stdin
    old_stdout = sys.stdout
    buf_in = io.StringIO(json.dumps(payload, ensure_ascii=False))
    buf_out = io.StringIO()
    try:
        sys.stdin = buf_in
        sys.stdout = buf_out
        hooks.run(event)
    finally:
        sys.stdin = old_stdin
        sys.stdout = old_stdout
    return buf_out.getvalue().strip()


def main(argv: list[str]) -> int:
    args = [a for a in argv if a != "--json"]
    want_json = "--json" in argv
    if len(args) < 2:
        print("usage: harness_hook.py <agent> <event>", file=sys.stderr)
        return 0
    agent = AGENT_ALIASES.get(args[0].lower(), args[0].lower())
    event = EVENT_ALIASES.get(args[1], EVENT_ALIASES.get(args[1].lower(), args[1].lower()))
    payload = _read_stdin()
    event, payload = _normalize(agent, event, payload)
    try:
        context = _run_membrain(event, payload)
    except Exception as exc:  # never break the turn
        print(f"membrain harness_hook error: {exc}", file=sys.stderr)
        context = ""

    # Shape stdout for the calling harness.
    if agent == "cursor":
        if event == "session-start":
            print(json.dumps({"additional_context": context}, ensure_ascii=False))
        elif event == "prompt-submit":
            # Cursor beforeSubmitPrompt cannot inject; capture already happened.
            print(json.dumps({"continue": True}, ensure_ascii=False))
        elif event == "precompact":
            msg = "membrain auto-commit triggered" if context == "" else context[:200]
            print(json.dumps({"user_message": msg}, ensure_ascii=False))
        else:
            print(json.dumps({}, ensure_ascii=False))
        return 0

    if want_json:
        print(json.dumps({"context": context, "continue": True, "agent": agent, "event": event}, ensure_ascii=False))
        return 0

    # WorkBuddy / Claude-compatible: raw injection text on stdout
    if context:
        print(context)
    return 0


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception as exc:
        print(f"membrain harness_hook error: {exc}", file=sys.stderr)
    raise SystemExit(0)