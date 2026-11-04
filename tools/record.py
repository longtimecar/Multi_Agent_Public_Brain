"""Standalone capture tool: record the user's own words into membrain.

Usage:
    python tools/record.py --text "用户原话" [--agent "Claude Code"] [--session id]

Where it writes is decided the same way as the hooks: MEMBRAIN_HOME if set,
otherwise <this repo>/memory. Safe to call repeatedly — the same text is only
stored once. The dashboard service picks the turn up on its next extraction
pass, so no server restart is needed.

Most agents do not need this: the dashboard's watcher auto-captures from their
session logs. Use this tool only when an agent has no local session store to
watch, or for a one-off capture.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("MEMBRAIN_HOME", str(REPO / "memory"))

_venv_python = REPO / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
if _venv_python.exists() and Path(sys.executable).resolve() != _venv_python.resolve():
    if os.environ.get("_MEMBRAIN_REEXEC") != "1":
        os.environ["_MEMBRAIN_REEXEC"] = "1"
        os.execv(str(_venv_python), [str(_venv_python), str(Path(__file__).resolve()), *sys.argv[1:]])

sys.path.insert(0, str(REPO / "src"))

from membrain.hooks import record_prompt  # noqa: E402
from membrain.server import MemoryService  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Record one user turn into membrain.")
    parser.add_argument("--text", required=True, help="the user's own words, verbatim")
    parser.add_argument("--agent", default="record-cli", help="agent display name (default: record-cli)")
    parser.add_argument("--session", default="manual", help="session id (default: manual)")
    args = parser.parse_args()
    text = args.text.strip()
    if not text:
        print("record: empty text, nothing stored", file=sys.stderr)
        return 1
    try:
        svc = MemoryService()
        ok = record_prompt(svc, session=args.session, agent=args.agent, text=text)
        print("record: stored" if ok else "record: skipped (duplicate, empty, or disabled agent)")
        return 0 if ok else 2
    except Exception as exc:
        print(f"record error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
