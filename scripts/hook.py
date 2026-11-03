"""Hook shim for agent integrations.

Exists so hook configuration never depends on a shell, a quoted Windows path,
or where the package happens to be installed. Resolves MEMBRAIN_HOME relative
to this file's own repository, then hands off to the real CLI.

Exit code is always 0: a memory layer must never be able to break a user's turn.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("MEMBRAIN_HOME", str(REPO / "memory"))

# Use the project virtualenv when present, so hooks work with a bare `python`.
_venv_python = REPO / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
if _venv_python.exists() and Path(sys.executable).resolve() != _venv_python.resolve():
    if os.environ.get("_MEMBRAIN_REEXEC") != "1":
        os.environ["_MEMBRAIN_REEXEC"] = "1"
        os.execv(str(_venv_python), [str(_venv_python), str(Path(__file__).resolve()), *sys.argv[1:]])

sys.path.insert(0, str(REPO / "src"))

from membrain.cli import main  # noqa: E402

if __name__ == "__main__":
    try:
        code = main(["hook", *sys.argv[1:]])
    except Exception as exc:  # never propagate into the agent's turn
        print(f"membrain hook error: {exc}", file=sys.stderr)
        code = 0
    raise SystemExit(0 if code is None else 0)
