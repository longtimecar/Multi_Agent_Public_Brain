"""Create a desktop shortcut that opens the membrain dashboard without a console.

Run once: .venv/Scripts/python.exe scripts/create_shortcut.py
The shortcut starts the dashboard with pythonw.exe (no black window). If the
dashboard is already running it only opens the browser, so clicking it twice
never starts a second background process.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PYTHONW = REPO / ".venv" / "Scripts" / "pythonw.exe"
SRC_DIR = REPO / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from membrain import autostart  # noqa: E402

NAME = autostart.DESKTOP_SHORTCUT_NAME


def main() -> int:
    try:
        result = autostart.create_desktop_shortcut()
    except autostart.AutostartError as exc:
        msg = f"{exc}\n"
        sys.stderr.buffer.write(msg.encode(sys.stderr.encoding or "utf-8", "replace"))
        return 1
    message = str(result.get("path") or NAME)
    # The console may use a legacy code page that cannot show the path.
    sys.stdout.buffer.write((message + "\n").encode(sys.stdout.encoding or "utf-8", "replace"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
