"""Create a desktop shortcut that opens the membrain dashboard without a console.

Run once: .venv/Scripts/python.exe scripts/create_shortcut.py
The shortcut starts the dashboard with pythonw.exe (no black window). If the
dashboard is already running it only opens the browser, so clicking it twice
never starts a second background process.
"""

from __future__ import annotations

import base64
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PYTHONW = REPO / ".venv" / "Scripts" / "pythonw.exe"
NAME = "membrain 记忆后台.lnk"


def _ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def main() -> int:
    if not PYTHONW.exists():
        print(f"pythonw.exe not found: {PYTHONW}")
        return 1
    # The desktop may be redirected (OneDrive), so ask Windows for it. Paths
    # with non-ASCII user names travel inside a UTF-16 encoded command.
    script = "\n".join([
        "[Console]::OutputEncoding = [Text.Encoding]::UTF8",
        "$desktop = [Environment]::GetFolderPath('Desktop')",
        f"$path = Join-Path $desktop {_ps_quote(NAME)}",
        "$shell = New-Object -ComObject WScript.Shell",
        "$link = $shell.CreateShortcut($path)",
        f"$link.TargetPath = {_ps_quote(str(PYTHONW))}",
        "$link.Arguments = '-m membrain.cli dashboard'",
        f"$link.WorkingDirectory = {_ps_quote(str(REPO))}",
        "$link.IconLocation = 'imageres.dll,-1023'",
        "$link.Description = 'Open the membrain memory dashboard (127.0.0.1:1966)'",
        "$link.Save()",
        "Write-Output $path",
    ])
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    result = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
                            capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    message = (result.stdout or result.stderr).decode("utf-8", "replace").strip()
    # The console may use a legacy code page that cannot show the path.
    sys.stdout.buffer.write((message + "\n").encode(sys.stdout.encoding or "utf-8", "replace"))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
