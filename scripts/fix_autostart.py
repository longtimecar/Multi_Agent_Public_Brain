"""Point the Startup shortcut at the launcher script and verify the bytes.

The shortcut kept losing its Arguments field: the target survived, the command
line did not, so signing in would have started an interpreter with nothing to
run. Aiming the shortcut at start-dashboard.vbs needs no arguments at all — the
shell runs a .vbs through its own host and the script holds the real command,
including the proxy reset and --restart.
"""

from __future__ import annotations

import base64
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STARTUP = Path.home() / "AppData/Roaming/Microsoft/Windows/Start Menu/Programs/Startup"
LINK = STARTUP / "membrain-dashboard.lnk"


def main() -> int:
    launcher = REPO / "start-dashboard.vbs"
    if not launcher.exists():
        print(f"FAIL: {launcher} 不存在")
        return 1
    script = "\n".join([
        "[Console]::OutputEncoding = [Text.Encoding]::UTF8",
        "$startup = [Environment]::GetFolderPath('Startup')",
        "$link = Join-Path $startup 'membrain-dashboard.lnk'",
        "$shell = New-Object -ComObject WScript.Shell",
        "$shortcut = $shell.CreateShortcut($link)",
        f"$shortcut.TargetPath = '{launcher}'",
        "$shortcut.Arguments = ''",
        f"$shortcut.WorkingDirectory = '{REPO}'",
        "$shortcut.Description = 'membrain memory dashboard (auto-start)'",
        "$shortcut.Save()",
        "Write-Output $link",
    ])
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    done = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
        capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    out = (done.stdout or done.stderr).decode("utf-8", "replace").strip()
    if out:
        print(out)
    if done.returncode != 0 or not LINK.exists():
        print("FAIL: 快捷方式没有写成功")
        return 1
    raw = LINK.read_bytes()
    hit = b"start-dashboard.vbs" in raw or "start-dashboard.vbs".encode("utf-16-le") in raw
    print(f"快捷方式 {len(raw)} 字节 | 目标含 start-dashboard.vbs: {hit}")
    print("结果:", "PASS" if hit else "FAIL —— 目标没写进去")
    return 0 if hit else 1


if __name__ == "__main__":
    sys.exit(main())
