"""Start the dashboard at Windows logon — in the background, without a browser.

Capture only happens while the dashboard process is alive, so it has to be
running from the moment the machine is usable. Opening a browser tab on every
sign-in is intrusive though, which is why the autostart entry points at a
silent launcher: it starts the server and leaves the browser alone.

The entry is a shortcut in the Startup folder whose target is that launcher.
A shortcut was chosen over writing a script into the folder (blocked by the
local security software) and over a command line (the shortcut's Arguments
field is silently dropped, which left an interpreter with nothing to run).
"""

from __future__ import annotations

import base64
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

LINK_NAME = "membrain-dashboard.lnk"
# Written by an earlier build; removed when the entry is (re)installed so two
# mechanisms never race at logon.
LEGACY_SCRIPT = "membrain-dashboard.vbs"
SILENT_LAUNCHER = "start-dashboard-silent.vbs"
VISIBLE_LAUNCHER = "start-dashboard.vbs"
PORT = 1966


class AutostartError(RuntimeError):
    """The entry could not be installed or removed."""


def supported() -> bool:
    return os.name == "nt"


def startup_dir() -> Path:
    return (Path(os.environ.get("APPDATA", "")) / "Microsoft" / "Windows"
            / "Start Menu" / "Programs" / "Startup")


def link_path() -> Path:
    return startup_dir() / LINK_NAME


def _read_launcher(path: Path) -> str:
    """Which launcher a shortcut points at, read straight from its bytes.

    pywin32 is not a dependency and shelling out to PowerShell to answer a
    status question would be both slow and fragile. Both the ANSI and UTF-16
    copies of the name are searched because the format stores the path twice.
    """
    try:
        raw = path.read_bytes().lower()
    except OSError:
        return ""
    for name in (SILENT_LAUNCHER, VISIBLE_LAUNCHER):
        for encoded in (name.lower().encode("ascii"), name.lower().encode("utf-16-le")):
            if encoded in raw:
                return name
    return ""


def describe() -> dict[str, Any]:
    """Everything the settings page needs to draw an honest toggle."""
    path = link_path()
    exists = path.exists()
    launcher = _read_launcher(path) if exists else ""
    return {
        "supported": supported(),
        "enabled": exists,
        # An entry that points at the browser launcher is "on" but not silent;
        # saying so is better than claiming the feature works as advertised.
        "silent": launcher.lower() == SILENT_LAUNCHER.lower(),
        "launcher": launcher,
        "link": str(path),
        "port": PORT,
    }


DESKTOP_SHORTCUT_NAME = "membrain 记忆后台.lnk"


def _ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def resolve_pythonw() -> Path:
    from .config import REPO_ROOT

    for candidate in (
        REPO_ROOT / ".venv" / "Scripts" / "pythonw.exe",
        Path(sys.executable).with_name("pythonw.exe"),
        Path(sys.executable),
    ):
        if candidate.exists():
            return candidate
    return Path(sys.executable)


def _run_powershell(script: str, error_prefix: str) -> str:
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
            capture_output=True,
            timeout=60,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise AutostartError(f"{error_prefix}：{exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or b"").decode("utf-8", "replace").strip()
        raise AutostartError(detail[:200] or f"{error_prefix}。")
    return (result.stdout or b"").decode("utf-8", "replace").strip()


def _launcher_body() -> str:
    from .config import REPO_ROOT

    pythonw = resolve_pythonw()
    lines = [
        "' Start the membrain dashboard in the background at logon.",
        "' Installed by the settings page (or `membrain install-autostart`).",
        "' No browser is opened: sign-in should not take over the screen. Use",
        "' start-dashboard.vbs when you want the dashboard in front of you.",
        'Set sh = CreateObject("WScript.Shell")',
        'Set env = sh.Environment("Process")',
        'env("HTTP_PROXY") = ""',
        'env("HTTPS_PROXY") = ""',
        'env("ALL_PROXY") = ""',
        f'sh.CurrentDirectory = "{REPO_ROOT}"',
        f'sh.Run """{pythonw}"" -m membrain.cli dashboard --port {PORT} --restart --no-open", 0, False',
        "",
    ]
    return "\r\n".join(lines)


def _write_launcher() -> Path:
    from .config import REPO_ROOT

    path = REPO_ROOT / SILENT_LAUNCHER
    body = _launcher_body()
    # VBScript reads ANSI unless the file carries a UTF-16 BOM, so a non-ASCII
    # install path has to switch encodings instead of being written as UTF-8.
    if body.isascii():
        path.write_bytes(body.encode("ascii"))
    else:
        path.write_bytes(b"\xff\xfe" + body.encode("utf-16-le"))
    return path


def _create_shortcut(target: Path) -> None:
    from .config import REPO_ROOT

    script = "\n".join([
        "[Console]::OutputEncoding = [Text.Encoding]::UTF8",
        "$startup = [Environment]::GetFolderPath('Startup')",
        f"$link = Join-Path $startup {_ps_quote(LINK_NAME)}",
        "$shell = New-Object -ComObject WScript.Shell",
        "$shortcut = $shell.CreateShortcut($link)",
        f"$shortcut.TargetPath = {_ps_quote(str(target))}",
        f"$shortcut.WorkingDirectory = {_ps_quote(str(REPO_ROOT))}",
        "$shortcut.Description = 'membrain dashboard (background, no browser)'",
        "$shortcut.Save()",
    ])
    _run_powershell(script, "无法写入开机启动项")


def create_desktop_shortcut(
    port: int = PORT,
    no_open: bool = False,
    link_dir: Path | None = None,
) -> dict[str, Any]:
    """Create a Windows desktop .lnk shortcut launching the dashboard without a console."""
    if not supported():
        raise AutostartError("桌面快捷方式目前只支持 Windows。")
    from .config import REPO_ROOT

    pythonw = resolve_pythonw()
    workdir = REPO_ROOT if REPO_ROOT.exists() else Path.home()
    args_parts = ["-m", "membrain.cli", "dashboard"]
    if int(port) != PORT:
        args_parts.extend(["--port", str(int(port))])
    if no_open:
        args_parts.append("--no-open")
    arguments = " ".join(args_parts)
    # The desktop may be redirected (OneDrive), so ask Windows for it unless
    # an explicit directory was supplied. Non-ASCII paths travel in UTF-16-LE.
    desktop_expr = _ps_quote(str(link_dir)) if link_dir is not None else "[Environment]::GetFolderPath('Desktop')"
    script = "\n".join([
        "[Console]::OutputEncoding = [Text.Encoding]::UTF8",
        f"$desktop = {desktop_expr}",
        f"$path = Join-Path $desktop {_ps_quote(DESKTOP_SHORTCUT_NAME)}",
        "$shell = New-Object -ComObject WScript.Shell",
        "$link = $shell.CreateShortcut($path)",
        f"$link.TargetPath = {_ps_quote(str(pythonw))}",
        f"$link.Arguments = {_ps_quote(arguments)}",
        f"$link.WorkingDirectory = {_ps_quote(str(workdir))}",
        "$link.IconLocation = 'imageres.dll,-1023'",
        f"$link.Description = {_ps_quote(f'Open the membrain memory dashboard (127.0.0.1:{int(port)})')}",
        "$link.Save()",
        "Write-Output $path",
    ])
    created_path = _run_powershell(script, "无法创建桌面快捷方式")
    if not created_path:
        created_path = str((link_dir or Path.home() / "Desktop") / DESKTOP_SHORTCUT_NAME)
    return {
        "ok": True,
        "supported": True,
        "name": DESKTOP_SHORTCUT_NAME,
        "path": created_path,
        "target": str(pythonw),
        "arguments": arguments,
        "port": int(port),
    }


def _remove(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise AutostartError(f"无法删除 {path.name}：{exc}") from exc


def enable() -> dict[str, Any]:
    """Install (or repair) the logon entry and report what was achieved."""
    if not supported():
        raise AutostartError("开机自启动目前只支持 Windows。")
    launcher = _write_launcher()
    _create_shortcut(launcher)
    _remove(startup_dir() / LEGACY_SCRIPT)
    state = describe()
    if not state["enabled"] or not state["silent"]:
        raise AutostartError("系统没有保留这条启动项，可能被安全软件拦截了。")
    return state


def disable() -> dict[str, Any]:
    if not supported():
        return describe()
    _remove(link_path())
    _remove(startup_dir() / LEGACY_SCRIPT)
    return describe()


def set_enabled(enabled: bool) -> dict[str, Any]:
    return enable() if enabled else disable()
