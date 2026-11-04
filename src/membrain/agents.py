"""Find the AI agents installed on this PC and connect them to membrain.

A config file changes only on an explicit connect request, after a backup.
Agents whose config the project protects, or whose settings live inside an app
window, get a copy-paste prompt instead of a file edit.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

from .config import IN_CHECKOUT, REPO_ROOT

SERVER_NAME = "membrain"


class AgentError(ValueError):
    """A request the owner can understand and fix; the message is shown as is."""


@dataclass(frozen=True)
class AgentSpec:
    key: str
    name: str
    match: tuple[str, ...]                 # lowercase tokens that identify this agent in scans
    commands: tuple[str, ...] = ()         # found on PATH
    paths: tuple[str, ...] = ()            # "~/" = home, "%APPDATA%/" = Roaming
    config: str | None = None              # JSON file a one-click connect may edit
    servers_key: str = "mcpServers"
    style: str = "standard"                # "standard" | "opencode"
    protected: bool = False                # project rules require owner approval to edit
    capture: str = "relay"                 # "hooks" captures by itself; "relay" needs the tool
    where: str = "在该软件的 MCP / 工具设置里新增一个自定义 MCP 服务"


SPECS: tuple[AgentSpec, ...] = (
    AgentSpec("claude-code", "Claude Code", ("claude code", "claude-code"), commands=("claude",),
              paths=("~/.claude",), protected=True, capture="hooks"),
    AgentSpec("codex", "Codex", ("codex",), commands=("codex",), paths=("~/.codex", "%APPDATA%/Codex"),
              protected=True, capture="hooks"),
    AgentSpec("opencode", "OpenCode", ("opencode",), commands=("opencode",), paths=("~/.config/opencode",),
              config="~/.config/opencode/opencode.jsonc", servers_key="mcp", style="opencode"),
    AgentSpec("workbuddy", "WorkBuddy", ("workbuddy",), paths=("~/.workbuddy",), config="~/.workbuddy/mcp.json"),
    AgentSpec("cursor", "Cursor", ("cursor",), commands=("cursor",), paths=("~/.cursor", "%APPDATA%/Cursor"),
              config="~/.cursor/mcp.json"),
    AgentSpec("gemini-cli", "Gemini CLI", ("gemini cli", "gemini-cli"), commands=("gemini",),
              paths=("~/.gemini",), config="~/.gemini/settings.json"),
    AgentSpec("trae", "Trae / Trae CN", ("trae",), paths=("%APPDATA%/Trae", "%APPDATA%/Trae CN", "~/.trae")),
    AgentSpec("qoder", "Qoder", ("qoder",),
              paths=("%APPDATA%/com.qoder.app.stable", "%APPDATA%/com.qodercn.app.stable",
                     "~/.qoder", "~/.qoder-cn")),
    AgentSpec("deepseek", "DeepSeek", ("deepseek",),
              paths=("%APPDATA%/DeepSeek Bot", "%APPDATA%/DeepSeek", "~/.deepseek"),
              config="~/.deepseek/mcp.json"),
    AgentSpec("deepseek-harness", "DeepSeek Harness", ("deepseek harness", "deepseek-harness", "dsh"),
              paths=("%LOCALAPPDATA%/@deepseek-aidsh-desktop-updater",
                     "%APPDATA%/DeepSeek Harness", "~/.deepseek-harness"),
              config="~/.deepseek-harness/mcp.json"),
    AgentSpec("doubao", "Doubao", ("doubao",),
              paths=("%APPDATA%/Doubao", "~/.doubao"),
              config="~/.doubao/mcp.json"),
    AgentSpec("qwen", "千问 (Qwen)", ("qwen", "qianwen", "千问"),
              paths=("%LOCALAPPDATA%/Qianwen", "%LOCALAPPDATA%/QianwenUpdater", "~/.qwen-agent"),
              where="在千问的 设置 → MCP / 工具 里新增一个自定义 MCP 服务"),
    AgentSpec("vscode", "VS Code (Copilot 等插件)", ("visual studio code", "vscode"), commands=("code",),
              paths=("%APPDATA%/Code",)),
)

# Names worth reporting when found by the scan but not described above.
KEYWORDS = ("claude", "codex", "cursor", "gemini", "copilot", "trae", "qoder", "kimi", "doubao", "deepseek",
            "codebuddy", "windsurf", "cline", "aider", "hermes", "qwen", "qianwen", "agent")

# A trailing version turns one product into several entries in the Windows
# uninstall list: "DeepSeek Bot" and "DeepSeek Bot 0.2.3" are the same app.
_VERSION_TAIL = re.compile(
    r"\s+v?\d+(?:\.\d+)*(?:[-+][A-Za-z0-9.]+)*(?:\s*\((?:User|系统|机器)\))?$", re.IGNORECASE)
# Electron apps installed through npm leave an "@scope" folder behind. It is
# not an application and must not be reported as one.
_SCOPE = re.compile(r"^@[\w.-]+$")


def _clean_name(name: str) -> str:
    """Strip the version and installer noise an uninstall entry carries."""
    cleaned = _VERSION_TAIL.sub("", name).strip()
    return cleaned or name.strip()


def _app_folders(base: Path) -> list[str]:
    """Directory names under base that look like applications."""
    if not base.is_dir():
        return []
    names: list[str] = []
    for path in base.iterdir():
        if not path.is_dir():
            continue
        if _SCOPE.match(path.name):
            continue
        names.append(path.name)
    return names


def _resolve(raw: str, home: Path, appdata: Path) -> Path:
    if raw.startswith("~/"):
        return home / raw[2:]
    if raw.startswith("%APPDATA%/"):
        return appdata / raw[len("%APPDATA%/"):]
    if raw.startswith("%LOCALAPPDATA%/"):
        return appdata.parent / "Local" / raw[len("%LOCALAPPDATA%/"):]
    return Path(raw)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise AgentError("这个配置文件含有注释或格式特殊，不能自动修改。请改用复制提示词。") from exc
    if not isinstance(value, dict):
        raise AgentError("这个配置文件的内容不是对象，不能自动修改。请改用复制提示词。")
    return value


def _python() -> Path:
    venv = REPO_ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return venv if venv.exists() else Path(sys.executable)


def server_entry(style: str, memory_home: Path) -> dict[str, Any]:
    """Launch with the project's own python so no console window opens."""
    command = [_python().as_posix(), "-m", "membrain.server"]
    env = {"MEMBRAIN_HOME": Path(memory_home).as_posix()}
    if style == "opencode":
        return {"type": "local", "command": command, "environment": env, "enabled": True}
    return {"command": command[0], "args": command[1:], "env": env}


def _installed(spec: AgentSpec, home: Path, appdata: Path, which: Callable[[str], str | None]) -> bool:
    return any(which(c) for c in spec.commands) or any(
        _resolve(p, home, appdata).exists() for p in spec.paths)


def _has_text(path: Path, needle: str) -> bool:
    try:
        return needle in path.read_text(encoding="utf-8-sig", errors="ignore")
    except OSError:
        return False


def _status(spec: AgentSpec, home: Path, appdata: Path) -> str:
    """connected | partial | not_connected | unreadable | manual (cannot be checked)."""
    if spec.key == "claude-code":
        mcp = _has_text(home / ".claude.json", '"membrain"')
        hooks = _has_text(home / ".claude" / "settings.json", "scripts/hook.py")
        return "connected" if mcp and hooks else "partial" if mcp or hooks else "not_connected"
    if spec.key == "codex":
        mcp = _has_text(home / ".codex" / "config.toml", "mcp_servers.membrain")
        hooks = _has_text(home / ".codex" / "hooks.json", "hook.py")
        return "connected" if mcp and hooks else "partial" if mcp or hooks else "not_connected"
    if spec.config is None:
        return "manual"
    path = _resolve(spec.config, home, appdata)
    if not path.exists():
        return "not_connected"
    try:
        servers = _read_json(path).get(spec.servers_key, {})
    except AgentError:
        return "unreadable"
    return "connected" if isinstance(servers, dict) and SERVER_NAME in servers else "not_connected"


def _registry_names() -> list[str]:
    """Display names from the Windows uninstall list (empty off Windows)."""
    if os.name != "nt":
        return []
    import winreg

    names: list[str] = []
    for root in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
        for sub in (r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall",
                    r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"):
            try:
                with winreg.OpenKey(root, sub) as key:
                    for i in range(winreg.QueryInfoKey(key)[0]):
                        try:
                            with winreg.OpenKey(key, winreg.EnumKey(key, i)) as item:
                                names.append(str(winreg.QueryValueEx(item, "DisplayName")[0]))
                        except OSError:
                            continue
            except OSError:
                continue
    return names


RELAY_RULE = (
    "每次收到用户新消息，先调用 membrain 的 record_user_message，text 填用户原话，agent 填你的名字，再正常回答。"
    "只记录用户亲手输入的文字，不要记录你自己的回复、工具输出或检索到的记忆。"
    "需要回忆时调用 search_memory。不要写入 constraint 类型。"
)


def _hooks_json(command_py: str, script: str, codex: bool) -> dict[str, Any]:
    prompt = "codex-prompt-submit" if codex else "prompt-submit"
    def hook(event: str) -> dict[str, Any]:
        return {"hooks": [{"type": "command", "command": f'"{command_py}" "{script}" {event}', "timeout": 30}]}
    events = {"SessionStart": hook("session-start"), "UserPromptSubmit": hook(prompt)}
    if not codex:
        events.update({"Stop": hook("capture"), "PreCompact": hook("capture"), "SessionEnd": hook("capture")})
    return {"hooks": {k: [v] for k, v in events.items()}}


def build_prompt(spec: AgentSpec, memory_home: Path) -> str:
    """Instructions the owner can paste into the agent so it connects itself."""
    py = _python().as_posix()
    # The hook script ships with the repository, not with the wheel. From a
    # checkout it sits next to the package; from an installed wheel the owner
    # has to point at their own copy.
    root = REPO_ROOT if IN_CHECKOUT else Path.cwd()
    script = (root / "scripts" / "hook.py").as_posix()
    entry = server_entry("standard", memory_home)
    server_json = json.dumps({SERVER_NAME: entry}, ensure_ascii=False, indent=2)
    head = (f"请帮我把本机的 membrain 个人长期记忆接入 {spec.name}。先备份要改的文件，改动前把差异给我看，"
            "我确认后再保存。不要打印任何密钥或 token，也不要改动与 membrain 无关的内容。\n\n")
    if spec.key == "codex":
        toml = (f'[mcp_servers.membrain]\ncommand = "{py}"\nargs = ["-m", "membrain.server"]\n\n'
                f'[mcp_servers.membrain.env]\nMEMBRAIN_HOME = "{Path(memory_home).as_posix()}"')
        hooks = json.dumps(_hooks_json(py, script, True), ensure_ascii=False, indent=2)
        return (head + "1. 在 ~/.codex/config.toml 末尾追加下面的段落（已存在则跳过）：\n" + toml +
                "\n\n2. 在 ~/.codex/hooks.json 中合并下面的 hooks（保留已有的钩子）：\n" + hooks +
                "\n\n3. 完成后让我重启 Codex，并在 Codex 里信任这些钩子。")
    if spec.key == "claude-code":
        hooks = json.dumps(_hooks_json(py, script, False), ensure_ascii=False, indent=2)
        return (head + "1. 运行：claude mcp add --scope user membrain -- " + f'"{py}" -m membrain.server' +
                "\n2. 在 ~/.claude/settings.json 的 hooks 中合并下面的内容（保留已有的钩子，不要重复）：\n" + hooks +
                "\n3. 完成后让我重新打开 Claude Code。")
    return (head + f"位置：{spec.where}。\n添加下面这个 MCP 服务（stdio 方式）：\n{server_json}\n\n"
            "添加后确认它显示为已连接。然后把下面这条规则加入你的长期指令或规则文件，让它每轮自动执行：\n" + RELAY_RULE)


def _spec(key: str) -> AgentSpec:
    for spec in SPECS:
        if spec.key == key:
            return spec
    raise AgentError("没有找到这个 Agent。")


def connect(key: str, memory_home: Path, home: Path | None = None, appdata: Path | None = None) -> dict[str, Any]:
    """Add membrain to one agent's MCP config after backing the file up."""
    home = home or Path.home()
    appdata = appdata or Path(os.environ.get("APPDATA") or home / "AppData" / "Roaming")
    spec = _spec(key)
    if spec.protected:
        raise AgentError("这是受保护的配置，不会自动修改。请点“复制提示词”，交给该 Agent 自己配置。")
    if spec.config is None:
        raise AgentError("这个软件没有可自动修改的配置文件。请点“复制提示词”，按提示在软件里添加。")
    path = _resolve(spec.config, home, appdata)
    if not path.exists() and not path.parent.exists():
        raise AgentError("没有检测到这个软件的配置目录，请先安装并启动一次。")
    data = _read_json(path) if path.exists() else {}
    servers = data.setdefault(spec.servers_key, {})
    if not isinstance(servers, dict):
        raise AgentError("配置里的 MCP 部分格式特殊，不能自动修改。请改用复制提示词。")
    if SERVER_NAME in servers:
        return {"ok": True, "changed": False, "backup": None}
    backup = None
    if path.exists():
        backup = path.with_name(f"{path.name}.bak-membrain-{datetime.now():%Y%m%d-%H%M%S}")
        shutil.copy2(path, backup)
    servers[SERVER_NAME] = server_entry(spec.style, memory_home)
    temp = path.with_name(path.name + ".membrain-tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)
    return {"ok": True, "changed": True, "backup": backup.name if backup else None}


def _others(known: Iterable[AgentSpec], names: Iterable[str]) -> list[str]:
    """Applications the scan found that no spec describes, de-duplicated."""
    tokens = {t for spec in known for t in spec.match}
    found: dict[str, str] = {}
    for name in names:
        cleaned = _clean_name(name)
        if not cleaned or _SCOPE.match(cleaned):
            continue
        low = cleaned.lower()
        if any(k in low for k in KEYWORDS) and not any(t in low for t in tokens):
            found.setdefault(low, cleaned)
    return sorted(found.values(), key=str.lower)[:20]


def scan(memory_home: Path, home: Path | None = None, appdata: Path | None = None,
         which: Callable[[str], str | None] = shutil.which, registry: Callable[[], list[str]] = _registry_names,
         ) -> list[dict[str, Any]]:
    """Report every known agent and anything else that looks like one."""
    home = home or Path.home()
    appdata = appdata or Path(os.environ.get("APPDATA") or home / "AppData" / "Roaming")
    items: list[dict[str, Any]] = []
    for spec in SPECS:
        installed = _installed(spec, home, appdata, which)
        items.append({
            "key": spec.key, "name": spec.name, "installed": installed,
            "status": _status(spec, home, appdata) if installed else "not_installed",
            "capture": spec.capture, "protected": spec.protected,
            "can_connect": spec.config is not None and not spec.protected,
            "prompt": build_prompt(spec, memory_home),
        })
    folders: list[str] = []
    for base in (appdata, home / "AppData" / "Local" / "Programs"):
        folders.extend(_app_folders(base))
    for name in _others(SPECS, [*registry(), *folders]):
        generic = AgentSpec("other:" + re.sub(r"\W+", "-", name.lower()), name, ())
        items.append({"key": generic.key, "name": name, "installed": True, "status": "manual",
                      "capture": "relay", "protected": False, "can_connect": False,
                      "prompt": build_prompt(generic, memory_home)})
    return items
