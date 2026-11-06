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
    needs_uninstaller: bool = False        # desktop apps: leftover data dirs must not count as installed


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
                     "~/.qoder", "~/.qoder-cn"),
              config="~/.qoder-cn/settings.json"),
    AgentSpec("deepseek", "DeepSeek", ("deepseek",),
              paths=("%APPDATA%/DeepSeek", "~/.deepseek"),
              config="~/.deepseek/mcp.json"),
    AgentSpec("deepseek-harness", "DeepSeek Harness", ("deepseek harness", "deepseek-harness", "dsh"),
              paths=("%APPDATA%/@deepseek-ai", "%LOCALAPPDATA%/@deepseek-aidsh-desktop-updater",
                     "~/.deepseek-harness", "~/.dsh"),
              where="在 DeepSeek Harness 的 设置 → MCP 里新增一个自定义 MCP 服务"),
    # A separate product whose name merely starts with "DeepSeek". Declaring it
    # here is what keeps its leftover data directory and uninstall entry from
    # being reported as the DeepSeek desktop app — the residue guard below skips
    # registry names claimed by a sibling spec with a longer match token.
    AgentSpec("deepseek-bot", "DeepSeek Bot", ("deepseek bot", "deepseekbot"),
              paths=("%APPDATA%/DeepSeek Bot",),
              needs_uninstaller=True,
              where="DeepSeek Bot 与 DeepSeek 是两个产品；如需接入请在它自己的设置里添加 MCP 服务"),
    AgentSpec("doubao", "Doubao", ("doubao",),
              paths=("%APPDATA%/Doubao", "~/.doubao"),
              config="~/.doubao/mcp.json", needs_uninstaller=True),
    AgentSpec("qwen", "千问 (Qwen)", ("qwen", "qianwen", "千问"),
              paths=("%LOCALAPPDATA%/Qianwen", "%LOCALAPPDATA%/QianwenUpdater", "~/.qwen-agent"),
              where="在千问的 设置 → MCP / 工具 里新增一个自定义 MCP 服务"),
    AgentSpec("vscode", "VS Code (Copilot 等插件)", ("visual studio code", "vscode"), commands=("code",),
              paths=("%APPDATA%/Code",),
              config="%APPDATA%/Code/User/mcp.json", servers_key="servers"),
    AgentSpec("grok-bot", "Grok Bot", ("grok bot", "grokbot", "grok"),
              paths=("%APPDATA%/Grok Bot", "~/.grokbot"),
              config="~/.grokbot/settings.json", servers_key="mcpBoxServers", style="grokbot",
              where="在 Grok Bot 的 设置 → MCP 里添加一个自定义 MCP 服务"),
    AgentSpec("marvis", "Marvis", ("marvis",),
              paths=("%LOCALAPPDATA%/Marvis", "%APPDATA%/marvis", "~/.marvis"),
              where="在 Marvis 的设置或技能里添加自定义 MCP 服务；没有该入口就写进它的长期指令"),
)

# Names worth reporting when found by the scan but not described above.
KEYWORDS = ("claude", "codex", "cursor", "gemini", "copilot", "trae", "qoder", "kimi", "doubao", "deepseek",
            "grok", "marvis", "windsurf", "cline", "roo", "aider", "zed",
            "codebuddy", "windsurf", "cline", "aider", "hermes", "qwen", "qianwen", "agent")

# A trailing version turns one product into several entries in the Windows
# uninstall list: "DeepSeek Bot" and "DeepSeek Bot 0.2.3" are the same app.
_VERSION_TAIL = re.compile(
    r"\s+v?\d+(?:\.\d+)*(?:[-+][A-Za-z0-9.]+)*(?:\s*\((?:User|系统|机器)\))?$", re.IGNORECASE)
# Electron apps installed through npm leave an "@scope" folder behind. It is
# not an application and must not be reported as one.
_SCOPE = re.compile(r"^@[\w.-]+$")


def agent_key(name: str) -> str:
    """Canonical identity for an agent name, for isolation comparisons.

    One agent reaches the store under several spellings: the watcher labelled a
    turn "WorkBuddy" while hooks and the relay protocol wrote "workbuddy", and
    "Claude Code" against the spec key "claude-code". Isolation compared those
    exactly, so disconnecting an agent left every memory captured under another
    spelling visible — a control that silently half-worked. Stored data is
    immutable, so the comparison normalises instead: case, and the
    space/hyphen/underscore separators that differ between display names and
    keys, all folded away.
    """
    return "-".join((name or "").replace("_", " ").replace("-", " ").split()).casefold()


def is_disabled(name: str, disabled: Iterable[str]) -> bool:
    """True when `name` is the same agent as one of the disconnected keys."""
    if not name or not disabled:
        return False
    key = agent_key(name)
    return any(key == agent_key(other) for other in disabled)


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


def _launch_parts(entry: dict[str, Any]) -> list[str]:
    """The argv an entry would run, whether it stores it as a list or as command+args."""
    command = entry.get("command")
    if isinstance(command, list):
        return [str(part) for part in command]
    return [str(command or "")] + [str(a) for a in (entry.get("args") or [])]


def _same_server(existing: Any, desired: dict[str, Any]) -> bool:
    """True when an entry already launches this brain from this install.

    A reinstall moves the project to a new path; an entry left pointing at the
    old one looks connected in every config but fails at launch, so the launch
    command — not the key name — decides whether a config is already correct.
    """
    if not isinstance(existing, dict):
        return False
    if str(existing.get("type") or "local") != str(desired.get("type") or "local"):
        return False
    have = [p.replace("\\", "/") for p in _launch_parts(existing)]
    want = [p.replace("\\", "/") for p in _launch_parts(desired)]
    if have != want:
        return False
    have_env = existing.get("env") or existing.get("environment") or {}
    want_env = desired.get("env") or desired.get("environment") or {}
    if not isinstance(have_env, dict):
        return False
    return {k: str(v).replace("\\", "/") for k, v in have_env.items()} == \
           {k: str(v).replace("\\", "/") for k, v in want_env.items()}


def _installed(spec: AgentSpec, home: Path, appdata: Path, which: Callable[[str], str | None],
               registry_names: Iterable[str] = ()) -> bool:
    if any(which(c) for c in spec.commands):
        return True
    if not any(_resolve(p, home, appdata).exists() for p in spec.paths):
        return False
    if spec.needs_uninstaller:
        # A desktop app whose data directory survived an uninstall is not installed.
        # Require the Windows uninstall list to still list the app itself.
        lowered = (name.lower() for name in registry_names)
        if not any(token in name for token in spec.match for name in lowered):
            return False
        return True
    if spec.config is not None:
        # Residue guard: a data folder alone is not "installed". The app either
        # left its config directory behind (it ran at least once) or is still
        # listed in the Windows uninstall list. Names claimed by a sibling spec
        # with a longer match token ("DeepSeek Harness" vs "DeepSeek") don't count.
        if _resolve(spec.config, home, appdata).parent.exists():
            return True
        own = max((len(t) for t in spec.match), default=0)
        longer = [t for s2 in SPECS if s2 is not spec for t in s2.match if len(t) > own]
        lowered = [name.lower() for name in registry_names]
        return any(token in name for token in spec.match
                   for name in lowered if not any(t in name for t in longer))
    return True


def _running_images() -> set[str]:
    """Lowercase executable names of running processes (Windows only, fail-open)."""
    if os.name != "nt":
        return set()
    try:
        import subprocess

        # Read bytes, not text. tasklist prints in the console codepage, and a
        # single non-ASCII process name used to make the text-mode reader fail,
        # leaving stdout None and taking the whole agent scan down with it.
        out = subprocess.run(["tasklist", "/fo", "csv", "/nh"], capture_output=True,
                             timeout=8, creationflags=0x08000000)
    except Exception:
        return set()
    raw = out.stdout or b""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = raw.decode("mbcs", "replace")
        except (LookupError, UnicodeDecodeError):
            text = raw.decode("utf-8", "replace")
    names: set[str] = set()
    for line in text.splitlines():
        match = re.match(r'"([^"]+)"', line.strip())
        if match:
            names.add(match.group(1).lower())
    return names


def _is_running(spec: AgentSpec, images: set[str]) -> bool:
    """True when one of THIS product's own processes is running.

    A process name claimed by a sibling spec with a longer token does not count.
    Without that, a running "deepseek harness.exe" reported the DeepSeek desktop
    app as installed — the same confusion the uninstall-list guard already
    handles, arriving through the process table instead.
    """
    own = max((len(t) for t in spec.match), default=0)
    longer = [t for other in SPECS if other is not spec for t in other.match if len(t) > own]
    for image in images:
        base = image[:-4] if image.endswith(".exe") else image
        if any(t in base for t in longer):
            continue
        if any(token in base for token in spec.match):
            return True
    return False


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
    if isinstance(servers, list):  # e.g. Grok Bot's mcpBoxServers
        return "connected" if any(
            isinstance(s, dict) and s.get("name") == SERVER_NAME
            and _launches_this_install(s) for s in servers) else "not_connected"
    if not isinstance(servers, dict) or SERVER_NAME not in servers:
        return "not_connected"
    # An entry that still points at a previous install reads as connected in
    # every config file but dies at launch, which is how OpenCode kept failing
    # while the dashboard called it healthy.
    return "connected" if _launches_this_install(servers[SERVER_NAME]) else "not_connected"


def _launches_this_install(entry: Any) -> bool:
    """True when the entry starts this project's python with this module.

    The memory home is deliberately not compared: the status question is
    "will this launch work", and a different MEMBRAIN_HOME is a valid choice.
    """
    if not isinstance(entry, dict):
        return False
    parts = _launch_parts(entry)
    want = [_python().as_posix(), "-m", "membrain.server"]
    return [p.replace("\\", "/") for p in parts] == want


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
    "本机的 membrain 后台服务会自动从你的会话日志捕获用户输入并提炼成长期记忆，通常不需要你写入。"
    "需要回忆时调用 search_memory。若发现自动捕获没有生效、需要补记一条用户原话时，"
    "调用 record_user_message（text 填用户原话，agent 填你的名字），或运行下面给出的独立记录工具。"
    "只记录用户亲手输入的文字，不要记录你自己的回复、工具输出或检索到的记忆。不要写入 constraint 类型。"
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
    record_tool = (root / "tools" / "record.py").as_posix()
    return (head + f"位置：{spec.where}。\n添加下面这个 MCP 服务（stdio 方式）：\n{server_json}\n\n"
            "添加后确认它显示为已连接。然后把下面这条规则加入你的长期指令或规则文件，让它每轮自动执行：\n" + RELAY_RULE +
            f"\n\n备用的独立记录工具（自动捕获失效时才需要，可直接运行）：{record_tool}\n"
            f"用法：python \"{record_tool}\" --text \"用户原话\" --agent \"{spec.name}\"")


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
        raise AgentError(
            f"没有找到 {spec.name} 的配置目录（期望：{path.parent}）。"
            f"请先安装并启动一次 {spec.name}，让它生成配置目录后再点一键接入；"
            f"或现在就点“复制提示词”，在软件里手动添加。")
    data = _read_json(path) if path.exists() else {}
    servers = data.setdefault(spec.servers_key, [] if spec.style == "grokbot" else {})
    if spec.style == "grokbot":
        if not isinstance(servers, list):
            raise AgentError("配置里的 MCP 列表格式特殊，不能自动修改。请改用复制提示词。")
        desired = server_entry("standard", memory_home)
        existing = next((s for s in servers
                         if isinstance(s, dict) and s.get("name") == SERVER_NAME), None)
        if existing is not None and _same_server(existing, desired):
            return {"ok": True, "changed": False, "backup": None}
    else:
        if not isinstance(servers, dict):
            raise AgentError("配置里的 MCP 部分格式特殊，不能自动修改。请改用复制提示词。")
        desired = server_entry(spec.style, memory_home)
        existing = servers.get(SERVER_NAME)
        if existing is not None and _same_server(existing, desired):
            return {"ok": True, "changed": False, "backup": None}
    repaired = existing is not None
    backup = None
    if path.exists():
        backup = path.with_name(f"{path.name}.bak-membrain-{datetime.now():%Y%m%d-%H%M%S}")
        shutil.copy2(path, backup)
    if spec.style == "grokbot":
        entry = server_entry("standard", memory_home)
        replacement = {"name": SERVER_NAME, "command": entry["command"],
                       "args": entry.get("args", []), "env": entry.get("env", {})}
        for index, item in enumerate(servers):
            if isinstance(item, dict) and item.get("name") == SERVER_NAME:
                servers[index] = replacement
                break
        else:
            servers.append(replacement)
    else:
        servers[SERVER_NAME] = server_entry(spec.style, memory_home)
    temp = path.with_name(path.name + ".membrain-tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)
    return {"ok": True, "changed": True, "repaired": repaired,
            "backup": backup.name if backup else None}


def disconnect(key: str, home: Path | None = None, appdata: Path | None = None) -> dict[str, Any]:
    """Remove membrain from one agent's MCP config when the file is ours to edit.

    Protected configs (Claude Code, Codex) are never edited automatically; the
    owner removes those entries by hand or through the agent itself. Memory
    isolation is handled separately by the service and always applies.
    """
    home = home or Path.home()
    appdata = appdata or Path(os.environ.get("APPDATA") or home / "AppData" / "Roaming")
    spec = _spec(key)
    if spec.protected:
        return {"ok": True, "config_removed": False, "manual": True,
                "note": f"{spec.name} 的配置受保护，请按复制提示词里的位置手动移除 membrain 条目。"}
    if spec.config is None:
        return {"ok": True, "config_removed": False, "manual": True,
                "note": f"{spec.name} 的 MCP 配置在软件界面里，请在 {spec.where} 手动移除。"}
    path = _resolve(spec.config, home, appdata)
    if not path.exists():
        return {"ok": True, "config_removed": False, "manual": False}
    data = _read_json(path)
    servers = data.get(spec.servers_key)
    keep: list[Any] | None = None
    if isinstance(servers, list):  # e.g. Grok Bot's mcpBoxServers
        keep = [s for s in servers if not (isinstance(s, dict) and s.get("name") == SERVER_NAME)]
        if len(keep) == len(servers):
            return {"ok": True, "config_removed": False, "manual": False}
    elif not isinstance(servers, dict) or SERVER_NAME not in servers:
        return {"ok": True, "config_removed": False, "manual": False}
    backup = path.with_name(f"{path.name}.bak-membrain-{datetime.now():%Y%m%d-%H%M%S}")
    shutil.copy2(path, backup)
    if keep is not None:
        data[spec.servers_key] = keep
    else:
        servers.pop(SERVER_NAME)
    temp = path.with_name(path.name + ".membrain-tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)
    return {"ok": True, "config_removed": True, "backup": backup.name}


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
         images: Callable[[], set[str]] = _running_images,
         ) -> list[dict[str, Any]]:
    """Report every known agent and anything else that looks like one."""
    home = home or Path.home()
    appdata = appdata or Path(os.environ.get("APPDATA") or home / "AppData" / "Roaming")
    items: list[dict[str, Any]] = []
    names = registry()
    running_images = images()
    for spec in SPECS:
        running = _is_running(spec, running_images)
        installed = running or _installed(spec, home, appdata, which, names)
        items.append({
            "key": spec.key, "name": spec.name, "installed": installed,
            "running": running,
            "status": _status(spec, home, appdata) if installed else "not_installed",
            "capture": spec.capture, "protected": spec.protected,
            "can_connect": spec.config is not None and not spec.protected,
            "prompt": build_prompt(spec, memory_home),
        })
    folders: list[str] = []
    for base in (appdata, home / "AppData" / "Local" / "Programs"):
        folders.extend(_app_folders(base))
    for name in _others(SPECS, [*names, *folders]):
        generic = AgentSpec("other:" + re.sub(r"\W+", "-", name.lower()), name, ())
        items.append({"key": generic.key, "name": name, "installed": True, "running": False,
                      "status": "manual", "capture": "relay", "protected": False, "can_connect": False,
                      "prompt": build_prompt(generic, memory_home)})
    return items


def display_name(name: str) -> str:
    """Human label for whatever spelling an agent was stored under.

    The overview counted "WorkBuddy" and "workbuddy" as two sources, so one
    agent showed up twice with half the numbers each. Grouping by the canonical
    key and printing the spec name fixes the display without rewriting evidence.
    """
    key = agent_key(name)
    for spec in SPECS:
        if spec.key == key:
            return spec.name
    return (name or "unknown").strip() or "unknown"
