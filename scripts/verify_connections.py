"""Prove each agent's membrain entry actually launches and answers.

Reading a config file only shows that a key exists. This spawns the exact
command each client would spawn, performs the MCP handshake, and runs one real
recall — the only evidence that a client is connected rather than configured.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

from membrain import agents  # noqa: E402

QUERY = "我昨晚说今天的任务是什么"


def entries(memory_home: Path) -> list[tuple[str, list[str], dict[str, str]]]:
    """(name, argv, env) for every client whose config names membrain."""
    home = Path.home()
    appdata = Path(os.environ.get("APPDATA") or home / "AppData" / "Roaming")
    found: list[tuple[str, list[str], dict[str, str]]] = []

    def add(name: str, entry: object) -> None:
        if not isinstance(entry, dict):
            return
        command = entry.get("command")
        argv = [str(p) for p in command] if isinstance(command, list) else \
               [str(command or "")] + [str(a) for a in (entry.get("args") or [])]
        env = entry.get("env") or entry.get("environment") or {}
        if argv and argv[0]:
            found.append((name, argv, {str(k): str(v) for k, v in env.items()}))

    for spec in agents.SPECS:
        if spec.config is None:
            continue
        path = agents._resolve(spec.config, home, appdata)
        if not path.exists():
            continue
        try:
            data = agents._read_json(path)
        except agents.AgentError:
            continue
        servers = data.get(spec.servers_key)
        if isinstance(servers, dict):
            add(spec.name, servers.get(agents.SERVER_NAME))
        elif isinstance(servers, list):
            for item in servers:
                if isinstance(item, dict) and item.get("name") == agents.SERVER_NAME:
                    add(spec.name, item)

    claude = home / ".claude.json"
    if claude.exists():
        try:
            add("Claude Code", json.loads(claude.read_text(encoding="utf-8-sig"))
                .get("mcpServers", {}).get(agents.SERVER_NAME))
        except ValueError:
            pass
    codex = home / ".codex" / "config.toml"
    if codex.exists():
        import re

        text = codex.read_text(encoding="utf-8", errors="replace")
        block = re.search(r"\[mcp_servers\.membrain\](.*?)(?=\n\[|\Z)", text, re.S)
        if block:
            cmd = re.search(r'command\s*=\s*"([^"]+)"', block.group(1))
            args = re.findall(r'"([^"]*)"', (re.search(r"args\s*=\s*\[(.*?)\]", block.group(1), re.S)
                                             or type("x", (), {"group": lambda s, i: ""})()).group(1))
            if cmd:
                found.append(("Codex", [cmd.group(1), *args], {}))
    return found


async def check(name: str, argv: list[str], env: dict[str, str]) -> tuple[str, str]:
    merged = {**os.environ, **env}
    params = StdioServerParameters(command=argv[0], args=argv[1:], env=merged)
    try:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await asyncio.wait_for(session.initialize(), timeout=40)
                tools = await session.list_tools()
                result = await asyncio.wait_for(
                    session.call_tool("search_memory", {"query": QUERY, "top_k": 3}), timeout=60)
                text = " ".join(getattr(c, "text", "") for c in result.content)[:120]
                return "OK", f"{len(tools.tools)} tools | 召回: {text[:90]}"
    except Exception as exc:  # a broken client must not stop the sweep
        return "FAIL", f"{type(exc).__name__}: {str(exc)[:110]}"


async def main() -> int:
    rows = entries(Path(os.environ.get("MEMBRAIN_HOME") or (Path(__file__).resolve().parents[1] / "memory")))
    seen: set[str] = set()
    failures = 0
    for name, argv, env in rows:
        if name in seen:
            continue
        seen.add(name)
        status, detail = await check(name, argv, env)
        failures += status == "FAIL"
        print(f"[{status}] {name:14} {argv[0][-28:]:30} {detail}")
    print(f"\n{len(seen)} 个客户端，失败 {failures} 个")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
