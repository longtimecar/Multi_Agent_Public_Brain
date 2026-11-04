from __future__ import annotations

import json

import pytest

from membrain import agents

NO_CMD = lambda _name: None  # noqa: E731


def _scan(tmp_path, registry=(), which=NO_CMD):
    return {i["key"]: i for i in agents.scan(tmp_path / "memory", home=tmp_path, appdata=tmp_path / "Roaming",
                                             which=which, registry=lambda: list(registry))}


def test_scan_reports_installed_state_and_never_marks_missing_agents_connectable(tmp_path):
    (tmp_path / ".workbuddy").mkdir()
    (tmp_path / ".workbuddy" / "mcp.json").write_text('{"mcpServers": {}}', encoding="utf-8")
    items = _scan(tmp_path)
    assert items["workbuddy"]["installed"] and items["workbuddy"]["status"] == "not_connected"
    assert items["workbuddy"]["can_connect"]
    assert items["gemini-cli"]["status"] == "not_installed"
    assert items["codex"]["protected"] and not items["codex"]["can_connect"]


def test_connect_backs_up_adds_server_and_keeps_other_settings(tmp_path):
    cfg = tmp_path / ".workbuddy" / "mcp.json"
    cfg.parent.mkdir()
    cfg.write_text(json.dumps({"mcpServers": {"other": {"command": "x"}}, "keep": 1}), encoding="utf-8")
    result = agents.connect("workbuddy", tmp_path / "memory", home=tmp_path, appdata=tmp_path / "Roaming")
    assert result["changed"] and (cfg.parent / result["backup"]).exists()
    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert data["keep"] == 1 and "other" in data["mcpServers"]
    entry = data["mcpServers"]["membrain"]
    assert entry["args"] == ["-m", "membrain.server"] and entry["env"]["MEMBRAIN_HOME"].endswith("memory")
    assert _scan(tmp_path)["workbuddy"]["status"] == "connected"
    again = agents.connect("workbuddy", tmp_path / "memory", home=tmp_path, appdata=tmp_path / "Roaming")
    assert not again["changed"]


def test_opencode_uses_its_own_schema(tmp_path):
    cfg = tmp_path / ".config" / "opencode" / "opencode.jsonc"
    cfg.parent.mkdir(parents=True)
    cfg.write_text('{"mcp": {}}', encoding="utf-8")
    agents.connect("opencode", tmp_path / "memory", home=tmp_path, appdata=tmp_path / "Roaming")
    entry = json.loads(cfg.read_text(encoding="utf-8"))["mcp"]["membrain"]
    assert entry["type"] == "local" and entry["command"][-2:] == ["-m", "membrain.server"]


@pytest.mark.parametrize("key", ["codex", "claude-code", "trae", "qoder"])
def test_protected_and_configless_agents_are_refused_with_a_prompt_instead(tmp_path, key):
    with pytest.raises(agents.AgentError):
        agents.connect(key, tmp_path / "memory", home=tmp_path, appdata=tmp_path / "Roaming")
    prompt = _scan(tmp_path)[key]["prompt"]
    assert "membrain" in prompt and "备份" in prompt


def test_commented_config_is_not_rewritten(tmp_path):
    cfg = tmp_path / ".cursor" / "mcp.json"
    cfg.parent.mkdir()
    original = '// my notes\n{"mcpServers": {}}'
    cfg.write_text(original, encoding="utf-8")
    with pytest.raises(agents.AgentError):
        agents.connect("cursor", tmp_path / "memory", home=tmp_path, appdata=tmp_path / "Roaming")
    assert cfg.read_text(encoding="utf-8") == original
    assert _scan(tmp_path)["cursor"]["status"] == "unreadable"


def test_connect_requires_an_existing_app_folder(tmp_path):
    with pytest.raises(agents.AgentError):
        agents.connect("cursor", tmp_path / "memory", home=tmp_path, appdata=tmp_path / "Roaming")
    assert not (tmp_path / ".cursor").exists()


def test_registry_scan_lists_unknown_agents_once_and_skips_known_ones(tmp_path):
    items = _scan(tmp_path, registry=["Kimi Desktop", "kimi desktop", "Cursor 0.4", "Codex", "Notepad++"])
    others = [i for i in items.values() if i["key"].startswith("other:")]
    assert [i["name"] for i in others] == ["Kimi Desktop"]
    assert others[0]["status"] == "manual" and "record_user_message" in others[0]["prompt"]


def test_leftover_data_folder_alone_is_not_installed(tmp_path):
    home = tmp_path / "home"
    appdata = home / "AppData" / "Roaming"
    (appdata / "SomeBot").mkdir(parents=True)
    spec = agents.AgentSpec("xbot", "XBot", ("xbot",),
                            paths=("%APPDATA%/SomeBot", "~/.xbot"), config="~/.xbot/mcp.json")
    which = lambda c: None  # noqa: E731
    # 只剩数据残留：既没配置目录也没注册表 → 不算安装
    assert agents._installed(spec, home, appdata, which=which, registry_names=[]) is False
    # 注册表里还挂着 → 算安装
    assert agents._installed(spec, home, appdata, which=which, registry_names=["XBot 1.0"]) is True
    # 启动过（留下配置目录）→ 算安装
    (home / ".xbot").mkdir(parents=True)
    assert agents._installed(spec, home, appdata, which=which, registry_names=[]) is True


def test_sibling_registry_names_do_not_count_for_deepseek(tmp_path):
    home = tmp_path / "home"
    appdata = home / "AppData" / "Roaming"
    (appdata / "DeepSeek").mkdir(parents=True)  # 残留数据
    spec = agents._spec("deepseek")
    which = lambda c: None  # noqa: E731
    # 注册表里只有更长名字的兄弟产品（DeepSeek Harness）→ 不算 DeepSeek 已安装
    assert agents._installed(spec, home, appdata, which=which,
                             registry_names=["DeepSeek Harness 0.2.1-rc.3"]) is False
    assert agents._installed(spec, home, appdata, which=which,
                             registry_names=["DeepSeek 3.0"]) is True
