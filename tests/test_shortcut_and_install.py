"""Tests for Windows desktop shortcut generation, CLI/HTTP routes, and install.ps1."""

from __future__ import annotations

import base64
import importlib.util
import json
import subprocess
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from membrain import autostart, cli
from membrain.dashboard import DashboardRuntime, create_server
from membrain.server import MemoryService


def _decode_encoded_ps(cmd: list[str]) -> str:
    idx = cmd.index("-EncodedCommand")
    return base64.b64decode(cmd[idx + 1]).decode("utf-16-le")


def test_desktop_shortcut_requires_windows(monkeypatch):
    monkeypatch.setattr(autostart, "supported", lambda: False)
    with pytest.raises(autostart.AutostartError, match="Windows"):
        autostart.create_desktop_shortcut()


def test_desktop_shortcut_builds_utf16_powershell_payload(tmp_path, monkeypatch):
    monkeypatch.setattr(autostart, "supported", lambda: True)
    seen: list[list[str]] = []
    expected_path = tmp_path / "桌面" / autostart.DESKTOP_SHORTCUT_NAME

    def fake_run(cmd, **kwargs):
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=(str(expected_path) + "\n").encode("utf-8"), stderr=b"")

    monkeypatch.setattr(autostart.subprocess, "run", fake_run)
    res = autostart.create_desktop_shortcut(port=2048, no_open=True, link_dir=tmp_path / "桌面")

    assert res["ok"] is True
    assert res["path"] == str(expected_path)
    assert res["arguments"] == "-m membrain.cli dashboard --port 2048 --no-open"
    assert len(seen) == 1
    ps_script = _decode_encoded_ps(seen[0])
    assert "$link.Arguments = '-m membrain.cli dashboard --port 2048 --no-open'" in ps_script
    assert "membrain 记忆后台.lnk" in ps_script
    assert str(tmp_path / "桌面") in ps_script


def test_cli_shortcut_and_script_wrapper(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(autostart, "supported", lambda: False)
    assert cli.main(["shortcut"]) == 1
    assert "Windows" in capsys.readouterr().err

    fake_path = str(tmp_path / autostart.DESKTOP_SHORTCUT_NAME)
    monkeypatch.setattr(
        autostart,
        "create_desktop_shortcut",
        lambda port=autostart.PORT, no_open=False: {
            "ok": True,
            "supported": True,
            "name": autostart.DESKTOP_SHORTCUT_NAME,
            "path": fake_path,
            "port": port,
        },
    )
    assert cli.main(["shortcut", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["path"] == fake_path

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "create_shortcut.py"
    spec = importlib.util.spec_from_file_location("create_shortcut_mod", script_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main() == 0


def test_dashboard_shortcut_endpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMBRAIN_HOME", str(tmp_path))
    monkeypatch.setenv("MEMBRAIN_WATCH", "0")
    svc = MemoryService()
    runtime = DashboardRuntime(svc, auto_extract_chat=False)
    server = create_server("127.0.0.1", 0, runtime=runtime)
    host, port = server.server_address
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def post(path: str, payload: dict) -> tuple[int, dict]:
        req = urllib.request.Request(
            f"http://{host}:{port}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            return err.code, json.loads(err.read().decode("utf-8"))

    try:
        monkeypatch.setattr(autostart, "supported", lambda: False)
        status, err_data = post("/api/shortcut", {})
        assert status == 400
        assert "Windows" in err_data["error"]

        monkeypatch.setattr(
            autostart,
            "create_desktop_shortcut",
            lambda port=1966, no_open=False: {"ok": True, "path": f"C:/Desktop/membrain.lnk", "port": port},
        )
        status, ok_data = post("/api/shortcut", {"port": 1966})
        assert status == 200
        assert ok_data["ok"] is True
        assert ok_data["path"] == "C:/Desktop/membrain.lnk"

        status, bad_port = post("/api/shortcut", {"port": 99999})
        assert status == 400
        assert "65535" in bad_port["error"]
    finally:
        server.shutdown()
        server.server_close()
        svc.index.close()


def test_install_ps1_is_ascii_safe_and_preserves_env_and_autostart_default():
    repo = Path(__file__).resolve().parents[1]
    ps1 = repo / "install.ps1"
    assert ps1.is_file()
    text = ps1.read_text(encoding="utf-8")
    assert text.isascii(), "Pure ASCII avoids Windows PowerShell 5.1 UTF-8 without BOM encoding bugs"
    assert "uv sync" in text
    assert 'if (-not (Test-Path ".env"))' in text
    assert "membrain init" in text
    assert "membrain shortcut" in text
    assert "install-autostart" not in text
