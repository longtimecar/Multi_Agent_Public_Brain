"""The logon entry must start the service without taking over the screen."""

from __future__ import annotations

from pathlib import Path

import pytest

from membrain import autostart


@pytest.fixture
def startup(tmp_path, monkeypatch):
    folder = tmp_path / "Startup"
    folder.mkdir()
    monkeypatch.setattr(autostart, "startup_dir", lambda: folder)
    monkeypatch.setattr(autostart, "supported", lambda: True)
    return folder


def _write_link(folder: Path, launcher: str) -> None:
    # A .lnk stores its target more than once, in ANSI and in UTF-16; the
    # reader has to cope with either copy.
    (folder / autostart.LINK_NAME).write_bytes(launcher.encode("utf-16-le"))


def test_no_entry_means_autostart_is_off(startup):
    state = autostart.describe()
    assert state["enabled"] is False
    assert state["silent"] is False


def test_an_entry_that_opens_a_browser_is_not_called_silent(startup):
    """The old shortcut pointed at the launcher that opens a tab.

    Reporting that as a working "开机自启" is exactly the kind of half-truth
    the toggle exists to avoid: the service does start, and so does a browser.
    """
    _write_link(startup, autostart.VISIBLE_LAUNCHER)
    state = autostart.describe()
    assert state["enabled"] is True
    assert state["silent"] is False
    assert state["launcher"] == autostart.VISIBLE_LAUNCHER


def test_enable_writes_a_launcher_that_stays_in_the_background(startup, tmp_path, monkeypatch):
    launcher = tmp_path / autostart.SILENT_LAUNCHER

    def write_launcher() -> Path:
        launcher.write_text(autostart._launcher_body(), encoding="ascii")
        return launcher

    monkeypatch.setattr(autostart, "_write_launcher", write_launcher)
    monkeypatch.setattr(autostart, "_create_shortcut",
                        lambda target: _write_link(startup, target.name))

    state = autostart.enable()
    assert state["enabled"] is True and state["silent"] is True

    body = launcher.read_text(encoding="ascii")
    assert "--no-open" in body, "登录时不能打开浏览器"
    assert "--restart" in body, "还要换掉上一次会话留下的进程"


def test_enable_rewrites_the_old_browser_entry(startup, tmp_path, monkeypatch):
    _write_link(startup, autostart.VISIBLE_LAUNCHER)
    launcher = tmp_path / autostart.SILENT_LAUNCHER
    monkeypatch.setattr(autostart, "_write_launcher", lambda: launcher)
    monkeypatch.setattr(autostart, "_create_shortcut",
                        lambda target: _write_link(startup, target.name))

    assert autostart.enable()["silent"] is True
    assert autostart.describe()["launcher"] == autostart.SILENT_LAUNCHER


def test_a_shortcut_that_the_system_drops_is_reported(startup, tmp_path, monkeypatch):
    """安全软件拦下启动项时不能假装已开启。"""
    monkeypatch.setattr(autostart, "_write_launcher", lambda: tmp_path / autostart.SILENT_LAUNCHER)
    monkeypatch.setattr(autostart, "_create_shortcut", lambda target: None)

    with pytest.raises(autostart.AutostartError):
        autostart.enable()


def test_disable_removes_the_entry_and_any_stray_script(startup):
    _write_link(startup, autostart.SILENT_LAUNCHER)
    (startup / autostart.LEGACY_SCRIPT).write_text("' old", encoding="ascii")

    state = autostart.disable()
    assert state["enabled"] is False
    assert not (startup / autostart.LINK_NAME).exists()
    assert not (startup / autostart.LEGACY_SCRIPT).exists(), "两个机制同时存在会在登录时抢着启动"
