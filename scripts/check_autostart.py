"""Report where the logon entry points (read-only).

One launcher starts the dashboard and opens the browser; the other starts it in
the background only. Both are valid entries, but only the silent one matches
"开机自启但不弹浏览器", so the difference is worth printing.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from membrain import autostart  # noqa: E402


def main() -> int:
    state = autostart.describe()
    link = autostart.link_path()
    if not state["enabled"]:
        print(f"未开启：{link.name} 不存在")
        print("在后台「模型连接 → 服务与启动」里打开开关，或运行 membrain install-autostart。")
        return 1
    print(f"{link.name}: {link.stat().st_size} 字节")
    print(f"  指向: {state['launcher'] or '（无法识别）'}")
    stray = autostart.startup_dir() / autostart.LEGACY_SCRIPT
    print(f"  启动文件夹里的多余脚本: {'有（需清理）' if stray.exists() else '无'}")
    if state["silent"]:
        print("\n结果: PASS —— 登录后在后台启动，不会弹出浏览器")
        return 0
    print("\n结果: 会打开浏览器 —— 关掉再打开一次开关即可改为静默启动")
    return 0


if __name__ == "__main__":
    sys.exit(main())
