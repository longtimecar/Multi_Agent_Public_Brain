"""Verify the Startup shortcut points at the launcher (read-only).

The shortcut carries no arguments on purpose: it aims at start-dashboard.vbs,
which holds the real command line. Checking for command flags in the .lnk
itself would report a failure for a shortcut that works.
"""

from __future__ import annotations

import sys
from pathlib import Path

STARTUP = Path.home() / "AppData/Roaming/Microsoft/Windows/Start Menu/Programs/Startup"
LINK = STARTUP / "membrain-dashboard.lnk"
STRAY_SCRIPT = STARTUP / "membrain-dashboard.vbs"


def main() -> int:
    if not LINK.exists():
        print(f"FAIL: {LINK.name} 不存在")
        return 1
    raw = LINK.read_bytes()
    print(f"{LINK.name}: {len(raw)} 字节")
    target_ok = (b"start-dashboard.vbs" in raw
                 or "start-dashboard.vbs".encode("utf-16-le") in raw)
    print(f"  目标指向启动脚本: {target_ok}")
    print(f"  启动文件夹里的多余脚本: {'有（需清理）' if STRAY_SCRIPT.exists() else '无'}")
    if not target_ok:
        print("  回退目标 pythonw:", b"pythonw" in raw)
    print("\n结果:", "PASS —— 登录时会运行 start-dashboard.vbs 启动服务" if target_ok
          else "FAIL —— 快捷方式没有指向启动脚本")
    return 0 if target_ok else 1


if __name__ == "__main__":
    sys.exit(main())
