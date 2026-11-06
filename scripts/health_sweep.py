"""Read-only functional sweep of the running dashboard.

Every feature is exercised through the same HTTP surface the page uses; a green
result means the endpoint answered with real data, not that the code compiles.
Nothing here writes: no import, no reindex, no chat.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:1966"
for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
    os.environ.pop(key, None)


def get(path: str, timeout: float = 45) -> tuple[str, str]:
    started = time.monotonic()
    try:
        with urllib.request.urlopen(BASE + path, timeout=timeout) as response:
            body = response.read().decode("utf-8", "replace")
        took = time.monotonic() - started
        return "OK", f"{took:5.1f}s {body[:150]}"
    except urllib.error.HTTPError as exc:
        return "HTTP", f"{exc.code} {exc.read()[:120].decode('utf-8', 'replace')}"
    except Exception as exc:  # noqa: BLE001 - a probe must report, never raise
        return "FAIL", f"{type(exc).__name__}: {exc}"


CHECKS = [
    ("模型配置", "/api/model"),
    ("总览统计", "/api/overview"),
    ("个人档案", "/api/profile"),
    ("记忆列表", "/api/facts?limit=3"),
    ("记忆检索", "/api/facts?q=%E4%BB%BB%E5%8A%A1&limit=3"),
    ("捕获记录", "/api/episodes?limit=3"),
    ("Agent 接入", "/api/agents"),
    ("采集器状态", "/api/watcher"),
    ("聊天会话列表", "/api/chat-sessions"),
    ("会话归档", "/api/archive?limit=2"),
    ("资料库文档", "/api/rag/documents"),
    ("导入批次", "/api/import/batches"),
    ("会话差异", "/api/session-diffs"),
    ("记忆健康(单类)", "/api/health?kinds=command_misfiled"),
    ("召回验证状态", "/api/recall/verify/status"),
    ("提炼任务状态", "/api/jobs/extract"),
    ("待办任务(排除项)", "/api/tasks"),
]


def main() -> int:
    rows = []
    for label, path in CHECKS:
        status, detail = get(path)
        rows.append((label, path, status, detail))
    width = max(len(r[0]) for r in rows)
    bad = 0
    for label, path, status, detail in rows:
        if status != "OK":
            bad += 1
        print(f"[{status}] {label:<{width}} {detail}")
    print(f"\n{len(rows)} 项，异常 {bad} 项")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
