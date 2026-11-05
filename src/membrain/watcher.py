"""Background auto-capture watcher.

Tails the local session stores of every known agent and feeds user turns into
the same capture pipeline the hooks use, so writes never depend on a model
choosing to call a recording tool. Parsing is shared with session_import;
dedupe is shared with hooks.record_capture (stable turn uids), which makes
repeated passes cheap and idempotent.

Runs as a daemon thread inside the dashboard service. Opt out with
MEMBRAIN_WATCH=0; tune the cadence with MEMBRAIN_WATCH_INTERVAL (seconds).
"""
from __future__ import annotations

import hashlib
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable

from . import hooks, session_import

DEFAULT_INTERVAL = 300.0
# A session quiet this long is treated as finished — the only "that conversation
# is over" signal most agents give us.
DEFAULT_CONSOLIDATE_IDLE_MINUTES = 60
CONSOLIDATE_LIMIT = 5

AGENT_LABELS = {
    "claude-code": "Claude Code",
    "codex": "Codex",
    "opencode": "OpenCode",
    "workbuddy": "WorkBuddy",
    "cursor": "Cursor",
    "dsh": "DeepSeek Harness",
    "vscode": "VS Code (Copilot)",
    "trae": "Trae",
    "qoder": "Qoder",
}


def enabled() -> bool:
    return os.environ.get("MEMBRAIN_WATCH", "1").strip().lower() not in ("0", "false", "off", "no")


def consolidate_enabled() -> bool:
    return os.environ.get("MEMBRAIN_CONSOLIDATE", "1").strip().lower() not in ("0", "false", "off", "no")


def consolidate_idle_minutes() -> int:
    try:
        value = int(os.environ.get("MEMBRAIN_CONSOLIDATE_IDLE_MINUTES", "")
                    or DEFAULT_CONSOLIDATE_IDLE_MINUTES)
    except ValueError:
        return DEFAULT_CONSOLIDATE_IDLE_MINUTES
    return max(5, value)


def interval() -> float:
    try:
        value = float(os.environ.get("MEMBRAIN_WATCH_INTERVAL", "") or DEFAULT_INTERVAL)
    except ValueError:
        return DEFAULT_INTERVAL
    return max(30.0, value)


def _turn_uid(origin: str, index: int, text: str) -> str:
    """Stable across watcher passes so re-parsing a grown session never duplicates."""
    return hashlib.sha256(f"watch|{origin}|{index}|{text}".encode("utf-8")).hexdigest()[:16]


def watch_once(svc: Any) -> dict[str, Any]:
    """One capture pass over all agent stores. Returns run stats."""
    stats: dict[str, Any] = {"sessions": 0, "user_turns": 0, "captured": 0,
                             "exact_ts": 0, "estimated_ts": 0, "errors": []}
    try:
        sessions = session_import.collect()
    except Exception as exc:
        stats["errors"].append(str(exc))
        return stats
    stats["sessions"] = len(sessions)
    for rs in sessions:
        try:
            turns = []
            for index, (role, text) in enumerate(rs.turns or []):
                if role != "user" or not text.strip():
                    continue
                # The turn's own log time when the source kept one; otherwise a
                # session-level anchor. Either way it is never the capture time:
                # stamping "now" onto a backfilled turn makes every later
                # time-based recall ("昨晚说的") answer with the wrong day.
                ts, exact = rs.ts_for(index)
                if ts is None:
                    ts, exact = datetime.now(timezone.utc), False
                turns.append(("user", text, _turn_uid(rs.origin, index, text), ts,
                              "exact" if exact else "estimated"))
            if not turns:
                continue
            stats["user_turns"] += len(turns)
            stats["exact_ts"] += sum(1 for t in turns if t[4] == "exact")
            stats["estimated_ts"] += sum(1 for t in turns if t[4] != "exact")
            agent = AGENT_LABELS.get(rs.source, rs.source)
            stats["captured"] += hooks.record_capture(svc, session=rs.origin, agent=agent, turns=turns)
        except Exception as exc:
            stats["errors"].append(f"{rs.source}:{rs.origin}: {exc}")
    return stats


class Watcher:
    """Daemon loop with status reporting for the dashboard."""

    def __init__(self, svc: Any, on_capture: Callable[[], None] | None = None,
                 poll_interval: float | None = None) -> None:
        self.svc = svc
        self.on_capture = on_capture
        self.poll_interval = poll_interval or interval()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._status: dict[str, Any] = {
            "enabled": enabled(),
            "interval_seconds": self.poll_interval,
            "last_run": None,
            "total_captured": 0,
            "runs": 0,
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def _update(self, stats: dict[str, Any]) -> None:
        with self._lock:
            self._status["last_run"] = {
                "at": datetime.now(timezone.utc).isoformat(),
                **stats,
            }
            self._status["total_captured"] += int(stats.get("captured", 0))
            self._status["runs"] += 1

    def run_once(self) -> dict[str, Any]:
        stats = watch_once(self.svc)
        self._update(stats)
        if stats.get("captured") and self.on_capture is not None:
            try:
                self.on_capture()
            except Exception:
                pass
        self._consolidate_idle(stats)
        return stats

    def _consolidate_idle(self, stats: dict[str, Any]) -> None:
        """Summarise sessions that went quiet, on the same pass that captures them.

        Idempotent and cheap: a session already summarised is skipped by the
        existing-file check, so this costs nothing on a pass with nothing new.
        Runs here rather than in the dashboard so capture and summarisation both
        survive the browser being closed.
        """
        if not consolidate_enabled():
            return
        try:
            from . import consolidate

            report = consolidate.consolidate(self.svc, idle_minutes=consolidate_idle_minutes(),
                                             limit=CONSOLIDATE_LIMIT)
            stats["summarised"] = len(report.written)
        except Exception:
            pass  # a summary is a convenience; it must never break capture

    def _loop(self) -> None:
        # First pass soon after boot so a restart catches up quickly.
        self._stop.wait(8)
        while not self._stop.is_set():
            try:
                self.run_once()
            except Exception:
                pass
            self._stop.wait(self.poll_interval)

    def start(self) -> None:
        if not enabled() or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="membrain-watcher", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def serve(self) -> None:
        """Run the capture loop in the foreground, with no dashboard involved.

        Capture is the only write path that does not depend on a model choosing
        to call a tool, so it must not depend on a UI process being open either.
        Hosting it inside the dashboard meant a night with no browser tab open
        was a night with no memory.
        """
        if not enabled():
            return
        while not self._stop.is_set():
            try:
                self.run_once()
            except Exception:
                pass  # one bad pass must not end the watcher
            self._stop.wait(self.poll_interval)
