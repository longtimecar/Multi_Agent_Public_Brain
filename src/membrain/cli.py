"""CLI. The human-side surface: constraints, review, and conflict resolution.

Everything an agent must not be able to do through MCP lives here — writing a
hard constraint, retracting by hand, and resolving a contested pair.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

from .config import load_settings
from .models import Evidence, FactType, Status
from .server import MemoryService
from .store import StoreError

PROFILE_TEMPLATE = """\
# Profile

<!-- Hand-edited. Always injected at session start. Keep it under ~400 tokens. -->

## Identity

- Name:
- Languages:
- Location / timezone:

## Hard constraints

<!-- Put durable musts here, not in agent-written memory. -->

-

## Communication preferences

-
"""


def _svc() -> MemoryService:
    return MemoryService()


def cmd_init(args: argparse.Namespace) -> int:
    svc = _svc()
    settings = svc.settings
    settings.home.mkdir(parents=True, exist_ok=True)
    settings.facts_dir.mkdir(exist_ok=True)
    settings.episodes_dir.mkdir(exist_ok=True)
    if not settings.profile_path.exists():
        settings.profile_path.write_text(PROFILE_TEMPLATE, encoding="utf-8")
        print(f"created {settings.profile_path}")
    else:
        print(f"profile already exists: {settings.profile_path}")
    print(f"memory home: {settings.home}")
    return 0


def cmd_add(args: argparse.Namespace) -> int:
    svc = _svc()
    evidence = [
        Evidence(
            episode=args.episode or f"cli_{date.today().isoformat()}",
            quote=args.quote or args.claim,
            date=date.fromisoformat(args.date) if args.date else date.today(),
            origin="stated",
        )
    ]
    try:
        fact = svc.store.draft_fact(
            type=FactType(args.type),
            subject=args.subject,
            claim=args.claim,
            evidence=evidence,
            source_agent="cli",
        )
        stored = svc.store.write_fact(fact)
        svc.retriever.sync_fact(stored)
        flagged = svc.retriever.scan_conflicts()
    except (StoreError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"{stored.id}  conf={stored.confidence}  {stored.claim}")
    if flagged:
        print(f"! {len(flagged)} conflict(s) detected - resolve with `membrain resolve`:")
        for c in flagged:
            print(f"  {c.a.id} ({c.a.claim})  <->  {c.b.id} ({c.b.claim})  [{c.reason}]")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    svc = _svc()
    facts = svc.store.all_facts()
    if args.status:
        facts = [f for f in facts if f.status.value == args.status]
    for f in facts:
        sens = " sensitive" if f.sensitive else ""
        uri = f"  {f.uri}" if f.uri else ""
        print(f"{f.id}  {f.type.value:<10} {f.status.value:<10} conf={f.confidence:.2f}{sens}  [{f.subject}] {f.claim}{uri}")
    print(f"-- {len(facts)} fact(s)")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    svc = _svc()
    try:
        fact = svc.store.get_fact(args.id)
    except StoreError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(fact.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


def cmd_correct(args: argparse.Namespace) -> int:
    svc = _svc()
    try:
        svc.store.retract(args.id, args.reason)
        svc.retriever.sync_fact(svc.store.get_fact(args.id))
    except StoreError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"retracted {args.id}")
    return 0


def cmd_contest(args: argparse.Namespace) -> int:
    svc = _svc()
    svc.store.contest(args.ids, args.note or "manual contest")
    for fid in args.ids:
        svc.retriever.sync_fact(svc.store.get_fact(fid))
    print(f"contested: {', '.join(args.ids)}")
    return 0


def cmd_resolve(args: argparse.Namespace) -> int:
    svc = _svc()
    svc.store.resolve(args.keep, args.drop)
    for fid in (args.keep, args.drop):
        svc.retriever.sync_fact(svc.store.get_fact(fid))
    print(f"kept {args.keep}, retracted {args.drop}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    svc = _svc()
    llm = None
    if getattr(args, "use_intent", False) or getattr(args, "rerank_llm", False) or str(getattr(args, "mode", "") or "").lower() == "thinking":
        from .llm import LLMClient
        llm = LLMClient(svc.settings)
    mode = getattr(args, "mode", None)
    use_intent = True if getattr(args, "use_intent", False) else None
    use_rerank = True if getattr(args, "rerank", False) else None
    if getattr(args, "rerank_provider", None):
        svc.settings.rerank_provider = str(args.rerank_provider)
    if getattr(args, "rerank_llm", False):
        svc.settings._rerank_use_llm = True  # type: ignore[attr-defined]
        if not getattr(args, "rerank_provider", None):
            if getattr(svc.settings, "rerank_provider", "heuristic") == "heuristic":
                svc.settings.rerank_provider = "llm"
    # Explicit false when --mode quick without flags stays None so mode resolves.
    if mode and str(mode).lower() == "quick" and not getattr(args, "use_intent", False):
        use_intent = False
    if mode and str(mode).lower() == "quick" and not getattr(args, "rerank", False):
        use_rerank = False
    result = svc.retriever.recall(
        args.query,
        top_k=args.top_k,
        scope=getattr(args, "scope", None),
        level=getattr(args, "level", 2),
        type=getattr(args, "type", None),
        include_sensitive=getattr(args, "include_sensitive", False),
        use_intent=use_intent,
        use_rerank=use_rerank,
        session=getattr(args, "session", None),
        llm=llm,
        mode=mode,
        peer=getattr(args, "peer", None),
        peer_scope=getattr(args, "peer_scope", "all") or "all",
        purpose=getattr(args, "purpose", None),
    )
    print(result.text or "(no matches)")
    print(
        f"-- mode={result.mode} level={result.level} tokens={result.tokens} notes={result.notes}",
        file=sys.stderr,
    )
    return 0


def cmd_migrate_scheme_a(args: argparse.Namespace) -> int:
    """Backfill uri/abstract/sensitive on existing facts and rebuild index."""
    svc = _svc()
    report = svc.store.migrate_scheme_a()
    stats = svc.reindex_all()
    print(json.dumps({"migrate": report, "reindex": stats}, ensure_ascii=False, indent=2))
    return 0


def cmd_auto_commit(args: argparse.Namespace) -> int:
    """Run idle/token auto-commit: Phase1 archive cut + Phase2 extract/diff."""
    from . import auto_commit, session_archive
    from .extract import Extractor
    from .llm import LLMClient

    svc = _svc()
    pending = auto_commit.check_idle(
        svc.store,
        threshold=svc.settings.auto_commit_token_threshold,
        idle_minutes=svc.settings.auto_commit_idle_minutes,
    )
    taken = auto_commit.take_pending(svc.store)
    sessions = sorted(set(pending) | set(taken))
    if args.session:
        sessions = sorted(set(sessions) | {args.session})
    if not sessions and not args.force:
        print(json.dumps({"pending": [], "phase1": [], "phase2": []}, ensure_ascii=False))
        return 0
    if args.force and args.session and args.session not in sessions:
        sessions.append(args.session)
    phase1 = session_archive.commit_sessions(svc.store, sessions, reason="cli-auto-commit") if sessions else []
    phase2 = []
    for item in phase1:
        if item.get("archived") and item.get("archive_path"):
            def _factory(s=svc):
                return Extractor(s.store, s.retriever, LLMClient(s.settings))
            phase2.append(session_archive.phase2_process(
                svc.store, item["archive_path"], run_extract=True, extractor_factory=_factory
            ))
    if args.force and not phase2:
        report = Extractor(svc.store, svc.retriever, LLMClient(svc.settings)).run(
            args.session if args.session else None
        )
        print(json.dumps({"pending": sessions, "phase1": phase1, "extract": report.as_dict()}, ensure_ascii=False, indent=2))
        return 0 if not report.errors else 1
    print(json.dumps({
        "pending": sessions,
        "phase1": phase1,
        "phase2": [{"status": p.get("status"), "archive_path": p.get("archive_path"),
                    "summary": (p.get("diff") or {}).get("summary")} for p in phase2],
    }, ensure_ascii=False, indent=2))
    return 0


def cmd_reindex(args: argparse.Namespace) -> int:
    svc = _svc()
    stats = svc.reindex_all()
    print(json.dumps(stats))
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    svc = _svc()
    live = svc.store.live_facts()
    print(
        json.dumps(
            {
                "home": str(svc.settings.home),
                "facts": len(svc.store.all_facts()),
                "live": len(live),
                "contested": len([f for f in live if f.status == Status.CONTESTED]),
                "episodes": sum(1 for _ in svc.store.iter_episodes()),
                **svc.settings.describe(),
            },
            indent=2,
        )
    )
    return 0


def cmd_tasks(args: argparse.Namespace) -> int:
    """List work the owner handed to a later session and has not closed."""
    svc = _svc()
    tasks = svc.retriever.open_tasks(limit=getattr(args, "limit", 50) or 50)
    print(json.dumps({
        "open_tasks": [
            {
                "id": f.id,
                "subject": f.subject,
                "claim": f.claim,
                "stated_on": f.valid_from.isoformat(),
                "confidence": round(f.confidence, 2),
                "status": f.status.value,
            }
            for f in tasks
        ],
        "count": len(tasks),
        "hint": "membrain correct --id <id> 关闭已完成的任务；未关闭的任务不会被遗忘机制降权。",
    }, ensure_ascii=False, indent=2))
    return 0


def cmd_consolidate(args: argparse.Namespace) -> int:
    """Write session summaries (a derived view; never evidence)."""
    from . import consolidate

    svc = _svc()
    report = consolidate.consolidate(
        svc, session=args.session, idle_minutes=args.idle_minutes,
        limit=args.limit, force=args.force,
    )
    print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    """Foreground auto-capture. Runs with or without the dashboard."""
    from .watcher import Watcher, enabled, interval

    if not enabled():
        print("MEMBRAIN_WATCH is off; nothing to do.")
        return 0
    svc = _svc()
    poll = float(args.interval) if args.interval else interval()
    watcher = Watcher(svc, poll_interval=poll)
    if args.once:
        stats = watcher.run_once()
        print(json.dumps(stats, ensure_ascii=False, default=str))
        return 0
    print(f"membrain watcher: polling every {poll:.0f}s. Ctrl-C to stop.")
    try:
        watcher.serve()
    except KeyboardInterrupt:
        watcher.stop()
        print("\nstopped.")
    return 0


def cmd_autostart(args: argparse.Namespace) -> int:
    """Install or remove capture-at-login, so memory does not depend on the UI."""
    import subprocess
    import sys

    task_name = "membrain-watcher"
    if sys.platform != "win32":
        print(json.dumps({
            "installed": False,
            "platform": sys.platform,
            "how": ("Run `membrain watch` from a systemd user unit or a launchd "
                    "agent; both should restart it on failure."),
        }, ensure_ascii=False, indent=2))
        return 0

    python = Path(sys.executable).with_name("pythonw.exe")
    interpreter = str(python) if python.exists() else sys.executable
    if args.remove:
        done = subprocess.run(["schtasks", "/Delete", "/TN", task_name, "/F"],
                              capture_output=True, text=True)
        print(json.dumps({"removed": done.returncode == 0,
                          "detail": (done.stdout or done.stderr).strip()}, ensure_ascii=False))
        return 0 if done.returncode == 0 else 1

    # ONLOGON keeps capture running for the whole session; /RL LIMITED means it
    # never needs elevation, and /F replaces a previous install instead of failing.
    cmd = [
        "schtasks", "/Create", "/TN", task_name, "/SC", "ONLOGON", "/RL", "LIMITED", "/F",
        "/TR", f'"{interpreter}" -m membrain.cli watch',
    ]
    done = subprocess.run(cmd, capture_output=True, text=True)
    print(json.dumps({
        "installed": done.returncode == 0,
        "task_name": task_name,
        "command": f'"{interpreter}" -m membrain.cli watch',
        "detail": (done.stdout or done.stderr).strip(),
        "note": "需要重启或注销后生效；也可在任务计划程序里手动运行一次。",
    }, ensure_ascii=False, indent=2))
    return 0 if done.returncode == 0 else 1


def cmd_hook(args: argparse.Namespace) -> int:
    from .hooks import run

    return run(args.event)


def cmd_extract(args: argparse.Namespace) -> int:
    from .extract import Extractor
    from .llm import LLMClient

    svc = _svc()
    report = Extractor(svc.store, svc.retriever, LLMClient(svc.settings)).run(args.session)
    print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    return 0 if not report.errors else 1


def cmd_dashboard(args: argparse.Namespace) -> int:
    from .dashboard import create_server, open_dashboard, running_dashboard

    # Windows rejects 127.0.0.0 as a bind address even though it is in the
    # loopback range. Keep the requested port and normalize to the usable host.
    host = "127.0.0.1" if args.host == "127.0.0.0" else args.host
    # One dashboard at a time: a second launch (for example the desktop
    # shortcut clicked twice) only opens the browser on the running one.
    if running_dashboard(host, args.port):
        if not args.no_open:
            open_dashboard(host, args.port)
        return 0
    try:
        server = create_server(host, args.port)
    except (OSError, ValueError) as exc:
        print(f"error: cannot start dashboard: {exc}", file=sys.stderr)
        return 1
    url = f"http://{host}:{args.port}/"
    print(f"membrain dashboard: {url}")
    if not args.no_open:
        open_dashboard(host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        server.runtime.service.index.close()  # type: ignore[attr-defined]
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    from .config import IN_CHECKOUT, REPO_ROOT
    from .embeddings import EmbeddingError
    from .llm import LLMClient, LLMError

    svc = _svc()
    result = {"configuration": svc.settings.describe(), "index": svc.index.stats()}
    try:
        result["embedding_reachable"] = svc.embedder.embed(["synthetic health check"]) is not None
    except (EmbeddingError, ValueError):
        result["embedding_reachable"] = False
    try:
        result["llm_reachable"] = LLMClient(svc.settings).complete_json(
            'Return the JSON object {"ok":true}.', "health check").get("ok") is True
    except LLMError:
        result["llm_reachable"] = False
    events = ("SessionStart", "UserPromptSubmit", "Stop", "PreCompact", "SessionEnd")

    def hook_events(path: Path) -> dict[str, bool]:
        # Report an event only when it points at this project's hook script.
        try:
            hooks = json.loads(path.read_text(encoding="utf-8-sig")).get("hooks", {})
            return {name: "scripts/hook.py" in json.dumps(hooks.get(name, []), ensure_ascii=False).replace("\\\\", "/")
                    for name in events}
        except (OSError, ValueError, AttributeError):
            return {}

    project_settings = (REPO_ROOT / ".claude" / "settings.json") if IN_CHECKOUT \
        else (Path.cwd() / ".claude" / "settings.json")
    result["project_hook_events"] = hook_events(project_settings)
    result["global_hook_events"] = hook_events(Path.home() / ".claude" / "settings.json")
    from .agents import scan
    result["agents"] = {item["name"]: item["status"] for item in scan(svc.settings.home) if item["installed"]}
    result["global_install"] = ("claude-code connected" if result["global_hook_events"]
                                and all(result["global_hook_events"].values()) else "not installed")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    svc.index.close()
    return 0 if result["embedding_reachable"] and result["llm_reachable"] else 1


def cmd_session_diff(args: argparse.Namespace) -> int:
    """Read-only memory_diff.json for an archive directory."""
    from . import session_archive

    svc = _svc()
    path = session_archive.resolve_archive_path(svc.store, args.archive)
    diff = session_archive.read_memory_diff(path)
    print(json.dumps(diff, ensure_ascii=False, indent=2))
    return 0


def cmd_session_archive_phase2(args: argparse.Namespace) -> int:
    """Process pending Phase2 archive tasks (background worker entry)."""
    from . import session_archive
    from .extract import Extractor
    from .llm import LLMClient

    svc = _svc()
    tasks = []
    if args.archive:
        tasks = [{"archive_path": str(session_archive.resolve_archive_path(svc.store, args.archive))}]
    else:
        tasks = session_archive.pending_phase2_tasks(svc.store)
    results = []
    for task in tasks:
        def _factory(s=svc):
            return Extractor(s.store, s.retriever, LLMClient(s.settings))
        results.append(session_archive.phase2_process(
            svc.store, task["archive_path"], run_extract=not args.skip_extract, extractor_factory=_factory
        ))
    # Auto-increment: land each finished session archive as a library note
    # (fail-open; the LLM failure path falls back to rule fields inside).
    if getattr(svc.settings, "library_auto_import", True):
        from . import session_import
        for r in results:
            apath = r.get("archive_path")
            if not apath:
                continue
            try:
                r["library_import"] = session_import.import_session_archive(
                    svc, LLMClient(svc.settings), apath)
            except Exception as exc:  # never break the background worker
                r["library_import"] = {"errors": [str(exc)[:160] or exc.__class__.__name__]}
    print(json.dumps([{"status": r.get("status"), "archive_path": r.get("archive_path"),
                       "summary": (r.get("diff") or {}).get("summary"),
                       "library_import": r.get("library_import")} for r in results],
                     ensure_ascii=False, indent=2))
    return 0



def cmd_evolution(args: argparse.Namespace) -> int:
    """Gap 9: case -> trajectory -> experience (default off)."""
    from . import evolution
    svc = MemoryService()
    if args.evolution_cmd == "status":
        print(json.dumps({"enabled": evolution.is_enabled(svc.settings), "home": str(svc.settings.home)}, ensure_ascii=False, indent=2))
        return 0
    if args.evolution_cmd == "list":
        print(json.dumps(evolution.list_experiences(svc.settings.home), ensure_ascii=False, indent=2))
        return 0
    if not evolution.is_enabled(svc.settings) and not getattr(args, "force", False):
        print(json.dumps({"ok": False, "error": "evolution_disabled", "hint": "set MEMBRAIN_EVOLUTION=1 or pass --force"}, ensure_ascii=False))
        return 2
    if args.evolution_cmd == "promote":
        result = evolution.promote_live_cases(svc.store, svc.settings, limit=int(args.limit or 50))
        if getattr(args, "force", False) and not result.get("enabled"):
            # force: temporarily treat as enabled for this invoke only
            class _S: ...
            s = _S()
            s.home = svc.settings.home
            s.evolution_enabled = True
            result = evolution.promote_live_cases(svc.store, s, limit=int(args.limit or 50))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.evolution_cmd == "distill":
        result = evolution.distill_experience(svc.settings.home, args.case_id, title=args.title)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("written") else 1
        print(json.dumps({"ok": False, "error": "unknown_evolution_cmd"}, ensure_ascii=False))
    return 2


def cmd_skills(args: argparse.Namespace) -> int:
    from . import skills_index
    svc = MemoryService()
    if args.skills_cmd == "ensure":
        path = skills_index.ensure_skills_dir(svc.settings.home)
        print(json.dumps({"path": str(path)}, ensure_ascii=False))
        return 0
    cards = skills_index.scan_skills(*skills_index.default_roots(svc.settings.home))
    print(json.dumps([{"name": c.name, "summary": c.summary, "path": c.path} for c in cards], ensure_ascii=False, indent=2))
    return 0


def cmd_resources(args: argparse.Namespace) -> int:
    from . import resources
    svc = MemoryService()
    if args.resources_cmd == "import":
        if args.file:
            result = resources.import_resource_file(svc.settings.home, args.file, category=args.category, title=args.title)
        else:
            body = Path(args.text_file).read_text(encoding="utf-8") if args.text_file else (args.body or "")
            result = resources.import_resource(svc.settings.home, body=body, category=args.category, title=args.title)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("written") or result.get("reason") == "duplicate" else 1
    if args.resources_cmd == "list":
        print(json.dumps(resources.list_resources(svc.settings.home, category=args.category), ensure_ascii=False, indent=2))
        return 0
    if args.resources_cmd == "read":
        print(json.dumps(resources.read_resource(svc.settings.home, args.uri_or_path, offset=args.offset, limit=args.limit), ensure_ascii=False, indent=2))
        return 0
    if args.resources_cmd == "overview":
        print(json.dumps(resources.category_overview(svc.settings.home, args.category or "misc"), ensure_ascii=False, indent=2))
        return 0
    return 2


def main(argv: list[str] | None = None) -> int:
    # pythonw.exe (the windowless desktop shortcut) has no console streams.
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")
    # Windows consoles default to a legacy codepage; hooks must emit UTF-8 or
    # non-ASCII memory (Chinese quotes, names) arrives mangled.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass

    p = argparse.ArgumentParser(prog="membrain", description="Portable cross-agent memory")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="create the memory tree and profile").set_defaults(func=cmd_init)

    a = sub.add_parser("add", help="add a fact (constraints allowed here only)")
    a.add_argument("claim")
    a.add_argument("--type", default="fact", choices=[t.value for t in FactType])
    a.add_argument("--subject", required=True)
    a.add_argument("--quote", default=None)
    a.add_argument("--episode", default=None)
    a.add_argument("--date", default=None)
    a.set_defaults(func=cmd_add)

    l = sub.add_parser("list", help="list facts")
    l.add_argument("--status", default=None)
    l.set_defaults(func=cmd_list)

    s = sub.add_parser("show", help="show one fact as JSON")
    s.add_argument("id")
    s.set_defaults(func=cmd_show)

    c = sub.add_parser("correct", help="retract a fact")
    c.add_argument("id")
    c.add_argument("--reason", required=True)
    c.set_defaults(func=cmd_correct)

    k = sub.add_parser("contest", help="mark facts as conflicting")
    k.add_argument("ids", nargs="+")
    k.add_argument("--note", default=None)
    k.set_defaults(func=cmd_contest)

    r = sub.add_parser("resolve", help="resolve a conflict in favour of one fact")
    r.add_argument("keep")
    r.add_argument("--drop", required=True)
    r.set_defaults(func=cmd_resolve)

    q = sub.add_parser("search", help="query memory")
    q.add_argument("query")
    q.add_argument("--top-k", type=int, default=8)
    q.add_argument("--scope", default=None, help="URI prefix e.g. mb://user/memories/preference")
    q.add_argument("--level", type=int, default=2, help="0=L0 abstract, 1=L1 overview, 2=claim")
    q.add_argument("--type", default=None, help="fact type filter")
    q.add_argument("--include-sensitive", action="store_true", help="include sensitive beliefs always")
    q.add_argument("--use-intent", action="store_true", help="optional IntentAnalyzer (default off)")
    q.add_argument("--rerank", action="store_true", help="optional L0 abstract rerank (default off)")
    q.add_argument("--rerank-llm", action="store_true", help="allow LLM scoring inside rerank (still fail-open)")
    q.add_argument("--rerank-provider", default=None,
                   choices=["heuristic", "llm", "openai_compatible", "cross_encoder"],
                   help="gap13 rerank provider (default from MEMBRAIN_RERANK_PROVIDER / heuristic)")
    q.add_argument("--session", default=None, help="session id for intent context")
    q.add_argument("--mode", choices=["quick", "thinking", "context"], default=None,
                   help="OV-aligned search mode: quick=find-like; thinking=intent+rerank; context=type quotas")
    q.add_argument("--peer", default=None, help="optional workspace peer id (default recall still cross-project)")
    q.add_argument("--peer-scope", choices=["all", "actor"], default="all",
                   help="all=ignore peer (default); actor=filter to peer or unscoped")
    q.add_argument("--purpose", choices=["chat", "coding"], default=None, help="context-mode quota preset")
    q.set_defaults(func=cmd_search)

    sub.add_parser("migrate-scheme-a", help="backfill URI/L0/sensitive on facts + reindex").set_defaults(
        func=cmd_migrate_scheme_a
    )

    ac = sub.add_parser("auto-commit", help="token/idle auto-commit check + extract")
    ac.add_argument("--session", default=None)
    ac.add_argument("--force", action="store_true")
    ac.set_defaults(func=cmd_auto_commit)

    sub.add_parser("reindex", help="rebuild the derived index").set_defaults(func=cmd_reindex)
    sub.add_parser("stats", help="show store status").set_defaults(func=cmd_stats)
    t = sub.add_parser("tasks", help="list open tasks handed to a later session")
    t.add_argument("--limit", type=int, default=50)
    t.set_defaults(func=cmd_tasks)

    w = sub.add_parser("watch", help="run auto-capture in the foreground (no dashboard needed)")
    w.add_argument("--once", action="store_true", help="run a single capture pass and exit")
    w.add_argument("--interval", type=float, default=None, help="seconds between passes")
    w.set_defaults(func=cmd_watch)

    au = sub.add_parser("autostart", help="install capture-at-login so memory survives a reboot")
    au.add_argument("--remove", action="store_true", help="remove the installed autostart entry")
    au.set_defaults(func=cmd_autostart)

    co = sub.add_parser("consolidate", help="write session summaries (derived view, not evidence)")
    co.add_argument("--session", default=None, help="one session id; default: sessions that went quiet")
    co.add_argument("--idle-minutes", type=int, default=60)
    co.add_argument("--limit", type=int, default=10)
    co.add_argument("--force", action="store_true", help="rewrite summaries that already exist")
    co.set_defaults(func=cmd_consolidate)
    sub.add_parser("doctor", help="check backends, index and project hooks").set_defaults(func=cmd_doctor)

    h = sub.add_parser("hook", help="hook entrypoint (reads JSON on stdin)")
    h.add_argument("event", choices=[
        "session-start", "prompt-submit", "codex-prompt-submit", "codex-verified-capture",
        "capture", "precompact", "auto-commit",
    ])
    h.set_defaults(func=cmd_hook)

    e = sub.add_parser("extract", help="extract durable facts from captured episodes")
    e.add_argument("--session", default=None)
    e.set_defaults(func=cmd_extract)

    sd = sub.add_parser("session-diff", help="read memory_diff.json for a session archive (read-only)")
    sd.add_argument("archive", help="archive dir path or sessions/<id>/history/archive_NNN")
    sd.set_defaults(func=cmd_session_diff)

    sp2 = sub.add_parser("session-archive-phase2", help="run pending session archive Phase2 workers")
    sp2.add_argument("--archive", default=None, help="optional specific archive path")
    sp2.add_argument("--skip-extract", action="store_true", help="write summaries/diff without extract")
    sp2.set_defaults(func=cmd_session_archive_phase2)


    evo = sub.add_parser("evolution", help="gap9 agent evolution thin layer (default off)")
    evo_sub = evo.add_subparsers(dest="evolution_cmd", required=True)
    evo_sub.add_parser("status", help="show whether evolution is enabled").set_defaults(func=cmd_evolution)
    evo_p = evo_sub.add_parser("promote", help="seed trajectories from live CASE facts")
    evo_p.add_argument("--limit", type=int, default=50)
    evo_p.add_argument("--force", action="store_true", help="run once even if MEMBRAIN_EVOLUTION=0")
    evo_p.set_defaults(func=cmd_evolution)
    evo_d = evo_sub.add_parser("distill", help="write experience note from a trajectory")
    evo_d.add_argument("case_id")
    evo_d.add_argument("--title", default=None)
    evo_d.add_argument("--force", action="store_true", help="run once even if MEMBRAIN_EVOLUTION=0")
    evo_d.set_defaults(func=cmd_evolution)
    evo_sub.add_parser("list", help="list experience notes").set_defaults(func=cmd_evolution)

    sk = sub.add_parser("skills", help="gap10 skills directory index")
    sk_sub = sk.add_subparsers(dest="skills_cmd", required=True)
    sk_sub.add_parser("list", help="list skill cards").set_defaults(func=cmd_skills)
    sk_sub.add_parser("ensure", help="create MEMBRAIN_HOME/skills scaffold").set_defaults(func=cmd_skills)

    rs = sub.add_parser("resources", help="gap11 resources namespace (not memory facts)")
    rs_sub = rs.add_subparsers(dest="resources_cmd", required=True)
    rs_i = rs_sub.add_parser("import", help="import a document into mb://user/resources/...")
    rs_i.add_argument("--file", default=None, help="path to source file")
    rs_i.add_argument("--text-file", default=None, help="read body from file")
    rs_i.add_argument("--body", default=None, help="inline body text")
    rs_i.add_argument("--category", default="misc")
    rs_i.add_argument("--title", default=None)
    rs_i.set_defaults(func=cmd_resources)
    rs_l = rs_sub.add_parser("list", help="list resources")
    rs_l.add_argument("--category", default=None)
    rs_l.set_defaults(func=cmd_resources)
    rs_r = rs_sub.add_parser("read", help="read one resource by uri or path")
    rs_r.add_argument("uri_or_path")
    rs_r.add_argument("--offset", type=int, default=0)
    rs_r.add_argument("--limit", type=int, default=4000)
    rs_r.set_defaults(func=cmd_resources)
    rs_o = rs_sub.add_parser("overview", help="L1-style category overview")
    rs_o.add_argument("--category", default="misc")
    rs_o.set_defaults(func=cmd_resources)

    d = sub.add_parser("dashboard", help="start the local visual memory dashboard")
    d.add_argument("--host", default="127.0.0.0")
    d.add_argument("--port", type=int, default=1966)
    d.add_argument("--no-open", action="store_true", help="do not open the browser automatically")
    d.set_defaults(func=cmd_dashboard)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
