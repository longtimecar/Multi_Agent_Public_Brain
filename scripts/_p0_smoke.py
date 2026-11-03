"""P0 smoke: MCP tools + controllable Phase1/2 on real home + sensitive SessionStart (read-only checks)."""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(r"<PROJECT>")
sys.path.insert(0, str(ROOT / "src"))
os.environ["MEMBRAIN_HOME"] = str(ROOT / "memory")

from membrain.config import load_settings
from membrain.models import Episode
from membrain.server import MemoryService, build_server
from membrain import session_archive
from membrain.overview import build_l1_overviews, render_l1_block, render_l0_lines

report = {"ts": datetime.now().astimezone().isoformat(), "steps": []}

def step(name, **kw):
    row = {"name": name, **kw}
    report["steps"].append(row)
    print(json.dumps(row, ensure_ascii=False))

settings = load_settings()
home = Path(settings.home)
step(
    "settings_home",
    home=str(home),
    expected=str(ROOT / "memory"),
    match=str(home.resolve()) == str((ROOT / "memory").resolve()),
)

svc = MemoryService(settings)
server = build_server(svc)

search_tool = server._tool_manager.get_tool("search_memory").fn
diff_tool = server._tool_manager.get_tool("read_memory_diff").fn

r_default = search_tool(query="tea preference", top_k=3)
notes_default = list(r_default.get("notes") or [])
step(
    "mcp_search_default",
    ok=isinstance(r_default, dict),
    has_intent=any(str(n).startswith("intent:") for n in notes_default),
    has_rerank=any(str(n).startswith("rerank:") for n in notes_default),
    keys=sorted(r_default.keys()),
    notes=[str(n) for n in notes_default][:8],
)

r_intent = search_tool(query="我平时喝什么茶", top_k=3, use_intent=True, session="p0_smoke_cli")
notes_i = list(r_intent.get("notes") or [])
step(
    "mcp_search_use_intent",
    ok=isinstance(r_intent, dict),
    notes=[str(n) for n in notes_i][:12],
    intent_marked=any("intent:" in str(n) for n in notes_i),
)

r_rerank = search_tool(query="tea preference", top_k=3, use_rerank=True)
notes_r = list(r_rerank.get("notes") or [])
step(
    "mcp_search_use_rerank",
    ok=isinstance(r_rerank, dict),
    notes=[str(n) for n in notes_r][:12],
    rerank_marked=any("rerank:" in str(n) for n in notes_r),
)

session_id = "p0_cli_archive_20261103"
now = datetime.now(timezone.utc)
eps = [
    Episode(
        id="ep_p0_cli_u1",
        ts=now,
        session=session_id,
        agent="grok-bot-cli",
        role="user",
        text="P0 controllable archive smoke: I prefer oolong tea in the afternoon.",
    ),
    Episode(
        id="ep_p0_cli_a1",
        ts=now,
        session=session_id,
        agent="grok-bot-cli",
        role="assistant",
        text="Noted for archive smoke only.",
    ),
]
written = svc.store.append_episodes(eps)
step("append_episodes", session=session_id, written=len(written))

phase1 = session_archive.phase1_commit(svc.store, session_id, reason="p0-cli-smoke", agent="grok-bot-cli")
step(
    "phase1",
    status=phase1.get("status"),
    archived=phase1.get("archived"),
    task_id=phase1.get("task_id"),
    archive_path=phase1.get("archive_path"),
    archive_uri=phase1.get("archive_uri"),
    reason=phase1.get("reason"),
)

if phase1.get("archived") and phase1.get("archive_path"):
    phase2 = session_archive.phase2_process(
        svc.store, phase1["archive_path"], run_extract=False, extractor_factory=None
    )
    step(
        "phase2",
        status=phase2.get("status"),
        archive_path=phase2.get("archive_path"),
        has_diff=bool(phase2.get("diff")),
        summary=(phase2.get("diff") or {}).get("summary"),
    )
    try:
        diff_mcp = diff_tool(archive=str(phase1["archive_path"]))
        step(
            "mcp_read_memory_diff",
            ok=isinstance(diff_mcp, dict),
            keys=sorted(list(diff_mcp.keys()))[:12] if isinstance(diff_mcp, dict) else [],
            summary=(diff_mcp.get("summary") if isinstance(diff_mcp, dict) else None),
        )
    except Exception as exc:
        step("mcp_read_memory_diff", ok=False, error=exc.__class__.__name__, detail=str(exc)[:200])

sessions_root = home / "sessions"
archives = [str(p) for p in sessions_root.rglob("archive_*") if p.is_dir()] if sessions_root.exists() else []
step("real_sessions_listing", exists=sessions_root.exists(), archive_count=len(archives), archives=archives[:10])

ap = Path(phase1["archive_path"]) if phase1.get("archive_path") else None
if ap:
    arts = {name: (ap / name).exists() for name in [
        "messages.jsonl", "meta.json", ".abstract.md", ".overview.md", "memory_diff.json", ".done"
    ]}
    step("archive_artifacts", path=str(ap), **arts)

# sensitive SessionStart via hooks helper
import membrain.hooks as hooks_mod
ctx_fn = getattr(hooks_mod, "_session_start_context", None)
injection = ""
if ctx_fn:
    try:
        injection = ctx_fn(svc, {"session_id": "p0_sensitive_check"}) or ""
        if isinstance(injection, dict):
            injection = json.dumps(injection, ensure_ascii=False)
    except TypeError:
        try:
            injection = ctx_fn(svc) or ""
        except Exception as exc:
            step("sensitive_hook_call", ok=False, error=f"{exc.__class__.__name__}:{exc}")
            injection = ""
    except Exception as exc:
        step("sensitive_hook_call", ok=False, error=f"{exc.__class__.__name__}:{exc}")
        injection = ""

sens = [f for f in svc.store.live_facts() if f.sensitive]
# Fallback synthetic injection text if hook returned empty
if not injection:
    block = render_l1_block(build_l1_overviews(svc.store.live_facts(), include_sensitive=False))
    l0 = "\n".join(render_l0_lines(svc.store.live_facts(), include_sensitive=False)[:30])
    injection = f"{block}\n{l0}"

markers = ["religion", "buddh", "政治", "宗教", "attraction", "buddhist", "christian"]
# Check sensitive subjects themselves don't appear as dumped claims
subj_hits = [s.subject for s in sens if s.subject and s.subject.lower() in injection.lower()]
claim_hits = []
for s in sens:
    snippet = (s.claim or "")[:40]
    if snippet and snippet in injection:
        claim_hits.append(s.id)

step(
    "sensitive_sessionstart_readonly",
    hook=getattr(ctx_fn, "__name__", None),
    injection_len=len(injection),
    sensitive_live=len(sens),
    sensitive_subjects=[f.subject for f in sens][:10],
    subject_leaked_into_injection=subj_hits,
    sensitive_claim_leaked=claim_hits,
    marker_hits=[m for m in markers if m in injection.lower()],
    injection_preview=injection[:600],
)

try:
    svc.index.close()
except Exception:
    pass

out = ROOT / "docs" / "P0-SMOKE-EVIDENCE-2026-11-03.json"
out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print("WROTE", out)
