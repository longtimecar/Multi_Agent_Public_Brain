"""Dry run of the OpenViking import against the real source, into a TEMPORARY store.

Uses the configured LLM and embedding services (a few dozen calls) and writes
only to a temp folder, so the owner's memory is untouched. Also prints the
similarity of sample queries, to calibrate Archive.SEMANTIC_FLOOR.
"""

import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
home = Path(tempfile.mkdtemp(prefix="membrain-dryrun-"))
os.environ["MEMBRAIN_HOME"] = str(home)

from membrain import openviking_import as ov  # noqa: E402
from membrain.llm import LLMClient  # noqa: E402
from membrain.server import MemoryService  # noqa: E402

QUERIES = [  # (query, a title fragment that must appear in the top 3, or "" when nothing should match)
    ("DHCP 交换机安全", "dhcp snooping"), ("电池识别问题 EC 固件", "EC 固件问题"),
    ("开机自启动修复", "autostart"), ("MCP server 集成分析", "MCP Server 集成分析报告"),
    ("what did we learn about the laptop battery firmware", "EC"), ("网络工程师考试题库", "网络工程师"),
    ("巴黎明天的天气预报", ""), ("红烧肉的做法", ""),
]

source_raw = os.environ.get("MEMBRAIN_OPENVIKING_ROOT", "").strip()
if not source_raw:
    sys.exit("Set MEMBRAIN_OPENVIKING_ROOT to the OpenViking user folder first.")
source = Path(source_raw).expanduser()

svc = MemoryService()
print("temp store:", home)
print("preview:", ov.preview(svc, source))
started = time.time()
report = ov.apply_import(svc, LLMClient(svc.settings), source,
                         progress=lambda p: print("  ", p, flush=True))
print(f"\napply took {time.time() - started:.0f}s")
print({k: v for k, v in report.items() if k != "errors"})
print("errors:", report["errors"][:5])

print("\nfacts distilled:")
for f in svc.store.live_facts():
    print(f"  {f.confidence:.2f} {f.type.value:10} [{f.subject}] {f.claim[:90]}")

ids = [r[0] for r in svc.archive.conn.execute("SELECT id FROM archive_vectors")]
mat = np.vstack([np.frombuffer(r[0], dtype=np.float32) for r in svc.archive.conn.execute("SELECT vec FROM archive_vectors")])
print(f"\nvectors: {len(ids)} of {svc.archive.browse()['tree'] and sum(sum(c.values()) for c in svc.archive.browse()['tree'].values())}")
print("\nquery calibration (floor =", svc.archive.SEMANTIC_FLOOR, ")")
for query, expect in QUERIES:
    q = svc.embedder.embed_queries([query])[0]
    top = float((mat @ q).max())
    hits, _ = svc.archive.search(query, top_k=3)
    titles = [h["title"] for h in hits]
    found = any(expect.lower() in title.lower() for title in titles) if expect else not hits
    print(f"  {'PASS' if found else 'MISS':5} top_sim={top:.3f} q={query!r} -> {[(h['category'], h['similarity'], h['title'][:24]) for h in hits]}")
svc.index.close()
