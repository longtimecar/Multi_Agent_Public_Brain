"""End-to-end workflow check in an isolated store.

Writes synthetic user turns as episodes, runs real extraction with the
configured LLM, then checks stored facts, storage location and semantic
recall. Uses a temporary MEMBRAIN_HOME, so the owner's memory is untouched.
"""

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

home = Path(tempfile.mkdtemp(prefix="membrain-e2e-"))
os.environ["MEMBRAIN_HOME"] = str(home)

from membrain.models import Episode  # noqa: E402
from membrain.server import MemoryService  # noqa: E402
from membrain.extract import Extractor  # noqa: E402
from membrain.llm import LLMClient  # noqa: E402

TURNS = [
    "我不吃辣，平时饮食要清淡。",
    "I work as a backend engineer at a fintech startup in Shanghai.",
    "我正在学习日语，目标是明年能看懂日剧。",
]
QUERIES = [
    ("any dietary considerations?", "spicy"),
    ("这个人的工作是什么", "engineer"),
    ("what language is the user learning", "Japanese"),
    ("what is the weather in Paris", None),
]

svc = MemoryService()
store = svc.store
now = datetime.now(timezone.utc)
eps = [
    Episode(id=f"ep_e2e_{i}", ts=now, session="e2e-session", agent="claude-code",
            role="user", text=t)
    for i, t in enumerate(TURNS)
]
store.append_episodes(eps)
print("episodes written:", len(eps))

ext = Extractor(store, svc.retriever, LLMClient(svc.settings))
report = ext.run()
print("extraction report:", json.dumps(report.as_dict(), ensure_ascii=False)[:600])

facts = store.live_facts()
print("\nstored facts (storage location:", store.facts_dir, ")")
for f in facts:
    print(" -", f.id, f.type.value, "|", f.claim, "| quotes:", [e.quote for e in f.evidence][:2])

svc.retriever.reindex()
print("\nsemantic recall:")
for query, expect in QUERIES:
    result = svc.retriever.recall(query, top_k=3)
    texts = [f.claim for f in result.hits]
    ok = None if expect is None else any(expect in t for t in texts)
    print(f" query={query!r} expect={expect!r} pass={ok} mode={result.mode} top={texts[:3]}")

print("\ntemp store:", home)
