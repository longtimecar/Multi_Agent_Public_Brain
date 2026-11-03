"""Measure labelled recall pairs against the configured embedding model.

Run from the repository root with ``uv run python scripts/calibrate.py``.
All facts and indexes are synthetic and live in a temporary directory.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import tempfile
from datetime import date
from pathlib import Path

from membrain.config import load_settings
from membrain.embeddings import Embedder, EmbeddingError
from membrain.index import Index
from membrain.models import Evidence, FactType
from membrain.retrieval import Retriever
from membrain.store import Store

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT / "tests" / "fixtures" / "recall_pairs.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--floor", type=float, help="override the configured semantic floor")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--temp-dir", type=Path, help="parent for the isolated temporary store")
    parser.add_argument("--strict", action="store_true", help="also fail on unrelated injections")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.floor is not None and not 0.0 <= args.floor <= 1.0):
        parser.error("batch size must be positive and floor must be between 0 and 1")
    pairs = json.loads(args.fixture.read_text(encoding="utf-8"))
    if not isinstance(pairs, list) or not pairs:
        parser.error("fixture must be a nonempty list")
    ids = set()
    for pair in pairs:
        if (not isinstance(pair, dict) or not isinstance(pair.get("related"), bool)
                or any(not isinstance(pair.get(k), str) or not pair[k].strip()
                       for k in ("id", "subject", "fact", "query"))
                or pair["id"] in ids):
            parser.error("each pair needs a unique id, a boolean label, and nonempty subject, fact, and query")
        ids.add(pair["id"])
    if {p["related"] for p in pairs} != {True, False}:
        parser.error("fixture must contain related and unrelated pairs")

    settings = load_settings()
    if args.floor is not None:
        settings = dataclasses.replace(settings, semantic_floor=args.floor)
    embedder = Embedder(settings)
    if not embedder.enabled:
        print("No embedding endpoint configured.", file=sys.stderr)
        return 2
    print(f"Model: {settings.embed_model}; floor: {settings.semantic_floor:.4f}; pairs: {len(pairs)}", flush=True)

    # Prime the same cache used by the retriever, with actual fact input shapes.
    try:
        for start in range(0, len(pairs), args.batch_size):
            batch = pairs[start:start + args.batch_size]
            embedder.embed([f"{p['subject']}\n{p['fact']}" for p in batch])
            embedder.embed_queries([p["query"] for p in batch])
    except EmbeddingError as exc:
        print(f"Embedding request failed: {type(exc.__cause__).__name__}", file=sys.stderr)
        return 2

    measurements: list[tuple[dict, float, bool, str]] = []
    with tempfile.TemporaryDirectory(prefix="membrain-calibration-", dir=args.temp_dir) as tmp:
        for n, pair in enumerate(pairs):
            local = dataclasses.replace(settings, home=Path(tmp) / str(n))
            store = Store(local.home)
            index = Index(local.db_path)
            try:
                fact = store.draft_fact(
                    type=FactType.FACT, subject=pair["subject"], claim=pair["fact"],
                    evidence=[Evidence(episode="synthetic", quote=pair["fact"], date=date.today())],
                    source_agent="calibration",
                )
                fact = store.write_fact(fact)
                retriever = Retriever(store, index, embedder, local)
                retriever.sync_fact(fact)
                doc = embedder.embed([f"{pair['subject']}\n{pair['fact']}"])[0]
                query = embedder.embed_queries([pair["query"]])[0]
                score = float(doc @ query)
                result = retriever.recall(pair["query"])
                matched = any(hit.id == fact.id for hit in result.hits)
                measurements.append((pair, score, matched, result.mode))
                ok = matched == pair["related"] and result.mode == "hybrid"
                label = "PASS" if ok else ("WARN" if not pair["related"] and matched and result.mode == "hybrid" else "FAIL")
                print(f"{label} {pair['id']}: {score:.4f}; injected={matched}; mode={result.mode}")
            finally:
                index.close()

    positives = [s for p, s, _, _ in measurements if p["related"]]
    negatives = [s for p, s, _, _ in measurements if not p["related"]]
    print(f"Lowest related: {min(positives):.4f}; highest unrelated: {max(negatives):.4f}")
    if max(negatives) >= min(positives):
        print("No single similarity floor separates all labelled pairs.")
    false_negatives = sum(p["related"] and not matched for p, _, matched, _ in measurements)
    false_positives = sum(not p["related"] and matched for p, _, matched, _ in measurements)
    degraded = sum(mode != "hybrid" for _, _, _, mode in measurements)
    print(f"Missed related: {false_negatives}; unrelated injected: {false_positives}; non-hybrid: {degraded}")
    return 1 if false_negatives or degraded or (args.strict and false_positives) else 0


if __name__ == "__main__":
    raise SystemExit(main())
