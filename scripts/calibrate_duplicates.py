"""Measure candidate similarities for the configured duplicate model."""
from __future__ import annotations

import json
from pathlib import Path

from membrain.config import load_settings
from membrain.embeddings import Embedder


def main() -> int:
    settings = load_settings()
    embedder = Embedder(settings)
    if not embedder.enabled:
        print("No embedding endpoint configured")
        return 2
    fixture = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "duplicate_pairs.json"
    pairs = json.loads(fixture.read_text(encoding="utf-8"))
    left = embedder.embed([f"{p['subject']}\n{p['a']}" for p in pairs])
    right = embedder.embed([f"{p['subject']}\n{p['b']}" for p in pairs])
    for pair, a, b in zip(pairs, left, right):
        score = float(a @ b)
        print(f"{pair['label']:9} {pair['subject']:10} {score:.4f} candidate={score >= settings.duplicate_floor}")
    print("Candidates still require model confirmation; high similarity alone does not prove equivalence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
