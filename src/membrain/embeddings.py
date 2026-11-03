"""Embeddings via any OpenAI-compatible endpoint.

Uses urllib so there is no SDK dependency and no provider lock-in: OpenAI,
Voyage, Jina, a local llama.cpp server, anything with /embeddings works.
When no endpoint is configured, retrieval degrades to FTS-only and says so —
a personal memory store must not become unusable because an API key is missing.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Sequence

import numpy as np

from .config import Settings


class EmbeddingError(Exception):
    pass


def embeddings_url(base_url: str) -> str:
    """Accept a base URL with or without the `/embeddings` suffix.

    Providers document both forms (`.../v1` and `.../v1/embeddings`), and
    appending the suffix twice produces a 404 that looks like a bad key.
    """
    base = base_url.rstrip("/")
    return base if base.endswith("/embeddings") else base + "/embeddings"


class Embedder:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._cache: dict[str, np.ndarray] = {}

    @property
    def enabled(self) -> bool:
        return self.settings.embeddings_enabled

    @property
    def model(self) -> str:
        return self.settings.embed_model

    def _post(self, texts: Sequence[str]) -> list[list[float]]:
        url = embeddings_url(self.settings.embed_base_url)
        body = json.dumps({"model": self.settings.embed_model, "input": list(texts)}).encode()
        headers = {"Content-Type": "application/json"}
        if self.settings.embed_api_key:
            headers["Authorization"] = f"Bearer {self.settings.embed_api_key}"
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.settings.embed_timeout) as resp:
                payload = json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise EmbeddingError(f"embedding request failed: {exc}") from exc
        data = sorted(payload.get("data", []), key=lambda d: d.get("index", 0))
        return [d["embedding"] for d in data]

    def embed(self, texts: Sequence[str]) -> np.ndarray | None:
        """Return an (n, d) float32 L2-normalised matrix, or None if disabled."""
        if not self.enabled or not texts:
            return None
        pending = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if pending:
            for text, vec in zip(pending, self._post(pending)):
                arr = np.asarray(vec, dtype=np.float32)
                self._cache[text] = arr / (np.linalg.norm(arr) + 1e-9)
        return np.vstack([self._cache[t] for t in texts]).astype(np.float32)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray | None:
        """Embed retrieval queries with their instruction; facts stay unprefixed.

        The full input, including the instruction, is the ordinary cache key.
        Changing an instruction therefore cannot reuse an old query vector.
        """
        instruction = self.settings.query_instruction
        if instruction:
            texts = [f"Instruct: {instruction}\nQuery: {text}" for text in texts]
        return self.embed(texts)
