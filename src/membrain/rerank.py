"""L0/abstract rerank for recall candidates (OpenViking gaps 3 + 13).

Uses only L0 abstract text (never full L2 claim bodies) to control tokens.
Providers (unified switch, default fail-open):
  - heuristic          lexical overlap (default when rerank on, no remote)
  - llm                optional JSON LLM scoring
  - openai_compatible  HTTP rerank API (OpenAI-compatible /v1/rerank-style)
  - cross_encoder      optional local sentence-transformers CrossEncoder

Any failure returns fallback_scores unchanged (fail-open).
Feature still defaults OFF via use_rerank_default / MEMBRAIN_USE_RERANK.
"""

from __future__ import annotations

import json
import math
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

PROVIDERS = ("heuristic", "llm", "openai_compatible", "cross_encoder")


class JSONClient(Protocol):
    def complete_json(self, system: str, user: str) -> dict[str, Any]: ...


@dataclass(frozen=True)
class RerankResult:
    scores: list[float]
    reranked: list[bool]
    notes: list[str]

    @property
    def model_used(self) -> bool:
        return any(self.reranked)


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return default
    return score if math.isfinite(score) else default


_TOKEN = re.compile(r"[\u4e00-\u9fff]|[a-z0-9_]+", re.I)


def _tokens(text: str) -> set[str]:
    return {t.casefold() for t in _TOKEN.findall(text or "") if t.strip()}


def heuristic_scores(query: str, documents: Sequence[str]) -> list[float]:
    """Lightweight lexical overlap on L0 abstracts; no external model."""
    q = _tokens(query)
    if not q:
        return [0.0] * len(documents)
    scores: list[float] = []
    for doc in documents:
        d = _tokens(doc)
        if not d:
            scores.append(0.0)
            continue
        overlap = len(q & d)
        scores.append(overlap / float(len(q)))
    return scores


def normalize_provider(name: str | None, *, use_llm: bool = False) -> str:
    raw = (name or "").strip().lower().replace("-", "_")
    if use_llm and not raw:
        return "llm"
    if raw in {"", "auto", "default", "heuristic"}:
        return "heuristic"
    if raw in {"llm", "chat", "json_llm"}:
        return "llm"
    if raw in {"openai", "openai_compatible", "http", "api", "cohere", "voyage"}:
        return "openai_compatible"
    if raw in {"cross_encoder", "crossencoder", "ce", "local"}:
        return "cross_encoder"
    return "heuristic"


def _blend(scores: Sequence[float], fallback: Sequence[float], weight: float = 0.65) -> list[float]:
    n = len(scores)
    return [weight * float(scores[i]) + (1.0 - weight) * float(fallback[i]) for i in range(n)]


def _openai_compatible_scores(
    query: str,
    documents: Sequence[str],
    *,
    base_url: str,
    api_key: str | None,
    model: str | None,
    timeout: float = 20.0,
) -> list[float]:
    """Call an OpenAI-compatible rerank endpoint.

    Tries JSON body shapes common in community rerank proxies:
      POST {base}/rerank  or  {base}/v1/rerank
      {"model", "query", "documents"} -> {"results":[{"index","relevance_score"}]}
    Never logs the API key.
    """
    base = (base_url or "").rstrip("/")
    if not base:
        raise ValueError("rerank_base_url missing")
    payload = {
        "model": model or "rerank",
        "query": query,
        "documents": list(documents),
        "top_n": len(documents),
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    last_err: Exception | None = None
    for suffix in ("/rerank", "/v1/rerank", "/v1/rerankers"):
        url = base + suffix if not base.endswith(suffix) else base
        # If base already ends with /rerank, only try once
        if base.endswith("/rerank") or base.endswith("/v1/rerank"):
            url = base
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
            scores = [0.0] * len(documents)
            results = None
            if isinstance(raw, dict):
                results = raw.get("results") or raw.get("data")
                if results is None and isinstance(raw.get("scores"), list):
                    return [_finite(s) for s in raw["scores"]]
            if isinstance(results, list):
                for item in results:
                    if not isinstance(item, dict):
                        continue
                    idx = item.get("index")
                    score = item.get("relevance_score", item.get("score"))
                    if isinstance(idx, int) and 0 <= idx < len(scores):
                        scores[idx] = _finite(score)
                return scores
            raise ValueError("unexpected rerank response shape")
        except Exception as exc:  # try next suffix
            last_err = exc
            if base.endswith("/rerank") or base.endswith("/v1/rerank"):
                break
            continue
    raise RuntimeError(f"openai_compatible_rerank_failed:{type(last_err).__name__ if last_err else 'unknown'}")


def _cross_encoder_scores(query: str, documents: Sequence[str], *, model: str | None) -> list[float]:
    """Optional local CrossEncoder; ImportError/runtime -> raise for fail-open."""
    try:
        from sentence_transformers import CrossEncoder  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dep
        raise RuntimeError(f"cross_encoder_unavailable:{type(exc).__name__}") from exc
    model_name = model or "cross-encoder/ms-marco-MiniLM-L-6-v2"
    encoder = CrossEncoder(model_name)
    pairs = [(query, doc) for doc in documents]
    raw_scores = encoder.predict(pairs)
    out: list[float] = []
    for s in raw_scores:
        out.append(_finite(s))
    # Min-max to 0..1 when possible
    if out:
        lo, hi = min(out), max(out)
        if hi > lo:
            out = [(s - lo) / (hi - lo) for s in out]
    return out


def rerank_scores(
    query: str,
    documents: Sequence[str],
    fallback_scores: Sequence[float],
    *,
    llm: JSONClient | None = None,
    use_llm: bool = False,
    provider: str | None = None,
    settings: Any | None = None,
) -> RerankResult:
    """Return per-document scores; on any failure keep fallback_scores."""
    n = len(documents)
    fallback = [_finite(s) for s in fallback_scores]
    if len(fallback) != n:
        fallback = (fallback + [0.0] * n)[:n]
    if n == 0:
        return RerankResult([], [], ["empty"])

    settings_provider = getattr(settings, "rerank_provider", None) if settings is not None else None
    chosen = normalize_provider(provider or settings_provider, use_llm=use_llm)
    notes: list[str] = [f"provider:{chosen}"]

    # --- llm ---
    if chosen == "llm":
        if llm is None:
            return RerankResult(fallback, [False] * n, notes + ["llm_missing_fallback"])
        try:
            payload = {
                "query": query,
                "documents": [{"index": i, "abstract": documents[i]} for i in range(n)],
            }
            system = (
                "Score how relevant each abstract is to the query. "
                'Return JSON {"scores": [float, ...]} same length/order as documents. '
                "Higher means more relevant. Scores in 0..1."
            )
            raw = llm.complete_json(system, json.dumps(payload, ensure_ascii=False))
            scores_raw = raw.get("scores") if isinstance(raw, dict) else None
            if not isinstance(scores_raw, list) or len(scores_raw) != n:
                return RerankResult(fallback, [False] * n, notes + ["llm_bad_shape_fallback"])
            scores = [_finite(s, default=fallback[i]) for i, s in enumerate(scores_raw)]
            return RerankResult(scores, [True] * n, notes + ["llm"])
        except Exception as exc:
            return RerankResult(fallback, [False] * n, notes + [f"llm_failed:{type(exc).__name__}"])

    # --- openai_compatible ---
    if chosen == "openai_compatible":
        try:
            base_url = getattr(settings, "rerank_base_url", None) if settings is not None else None
            api_key = getattr(settings, "rerank_api_key", None) if settings is not None else None
            model = getattr(settings, "rerank_model", None) if settings is not None else None
            timeout = float(getattr(settings, "rerank_timeout", 20.0) or 20.0) if settings is not None else 20.0
            scores = _openai_compatible_scores(
                query,
                documents,
                base_url=str(base_url or ""),
                api_key=api_key,
                model=model,
                timeout=timeout,
            )
            if len(scores) != n:
                return RerankResult(fallback, [False] * n, notes + ["openai_len_fallback"])
            return RerankResult(_blend(scores, fallback), [True] * n, notes + ["openai_compatible"])
        except Exception as exc:
            # fail-open: try heuristic before giving up entirely
            try:
                scores = heuristic_scores(query, documents)
                return RerankResult(
                    _blend(scores, fallback),
                    [True] * n,
                    notes + [f"openai_failed:{type(exc).__name__}", "heuristic_fallback"],
                )
            except Exception as exc2:
                return RerankResult(
                    fallback,
                    [False] * n,
                    notes + [f"openai_failed:{type(exc).__name__}", f"heuristic_failed:{type(exc2).__name__}"],
                )

    # --- cross_encoder ---
    if chosen == "cross_encoder":
        try:
            model = getattr(settings, "rerank_model", None) if settings is not None else None
            scores = _cross_encoder_scores(query, documents, model=model)
            if len(scores) != n:
                return RerankResult(fallback, [False] * n, notes + ["ce_len_fallback"])
            return RerankResult(_blend(scores, fallback), [True] * n, notes + ["cross_encoder"])
        except Exception as exc:
            try:
                scores = heuristic_scores(query, documents)
                return RerankResult(
                    _blend(scores, fallback),
                    [True] * n,
                    notes + [f"ce_failed:{type(exc).__name__}", "heuristic_fallback"],
                )
            except Exception as exc2:
                return RerankResult(
                    fallback,
                    [False] * n,
                    notes + [f"ce_failed:{type(exc).__name__}", f"heuristic_failed:{type(exc2).__name__}"],
                )

    # --- heuristic (default) ---
    try:
        scores = heuristic_scores(query, documents)
        if len(scores) != n:
            return RerankResult(fallback, [False] * n, notes + ["heuristic_len_fallback"])
        blended = _blend(scores, fallback)
        return RerankResult(blended, [True] * n, notes + ["heuristic"])
    except Exception as exc:
        return RerankResult(fallback, [False] * n, notes + [f"heuristic_failed:{type(exc).__name__}"])
