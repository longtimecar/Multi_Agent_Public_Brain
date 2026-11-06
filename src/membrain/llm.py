"""Small JSON-only client for configured OpenAI or Anthropic endpoints."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Any

from .config import Settings


class LLMError(Exception):
    pass


def _strict_object(text: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"invalid JSON constant {value}")
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    def load(value: str) -> Any:
        return json.loads(value, parse_constant=reject_constant, object_pairs_hook=unique_object)

    try:
        parsed = load(text)
    except (TypeError, ValueError):
        # Models intermittently wrap the object in one line of prose or a code
        # fence. Accept exactly one object and nothing else; two objects or none
        # is still an error, so the caller never acts on an ambiguous answer.
        extracted = _single_object(text)
        if extracted is None:
            raise LLMError("model returned invalid JSON") from None
        try:
            parsed = load(extracted)
        except (TypeError, ValueError) as exc:
            raise LLMError("model returned invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise LLMError("model response must be a JSON object")
    return parsed


def _single_object(text: str) -> str | None:
    """The one JSON object embedded in `text`, or None.

    String-aware: braces inside quoted values do not count. Returns None when
    there is no object or when a second one follows, so "pick the first object"
    can never silently discard a conflicting instruction.
    """
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                if "{" in text[index + 1:]:
                    return None
                return text[start:index + 1]
    return None


def _endpoint(base: str, api_format: str) -> str:
    base = base.rstrip("/")
    if api_format == "openai":
        return base if base.endswith("/chat/completions") else base + "/chat/completions"
    if api_format == "anthropic":
        if base.endswith("/v1/messages"):
            return base
        return base + ("/messages" if base.endswith("/v1") else "/v1/messages")
    raise LLMError("unsupported LLM API format")


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _complete_text(self, system: str, messages: Sequence[dict[str, str]], *, json_mode: bool = False) -> str:
        settings = self.settings
        if not settings.llm_enabled:
            raise LLMError("LLM is not configured")
        api_format = settings.llm_api_format
        headers = {"Content-Type": "application/json"}
        if api_format == "openai":
            body = {"model": settings.llm_model,
                    "messages": [{"role": "system", "content": system}, *messages],
                    "temperature": 0 if json_mode else 0.2}
            if json_mode:
                body["response_format"] = {"type": "json_object"}
            if settings.llm_api_key:
                headers["Authorization"] = f"Bearer {settings.llm_api_key}"
        elif api_format == "anthropic":
            body = {"model": settings.llm_model, "system": system, "messages": list(messages),
                    "max_tokens": 2048, "temperature": 0.2}
            headers["anthropic-version"] = "2023-06-01"
            if settings.llm_api_key:
                headers["x-api-key"] = settings.llm_api_key
        else:
            raise LLMError("unsupported LLM API format")
        request = urllib.request.Request(_endpoint(settings.llm_base_url, api_format),
                                         data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                         headers=headers, method="POST")
        for attempt in range(2):
            try:
                with urllib.request.urlopen(request, timeout=settings.llm_timeout) as response:
                    envelope = json.loads(response.read())
                break
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if attempt:
                    raise LLMError("LLM request failed") from exc
            except (ValueError, UnicodeDecodeError) as exc:
                raise LLMError("LLM returned an invalid response envelope") from exc
        try:
            if api_format == "openai":
                text = envelope["choices"][0]["message"]["content"]
            else:
                parts = envelope["content"]
                text = "".join(part["text"] for part in parts if part.get("type") == "text")
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise LLMError("LLM response is missing text") from exc
        if not isinstance(text, str) or not text.strip():
            raise LLMError("LLM response is empty")
        return text.strip()

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        text = self._complete_text(system, [{"role": "user", "content": user}], json_mode=True)
        return _strict_object(text)

    def complete_text(self, system: str, user: str,
                      history: Sequence[dict[str, str]] | None = None) -> str:
        messages = [*(history or []), {"role": "user", "content": user}]
        clean = [
            {"role": item["role"], "content": item["content"]}
            for item in messages
            if item.get("role") in {"user", "assistant"} and isinstance(item.get("content"), str)
        ]
        return self._complete_text(system, clean, json_mode=False)


class FakeLLMClient:
    """Scriptable in-process fake; never calls a provider."""
    def __init__(self, responses: list[dict[str, Any] | Exception]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        self.calls.append((system, user))
        if not self.responses:
            raise LLMError("fake has no response")
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    def complete_text(self, system: str, user: str,
                      history: Sequence[dict[str, str]] | None = None) -> str:
        self.calls.append((system, user))
        if not self.responses:
            raise LLMError("fake has no response")
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        if isinstance(value, dict):
            return str(value.get("text", ""))
        return str(value)
