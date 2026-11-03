from __future__ import annotations

import dataclasses
import json
import urllib.error

import pytest

from membrain.llm import LLMClient, LLMError, _endpoint
from test_core import make_settings


class Response:
    def __init__(self, body):
        self.body = json.dumps(body).encode()
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def read(self):
        return self.body


@pytest.mark.parametrize("api_format", ["openai", "anthropic"])
def test_request_shapes_and_json_reply(tmp_path, monkeypatch, api_format):
    settings = dataclasses.replace(make_settings(tmp_path), llm_base_url="https://example.test/v1",
                                   llm_api_key="secret", llm_model="fake", llm_api_format=api_format)
    requests = []
    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        if api_format == "openai":
            return Response({"choices":[{"message":{"content":"{\"answer\": 1}"}}]})
        return Response({"content":[{"type":"text","text":"{\"answer\": 1}"}]})
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert LLMClient(settings).complete_json("system", "user") == {"answer":1}
    request, timeout = requests[0]
    body = json.loads(request.data)
    assert timeout == 30 and body["model"] == "fake"
    if api_format == "openai":
        assert request.full_url.endswith("/v1/chat/completions")
        assert body["response_format"] == {"type":"json_object"}
        assert body["messages"][0]["role"] == "system"
    else:
        assert request.full_url.endswith("/v1/messages")
        assert body["system"] == "system" and body["max_tokens"] > 0
        assert request.get_header("X-api-key") == "secret"


@pytest.mark.parametrize("content", ["not json", "[]", '{"x":NaN}', '{"x":1,"x":2}', '```json\n{}\n```'])
def test_invalid_json_is_rejected(tmp_path, monkeypatch, content):
    settings = dataclasses.replace(make_settings(tmp_path), llm_base_url="https://example.test/v1", llm_model="fake")
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Response({"choices":[{"message":{"content":content}}]}))
    with pytest.raises(LLMError):
        LLMClient(settings).complete_json("s", "u")


def test_retry_once_on_transport_error(tmp_path, monkeypatch):
    settings = dataclasses.replace(make_settings(tmp_path), llm_base_url="https://example.test/v1", llm_model="fake")
    attempts = []
    def urlopen(*args, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise urllib.error.URLError("transient")
        return Response({"choices":[{"message":{"content":"{}"}}]})
    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    assert LLMClient(settings).complete_json("s", "u") == {} and len(attempts) == 2


def test_endpoint_accepts_provider_base_and_complete_url():
    assert _endpoint("https://x/v1/messages", "anthropic") == "https://x/v1/messages"
    assert _endpoint("https://x", "anthropic") == "https://x/v1/messages"
    assert _endpoint("https://x/v1/chat/completions", "openai") == "https://x/v1/chat/completions"


@pytest.mark.parametrize("api_format", ["openai", "anthropic"])
def test_text_completion_sends_conversation_history_without_json_mode(tmp_path, monkeypatch, api_format):
    settings = dataclasses.replace(make_settings(tmp_path), llm_base_url="https://example.test/v1",
                                   llm_model="fake", llm_api_format=api_format)
    requests = []

    def fake_urlopen(request, timeout):
        requests.append(json.loads(request.data))
        if api_format == "openai":
            return Response({"choices": [{"message": {"content": "A useful answer."}}]})
        return Response({"content": [{"type": "text", "text": "A useful answer."}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    answer = LLMClient(settings).complete_text(
        "system",
        "new question",
        history=[{"role": "user", "content": "earlier question"},
                 {"role": "assistant", "content": "earlier answer"}],
    )
    assert answer == "A useful answer."
    body = requests[0]
    if api_format == "openai":
        assert [item["role"] for item in body["messages"]] == ["system", "user", "assistant", "user"]
        assert "response_format" not in body
    else:
        assert [item["role"] for item in body["messages"]] == ["user", "assistant", "user"]
