import json

import httpx
import pytest

from fusion_agent.config import APISettings, LLMSettings, OllamaSettings
from fusion_agent.llm import LLMClient, LLMError


@pytest.mark.parametrize("provider", ["api", "ollama"])
def test_both_provider_payloads_and_cache(tmp_path, provider):
    calls = []

    def handler(request):
        payload = json.loads(request.content)
        calls.append((str(request.url), payload))
        if provider == "api":
            assert request.headers["Authorization"] == "Bearer secret-test"
            assert payload["response_format"] == {"type": "json_object"}
            return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": true}'}}]})
        assert payload["stream"] is False
        assert payload["format"]["type"] == "object"
        assert payload["think"] is False
        assert payload["options"]["num_predict"] == 512
        return httpx.Response(200, json={"message": {"content": '{"ok": true}'}})

    settings = LLMSettings(provider=provider, api=APISettings(model="test", api_key="secret-test"),
                           ollama=OllamaSettings(model="test", think=False, num_predict=512))
    client = LLMClient(settings, tmp_path)
    client.http.close()
    client.http = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        for _ in range(2):
            assert client.complete_json("JSON", "test", {"type": "object"}) == {"ok": True}
        assert len(calls) == 1
        assert client.cache_hits == 1
        assert calls[0][0].endswith("/chat/completions" if provider == "api" else "/api/chat")
        assert all("secret-test" not in p.read_text() for p in tmp_path.rglob("*.json"))
    finally:
        client.close()


def test_budget_caps_actual_http_requests(tmp_path):
    settings = LLMSettings(provider="ollama", ollama=OllamaSettings(model="test"), max_calls=1, retries=0, cache=False)
    client = LLMClient(settings, tmp_path)
    client.http.close()
    client.http = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"message": {"content": "{}"}})))
    try:
        client.complete_json("JSON", "a", {})
        with pytest.raises(LLMError, match="预算"):
            client.complete_json("JSON", "b", {})
        assert client.calls == 1
    finally:
        client.close()


@pytest.mark.parametrize("option,new_value", [("think", True), ("num_predict", 1024)])
def test_ollama_generation_options_invalidate_cache(tmp_path, option, new_value):
    settings = LLMSettings(ollama=OllamaSettings(model="test", think=False, num_predict=512))
    client = LLMClient(settings, tmp_path)
    client.http.close()
    client.http = httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(200, json={"message": {"content": '{"ok":true}'}})))
    try:
        client.complete_json("JSON", "test", {})
        setattr(settings.ollama, option, new_value)
        client.complete_json("JSON", "test", {})
        assert client.calls == 2
        client.complete_json("JSON", "test", {})
        assert client.cache_hits == 1
    finally:
        client.close()

