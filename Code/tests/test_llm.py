import json
from concurrent.futures import ThreadPoolExecutor
import time

import httpx
import pytest

from fusion_agent.config import APISettings, LLMSettings, OllamaSettings, LlamaCppSettings, AdaptiveConcurrencySettings
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


def test_llama_cpp_schema_payload_optional_auth_and_cache(tmp_path):
    calls = []

    def handler(request):
        payload = json.loads(request.content)
        calls.append(payload)
        assert str(request.url) == "http://localhost:8080/v1/chat/completions"
        assert "Authorization" not in request.headers
        assert payload["response_format"] == {"type": "json_schema", "schema": {"type": "object"}}
        assert payload["stream"] is False
        assert payload["max_tokens"] in {512, 1024}
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": '{"ok":true}'}}]})

    settings = LLMSettings(provider="llama_cpp", llama_cpp=LlamaCppSettings(num_predict=512))
    client = LLMClient(settings, tmp_path)
    client.http.close()
    client.http = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        for _ in range(2):
            assert client.complete_json("JSON", "test", {"type": "object"}) == {"ok": True}
        assert len(calls) == 1 and client.cache_hits == 1
        settings.llama_cpp.num_predict = 1024
        client.complete_json("JSON", "test", {"type": "object"})
        assert [p["max_tokens"] for p in calls] == [512, 1024]
    finally:
        client.close()


def test_llama_cpp_truncated_response_is_not_cached(tmp_path):
    client = LLMClient(LLMSettings(provider="llama_cpp", retries=0), tmp_path)
    client.http.close()
    client.http = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={
        "choices": [{"finish_reason": "length", "message": {"content": "{}"}}]})))
    try:
        with pytest.raises(LLMError, match="解析失败"):
            client.complete_json("JSON", "test", {})
        assert not list(tmp_path.rglob("*.json"))
    finally:
        client.close()


def test_llama_cpp_props_auth_and_missing_slots_fail_closed(tmp_path):
    settings = LLMSettings(provider="llama_cpp", llama_cpp=LlamaCppSettings(api_key="private-key"))
    client = LLMClient(settings, tmp_path)
    client.http.close()

    def handler(request):
        assert request.headers["Authorization"] == "Bearer private-key"
        return httpx.Response(200, json={})

    client.http = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(LLMError, match="total_slots"):
            client.prepare_concurrency()
    finally:
        client.close()


def test_remote_llama_cpp_cannot_use_local_vram_for_tuning(tmp_path):
    settings = LLMSettings(provider="llama_cpp", adaptive=AdaptiveConcurrencySettings(enabled=True),
                           llama_cpp=LlamaCppSettings(base_url="http://remote-host:8080/v1"))
    client = LLMClient(settings, tmp_path)
    try:
        with pytest.raises(LLMError, match="同一台机器"):
            client.validate()
    finally:
        client.close()


def test_identical_concurrent_requests_share_one_inference_and_atomic_cache(tmp_path):
    client = LLMClient(LLMSettings(provider="llama_cpp"), tmp_path)
    client.http.close()

    def handler(request):
        time.sleep(0.02)
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok":true}'}}]})

    client.http = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            replies = list(pool.map(lambda _: client.complete_json("JSON", "same article", {}), range(4)))
        assert replies == [{"ok": True}] * 4
        assert client.calls == 1 and client.cache_hits == 3
        assert len(list(tmp_path.rglob("*.json"))) == 1
    finally:
        client.close()


def test_llama_cpp_detects_reduced_context_after_slot_increase(tmp_path):
    client = LLMClient(LLMSettings(provider="llama_cpp"), tmp_path)
    client.http.close()
    client.http = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={
        "total_slots": 4, "default_generation_settings": {"n_ctx": 2048}})))
    try:
        with pytest.raises(LLMError, match="num_ctx=8192"):
            client.prepare_concurrency()
    finally:
        client.close()

