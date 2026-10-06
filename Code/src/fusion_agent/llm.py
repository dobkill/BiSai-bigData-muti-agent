"""统一 HTTP 模型入口：Ollama 与兼容 API，含预算、重试、缓存。"""
import json
import os
from pathlib import Path
import re
from threading import Lock
import time

import httpx

from .config import LLMSettings
from .utils import stable_id, write_json


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, settings: LLMSettings, cache_dir: Path):
        self.settings = settings
        self.cache_dir = cache_dir / "llm"
        self.calls = 0
        self.cache_hits = 0
        self.lock = Lock()
        self.http = httpx.Client(timeout=httpx.Timeout(settings.timeout_seconds, connect=15))

    def close(self):
        self.http.close()

    def validate(self):
        s = self.settings
        endpoint = s.api if s.provider == "api" else s.ollama
        if not endpoint.model:
            raise LLMError(f"请在 config.toml 中填写 llm.{s.provider}.model")
        if s.provider == "api" and not (s.api.api_key or os.environ.get(s.api.api_key_env)):
            raise LLMError(f"请设置 {s.api.api_key_env} 环境变量或 llm.api.api_key")

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        self.validate()
        s = self.settings
        endpoint = s.api if s.provider == "api" else s.ollama
        # 密钥永不进入缓存键、缓存内容、日志。
        key = stable_id(json.dumps({"provider": s.provider, "url": endpoint.base_url, "model": endpoint.model,
                                   "temperature": s.temperature, "schema": schema, "system": system,
                                   "user": user, "num_ctx": s.ollama.num_ctx, "json_mode": s.api.json_mode,
                                   "think": s.ollama.think, "num_predict": s.ollama.num_predict},
                                  ensure_ascii=False, sort_keys=True))
        path = self.cache_dir / (key + ".json")
        if s.cache and path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    raise ValueError("缓存必须是 JSON 对象")
                with self.lock:
                    self.cache_hits += 1
                return data
            except (ValueError, OSError):
                pass
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if s.provider == "api":
            url = s.api.base_url.rstrip("/") + "/chat/completions"
            payload = {"model": s.api.model, "messages": messages, "temperature": s.temperature}
            if s.api.json_mode:
                payload["response_format"] = {"type": "json_object"}
            headers = {"Authorization": "Bearer " + (s.api.api_key or os.environ[s.api.api_key_env])}
        else:
            url = s.ollama.base_url.rstrip("/") + "/api/chat"
            payload = {"model": s.ollama.model, "messages": messages, "stream": False, "format": schema,
                       "options": {"temperature": s.temperature, "num_ctx": s.ollama.num_ctx,
                                   "num_predict": s.ollama.num_predict},
                       "keep_alive": s.ollama.keep_alive}
            if s.ollama.think is not None:
                payload["think"] = s.ollama.think
            headers = {}
        for attempt in range(s.retries + 1):
            with self.lock:
                if self.calls >= s.max_calls:
                    raise LLMError("已达到 llm.max_calls 请求预算")
                self.calls += 1
            try:
                response = self.http.post(url, json=payload, headers=headers)
                response.raise_for_status()
                body = response.json()
                if s.provider == "api":
                    content = body["choices"][0]["message"]["content"]
                else:
                    content = body["message"]["content"]
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                if content.startswith("```"):
                    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content).strip()
                data = json.loads(content)
                if not isinstance(data, dict):
                    raise ValueError("模型必须返回 JSON 对象")
                if s.cache:
                    write_json(path, data)
                return data
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status not in {408, 429} and status < 500:
                    raise LLMError(f"模型服务返回 HTTP {status}，请检查配置") from None
                error = f"模型服务返回 HTTP {status}"
            except (httpx.RequestError, ValueError, KeyError, IndexError, TypeError) as exc:
                error = f"模型请求/JSON 解析失败：{type(exc).__name__}"
            if attempt < s.retries:
                time.sleep(min(2 ** attempt, 8))
        raise LLMError(error)

