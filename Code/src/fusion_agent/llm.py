"""统一 HTTP 模型入口：Ollama、llama.cpp 与兼容 API。"""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
import json
import logging
import os
from pathlib import Path
import re
from threading import Lock, local
import time
from urllib.parse import urlsplit, urlunsplit

import httpx

from .config import LLMSettings
from .concurrency import ConcurrencyController, RequestMeasurement
from .utils import stable_id, write_json

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


def validate_response_schema(data, schema, root=None):
    """Validate the JSON-schema subset produced by our Pydantic contracts."""
    root = schema if root is None else root
    if "$ref" in schema:
        target = root
        for part in schema["$ref"].removeprefix("#/").split("/"):
            target = target[part]
        return validate_response_schema(data, target, root)
    if "anyOf" in schema:
        for option in schema["anyOf"]:
            try:
                validate_response_schema(data, option, root)
                return
            except ValueError:
                pass
        raise ValueError("JSON 值不符合 schema 的任何允许类型")
    kind = schema.get("type")
    valid = {"object": isinstance(data, dict), "array": isinstance(data, list),
             "string": isinstance(data, str), "integer": isinstance(data, int) and not isinstance(data, bool),
             "number": isinstance(data, (int, float)) and not isinstance(data, bool),
             "boolean": isinstance(data, bool), "null": data is None}
    if kind in valid and not valid[kind]:
        raise ValueError("JSON 值类型不符合 schema")
    if "enum" in schema and data not in schema["enum"]:
        raise ValueError("JSON 枚举值不合法")
    if kind == "object":
        if any(key not in data for key in schema.get("required", [])):
            raise ValueError("JSON 缺少 schema 必填字段")
        for key, value in data.items():
            if key in schema.get("properties", {}):
                validate_response_schema(value, schema["properties"][key], root)
    elif kind == "array":
        for value in data:
            validate_response_schema(value, schema.get("items", {}), root)


class LLMClient:
    def __init__(self, settings: LLMSettings, cache_dir: Path):
        self.settings = settings
        self.cache_dir = cache_dir / "llm"
        self.calls = 0
        self.cache_hits = 0
        self.validation_repairs = 0
        self.lock = Lock()
        self.key_locks = {}
        self.local = local()
        self.controller = None
        self.http = httpx.Client(timeout=httpx.Timeout(settings.timeout_seconds, connect=15))

    def close(self):
        self.http.close()

    @contextmanager
    def review_context(self):
        previous = self.settings.ollama.num_ctx
        if self.settings.provider == "ollama" and self.settings.ollama.review_num_ctx is not None:
            self.settings.ollama.num_ctx = max(previous, self.settings.ollama.review_num_ctx)
        try:
            yield
        finally:
            self.settings.ollama.num_ctx = previous

    def validate(self):
        s = self.settings
        # Revalidate CLI overrides as well as values loaded from TOML.
        LLMSettings.model_validate(s.model_dump())
        endpoint = s.endpoint
        if not endpoint.model:
            raise LLMError(f"请在 config.toml 中填写 llm.{s.provider}.model")
        if s.provider == "api" and not (s.api.api_key or os.environ.get(s.api.api_key_env)):
            raise LLMError(f"请设置 {s.api.api_key_env} 环境变量或 llm.api.api_key")
        if s.adaptive.enabled and urlsplit(endpoint.base_url).hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise LLMError("显存自适应需要 llama_cpp 服务与程序运行在同一台机器；远程服务请关闭 adaptive")

    def _llama_props_url(self):
        parts = urlsplit(self.settings.llama_cpp.base_url.rstrip("/"))
        path = parts.path.removesuffix("/v1")
        return urlunsplit((parts.scheme, parts.netloc, path + "/props", "", ""))

    def prepare_concurrency(self):
        if self.controller is not None:
            return
        self.validate()
        slots = None
        if self.settings.provider == "llama_cpp":
            endpoint = self.settings.llama_cpp
            key = endpoint.api_key or os.environ.get(endpoint.api_key_env)
            headers = {"Authorization": "Bearer " + key} if key else {}
            try:
                response = self.http.get(self._llama_props_url(), headers=headers, timeout=15)
                response.raise_for_status()
                properties = response.json()
                slots = properties["total_slots"]
                if isinstance(slots, bool) or not isinstance(slots, int) or slots < 1:
                    raise ValueError("invalid total_slots")
                context = properties.get("default_generation_settings", {}).get("n_ctx")
                if isinstance(context, int) and context < endpoint.num_ctx:
                    raise LLMError(f"llama-server 每路上下文只有 {context}，小于配置 num_ctx={endpoint.num_ctx}；请按每路容量重新启动服务")
            except (httpx.HTTPError, ValueError, KeyError, TypeError):
                raise LLMError("无法读取 llama-server /props 的 total_slots；请检查服务地址、鉴权和版本，避免请求只在服务端排队") from None
        self.controller = ConcurrencyController(self.settings.concurrency, self.settings.adaptive, slots)

    def concurrency_snapshot(self):
        return self.controller.snapshot() if self.controller else None

    def map_tasks(self, function, items, succeeded=None):
        """Keep only admitted tasks in flight, preserving input order in the result.

        Cached replies do not train the controller. On failure queued work is cancelled;
        already running HTTP requests finish so their successful caches can be reused.
        """
        self.prepare_concurrency()
        controller = self.controller
        controller.begin_batch()
        items = list(items)
        results = [None] * len(items)
        next_index = 0
        pending = {}

        def measured(item):
            self.local.measurement = RequestMeasurement()
            try:
                result = function(item)
                if succeeded is not None and not succeeded(result):
                    self.local.measurement.success = False
                return result, self.local.measurement, None
            except Exception as exc:
                self.local.measurement.success = False
                return None, self.local.measurement, exc

        with ThreadPoolExecutor(max_workers=controller.maximum) as pool:
            try:
                while next_index < len(items) or pending:
                    controller.poll_memory()
                    while next_index < len(items) and len(pending) < controller.limit:
                        saturated = len(items) - next_index + len(pending) >= controller.limit
                        future = pool.submit(measured, items[next_index])
                        pending[future] = (next_index, controller.epoch, saturated)
                        next_index += 1
                    done, _ = wait(pending, timeout=self.settings.adaptive.memory_poll_seconds,
                                   return_when=FIRST_COMPLETED)
                    for future in done:
                        index, epoch, saturated = pending.pop(future)
                        result, measurement, error = future.result()
                        controller.observe(measurement, epoch, saturated)
                        if error is not None:
                            raise error
                        results[index] = result
            except BaseException:
                for future in pending:
                    future.cancel()
                raise
        return results

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        self.validate()
        s = self.settings
        endpoint = s.endpoint
        measurement = getattr(self.local, "measurement", None)
        if measurement is None:
            measurement = RequestMeasurement()
        path = self.response_path(system, user, schema)
        if s.cache:
            with self.lock:
                key_lock = self.key_locks.setdefault(path.stem, Lock())
            with key_lock:
                return self._complete_json(s, endpoint, system, user, schema, path, measurement)
        return self._complete_json(s, endpoint, system, user, schema, path, measurement)

    def validated_json(self, system, user, schema, validator):
        """Ask the model to repair a rejected reply without relaxing evidence checks.

        A repaired request has its own cache key. Rejected replies are archived
        for audit and removed from the successful response cache.
        """
        request = user
        for attempt in range(self.settings.retries + 1):
            data = self.complete_json(system, request, schema)
            try:
                return validator(data)
            except ValueError as exc:
                rejection = self.cache_dir.parent / 'llm_rejections' / (self.response_path(system, request, schema).stem +
                                                                       f'-{time.time_ns()}.json')
                write_json(rejection, {'system': system, 'user': request, 'schema': schema,
                                       'response': data, 'validation_error': str(exc)})
                self.invalidate_response(system, request, schema)
                if attempt == self.settings.retries:
                    raise
                with self.lock:
                    self.validation_repairs += 1
                logger.warning('模型回复未通过证据校验，要求纠正（%d/%d）：%s', attempt + 1, self.settings.retries, exc)
                request = (user + '\n\n上次 JSON 未通过代码证据校验，请重新审查并纠正，仍须满足全部原任务要求。'
                           '\nquote 必须逐字复制原文连续片段，不改写时间、不拼接日期；未知事实使用空数组，不填 null/空引用条目。'
                           '\n不得删除不利事实、编造引用或用多数票覆盖同活动反证。'
                           '\n校验错误：' + str(exc) + '\n上次 JSON：' + json.dumps(data, ensure_ascii=False))

    def response_path(self, system, user, schema):
        s, endpoint = self.settings, self.settings.endpoint
        # 密钥永不进入缓存键、缓存内容、日志。
        cache_values = {"provider": s.provider, "url": endpoint.base_url, "model": endpoint.model,
                        "temperature": s.temperature, "schema": schema, "system": system,
                        "user": user, "num_ctx": s.ollama.num_ctx, "json_mode": s.api.json_mode,
                        "think": s.ollama.think, "num_predict": s.ollama.num_predict}
        # Keep existing Ollama/API keys unchanged so full runs reuse previous replies.
        if s.provider == "llama_cpp":
            cache_values.update(num_ctx=endpoint.num_ctx, think=endpoint.think,
                                num_predict=endpoint.num_predict, json_mode="schema")
        key = stable_id(json.dumps(cache_values, ensure_ascii=False, sort_keys=True))
        return self.cache_dir / (key + ".json")

    def invalidate_response(self, system, user, schema):
        # Application-level citation/coverage failures must not survive reruns.
        self.response_path(system, user, schema).unlink(missing_ok=True)

    def _complete_json(self, s, endpoint, system, user, schema, path, measurement):
        if s.cache and path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    raise ValueError("缓存必须是 JSON 对象")
                validate_response_schema(data, schema)
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
        elif s.provider == "llama_cpp":
            url = endpoint.base_url.rstrip("/") + "/chat/completions"
            payload = {"model": endpoint.model, "messages": messages, "stream": False,
                       "temperature": s.temperature, "max_tokens": endpoint.num_predict,
                       "response_format": {"type": "json_schema", "schema": schema}}
            if endpoint.think is not None:
                payload["chat_template_kwargs"] = {"enable_thinking": endpoint.think}
                if endpoint.think is False:
                    payload["reasoning_effort"] = "none"
            api_key = endpoint.api_key or os.environ.get(endpoint.api_key_env)
            headers = {"Authorization": "Bearer " + api_key} if api_key else {}
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
            measurement.network_requests += 1
            measurement.started = measurement.started or time.perf_counter()
            try:
                response = self.http.post(url, json=payload, headers=headers)
                response.raise_for_status()
                body = response.json()
                if s.provider in {"api", "llama_cpp"}:
                    if s.provider == "llama_cpp" and body["choices"][0].get("finish_reason") == "length":
                        raise ValueError("模型输出被 token 上限截断")
                    content = body["choices"][0]["message"]["content"]
                else:
                    if body.get("done_reason") == "length":
                        raise ValueError("模型输出被 token 上限截断")
                    content = body["message"]["content"]
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                if content.startswith("```"):
                    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content).strip()
                data = json.loads(content)
                if not isinstance(data, dict):
                    raise ValueError("模型必须返回 JSON 对象")
                validate_response_schema(data, schema)
                if s.cache:
                    write_json(path, data)
                measurement.success = True
                measurement.output_tokens = (body.get("usage", {}).get("completion_tokens", 0)
                                             if s.provider != "ollama" else body.get("eval_count", 0)) or 0
                return data
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status in {408, 429} or status >= 500:
                    measurement.pressure = True
                if status not in {408, 429} and status < 500:
                    raise LLMError(f"模型服务返回 HTTP {status}，请检查配置") from None
                error = f"模型服务返回 HTTP {status}"
            except (httpx.RequestError, ValueError, KeyError, IndexError, TypeError) as exc:
                if isinstance(exc, httpx.RequestError):
                    measurement.pressure = True
                error = f"模型请求/JSON 解析失败：{type(exc).__name__}: {exc}"
            if attempt < s.retries:
                time.sleep(min(2 ** attempt, 8))
        raise LLMError(error)

