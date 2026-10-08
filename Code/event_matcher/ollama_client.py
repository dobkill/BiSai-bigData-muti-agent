import hashlib
import json
import time
import threading
import uuid
from pathlib import Path

import requests
from pydantic import BaseModel

VERSION = "matcher-v1.0"


def dump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + "." + uuid.uuid4().hex + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    for attempt in range(6):
        try:
            temp.replace(path)
            break
        except PermissionError:
            # Windows file watchers/scanners may briefly deny replacing an
            # existing checkpoint. Keep the previous file intact while retrying.
            if attempt==5:
                raise
            time.sleep(min(.05*2**attempt,.8))


class OllamaClient:
    def __init__(self, cache: Path, model: str, embedding_model: str, url: str, workers=1, num_ctx=12288, chat_url=None):
        self.cache, self.model, self.embedding_model, self.url = cache, model, embedding_model, url.rstrip("/")
        if workers < 1 or num_ctx < 1:
            raise ValueError("workers and num_ctx must be positive")
        self.workers, self.num_ctx = workers, num_ctx
        self.chat_url = chat_url.rstrip("/") if chat_url else None
        self.calls = self.hits = 0
        self.active_requests = self.max_active_requests = 0
        self._local = threading.local()
        self._lock = threading.Lock()

    @property
    def session(self):
        if not hasattr(self._local, "session"):
            self._local.session = requests.Session()
            self._local.session.trust_env = False
        return self._local.session

    def cache_hit(self, count=1):
        with self._lock:
            self.hits += count

    def request(self, endpoint, body):
        start = time.monotonic()
        request_url = self.url + endpoint
        request_body = body
        external_chat = endpoint == "/api/chat" and self.chat_url is not None
        if external_chat:
            request_url = self.chat_url + "/v1/chat/completions"
            options = body["options"]
            request_body = {"model":body["model"],"messages":body["messages"],"stream":False,
                            "temperature":options["temperature"],"max_tokens":options["num_predict"],"seed":options["seed"],
                            "top_k":20,"top_p":.95,"min_p":0,"presence_penalty":1.5,"repeat_penalty":1.0,"n_keep":4,
                            "chat_template_kwargs":{"enable_thinking":False},"cache_prompt":True,
                            "response_format":{"type":"json_schema","json_schema":{"name":"evidence","schema":body["format"]}}}
        with self._lock:
            self.active_requests += 1
            self.max_active_requests = max(self.max_active_requests, self.active_requests)
        try:
            for attempt in range(4):
                try:
                    r = self.session.post(request_url, json=request_body, timeout=(10, 600))
                    if r.status_code in {429, 500, 502, 503, 504} and attempt < 3:
                        time.sleep(.5 * 2**attempt)
                        continue
                    r.raise_for_status()
                    response = r.json()
                    if external_chat:
                        raw_response = response
                        choice = response["choices"][0]
                        response = {"message":choice["message"],"done_reason":choice["finish_reason"],
                                    "eval_count":response.get("usage",{}).get("completion_tokens"),
                                    "prompt_eval_count":response.get("usage",{}).get("prompt_tokens"),
                                    "backend":"llama-server","raw_backend_response":raw_response}
                    break
                except (requests.ConnectionError, requests.Timeout):
                    if attempt == 3:
                        raise
                    time.sleep(.5 * 2**attempt)
            self.cache.mkdir(parents=True, exist_ok=True)
            with self._lock:
                self.calls += 1
                with (self.cache / "requests.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"endpoint": endpoint, "model": body["model"], "seconds": round(time.monotonic()-start, 2),
                                        "eval_count": response.get("eval_count"), "prompt_eval_count": response.get("prompt_eval_count"),
                                        "done_reason": response.get("done_reason"), "attempts":attempt+1,
                                        "request_url":request_url}) + "\n")
        finally:
            with self._lock:
                self.active_requests -= 1
        return response

    def unload(self, model):
        if self.chat_url and model == self.model:
            return {"done_reason":"external_text_server_remains_loaded"}
        return self.request("/api/generate", {"model": model, "keep_alive": 0})

    def runtime_info(self):
        return {"workers":self.workers,"num_ctx":self.num_ctx,"ollama_url":self.url,
                "chat_url":self.chat_url,"chat_backend":"llama-server" if self.chat_url else "ollama",
                "server_info":getattr(self,"server_info",{}),
                "cache_directory":str(self.cache),"max_active_http_requests":self.max_active_requests}

    def wait_ready(self, timeout=300):
        if not self.chat_url:
            return
        started=time.monotonic()
        while time.monotonic()-started < timeout:
            try:
                response=self.session.get(self.chat_url+"/health",timeout=(5,10))
                if response.status_code == 200:
                    props=self.session.get(self.chat_url+"/props",timeout=(5,10))
                    props.raise_for_status()
                    settings=props.json()
                    slots=settings.get("total_slots",0)
                    if slots < self.workers:
                        raise RuntimeError(f"text server has {slots} slots, but client requested {self.workers} workers")
                    context=settings.get("default_generation_settings",{}).get("n_ctx",0)
                    if context < self.num_ctx:
                        raise RuntimeError(f"text server has {context} tokens per slot, but client needs {self.num_ctx}")
                    self.server_info={"total_slots":slots,"context_per_slot":context}
                    return
                if response.status_code != 503:
                    response.raise_for_status()
            except (requests.ConnectionError,requests.Timeout):
                pass
            time.sleep(2)
        raise TimeoutError("Text inference server did not become ready")

    def chat(self, role: str, instructions: str, payload: dict, schema: type[BaseModel], validate=None, schema_override=None, compatible_schema=None):
        output_schema=schema_override or schema.model_json_schema()
        key_data = [VERSION, self.model, role, instructions, payload, output_schema]
        runtime_key = {"chat_backend":"llama-server","sampling":"qwen35-default-v1","num_ctx":self.num_ctx} if self.chat_url else ({"num_ctx":self.num_ctx} if self.num_ctx != 12288 else None)
        if runtime_key:
            key_data.append(runtime_key)
        key = hashlib.sha256(json.dumps(key_data, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        path = self.cache / role / (key + ".json")
        # Only an explicitly supplied older schema for identical semantics may be migrated.
        # Inputs, instructions, model and version remain part of both hashes; validation is mandatory.
        if not path.exists() and compatible_schema is not None:
            schemas=compatible_schema if isinstance(compatible_schema,list) else [compatible_schema]
            for earlier_schema in schemas:
                prior_data = [VERSION,self.model,role,instructions,payload,earlier_schema]
                if runtime_key:
                    prior_data.append(runtime_key)
                prior_key=hashlib.sha256(json.dumps(prior_data,
                                                  ensure_ascii=False,sort_keys=True).encode()).hexdigest()
                prior=self.cache/role/(prior_key+".json")
                if not prior.exists():
                    continue
                cached=json.loads(prior.read_text(encoding="utf-8"))
                try:
                    proposed=schema.model_validate(cached["result"])
                    if validate:
                        validate(proposed)
                except (ValueError,KeyError):
                    pass  # Invalid earlier output must reach the model under the constrained schema.
                else:
                    dump(path,{**cached,"result":proposed.model_dump(),"revalidated_from":str(prior),
                               "schema_migration":"仅收窄输出约束；同输入同指令的旧响应通过当前完整校验后复用"})
                    break
        if path.exists():
            result = schema.model_validate(json.loads(path.read_text(encoding="utf-8"))["result"])
            if validate:
                validate(result)
            self.cache_hit()
            return result
        self.cache.mkdir(parents=True, exist_ok=True)
        messages = [{"role": "system", "content": instructions},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
        for attempt in range(3):
            response = self.request("/api/chat", {"model": self.model, "messages": messages,
                "format": output_schema, "stream": False, "think": False, "keep_alive": "30m",
                "options": {"temperature": 0, "num_ctx": self.num_ctx, "num_predict": 5000, "seed": 42}})
            try:
                if response.get("done_reason") == "length":
                    raise ValueError("模型输出被截断")
                result = schema.model_validate_json(response["message"]["content"])
                if validate:
                    validate(result)
                dump(path, {"role": role, "model": self.model, "payload": payload, "result": result.model_dump(), "raw_response": response})
                return result
            except (ValueError, KeyError) as e:
                dump(self.cache / "failed" / f"{key}-{attempt}.json", {"role": role, "error": str(e), "response": response})
                if attempt == 2:
                    raise RuntimeError(f"{role} validated output failed: {e}") from e
                # Keep the evidence intact; appending failed long outputs can evict the original news.
                messages = [{"role":"system","content":instructions},
                            {"role":"user","content":json.dumps(payload,ensure_ascii=False)},
                            {"role":"user","content":f"上次输出校验失败，请依据上述原始材料重新输出完整JSON：{e}"}]

    def embeddings(self, texts: list[str]):
        key = hashlib.sha256(json.dumps([self.embedding_model, texts], ensure_ascii=False).encode()).hexdigest()
        path = self.cache / "embeddings" / (key + ".json")
        if path.exists():
            self.cache_hit()
            vectors=json.loads(path.read_text(encoding="utf-8"))["vectors"]
            self.seed_embeddings(texts,vectors)
            return vectors
        self.cache.mkdir(parents=True, exist_ok=True)
        paths=[self.embedding_path(text) for text in texts]
        vectors=[json.loads(p.read_text(encoding="utf-8"))["vector"] if p.exists() else None for p in paths]
        missing=[i for i,v in enumerate(vectors) if v is None]
        self.cache_hit(len(texts)-len(missing))
        for offset in range(0,len(missing),24):
            indices=missing[offset:offset+24]
            response = self.request("/api/embed", {"model": self.embedding_model,
                "input": [texts[i] for i in indices], "truncate": False, "keep_alive": "30m"})
            if len(response["embeddings"])!=len(indices):
                raise ValueError("embedding response count mismatch")
            for i,v in zip(indices,response["embeddings"]):
                vectors[i]=v
        if any(v is None for v in vectors):
            raise ValueError("embedding count mismatch")
        self.seed_embeddings(texts,vectors)
        dump(path, {"model": self.embedding_model, "vectors": vectors})
        return vectors

    def embedding_path(self,text):
        key=hashlib.sha256(json.dumps([self.embedding_model,text],ensure_ascii=False).encode()).hexdigest()
        return self.cache/"embedding_documents"/(key+".json")

    def seed_embeddings(self,texts,vectors):
        for text,vector in zip(texts,vectors):
            path=self.embedding_path(text)
            if not path.exists():
                dump(path,{"model":self.embedding_model,"text":text,"vector":vector})
