import hashlib
import json
import time
from pathlib import Path

import requests
from pydantic import BaseModel

VERSION = "matcher-v1.0"


def dump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


class OllamaClient:
    def __init__(self, cache: Path, model: str, embedding_model: str, url: str):
        self.cache, self.model, self.embedding_model, self.url = cache, model, embedding_model, url.rstrip("/")
        self.calls = self.hits = 0
        self.session = requests.Session()
        self.session.trust_env = False

    def request(self, endpoint, body):
        start = time.monotonic()
        r = self.session.post(self.url + endpoint, json=body, timeout=(10, 600))
        r.raise_for_status()
        response = r.json()
        self.calls += 1
        with (self.cache / "requests.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"endpoint": endpoint, "model": body["model"], "seconds": round(time.monotonic()-start, 2),
                                "eval_count": response.get("eval_count"), "done_reason": response.get("done_reason")}) + "\n")
        return response

    def chat(self, role: str, instructions: str, payload: dict, schema: type[BaseModel], validate=None, schema_override=None):
        output_schema=schema_override or schema.model_json_schema()
        key_data = [VERSION, self.model, role, instructions, payload, output_schema]
        key = hashlib.sha256(json.dumps(key_data, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        path = self.cache / role / (key + ".json")
        if path.exists():
            result = schema.model_validate(json.loads(path.read_text(encoding="utf-8"))["result"])
            if validate:
                validate(result)
            self.hits += 1
            return result
        self.cache.mkdir(parents=True, exist_ok=True)
        messages = [{"role": "system", "content": instructions},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
        for attempt in range(3):
            response = self.request("/api/chat", {"model": self.model, "messages": messages,
                "format": output_schema, "stream": False, "think": False, "keep_alive": "30m",
                "options": {"temperature": 0, "num_ctx": 12288, "num_predict": 5000, "seed": 42}})
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
                messages += [{"role": "assistant", "content": response["message"]["content"]},
                             {"role": "user", "content": f"输出校验失败，请重新输出完整结果，修正这些问题：{e}"}]

    def embeddings(self, texts: list[str]):
        key = hashlib.sha256(json.dumps([self.embedding_model, texts], ensure_ascii=False).encode()).hexdigest()
        path = self.cache / "embeddings" / (key + ".json")
        if path.exists():
            self.hits += 1
            vectors=json.loads(path.read_text(encoding="utf-8"))["vectors"]
            self.seed_embeddings(texts,vectors)
            return vectors
        self.cache.mkdir(parents=True, exist_ok=True)
        paths=[self.embedding_path(text) for text in texts]
        vectors=[json.loads(p.read_text(encoding="utf-8"))["vector"] if p.exists() else None for p in paths]
        missing=[i for i,v in enumerate(vectors) if v is None]
        self.hits+=len(texts)-len(missing)
        for offset in range(0,len(missing),24):
            indices=missing[offset:offset+24]
            response = self.request("/api/embed", {"model": self.embedding_model,
                "input": [texts[i] for i in indices], "truncate": False, "keep_alive": "0"})
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
