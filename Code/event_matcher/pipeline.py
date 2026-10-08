import json
from pathlib import Path

from .agents import extract_sources
from .data import load_radar, load_texts
from .ollama_client import OllamaClient, dump


def prepare_day(args):
    input_dir = args.input.resolve()
    output = args.output.resolve() / input_dir.name
    if input_dir == output or input_dir in output.parents:
        raise ValueError("输出目录不能位于原始日期输入内")
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "运行记录" / "模型缓存"
    cache_run = getattr(args, "cache_run", None)
    if cache_run:
        if Path(cache_run).name != cache_run or cache_run in {".", ".."}:
            raise ValueError("cache-run must be a directory name")
        cache = cache / cache_run
    client = OllamaClient(cache, args.model, args.embedding_model, args.ollama_url,
                          getattr(args,"workers",1),getattr(args,"num_ctx",12288),getattr(args,"chat_url",None))
    client.wait_ready()
    sources = load_texts(input_dir)
    radars, radar_manifest = load_radar(input_dir)
    dump(output / "运行记录" / "原始文件清单.json", {"texts": [{k:s[k] for k in ["id", "kind", "path", "sha256"]} for s in sources], "radar": radar_manifest})
    dump(output / "运行记录" / "雷达候选活动.json", radars)
    print(f"[input] {len(sources)} texts; {radar_manifest['rows']} radar rows; {len(radars)} candidate activities", flush=True)
    return {"args":args,"output":output,"client":client,"sources":sources,"radars":radars,"radar_manifest":radar_manifest}


def extract_day(day):
    texts = extract_sources(day["client"], day["sources"], lambda data: dump(day["output"] / "运行记录" / "文本抽取进度.json", data))
    dump(day["output"] / "运行记录" / "文本抽取.json", texts)
    day["texts"] = texts


def finish_day(day, prepared_index=None):
    from .matching import match
    from .export import export_results
    client, output, args = day["client"], day["output"], day["args"]
    groups, vector_manifest = match(client, day["texts"], day["radars"], output, prepared_index)
    manifest = export_results(groups, day["texts"], day["radars"], day["radar_manifest"], output, args, vector_manifest, client)
    concise=dict(manifest)
    concise["vector_index"]={k:v for k,v in manifest["vector_index"].items() if k!="ids"}
    concise["code_provenance"]={k:v for k,v in manifest["code_provenance"].items() if k!="files"}
    print(json.dumps(concise, ensure_ascii=False, indent=2), flush=True)
    return manifest


def run(args):
    day = prepare_day(args)
    extract_day(day)
    client = day["client"]
    if args.stage == "extract":
        print(f"[done extract] HTTP={client.calls}, cache={client.hits}", flush=True)
        return
    return finish_day(day)
