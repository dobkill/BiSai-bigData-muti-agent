import json
from pathlib import Path

from .agents import extract_sources
from .data import load_radar, load_texts
from .ollama_client import OllamaClient, dump


def run(args):
    input_dir = args.input.resolve()
    output = args.output.resolve() / input_dir.name
    if input_dir == output or input_dir in output.parents:
        raise ValueError("输出目录不能位于原始日期输入内")
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "运行记录" / "模型缓存"
    client = OllamaClient(cache, args.model, args.embedding_model, args.ollama_url)
    sources = load_texts(input_dir)
    radars, radar_manifest = load_radar(input_dir)
    dump(output / "运行记录" / "原始文件清单.json", {"texts": [{k:s[k] for k in ["id", "kind", "path", "sha256"]} for s in sources], "radar": radar_manifest})
    dump(output / "运行记录" / "雷达候选活动.json", radars)
    print(f"[input] {len(sources)} texts; {radar_manifest['rows']} radar rows; {len(radars)} candidate activities", flush=True)
    texts = extract_sources(client, sources, lambda data: dump(output / "运行记录" / "文本抽取进度.json", data))
    dump(output / "运行记录" / "文本抽取.json", texts)
    if args.stage == "extract":
        print(f"[done extract] HTTP={client.calls}, cache={client.hits}", flush=True)
        return
    from .matching import match
    from .export import export_results
    groups, vector_manifest = match(client, texts, radars, output)
    manifest = export_results(groups, texts, radars, radar_manifest, output, args, vector_manifest, client)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
