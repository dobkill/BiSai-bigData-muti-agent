"""从已完成的单日试跑生成中文阅读说明和逐字段模型对照，不调用模型。"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

from fusion_agent.models import DayResult, FACT_FIELDS
from fusion_agent.review_export import FIELD_NAMES, SOURCE_NAMES, csv_text, table_cell, write_bytes
from fusion_agent.utils import parse_date, write_json, write_text


def fact_signature(facts):
    return {key: sorted(values, key=str) for key, values in facts.items() if values}


def show(values):
    return "；".join(str(v) for v in values) if values else "未给出"


def summarize(trial_dir: Path, baseline_dir: Path, day: str) -> dict:
    trial_dir = trial_dir.resolve()
    manifest = json.loads((trial_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["processed_dates"] != [day] or not manifest["model_enabled"]:
        raise ValueError("需要已完成且只包含指定日期的模型试跑")
    trial = DayResult.model_validate_json((trial_dir / "daily" / f"{day}.json").read_text(encoding="utf-8"))
    baseline = DayResult.model_validate_json((baseline_dir / f"{day}.json").read_text(encoding="utf-8"))
    text_obs = [o for o in trial.observations if o.source != "radar"]
    modeled = [o for o in text_obs if o.metadata.get("extractor") in {"llm_validated+rules", "llm_semantic"}]
    if len(modeled) != len(text_obs) or manifest["llm_fallbacks"]:
        raise ValueError("部分文本未成功使用模型，请先查看运行日志")
    if {o.id for o in trial.observations} != {o.id for o in baseline.observations}:
        raise ValueError("试跑与基线覆盖的原始观测不同，不能直接比较")
    rows = []
    warnings = Counter()
    claims_count = 0
    added_documents = []
    added_fields = 0
    for obs in modeled:
        raw = obs.metadata["llm_extraction"]
        before = obs.metadata.get("rule_candidates", obs.metadata.get("rule_facts", {}))
        validated = obs.metadata.get("llm_validated_facts", obs.facts)
        has_added = False
        warnings.update(w for w in obs.warnings if w.startswith("llm_"))
        for field in FACT_FIELDS:
            claims = ([c for a in raw["activities"] for c in a.get(field, [])]
                      if "activities" in raw else raw.get(field, []))
            claims_count += len(claims)
            additions = [v for v in obs.facts.get(field, []) if v not in before.get(field, [])]
            if additions:
                has_added = True
                added_fields += 1
            hints = [w for w in obs.warnings if w.startswith("llm_") and w.endswith(":" + field)]
            if field == "platform" and obs.metadata.get("llm_ignored_descriptors"):
                hints.append("类别泛称保留为描述，不另计第二个平台")
            rows.append([day, SOURCE_NAMES[obs.source], obs.path, FIELD_NAMES[field],
                         show(before.get(field, [])), show([c["value"] for c in claims]),
                         "\n".join(c["quote"] for c in claims) or ("按区域派生" if field == "direction" else "未给出"),
                         show(validated.get(field, [])), show(obs.facts.get(field, [])), show(additions),
                         "；".join(hints)])
        if has_added:
            added_documents.append(obs.path)
    write_bytes(trial_dir / "模型抽取对照.csv", csv_text(
        ["日期", "来源", "原始文件", "字段", "规则抽取", "模型声明值", "模型原文引用",
         "验证后模型值", "最终抽取", "新增值", "校验提示"], rows).encode("utf-8"))
    old_observations = {o.id: o for o in baseline.observations}
    changed_observations = [o.path for o in trial.observations
                            if fact_signature(o.facts) != fact_signature(old_observations[o.id].facts)]
    old_events = {e.id: e for e in baseline.events}
    new_events = {e.id: e for e in trial.events}
    changed_events = sorted(key for key in old_events.keys() | new_events.keys()
                            if key not in old_events or key not in new_events
                            or old_events[key].model_dump() != new_events[key].model_dump())
    seconds = sum(t["seconds"] for t in trial.trace if t["agent"] == "extraction")
    inference_path = trial_dir / "inference_run.json"
    inference = manifest
    if inference_path.exists():
        saved = json.loads(inference_path.read_text(encoding="utf-8"))
        if all(saved.get(key) == manifest.get(key) for key in ("processed_dates", "model_provider", "model_name")):
            inference = saved
    elif manifest["llm_requests"]:
        inference = {**manifest, "extraction_seconds": seconds}
        write_json(inference_path, inference)
    initial_seconds = inference.get("extraction_seconds")
    initial_summary = trial_dir / "initial_summary.json"
    if initial_seconds is None and initial_summary.exists():
        saved = json.loads(initial_summary.read_text(encoding="utf-8"))
        if saved.get("date") == day and saved.get("model") == manifest["model_name"]:
            initial_seconds = saved["extraction_seconds"]
    if initial_seconds is None:
        initial_seconds = seconds
    ignored_descriptors = sum(len(o.metadata.get("llm_ignored_descriptors", [])) for o in modeled)
    counts = manifest["review_export"]["counts"]
    summary = {
        "date": day, "model": manifest["model_name"], "provider": manifest["model_provider"],
        "processed_dates": manifest["processed_dates"], "lookback_days": manifest["lookback_days"],
        "text_counts": dict(Counter(o.source for o in text_obs)), "model_processed_texts": len(modeled),
        "radar_batches": sum(o.source == "radar" for o in trial.observations),
        "radar_rows": sum(len(o.metadata["raw_rows"]) for o in trial.observations if o.source == "radar"),
        "initial_inference_http_requests": inference["llm_requests"],
        "latest_http_requests": manifest["llm_requests"], "latest_llm_cache_hits": manifest["llm_cache_hits"],
        "fallbacks": manifest["llm_fallbacks"], "initial_inference_extraction_seconds": initial_seconds,
        "latest_extraction_seconds": seconds, "ignored_platform_descriptors": ignored_descriptors,
        "model_claims": claims_count, "model_claim_validation_warnings": dict(warnings),
        "documents_with_added_facts": added_documents, "added_fields": added_fields,
        "changed_observations_vs_baseline": changed_observations, "changed_event_ids_vs_baseline": changed_events,
        "baseline_candidates": len(baseline.candidates), "trial_candidates": len(trial.candidates),
        "baseline_accepted": sum(e.accepted for e in baseline.events),
        "trial_accepted": sum(e.accepted for e in trial.events), "review_counts": counts,
    }
    write_json(trial_dir / "trial_summary.json", summary)
    index = list(csv.DictReader((trial_dir / "review" / "索引.csv").read_text(encoding="utf-8-sig").splitlines()))
    lines = [f"# {day} 本地 Ollama 单日试跑", "", "## 实际运行情况", "",
             f"模型：**{manifest['model_name']}**，服务：本地 Ollama。只读取 `{day}`，没有加载其他日期。",
             "", f"- 新闻 {summary['text_counts'].get('open_source', 0)} 篇；舆论 {summary['text_counts'].get('opinion', 0)} 篇。",
             f"- 雷达 {summary['radar_batches']} 个批次，共 {summary['radar_rows']} 条原始记录，由程序解析。",
             f"- 成功采用模型抽取的文本：{len(modeled)} 篇。首次完整推理实际 HTTP 请求 **{inference['llm_requests']}** 次；"
             f"规则回退 **{manifest['llm_fallbacks']}** 次。",
             f"- 首次抽取阶段约 {initial_seconds / 60:.1f} 分钟；首次连接测试另有一次小型模型调用。",
             f"- 最近一次运行：实际 HTTP 请求 {manifest['llm_requests']} 次、请求缓存命中 {manifest['llm_cache_hits']} 次，"
             f"抽取阶段 {seconds:.1f} 秒。首次请求记录在 `inference_run.json`，最新运行在 `manifest.json`。",
             "- 模型输出带原文引用的结构化事实；三源关联、字段核验和日报组织由程序执行。",
             "", "## 与已有规则基线比较", "", "| 项目 | 规则基线 | 本次 Ollama |", "| --- | --- | --- |",
             f"| 候选分组 | {len(baseline.candidates)} | {len(trial.candidates)} |",
             f"| 日报接纳 | {summary['baseline_accepted']} | {summary['trial_accepted']} |",
             "", f"本次分组：已核验 **{counts.get('已核验', 0)}**；待人工确认 **{counts.get('待人工确认', 0)}**；"
             f"未匹配 **{counts.get('未匹配', 0)}**。",
             "", f"模型共声明 {claims_count} 条字段值；{sum(warnings.values())} 条未通过引用或字段校验，"
             "均未直接作为事实写入。缺少连续原文引用、或无法从引用验证的声明会被过滤；规则已识别的显式冲突会保留。",
             "", f"另外合并了 **{ignored_descriptors}** 条平台类型泛称：例如同篇中的“歼-15”和“舰载战斗机”"
             "按具体型号与描述处理，不会误判为两个平台。真实的多个型号仍保留多平台标记。",
             "", f"相对同次抽取的规则结果，模型新增字段涉及 **{len(added_documents)}** 篇文本、**{added_fields}** 个字段。"
             f"相对已有规则基线，最终事实不同的观测有 **{len(changed_observations)}** 项，事件裁决不同的分组有 **{len(changed_events)}** 项。"]
    if not changed_observations and not changed_events:
        lines += ["", "这次最终关联与核验结果和规则基线一致。模型主要提供了更短的字段级原文引用；"
                  "当前策略仍要求用程序独立验证引用，且不会让模型删除规则发现的冲突，因此没有证据表明这次试跑提高了准确率。"]
    lines += ["", "[模型抽取对照.csv](模型抽取对照.csv) 每行对应一篇文本的一个字段，可用 Excel/WPS 筛选"
              "“新增值”或“校验提示”，并对照规则值、模型声明、原文引用、验证后值与最终值。",
              "", "## 建议先打开这些资料", "", "| 状态 | 示例 | 需要关注 |", "| --- | --- | --- |"]
    for status in ("已核验", "待人工确认", "未匹配"):
        options = [r for r in index if r["状态"] == status]
        if options:
            row = next((r for r in options if "数量" in r["需要确认的问题"]), options[0])
            title = table_cell(row["国家地区"] + " " + row["平台"])
            link = "review/" + row["分组目录"] + "/关联说明.md"
            reason = table_cell(row["需要确认的问题"] or "核对三源字段表与原文是否对应")
            lines.append(f"| {status} | [{title}](<{link}>) | {reason} |")
    lines += ["", "阅读入口：[分组导航](review/README.md)、[总索引](review/索引.csv)、"
              f"[当天日报](reports/{day}.md)。每组的 `人工复核.csv` 可以填写人工判断，意见目前不会自动重算。",
              "", "## 如何复跑", "", "在项目根目录执行：", "", "```powershell",
              f".\\Code\\scripts\\run.cmd --date {day} --single-day --use-llm --strict-llm --output-dir Code\\outputs\\ollama_trial",
              "& D:\\APP\\miniconda\\envs\\multimodal-agent\\python.exe .\\Code\\scripts\\summarize_trial.py --date " + day,
              "```", "", "复跑会复用缓存，实际 HTTP 请求可能变为 0；查看 `manifest.json` 区分请求与缓存。"
              "`--force` 只重算日结果，仍复用模型请求缓存。", "",
              "只有一天的数据，日报不具备跨日趋势依据。接纳数量和匹配分数都不是比赛准确率；仍需人工核对或标准答案评测。"]
    if (trial_dir / "export_validation.json").exists():
        lines += ["", "副本核对记录：[export_validation.json](export_validation.json)，包含文本逐字节比对、"
                  "雷达子表原始字段/行号检查与资料覆盖数量。"]
    write_text(trial_dir / "试跑说明.md", "\n".join(lines) + "\n")
    return summary


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    code_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--trial-dir", type=Path, default=code_root / "outputs" / "ollama_trial")
    parser.add_argument("--baseline-dir", type=Path, default=code_root / "outputs" / "daily")
    args = parser.parse_args()
    print(json.dumps(summarize(args.trial_dir, args.baseline_dir, parse_date(args.date)), ensure_ascii=False, indent=2))
