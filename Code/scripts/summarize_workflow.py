"""Compare an actual workflow trial against its saved baseline; no inference."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Code/src"))
from fusion_agent.config import load_settings
from fusion_agent.models import DayResult
from fusion_agent.utils import write_json, write_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--date", default="2025-05-01")
    args = parser.parse_args()
    out = args.analysis_dir.resolve()
    before = DayResult.model_validate_json((out / "baseline.json").read_text(encoding="utf-8"))
    after = DayResult.model_validate_json((out / "daily" / (args.date + ".json")).read_text(encoding="utf-8"))
    old_events, old_groups = {e.id: e for e in before.events}, {c.id: c for c in before.candidates}
    changes = []
    for candidate in after.candidates:
        old = old_groups[candidate.id]
        previous = old_events[candidate.id]
        event = next(e for e in after.events if e.id == candidate.id)
        if (old.observation_ids != candidate.observation_ids or previous.classification != event.classification
                or previous.accepted != event.accepted or previous.verification_reasons != event.verification_reasons):
            changes.append({"id": candidate.id, "news_number": candidate.assignment["news_number"],
                            "before_opinion": old.assignment.get("opinion_number"),
                            "after_opinion": candidate.assignment.get("opinion_number"),
                            "before_classification": previous.classification, "after_classification": event.classification,
                            "before_accepted": previous.accepted, "after_accepted": event.accepted,
                            "before_reasons": previous.verification_reasons, "after_reasons": event.verification_reasons,
                            "association": event.semantic_review.get("response", {}).get("association"),
                            "fact_statuses": {k: v.status for k, v in event.verdicts.items()},
                            "recovery": candidate.assignment.get("workflow", {}).get("recovery", [])})
    def metrics(day):
        return {"groups": len(day.candidates), "paired_opinions": sum(c.assignment.get("opinion_id") is not None for c in day.candidates),
                "classification": dict(Counter(e.classification for e in day.events)),
                "accepted": sum(e.accepted for e in day.events), "unassigned": len(day.unassigned_observation_ids)}
    old_observations = {o.id: o for o in before.observations}
    observation_changes = [{"id": o.id, "source": o.source, "path": o.path,
                            "before": old_observations[o.id].facts, "after": o.facts}
                           for o in after.observations if o.facts != old_observations[o.id].facts]
    workflow = next(t for t in after.trace if t["agent"] == "workflow")
    source_review = next((t for t in after.trace if t["agent"] == "source_completeness"), {})
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    settings = load_settings(ROOT / "Code/config.toml")
    saved = json.loads((out / "preservation_before.json").read_text(encoding="utf-8"))
    original_dir = settings.paths.data_dir / (args.date[:4] + "." + str(int(args.date[5:7])) + "." + str(int(args.date[8:10])))
    formal_review = settings.paths.review_dir / args.date
    preserved = {"original_files": len(saved["originals"]), "originals_unchanged": all(
        hashlib.sha256((original_dir / name).read_bytes()).hexdigest() == value for name, value in saved["originals"].items()),
        "human_review_files": len(saved["reviews"]), "human_reviews_unchanged": all(
        hashlib.sha256((formal_review / name).read_bytes()).hexdigest() == value for name, value in saved["reviews"].items()),
        "observation_ids_unchanged": set(old_observations) == {o.id for o in after.observations},
        "raw_model_extractions_unchanged": all(o.metadata.get("llm_extraction") == old_observations[o.id].metadata.get("llm_extraction")
                                               for o in after.observations),
        "raw_radar_rows_unchanged": all(o.metadata.get("raw_rows") == old_observations[o.id].metadata.get("raw_rows")
                                        for o in after.observations if o.source == "radar"),
        "non_entity_facts_unchanged": all({f: v for f, v in o.facts.items() if f != "platform"} ==
                                         {f: v for f, v in old_observations[o.id].facts.items() if f != "platform"}
                                         for o in after.observations)}
    target = next(e for e in after.events if e.id == "03f0424e704a90e8d1ff")
    result = {"date": args.date, "input_mode": manifest.get("input_mode"), "before": metrics(before), "after": metrics(after), "changes": changes,
              "observation_changes": observation_changes,
              "source_completeness": source_review,
              "entity_type_resolutions": [{"observation_id": o.id, **entry} for o in after.observations
                                          for entry in o.metadata.get("entity_type_resolutions", [])],
              "round_summaries": workflow["round_summaries"], "recovered_groups": workflow["recovered_groups"],
              "verification_reuses": workflow.get("verification_reuses", 0),
              "recovery_proposals": workflow["recovery_proposals"], "stop_reason": workflow["stop_reason"],
              "llm_requests": manifest["llm_requests"], "llm_cache_hits": manifest["llm_cache_hits"],
              "llm_validation_repairs": manifest["llm_validation_repairs"], "preservation": preserved,
              "target_event": {"classification": target.classification, "accepted": target.accepted,
                               "reasons": target.verification_reasons,
                               "support_sources": {f: v.support_sources for f, v in target.verdicts.items()},
                               "code_checks": target.semantic_review.get("code_checks", [])},
              "evidence_guards": dict(Counter(check for e in after.events for check in e.semantic_review.get("code_checks", []))),
              "scope": "Actual local-model run with evidence checks; no ground-truth accuracy claim."}
    write_json(out / "comparison.json", result)
    write_json(out / "workflow_trace.json", workflow)
    lines = ["# Agent 工作流试验", "", f"日期：{args.date}。实际本地模型：{manifest['model_name']}。", "",
             "本次从原正式结果的抽取快照出发，逐条核对原文引用和雷达行，再由本地模型执行新工作流。缺失核心字段及疑似漏抽的国别由源内复核补证；未重新抽取整日材料。输入模式：`" + manifest.get("input_mode", "unknown") + "`。", "",
             "| 指标 | 原正式结果 | 新工作流 |", "| --- | --- | --- |"]
    for label, old, new in (("新闻主组", result["before"]["groups"], result["after"]["groups"]),
                            ("已配舆论", result["before"]["paired_opinions"], result["after"]["paired_opinions"]),
                            ("可信事件", result["before"]["classification"].get("trusted", 0), result["after"]["classification"].get("trusted", 0)),
                            ("不可信事件", result["before"]["classification"].get("untrusted", 0), result["after"]["classification"].get("untrusted", 0)),
                            ("日报接纳", result["before"]["accepted"], result["after"]["accepted"])):
        lines.append(f"| {label} | {old} | {new} |")
    lines += ["", f"目标事件 `03f0424e704a90e8d1ff`：{target.classification}；核验阻断原因：{target.verification_reasons}。", "",
              f"源内复核 {source_review.get('reviewed_sources', 0)} 篇，补回 {source_review.get('added_claims', 0)} 条有原文引用的事实。名称、平台类型修正及新增事实均记录在日结果中，原始模型抽取保持不变。", "",
              "原正式结果与新工作流的差异同时包含源内复核、实体规范化、多路召回的影响。下面分轮记录异常复查带来的额外变化：", "",
              "| 工作流轮次 | 已配舆论 | 可信 | 不可信 | 日报接纳 | 新核验 / 复用 |", "| --- | --- | --- | --- | --- | --- |"]
    for step in workflow["round_summaries"]:
        lines.append(f"| {step['round']} | {step['paired_opinions']} | {step['classification']['trusted']} | {step['classification']['untrusted']} | {step['accepted']} | {step.get('new_verifications', 0)} / {step.get('reused_verifications', 0)} |")
    lines += ["", f"复查提议 {workflow['recovery_proposals']} 个候选，其中最终进入主组的事件 {workflow['recovered_groups']} 个。",
              f"真实 HTTP 请求 {manifest['llm_requests']} 次，模型缓存命中 {manifest['llm_cache_hits']} 次，证据校验修正 {manifest['llm_validation_repairs']} 次。",
              f"证据及所选关联未改变时复用已通过核验的结论，共 {workflow.get('verification_reuses', 0)} 组次；不会通过反复抽样把未知变成可信。",
              "这些计数属于完成的该次运行；此前失败尝试及独立排障请求不在其中。失败尝试记录保存在 attempts 目录。",
              "", "新增流程不会以复查提议代替活动关联或事实核验。搜索到预算上限时保留未确认及实际覆盖范围。",
              "", "这是同一数据集的实际模型试跑，可信数量变化不等于准确率变化；缺少官方真值，不能计算准确率。",
              "", f"原始文件 {preserved['original_files']} 份与人工复核文件 {preserved['human_review_files']} 份 SHA-256 保全：原文 {preserved['originals_unchanged']}，人工意见 {preserved['human_reviews_unchanged']}。",
              "", "阅读入口：[当天分组](review/" + args.date + "/README.md)。",
              "变化、逐轮结果、原始文件与人工复核保全情况见 [comparison.json](comparison.json)；完整状态迁移及复查原文引用见 [workflow_trace.json](workflow_trace.json)。", ""]
    write_text(out / "试验说明.md", "\n".join(lines))
    print(json.dumps({k: result[k] for k in ["before", "after", "target_event", "recovered_groups", "preservation"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
